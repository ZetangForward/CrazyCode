"""
Code for VQ-SVG-LLAMA
"""
import sys
import random
import torch  
import torch.nn as nn 
import torch.nn.functional as F
from transformers import T5Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from modelzipper.tutils import *

class VQSVGSeq2SeqModel(T5Model):  
    def __init__(self, config, tokenizer=None, vqvae=None, codebook_size=8192):  
        super(VQSVGSeq2SeqModel, self).__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.codebook_size = codebook_size + 1  # add one for svg end token
        self.svg_end_token_id = codebook_size
        self.vqvae = vqvae
        
        # decoder
        self.vqvae_embedding = nn.Embedding(self.codebook_size, config.hidden_size)
        self.vqvae_head = nn.Linear(config.hidden_size, self.codebook_size)

        self.post_init()
        
        if config.frozen_llm: 
            print_c("Attention! encoder is freezed!")
            self.encoder.requires_grad_ = False # only freeze the encoder
            self.shared.requires_grad_ = False  # freeze the text embedding 
    
    def init_vqvae(self, vqvae):
        self.vqvae = vqvae
        self.vqvae.model.eval()
        for param in self.vqvae.model.parameters():
            param.requires_grad = False

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
        
    def forward(self, text_input_ids=None, text_attention_mask=None, text_labels=None, svg_tensors=None, svg_padding_mask=None, **kwargs): 
        """
            text_input_ids: B x L 
            text_attention_mask: B x L,
            text_labels: B x L,
            svg_tensors: B x L (x l_bins),  depend on offline or online mode
            svg_padding_mask: B x L,
        """
        if self.config.frozen_llm:  # only calculate svg loss when freezen LLM
            self.encoder.requires_grad_ = False # only freeze the encoder
            self.shared.requires_grad_ = False  # freeze the text embedding 
        
        bsz = text_input_ids.size(0)
        
        # embedding text
        text_embeddings = self.shared(text_input_ids)
        
        encoder_outputs = self.encoder(
            input_ids=None,
            attention_mask=text_attention_mask,
            inputs_embeds=text_embeddings,
        )
        
        hidden_states = encoder_outputs[0]
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if svg_tensors is not None:
                svg_tensors = svg_tensors.to(self.decoder.first_device)
            if text_attention_mask is not None:
                text_attention_mask = text_attention_mask.to(self.decoder.first_device)
            if svg_padding_mask is not None:
                svg_padding_mask = svg_padding_mask.to(self.decoder.first_device)
        
        # quantizied svg tensors with vqvae
        if self.vqvae is not None: # online mode
            if self.vqvae.model.training: # deepspeed will make vqvae training again
                self.vqvae.model.eval()
                freeze_model(self.vqvae.model)
            svg_token_ids = self.vqvae.model.encode_no_grad(svg_tensors, start_level=0, end_level=1)
            svg_token_ids = svg_token_ids[0]  # first compress level
        else:  # offline mode
            svg_token_ids = svg_tensors
        
        compress_svg_max_length = svg_token_ids.size(1)
        # add svg end token id
        real_svg_lengths = svg_padding_mask.sum(dim=1)

        for i in range(bsz):
            cur_padding_pos = min(real_svg_lengths[i], compress_svg_max_length - 1)
            svg_token_ids[i, cur_padding_pos] = self.svg_end_token_id
            svg_padding_mask[i, cur_padding_pos] = True

        golden_svg_tokens = torch.where(svg_padding_mask, svg_token_ids, -100).to(svg_token_ids.device).long()
        svg_token_embeddings = self.vqvae_embedding(svg_token_ids) # Encode svg tokens
        svg_padding_mask = svg_padding_mask.to(text_attention_mask.dtype)  # prevent the type error
        # decode svg tokens
        
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=svg_padding_mask,
            inputs_embeds=svg_token_embeddings,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=text_attention_mask,
        )
        
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.vqvae_head = self.vqvae_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.vqvae_head.weight.device)
            
        svg_logits = self.vqvae_head(sequence_output)
        
        loss = None

        if golden_svg_tokens is not None:
            # Shift so that tokens < n predict n
            shift_svg_logits = svg_logits[:, :-1, :].contiguous()
            shift_golden_svg_tokens = golden_svg_tokens[:, 1:].contiguous()
            shift_svg_logits = shift_svg_logits.view(-1, self.codebook_size)
            shift_golden_svg_tokens = shift_golden_svg_tokens.view(-1)
            loss = F.cross_entropy(shift_svg_logits, shift_golden_svg_tokens)

        metrics = {}
        
        if not self.training:
            metrics['eval_loss'] = loss
        else:
            metrics['train_loss'] = loss
        
        return metrics
    
    
    @torch.no_grad()
    def generate(self, text_input_ids=None, text_attention_mask=None, past_key_values=None, max_generate_length=1024, do_sample=False, top_p=0.9, top_k=40, temperature=0.7, num_beams=1) -> List[torch.LongTensor]:
       
        if self.svg_begin_token_id in text_input_ids:
            svg_being_token_pos = text_input_ids == self.svg_begin_token_id
            text_input_ids[svg_being_token_pos] = self.tokenizer.pad_token_id 
            text_attention_mask[svg_being_token_pos] = 0
            
        assert self.svg_begin_token_id not in text_input_ids, "You should not add svg_begin_token_id in text_input_ids, since it will automactically add svg_begin_token_id in the beginning of svg_tensors during the inference!"
        
        batch_size = text_input_ids.size(0)
  
        # initial eos_generated_mask to False for all samples as no sample has generated eos_token yet
        eos_generated_mask = torch.zeros((batch_size, 1), dtype=torch.bool).to(text_input_ids.device)
        
        outputs = self.model(
            input_ids=text_input_ids,
            past_key_values=past_key_values,
            attention_mask=text_attention_mask,
            use_cache=True,
        )
        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        text_width = text_input_ids.size(1)
        
        generated_ids = [text_input_ids[:, i].unsqueeze(1) for i in range(text_width)]
        
        # create svg_begin token id and embeddings
        svg_begin_token_ids = torch.empty(text_input_ids.size(0)).fill_(self.svg_begin_token_id).long().to(last_hidden_state.device)
        
        prev_svg_token_ids = svg_begin_token_ids.unsqueeze(1)
        
        text_embedding_module = self.base_model.get_input_embeddings()
        input_embeddings = text_embedding_module(prev_svg_token_ids)

        for _ in range(max_generate_length - 1):
            outputs = self.model(
                input_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=input_embeddings, 
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            pred_h = self.down_adapter(last_hidden_state)
            pred_logits = self.vqvae_head(pred_h).float()
            
            if do_sample:
                pred_svg_idx = top_k_top_p_sampling(pred_logits[:, -1], top_k=top_k, top_p=top_p, temperature=temperature, num_samples=num_beams).view(batch_size, -1)
            else:
                pred_svg_idx = pred_logits[:, -1].argmax(dim=-1).unsqueeze(1)
            
            # update eos_generated_mask, as some samples generate svg_eos_token
            eos_generated_mask |= (pred_svg_idx == self.svg_end_token_id)  

            # add the predicted svg token embedding to input_embeddings according to pred_svg_idx
            current_step_ids = torch.full((batch_size, 1), self.svg_end_token_id, dtype=torch.long, device=last_hidden_state.device)  
           
            current_step_ids[~eos_generated_mask] = pred_svg_idx[~eos_generated_mask]  
            generated_ids.append(current_step_ids)  
            
            if eos_generated_mask.all():  # all samples have generated eos_token
                break
            
            prev_svg_token_ids = current_step_ids
            input_embeddings = self.vqvae_embedding(prev_svg_token_ids)
            input_embeddings = self.up_adapter(input_embeddings)
            
        generated_ids = torch.cat(generated_ids, dim=1)  # B x gen_length
        generated_mask = ~(generated_ids == self.svg_end_token_id)  # B x gen_length
        post_processed_ids = []  # List[Tensor]
        
        for i in range(batch_size):
            post_processed_ids.append(generated_ids[i, text_width: generated_mask[i].sum()])

        return generated_ids, post_processed_ids
        
    
    @property
    def model_device(self):
        return next(self.parameters()).device
    