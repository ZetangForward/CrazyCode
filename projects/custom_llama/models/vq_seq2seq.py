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
from transformers.modeling_outputs import Seq2SeqLMOutput


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
        
        
    def forward(self, text_input_ids=None, text_attention_mask=None, svg_tensors=None, svg_padding_mask=None, return_dict=None, **kwargs): 
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

        if not return_dict:
            output = (svg_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=svg_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    
    @property
    def model_device(self):
        return next(self.parameters()).device
    