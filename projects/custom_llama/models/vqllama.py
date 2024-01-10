"""
Code for VQ-SVG-LLAMA
"""
import sys
import json
import re 
import random
import torch  
import torch.nn as nn 
from tqdm import tqdm  
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM  
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from modelzipper.tutils import *


class VQSVGLlama(LlamaForCausalLM):  
    def __init__(self, config, vq_loss_weight=2.0, convert_token_weight=1.5, tokenizer=None, svg_end_token_id=None, svg_begin_token_id=None, vqvae=None, codebook_size=16384, compress_level=2, svg_pad_token_id=None):  
        super(VQSVGLlama, self).__init__(config)
        self.tokenizer = tokenizer
        self.svg_end_token_id = svg_end_token_id
        self.svg_begin_token_id = svg_begin_token_id
        self.vq_loss_weight = vq_loss_weight
        self.convert_token_weight = convert_token_weight
        self.codebook_size = codebook_size + 1  # add one for svg end token
        self.svg_end_token_id = codebook_size
        self.compress_level = compress_level
        self.svg_pad_token_id = svg_pad_token_id
        self.vqvae = vqvae
        self.vqvae_embedding = nn.Embedding(codebook_size, config.hidden_size)
        self.vqvae_head = nn.Linear(config.hidden_size, codebook_size)
        self.post_init()
        if config.frozen_llm: 
            print_c("Attention! Part of the parameters are freezed!")
            self.requires_grad_ = False 
            self.input_adapter.requires_grad_ = True
            self.output_adapter.requires_grad_ = True

    def set_svg_pad_token_id(self, svg_pad_token_id):
        self.svg_pad_token_id = svg_pad_token_id
    
    def init_vqvae(self, vqvae):
        self.vqvae = vqvae
        self.vqvae.model.eval()
        self.vqvae.model.requires_grad_ = False

    def add_svg_end_token_id(self, svg_end_token_id):
        self.svg_end_token_id = svg_end_token_id

    def add_svg_begin_token_id(self, svg_begin_token_id):
        self.svg_begin_token_id = svg_begin_token_id

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    
    def create_padding_mask(self, x, pad_token_id):
        padding_mask = torch.ones_like(x, dtype=torch.bool).to(x.device)
        # for idx, sequence in enumerate(x):
        #     pad_positions = (sequence == pad_token_id).nonzero(as_tuple=True)[0]
        #     if pad_positions.numel() > 0:  
        #         first_pad_position = pad_positions[0].item()
        #         padding_mask[idx, first_pad_position:] = False
        return padding_mask
        
        
    def forward(self, text_input_ids=None, text_attention_mask=None, text_labels=None, svg_tensors=None, svg_padding_mask=None, **kwargs): 
        """
            text_input_ids: B x L 
            text_attention_mask: B x L,
            text_labels: B x L,
            svg_tensors: B x L x l_bins,
            svg_padding_mask: B x L,
        """
        bsz = text_input_ids.size(0)
        # embedding text
        text_width = text_input_ids.size(1)
        text_embedding_module = self.base_model.get_input_embeddings()
        input_embeddings = text_embedding_module(text_input_ids)
        
        # quantizied svg tensors with vqvae
        if self.vqvae.model.training: # deepspeed will make vqvae training again
            self.vqvae.model.eval()
            freeze_model(self.vqvae.model)
        svg_token_ids = self.vqvae.model.encode(svg_tensors, start_level=0, end_level=1)
        svg_token_ids = svg_token_ids[0]  # first compress level
        compress_svg_max_length = svg_token_ids.size(1)
        # add svg end token id
        real_svg_lengths = svg_padding_mask.sum(dim=1)

        for i in range(bsz):
            cur_padding_pos = min(real_svg_lengths[i], compress_svg_max_length - 1)
            svg_token_ids[i, cur_padding_pos] = self.svg_end_token_id
            svg_padding_mask[i, cur_padding_pos] = True

        golden_svg_tokens = torch.where(svg_padding_mask, svg_token_ids, -100).to(svg_token_ids.device).long()
        svg_token_embeddings = self.vqvae_embedding(svg_token_ids) # Encode svg tokens
        
        input_embeddings = torch.cat([input_embeddings, svg_token_embeddings], dim=1) # concate the text embedding and svg token embedding
        attention_masks = torch.cat([text_attention_mask, svg_padding_mask], dim=1) # concate the text attention mask and svg padding mask 
                 
        outputs = self.model(
            input_ids=None, 
            attention_mask=attention_masks,
            inputs_embeds=input_embeddings, 
            **kwargs
        )
        hidden_states = outputs[0]

        # text modality, last token is svg special token
        text_logits = self.lm_head(hidden_states[:, :text_width, :]).float()
        # svg modality, first token is svg special token, last token should be svg end token
        svg_pred = self.vqvae_head(hidden_states[:, text_width:, :]).float() 
        
        total_loss, text_loss, svg_loss, convert_token_loss = None, None, None, None
        import pdb; pdb.set_trace()
        if text_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = text_logits[..., :-1, :].contiguous()  # last logits is convert_token logits
            shift_labels = text_labels[..., 1:].contiguous() # last token is convert_token
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = F.cross_entropy(shift_logits, shift_labels)
        
        import pdb; pdb.set_trace()
        if golden_svg_tokens is not None:
            # Shift so that tokens < n predict n
            shift_svg_logits = svg_pred[:, :-1, :].contiguous()
            shift_golden_svg_tokens = golden_svg_tokens[:, 1:].contiguous()
            shift_svg_logits = shift_svg_logits.view(-1, self.codebook_size)
            shift_golden_svg_tokens = shift_golden_svg_tokens.view(-1)
            svg_loss = F.cross_entropy(shift_svg_logits, shift_golden_svg_tokens)

        import pdb; pdb.set_trace()
        if text_labels is not None and svg_token_ids is not None:  # convert token loss is be significant as vocabularies are different
            bsz, _, dim_ = text_logits.size()

            golden_svg_end_token_ids = torch.empty(bsz, 1).fill_(self.svg_end_token_id).to(text_logits.device).long()

            # obtain the last text token logits
            real_text_lengths = text_attention_mask.sum(dim=1)  
            last_text_token_logits = torch.zeros(bsz, 1, dim_).to(text_logits.device)
            
            for i in range(bsz):  # convert the last text token (<svg>) to the first svg token
                last_text_token_logits[i] = self.vqvae_head(hidden_states[i, real_text_lengths[i] - 1][None, ])

            # obtain the last svg token logits
            # real_svg_lengths = svg_padding_mask.sum(dim=1)  
            # last_svg_token_logits = torch.zeros(bsz, dim_).to(text_logits.device)

            # for i in range(bsz):
            #     last_svg_token_logits[i] = self.lm_head(hidden_states[i, text_width + real_svg_lengths[i] - 1])
            import pdb; pdb.set_trace()
            # calculate CE Loss for last text token -> first svg token
            text2svg_loss = F.cross_entropy(
                last_text_token_logits.contiguous().view(-1, self.codebook_size), 
                svg_token_ids[:, 0].contiguous().view(-1), 
                reduction="mean",
                ignore_index=self.svg_pad_token_id,
            )
            
            # # calculate CE Loss for last svg token -> golden text token
            # svg2text_loss = F.cross_entropy(
            #     last_svg_token_logits.contiguous().view(-1, self.config.vocab_size), 
            #     golden_svg_end_token_ids.contiguous().view(-1), 
            #     reduction="mean",
            # )

            convert_token_loss = text2svg_loss

        import pdb; pdb.set_trace()
        if text_loss is not None and svg_loss is not None:  
            total_loss = text_loss + self.vq_loss_weight * svg_loss + self.convert_token_weight * convert_token_loss    

        metrics = dict(
            total_loss=total_loss, text_loss=text_loss, svg_loss=svg_loss, convert_token_loss=convert_token_loss
        )

        if self.training:
            return metrics
        
        return {
            "logits": text_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_length=None, min_length=None, **kwargs):
        return super().generate(input_ids, attention_mask, max_length, min_length, **kwargs)
    
    
    @property
    def model_device(self):
        return next(self.parameters()).device
    
if __name__ == "__main__":
    pretrained_model = LlamaForCausalLM.from_pretrained("/zecheng/model_hub/Llama-2-7b-hf", device_map="auto")
    llamaconfig = LlamaConfig.from_pretrained("/zecheng/model_hub/Llama-2-7b-hf")
    llamaconfig.svg_vocab_size = 1000
    llamaconfig.text_width = 64
    svgllama = SvgLlama(llamaconfig)
    svgllama.load_state_dict(pretrained_model.state_dict(), strict=False)