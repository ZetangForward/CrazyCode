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
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch import Tensor 
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from modelzipper.tutils import *


class VQSVGLlama(LlamaForCausalLM, GenerationMixin):  
    def __init__(self, config, vq_loss_weight=1.0, tokenizer=None, hidden_dims=None, numerical_token=None):  
        super(VQSVGLlama, self).__init__(config)
        
        self.tokenizer = tokenizer
        self.numerical_token = numerical_token
        self.vq_loss_weight = vq_loss_weight

        self.input_adapter = nn.Linear(config.svgcode_hidden_dims, config.hidden_dims)
        self.output_adapter = nn.Linear(config.hidden_dims, config.svgcode_hidden_dims)

        self.post_init()
        
        if config.frozen_llm: 
            print_c("Attention! Part of the parameters are freezed!")
            self.requires_grad_ = False 
            self.input_adapter.requires_grad_ = True
            self.output_adapter.requires_grad_ = True
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    

    def forward(self, text_input_ids=None, text_attention_mask=None, text_labels=None, svg_quantised=None, svg_padding_mask=None, **kwargs): 
        """
            text_input_ids: B x L 
            text_attention_mask: B x L,
            text_labels: B x L,
            svg_quantised: B x L x D,
            svg_padding_mask: B x L 
        """
        
        text_embedding_module = self.base_model.get_input_embeddings()
        input_embeddings = text_embedding_module(text_input_ids)
        
        svg_token_embeddings = self.input_adapter(svg_quantised) # Encode svg tokens
        input_embeddings = torch.cat([input_embeddings, svg_token_embeddings], dim=1) # concate the text embedding and svg token embedding
        
        # FIXME: check the dtype of two tensors 
        attention_mask = torch.cat([text_attention_mask, svg_padding_mask], dim=1) # concate the text attention mask and svg padding mask 

        
        #### Step 3.3: Replace all the numerical positions in text embeddings 
        #### with numerical embeddings 
        for i in range(batch_size):
            numerical_index = 0
            for j in range(seq_len):
                if text_inputs[i, j] == self.numerical_token_id:
                    input_embeddings[i, j, :] = zs[i][numerical_index, :]
                    numerical_index += 1
                if text_inputs[i, j] == self.tokenizer.eos_token_id:
                    assert zs[i].size(0) == numerical_index, "Numerical index not match"
                    break
                
        ### Step 4: pass the text embeddings to the base model        
        outputs = self.model(
            input_ids=None, 
            attention_mask=attention_masks,
            inputs_embeds=input_embeddings, 
            **kwargs
        )
        hidden_states = outputs[0]

        ## Decode numerical information
        ### Step 1: extract all the numerical embeddings
        numerical_res = []
        for i in range(batch_size):
            minibatch_numerical = []
            for j in range(seq_len):
                if text_inputs[i, j] == self.numerical_token_id:
                    minibatch_numerical.append(hidden_states[i, j-1, :].unsqueeze(0))
                if text_inputs[i, j] == self.tokenizer.eos_token_id:
                    break
            numerical_res.append(torch.cat(minibatch_numerical, dim=0)) # List[List[Tensor]]

        ### Step 2: decode numerical embeddings with VAE
        decode_results = [self.vae.decode(z.unsqueeze(0)) for z in numerical_res]  # List[Tensor]
        
        ### Step 3: calculate the reconstruction Loss
        recon_loss = 0
        for i in range(batch_size):
            # golden_numerical_inputs[i] 1 x 4 x l
            golden_numerical = golden_numerical_inputs[i].squeeze(0)
            decode_result = decode_results[i].squeeze(0).transpose(0, 1)
            assert golden_numerical.size() == decode_result.size(), "Numerical input size not match"
            recon_loss += F.mse_loss(decode_result, golden_numerical, reduction="mean") / batch_size
        
        ## Decode textual information
        ### Step 1: extract all the textual embeddings
        text_logits = self.lm_head(hidden_states).float()

        ### Step 2: calculate the LM Loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        total_loss = loss + self.vae_loss_weight * (kl_loss + recon_loss)
        # import pdb; pdb.set_trace()
        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "reconstruction_loss": recon_loss,
            "textual_loss": loss,
            "logits": text_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_length=None, min_length=None, **kwargs):
        
        return super().generate(input_ids, attention_mask, max_length, min_length, **kwargs)
    
    
    @property
    def numerical_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.numerical_token)
    
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