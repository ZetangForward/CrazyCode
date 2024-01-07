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
            svg_quantised: b x l,
            svg_padding_mask: B x L 
        """
        
        ## Encode textual information
        text_embedding_module = self.base_model.get_input_embeddings()
        input_embeddings = text_embedding_module(text_inputs)
        
        ## Encode numerical information
        ### Step 1: compress numerical inputs with torch.log
        numerical_inputs = torch.log(numerical_inputs + 1).to(next(self.parameters()).dtype) # prevent from log(0)
        numerical_inputs = numerical_inputs.unsqueeze(-1).repeat(1, 1, 4)  # construct 4 channels

        ### Step 2: compress numerical inputs with VAE
        numerical_inputs = numerical_inputs.permute(0, 2, 1)  # b x 4 x l
        numerical_embeddings = self.vae.encoder(numerical_inputs)  # b x 4 x l -> b x d_model x l
        numerical_embeddings = numerical_embeddings.permute(0, 2, 1)  # b x d_model x l -> b x l x d_model
        
        
        ### Step 3: calculate mu, sigma, and z with VAE 
        #### Step 3.1: extract all real encoded numerical embedding
        batch_size, seq_len = text_inputs.size()
        numerical_mask = text_inputs == self.numerical_token_id
        numerical_counts = numerical_mask.sum(dim=1).tolist() 
        encoded_numerical_h = [item[:count] for item, count in zip(numerical_embeddings, numerical_counts)]  # List[Tensor]
        mus = [self.vae.fc_mu(item) for item in encoded_numerical_h]
        log_vars = [self.vae.fc_var(item) for item in encoded_numerical_h]
        log_vars = [torch.clamp(item, -30.0, 20.0) for item in log_vars]
        mu_vars = [(mu, log_var) for mu, log_var in zip(mus, log_vars)]  # [List[Tuple[Tensor, Tensor]]]
        zs = [self.vae.reparameterize(mu, log_var) for mu, log_var in mu_vars]  # List[Tensor]
        
        #### Step 3.2: calculate the KL loss
        kl_loss = 0.
        kld_weight = 1e-2 # Account for the minibatch samples from the dataset
        for i in range(batch_size):
            kl_loss += self.vae.cal_kl_loss(mus[i], log_vars[i], kld_weight) / batch_size
            
        #### Step 3.3: construct the golden numerical inputs
        golden_numerical_inputs = [item[:, :count].transpose(0, 1) for item, count in zip(numerical_inputs, numerical_counts)]
        
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