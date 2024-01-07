# custom numerical llama

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

class CustomNonlinearity(nn.Module):  
    def forward(self, x):  
        return x * torch.sigmoid(x)

class VanillaVAE(nn.Module):
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 64, 256, 1024, 2048, 4096]

        # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv1d(
        #                 in_channels, out_channels=h_dim,
        #                 kernel_size=1, stride=1, padding=0
        #             ),
        #             nn.BatchNorm1d(h_dim),
        #             nn.ReLU()
        #         )
        #     )
        #     in_channels = h_dim
        
        for h_dim in hidden_dims:    
            modules.append(    
                nn.Sequential(    
                    nn.Linear(    
                        in_features=in_channels, out_features=h_dim    
                    ),    
                    nn.LayerNorm(h_dim),    
                    nn.Tanh()    
                )    
            )    
            in_channels = h_dim    

        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []
        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(  
        #             nn.Conv1d(  
        #                 hidden_dims[i],  
        #                 hidden_dims[i + 1],  
        #                 kernel_size=1,  
        #                 stride=1,  
        #                 padding=0,  
        #             ),  
        #             nn.BatchNorm1d(hidden_dims[i + 1]),  
        #             nn.ReLU()  
        #         )  
        #     )
        
        for i in range(len(hidden_dims) - 1):  
            modules.append(  
                nn.Sequential(  
                    nn.Linear(  
                        hidden_dims[i],  
                        hidden_dims[i + 1]  
                    ),  
                    nn.LayerNorm(hidden_dims[i + 1]),  
                    nn.ReLU()  
                )  
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(  
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),  
            nn.LayerNorm(hidden_dims[-1]),  
            CustomNonlinearity(),  
            nn.Linear(hidden_dims[-1], 4),  
            nn.ReLU(),  
        )  

        # self.final_layer = nn.Sequential(  
        #                         nn.Conv1d(  
        #                             hidden_dims[-1],  
        #                             hidden_dims[-1],  
        #                             kernel_size=1,  
        #                             stride=1,  
        #                             padding=0,   
        #                         ),  
        #                         nn.BatchNorm1d(hidden_dims[-1]),  
        #                         nn.ReLU(),  
        #                         nn.Conv1d(  
        #                             hidden_dims[-1], out_channels=4,  
        #                             kernel_size=1, padding=0,  
        #                         ),  
        #                         nn.ReLU(),  
        #                     )  

    
    def nonlinearity(self, x):
        return x * torch.sigmoid(x)
    
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B X L]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)
        # result = result.transpose(1, 2)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the numerical space.
        :param z: (Tensor) [B x l x D]
        :return: (Tensor) [B x l]
        """
        result = self.decoder_input(z)
        # result = result.transpose(1, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    
    def cal_kl_loss(self, mu, log_var, kld_weight):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_weight * kld_loss
    
    
    def loss_function(self,
                      kld_weight=1.0,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # kld_weight = kld_weight
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    
class SvgLlama(LlamaForCausalLM, GenerationMixin):  
    def __init__(self, config, vae_loss_weight=1.0, tokenizer=None, hidden_dims=None, numerical_token=None):  
        super(SvgLlama, self).__init__(config)
        
        self.vae = VanillaVAE(
            in_channels=4,  # 4 as default 
            latent_dim=config.hidden_size,
            hidden_dims=hidden_dims
        )
        self.vae_loss_weight = vae_loss_weight
        self.tokenizer = tokenizer
        self.numerical_token = numerical_token
        self.post_init()
        
        if config.frozen_llm: 
            print("Attention! Part of the parameters are freezed!")
            self.requires_grad_ = False 
            self.svg_embedding.requires_grad_ = True
            self.svg_lm_head.requires_grad_ = True
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    
    
    def forward(self, text_inputs=None, numerical_inputs=None, labels=None, attention_masks=None, **kwargs): 
        """
        input_ids: {
            "text_inputs": b x l, 
            "numerical_inputs": b x l,
            "batch_attn_mask": b x l,
            "batch_labels": b x l,
        }
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