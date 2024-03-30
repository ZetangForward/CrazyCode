import torch
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
from mamba_ssm.utils.generation import update_graph_cache, InferenceParams, sample
from transformers import MambaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from dataclasses import dataclass
from modelzipper.tutils import *


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    # from .mamba_kernel_fn import mamba_inner_fn  # custom module
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

is_fast_path_available = False  # as cuda_covn1d only support 4 width kernel


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config, layer_idx, use_multi_head=False, multi_head_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size=4
        if multi_head_config is not None:
            self.conv_kernel_size = multi_head_config['kernel_sizes']
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_multi_head = use_multi_head

        if self.use_multi_head:
            self.num_heads = multi_head_config['num_head']
            self.linear_free_multi_head = multi_head_config['linear_free_multi_head']
            assert self.intermediate_size % self.num_heads == 0, "d_model must be divisible by num_heads"
            self.per_head_size = int(self.intermediate_size / self.num_heads)
            
            if not self.linear_free_multi_head:  # use linear to transform the dimension and concatentate them together
                self.up_projector = nn.Linear(self.per_head_size, self.intermediate_size, bias=False)

        self.use_conv_bias = config.use_conv_bias

        if self.use_multi_head:
            self.conv1d = nn.Conv1d(
                in_channels=self.intermediate_size,
                out_channels=self.intermediate_size,
                bias=config.use_conv_bias,
                kernel_size=multi_head_config['kernel_sizes'],
                groups=self.num_heads,  # multi-head (split)
                padding=multi_head_config['kernel_sizes'] - 1,
                dilation=multi_head_config['dilation']
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.intermediate_size,
                out_channels=self.intermediate_size,
                bias=config.use_conv_bias,
                kernel_size=self.conv_kernel_size,
                groups=self.intermediate_size,
                padding=self.conv_kernel_size - 1,
            )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden statesa
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            print_c(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d", "red"
            )

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params=None, extra_kwargs=None):
        
        analysis_mode = False
            
        if extra_kwargs is not None: ## for analysis
            analysis_layers = [0, 11, 23, 35, 47]
            depth = extra_kwargs.get("depth", None)
            save_dir = extra_kwargs.get("save_dir", None)
            ctx_length = extra_kwargs.get("ctx_length", None)
            if ctx_length in [500, 1000, 2000, 4000, 8000, 16000, 32000] and depth in [0.05, 0.55, 1.0]:
                analysis_mode = True
                depth = str(depth).replace(".", "_")    

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:  # inference mode
            hidden_states, gate = projected_states.chunk(2, dim=1)
            
            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)

                if analysis_mode:  # save first hidden state
                    if self.layer_idx in analysis_layers:  # analysis step: save the 1-st hidden_state
                        save_path = os.path.join(f"{save_dir}/ctx_length_{ctx_length}-depth_{depth}/layer_{self.layer_idx}", f"hidden_states_{cache_params.seqlen_offset}.pkl")
                        auto_save_data(hidden_states[:, :, 0].cpu(), save_path)
            
            else:
                if cache_params is not None:
                    
                    if analysis_mode:  # save first hidden state
                        if self.layer_idx in analysis_layers:  # analysis step: save the 1-st hidden_state
                            save_path = os.path.join(f"{save_dir}/ctx_length_{ctx_length}-depth_{depth}/layer_{self.layer_idx}", "first_hidden_state.pkl")
                            auto_save_data(hidden_states.cpu(), save_path)

                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )


            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params=None, extra_kwargs=None):
        
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx]
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state.clone()
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)

                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(  # only save last conv_kernel_size states
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state.clone()
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )


        if selective_scan_fn is not None and selective_state_update is not None:
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            # import pdb; pdb.set_trace()
            # print(hidden_states.shape)
            # print(discrete_time_step.shape)
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
             # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

        else:
            discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
            discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

            # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
            A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
            discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
            discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
            deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            scan_outputs = []
            for i in range(seq_len):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
                scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
                scan_outputs.append(scan_output[:, :, 0])
            scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = (scan_output * self.act(gate))

            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx] = ssm_state.clone()

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        
        return contextualized_states
   

    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(
                hidden_states, 
                cache_params, 
                extra_kwargs=extra_kwargs['extra_kwargs'] if 'extra_kwargs' in extra_kwargs else None
            )
        return self.slow_forward(
            hidden_states, 
            cache_params, 
            extra_kwargs=extra_kwargs['extra_kwargs'] if 'extra_kwargs' in extra_kwargs else None
        )


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx, use_relative_position=False, max_position_embeddings=None, use_abs_position=False, use_multi_head=False, multi_head_config=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx, use_multi_head=use_multi_head, multi_head_config=multi_head_config)
        self.use_relative_position = use_relative_position
        self.max_position_embeddings = max_position_embeddings
        self.use_abs_position = use_abs_position

    def forward(self, hidden_states, cache_params=None, extra_kwargs=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None, multi_head_config=None ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = 4
        if multi_head_config is not None:
            conv_kernel_size = multi_head_config['kernel_sizes']

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


@dataclass
class MambaOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cache_params: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class CustomMambaModel(MambaPreTrainedModel):
    def __init__(self, config, use_relative_position=False, max_position_embeddings=None, use_abs_position=False, use_multi_head=False, multi_head_config=None) -> None:
        super().__init__(config)
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.max_position_embeddings = max_position_embeddings
        self.use_relative_position = use_relative_position
        self.use_abs_position = use_abs_position
        self.multi_head_config = multi_head_config

        if use_abs_position:
            self.wpe = nn.Embedding(max_position_embeddings, config.d_model)
        elif use_relative_position:
            freqs = torch.exp(-np.log(10000.0) / config.d_model * torch.arange(0, config.d_model, 2).float())
            self.register_buffer('freqs', freqs)
            
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    config, 
                    layer_idx=idx,
                    use_multi_head=use_multi_head, 
                    multi_head_config=multi_head_config,
                ) for idx in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
   
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype, multi_head_config= self.multi_head_config
            )

        position_embeds = None
        input_shape = input_ids.shape
        if self.use_relative_position or self.use_abs_position:
            if position_ids is None:
                if cache_params is not None:
                    position_ids = torch.arange(cache_params.seqlen_offset, input_shape[-1] + cache_params.seqlen_offset, dtype=torch.long).to(input_ids.device)
                    position_ids = position_ids.unsqueeze(0)
                else:
                    position_ids = torch.arange(input_shape[-1], dtype=torch.long).to(input_ids.device)
                    position_ids = position_ids.unsqueeze(0)

            if self.use_abs_position:
                position_embeds = self.wpe(position_ids).to(inputs_embeds.dtype)
            elif self.use_relative_position:  # TODO: DEBUG
                freqs = self.freqs.float().to(position_ids.device)
                position_ids = position_ids.float()
                # Add a small constant to avoid division by zero
                freqs += 1e-7
                angles = position_ids.unsqueeze(-1) / freqs.unsqueeze(0)
                position_embeds = torch.cat([angles.sin(), angles.cos()], dim=-1).to(inputs_embeds.dtype)

        hidden_states = inputs_embeds + position_embeds if position_embeds is not None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params, extra_kwargs=extra_kwargs)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, extra_kwargs=extra_kwargs)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class CustomMambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, use_relative_position=False, max_position_embeddings=None, use_abs_position=False, custom_conv1d_configs=None):
        super().__init__(config)
        self.backbone = CustomMambaModel(config, use_relative_position=use_relative_position, max_position_embeddings=max_position_embeddings, use_abs_position=use_abs_position, multi_head_config=custom_conv1d_configs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    def from_pretrained(self, path, dtype, is_from_pytorch_lightning=False):
        if self.dtype != dtype:
            self.to(dtype)
        state_dict = torch.load(path, map_location='cpu')
        if dtype is not None:
            state_dict = {k: v.type(dtype) for k, v in state_dict.items()}
        if is_from_pytorch_lightning:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('model.', '')] = v
            state_dict = new_state_dict
        self.load_state_dict(state_dict, strict=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs["cache_params"]
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params=None, inputs_embeds=None, attention_mask=None, extra_kwargs=None, **kwargs,
    ):  
        """
        extra_kwargs: for analysis like depth and ctx_length
        """
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        model_inputs["extra_kwargs"] = extra_kwargs
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_kwargs=kwargs,  # for analysis like depth and ctx_length
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
