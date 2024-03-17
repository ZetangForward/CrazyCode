import torch
import torch.nn as nn
from functools import wraps, partial
from transformers import PreTrainedModel
from typing import Dict
import torch.nn.functional as F


def dict_to(d: dict, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


class LMForwardAPI(nn.Module):
    def __init__(self, model, tokenizer, label_dict: Dict[int, str], device='cuda:0'):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.calibration_probs = None
        self.use_calibration_probs = False
        self.probs_from_results_fn = None
        self.results_args: dict = {}
        self.label_map = {tokenizer.encode(v, add_special_tokens=False)[0]: k for k, v in
                          label_dict.items()}
        self.position_offset = 0

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'LMForwardAPI: set device to {device}')
        self.model = self.model.to(device)
        # if self.past_key_values:
        #     self.past_key_values = self.past_key_values  # will reset device

    def cal_logits(self, inputs, **kwargs):
        self.model.eval()
        inputs = dict_to(inputs, self.device)

        results = self.model(
            input_ids=inputs['input_ids'],
            **kwargs,
        )
        logits = results['logits']
        # find last position before pad tokens
        input_ids = inputs['input_ids']
        eos_token_id: int = self.tokenizer.eos_token_id
        is_not_eos = input_ids != eos_token_id
        prediction_pos = is_not_eos.sum(dim=1) - 1
        is_not_eos = is_not_eos.float()
        # check all eos_tokens are at the end
        assert (is_not_eos[:, :-1] - is_not_eos[:, 1:] >= 0).all()
        # get logits for the last position
        logits = logits[torch.arange(input_ids.shape[0]), prediction_pos, :]
        return logits, results

    def _cal_probs(self, logits):
        interest_index = list(self.label_map.keys())
        logits = logits[:, interest_index]
        probs = F.softmax(logits, dim=-1)
        if self.use_calibration_probs:
            assert self.calibration_probs is not None
            probs = probs / self.calibration_probs
        return probs, logits

    def cal_probs(self, inputs, **kwargs):
        logits, results = self.cal_logits(inputs, **kwargs)
        probs, logits = self._cal_probs(logits)
        return probs, logits, results

    def cal_probs_from_results(self, inputs, results):
        return self.probs_from_results_fn(inputs, results)

    @property
    def past_key_values(self):
        return self._past_key_values

    @past_key_values.setter
    def past_key_values(self, past_key_values):
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            assert isinstance(past_key_values[0], tuple)
            assert len(past_key_values[0]) == 2
            assert isinstance(past_key_values[0][0], torch.Tensor)
            assert past_key_values[0][0].shape[0] == 1
            self._past_key_values = tuple(
                tuple(t.to(self.device) for t in tup) for tup in past_key_values)
        else:
            self._past_key_values = None

    @property
    def use_past_key_values(self):
        return self._use_past_key_values

    @use_past_key_values.setter
    def use_past_key_values(self, use_past_key_values):
        self._use_past_key_values = use_past_key_values

    def get_mask_with_past_key_values(self, mask):
        if self.past_key_values is None:
            raise ValueError('past_key_values is None, please set it first')
        batch_size = mask.shape[0]
        past_key_values_len = self.past_key_values[0][0].shape[2]
        mask = torch.cat(
            [torch.ones(batch_size, past_key_values_len, dtype=torch.bool, device=self.device),
             mask], dim=1)
        return mask

    def get_past_key_values(self, inputs):
        if self.past_key_values is None:
            raise ValueError('past_key_values is None, please set it first')
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = ()
        for layer_key, layer_value in self.past_key_values:
            past_key_values += (
                                   layer_key.expand(batch_size, -1, -1, -1),
                                   layer_value.expand(batch_size, -1, -1, -1)),

        return past_key_values

    @torch.no_grad()
    def forward_no_grad(self, inputs):
        ori_logits, results = self.cal_logits(inputs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        probs_from_results = self.cal_probs_from_results(inputs, results)
        probs_from_results['ori_logits'] = ori_logits
        return probs, probs_from_results

    def forward(self, **kwargs):
        ori_logits, results = self.cal_logits(kwargs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        result = {'probs': probs, 'logits': logits, 'results': results}
        if self.probs_from_results_fn:
            probs_from_results = self.cal_probs_from_results(kwargs, results)
            result['probs_from_results'] = probs_from_results
        result['ori_logits'] = ori_logits
        return result


class Conv1dAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, weights):
        if self.use_flag:
            return self._forward(weights)
        else:
            return weights

    def _forward(self, weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class Conv1dManagerBase:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.conv1d_adapters = self.register_conv1d_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for conv1d_adapter in self.conv1d_adapters:
            conv1d_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_conv1d_to_model(self):
        raise NotImplementedError

    def zero_grad(self,set_to_none=True):
        if set_to_none:
            for conv1d_adapter in self.conv1d_adapters:
                conv1d_adapter.params = None
        else:
            for conv1d_adapter in self.conv1d_adapters:
                conv1d_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad, use_abs = True):
        # assert len(grad.shape) == 4
        # import pdb; pdb.set_trace()
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self,*args,**kwargs):
        grads = []
        for conv1d_adapter in self.conv1d_adapters:
            grads.append(self.grad_process(conv1d_adapter.params.grad,*args,**kwargs))
        return grads


def manager_decoractor(manager: Conv1dManagerBase):
    
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class Conv1dAdapter(Conv1dAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def _forward(self, weights):
        if self.params is None:
            self.params = torch.ones_like(weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(weights)
        return weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)


def slow_forward(self, input_states, cache_params=None, extra_kwargs=None, adapter=None):
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
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        else:
            conv_state = nn.functional.pad(  # only save last conv_kernel_size states
                hidden_states,
                (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_state)

            tmp = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
            tmp = adapter(tmp)
            hidden_states = self.act(tmp)     # [batch, intermediate_size, seq_len]
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        tmp = self.conv1d(hidden_states)[:, :, :seq_len] # [batch, intermediate_size, seq_len]
        tmp = adapter(tmp)
        hidden_states = self.act(tmp)         # [batch, intermediate_size, seq_len]

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())                                             # [intermediate_size, ssm_state_size]
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
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))             # [batch, seq_len, hidden_size]
    return contextualized_states


class Conv1dManager(Conv1dManagerBase):
    def __init__(self, model: PreTrainedModel):
        super().__init__(model)

    def register_conv1d_to_model(self):
        conv1d_adapters = []
        for i, layer in enumerate(self.model.backbone.layers):
            conv1d_adapter = Conv1dAdapter()
            layer.mixer.slow_forward = partial(slow_forward, layer.mixer, adapter=conv1d_adapter)
            conv1d_adapters.append(conv1d_adapter)
        return conv1d_adapters
    


if __name__ == "__main__":
    import os 
    import sys
    sys.path.append(os.getcwd())
    from custom_mamba.custom_mamba_v2 import CustomMambaForCausalLM
    from modelzipper.tutils import *

    model = CustomMambaForCausalLM.from_pretrained("/nvme/hf_models/mamba-1.4b-hf").to('cuda:2')
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/mamba-1.4b-hf")
    
    data = auto_read_data("/nvme/zecheng/data/needle/processed_data/32k_500_insert.pkl")
    
    conv1d_manger = Conv1dManager(model)
    
    for idx, data in tqdm(enumerate(data)):
        token_ids = tokenizer(data['passkey_context'], return_tensors="pt").input_ids[0]
        data = token_ids.to(model.device).unsqueeze(0)
        conv1d_manger.zero_grad()
        
        output = model(input_ids=data, extra_kwargs=None)
        
        label = data[:, -1]
        loss = F.cross_entropy(output[0][:, -2, :], label)
       
        loss.backward(retain_graph=True)

        for i in range(len(conv1d_manger.conv1d_adapters)):
            import pdb; pdb.set_trace()
            saliency = conv1d_manger.grad(use_abs=True)[i]
            # pro = get_proportion(saliency, class_poss, final_poss)
            # pros.append(pro)
        # pros = np.array(pros)
        # pros = pros.T
        # pros_list.append(pros)