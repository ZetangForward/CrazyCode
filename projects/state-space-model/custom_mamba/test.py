import torch
import torch.nn as nn
from functools import wraps, partial
from transformers import PreTrainedModel
from typing import Dict
import torch.nn.functional as F
from utils import get_model_tokenizer

def dict_to(d: dict, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d

class LMForwardAPI(nn.Module):
    def __init__(self, model, model_name, tokenizer, label_dict: Dict[int, str], device='cuda:0'):
        super().__init__()
        self._use_past_key_values = False
        self._past_key_values = None
        self.model = model
        self.model_name = model_name
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

        assert model_name in ['gpt2-xl', 'gpt-j-6b']

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'LMForwardAPI: set device to {device}')
        self.model = self.model.to(device)
        if self.past_key_values:
            self.past_key_values = self.past_key_values  # will reset device

    def cal_logits(self, inputs, **kwargs):
        self.model.eval()
        inputs = dict_to(inputs, self.device)

        if self.use_past_key_values:
            past_key_values = self.get_past_key_values(inputs)
            kwargs['past_key_values'] = past_key_values
            inputs['attention_mask'] = self.get_mask_with_past_key_values(inputs['attention_mask'])
            if self.model_name in ['gpt-j-6b','gpt2-xl']:
                bsz, sql = inputs['input_ids'].shape
                position_ids = torch.arange(sql, dtype=torch.long, device=self.device).repeat(bsz, 1)
                position_ids = position_ids + self.position_offset
                kwargs['position_ids'] = position_ids

        results = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
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


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self,set_to_none=True):
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad, use_abs = True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self,*args,**kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(self.grad_process(attention_adapter.params.grad,*args,**kwargs))
        return grads


def manager_decoractor(manager: AttentionerManagerBase):
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

class AttentionAdapter(AttentionAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)



def gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class GPT2AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel):
        super().__init__(model)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter()
            layer.attn._attn = partial(gpt2_attn, layer.attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters
    

# attentionermanger = GPT2AttentionerManager(model.model)


# for idx, data in tqdm(enumerate(analysis_dataloader)):
#     data = dict_to(data, model.device)
#     print(data['input_ids'].shape)
#     attentionermanger.zero_grad()
#     output = model(**data)
#     label = data['labels']
#     loss = F.cross_entropy(output['logits'], label)
#     loss.backward()
#     class_poss, final_poss = predictor.get_pos({'input_ids': attentionermanger.input_ids})
#     pros = []
#     for i in range(len(attentionermanger.attention_adapters)):
#         saliency = attentionermanger.grad(use_abs=True)[i]
#         pro = get_proportion(saliency, class_poss, final_poss)
#         pros.append(pro)
#     pros = np.array(pros)
#     pros = pros.T
#     pros_list.append(pros)


if __name__ == "__main__":
    from custom_mamba.custom_mamba_v2 import CustomMambaForCausalLM
    model = CustomMambaForCausalLM.from_pretrained("/nvme/hf_models/mamba-1.4b-hf", dtype=torch.bfloat16)
