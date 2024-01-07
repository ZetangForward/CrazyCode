
import sys
import json
import re 
import random
import torch  
import torch.nn as nn 
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch.utils.data import DataLoader, Dataset 
from modelzipper.tutils import *


def pad_tensor(vec, pad_len, dim, pad_token_id):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad
            pad_token_id - padding token id
        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad_len - vec.size(dim)
        return torch.cat([vec, torch.empty(*pad_size).fill_(pad_token_id)], dim=dim)



class BasicDataset(Dataset):

    PROMPT_TEMPLATE = "Keywords: {keywords} #begin:"

    def __init__(self, content, tokenizer, mode="train", svg_token=None, max_text_length=64, max_svg_length=1024) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.content = content
        self.mode = mode
        self.svg_token = svg_token
        self.max_text_length = max_text_length
        self.max_svg_length = max_svg_length

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        svg_quantised = self.content[index]["xs_quantised"]
        keywords = self.content[index]["keywords"]
        prompts = self.PROMPT_TEMPLATE.format(keywords=keywords)
        
        # truncate the svg_quantised
        svg_quantised = svg_quantised[: self.max_svg_length]

        # process the input keywords
        if self.svg_token is not None:
            prompts = prompts + " " + self.svg_token

        seq_inputs = self.tokenizer(
            prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_input_ids = seq_inputs.input_ids[0]
        text_attention_mask = seq_inputs.attention_mask[0]
        text_labels = torch.where(
            text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100
        )

        # FIXME: check the dtype 和实际的修改是否正确(这里就是单纯删除</s> token，让文本部分结尾是svg_token)
        if self.svg_token is not None:  # utilize svg_token as the end of the text
            text_input_ids[text_attention_mask.sum() - 1] = self.tokenizer.pad_token_id
            text_labels[text_attention_mask.sum() - 1] = -100
            text_attention_mask[text_attention_mask.sum() - 1] = 0

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "svg_quantised": svg_quantised,
        }


class VQDataCollator:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, svg_pad_token_id=0, max_svg_length=1024, return_all_token_mask=False):
        self.svg_pad_token_id = svg_pad_token_id
        self.max_svg_length = max_svg_length
        self.return_all_token_mask = return_all_token_mask

    def pad_collate(self, batch):
        text_input_ids = list(map(lambda x: x['text_input_ids'], batch))
        text_attention_mask = list(map(lambda x: x['text_attention_mask'], batch))
        text_labels = list(map(lambda x: x['text_labels'], batch))
        svg_quantised = list(map(lambda x: x['svg_quantised'], batch))

        ## combine the text inputs
        text_input_ids = torch.stack(text_input_ids, dim=0)
        text_attention_mask = torch.stack(text_attention_mask, dim=0)
        text_labels = torch.stack(text_labels, dim=0)

        ## pad the vq svg quantised
        svg_quantised = list(map(lambda x: pad_tensor(x, self.max_svg_length, 0, self.svg_pad_token_id), svg_quantised))
        svg_quantised = torch.stack(svg_quantised, dim=0)

        ## obtain svg padding mask
        if self.return_all_token_mask:
            svg_padding_mask = ~(svg_quantised == self.svg_pad_token_id)
        else:
            svg_padding_mask = ~(svg_quantised == self.svg_pad_token_id).all(dim=2, keepdim=True).squeeze()

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "svg_quantised": svg_quantised,
            "svg_padding_mask": svg_padding_mask,
        }

    def __call__(self, batch):
        return self.pad_collate(batch)

    

class VQLLaMAData:
    def __init__(self, args, vq_svg_file, tokenizer: PreTrainedTokenizer, split="train"):  

        self.tokenizer = tokenizer  
        self.split = split
        self.max_seq_len = args.model_max_length
        content = auto_read_data(vq_svg_file) ## Load VQSVG data
        self.valid_data = content[:2000]
        self.train_data = content[2000:]


    def train_dataloader(self) -> DataLoader:
        pass


    def valid_dataloader(self) -> DataLoader:
        pass


    def predict_dataloader(self) -> DataLoader:
        pass
