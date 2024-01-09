
import sys
import json
import re 
import random
import torch  
import torch.nn as nn 
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch.utils.data import DataLoader, Dataset 
from modelzipper.tutils import *


EDGE = torch.tensor([  # after convert function
    [    0,    0,    0,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  199],
    [    1,    4,  199,    0,    0,    0,    0,  199,  199],
    [    1,  199,  199,    0,    0,    0,    0,  199,    4],
    [    1,  199,    4,    0,    0,    0,    0,    4,    4],
    [    1,    4,    4,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  104],
])

def pad_tensor_with_h(vec, pad_len, dim, pad_token_h):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad
            pad_token_h - represent of pad token
        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        return torch.cat([vec, pad_token_h.repeat(pad_len - vec.size(dim), 1)], dim=dim)

def pad_tensor(vec, pad, dim, pad_token_id):
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
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.empty(*pad_size).fill_(pad_token_id)], dim=dim)


class BasicDataset(Dataset):

    PROMPT_TEMPLATE = "Keywords: {keywords} #begin:"

    def __init__(self, content, tokenizer, svg_begin_token=None, svg_end_token=None, mode="train", min_path_nums=None, max_path_nums=None, max_text_length=64, cluster_batch=False) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.mode = mode
        self.svg_begin_token = svg_begin_token
        self.svg_end_token = svg_end_token
        self.max_text_length = max_text_length
        self.min_path_nums = min_path_nums
        self.max_path_nums = max_path_nums

        content = self.pre_process(content)
        if cluster_batch:
            # first sort the dataset by length
            print_c("you choose to cluster by batch length, begin to sort dataset by length, this may take some time ...", color='magenta')
            content = sorted(content, key=lambda x: x['mesh_data'].shape[0])
            print_c("sort done !", color='magenta')
        self.content = content

    def pre_process(self, dataset, min_length=0):   
        # just prevent too short path
        # length exceed max_seq_length will be cut off in __getitem__
        print_c(f"begin to sanity check the dataset and conduct pre_process, num of samples: {len(dataset)}, it will take some time...", color='magenta')
        new_dataset = []
        for item in dataset:
            sample = item['mesh_data']
            if sample is None:
                continue
            if sample[:7].equal(EDGE):
                sample = sample[7:]
            if min_length <= len(sample):
                new_dataset.append(
                    {
                        'keywords': item['keywords'],
                        'mesh_data': sample,
                    }
                )
        return new_dataset

    def custom_command(self, svg_tensor):
        col1 = svg_tensor[:, 0]
        col1[col1 == 1] = 100
        col1[col1 == 2] = 200
        svg_tensor[:, 0] = col1
        return svg_tensor

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        keywords, sample = item['keywords'], item['mesh_data']
        prompts = self.PROMPT_TEMPLATE.format(keywords=keywords)

        sample = sample[:self.max_path_nums]  # prevent too long num path
        sample = self.custom_command(sample)

        # process the input keywords
        if self.svg_begin_token is not None:
            prompts = prompts + " " + self.svg_begin_token

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

        if self.svg_begin_token is not None:  # utilize svg_token as the end of the text
            text_input_ids[text_attention_mask.sum() - 1] = self.tokenizer.pad_token_id
            text_labels[text_attention_mask.sum() - 1] = -100
            text_attention_mask[text_attention_mask.sum() - 1] = 0

        if self.svg_end_token is not None:
            svg_end_token_id = self.tokenizer.convert_tokens_to_ids(self.svg_end_token)

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "text_labels": text_labels,
            "svg_path": sample.long,
            "svg_end_token_id": svg_end_token_id, 
        }


class VQDataCollator:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, svg_pad_token_h, max_svg_length=1024, pad_token_id=0, cluster_batch=False):
        self.max_svg_length = max_svg_length
        self.svg_pad_token_h = svg_pad_token_h
        self.pad_token_id = pad_token_id
        self.cluster_batch = cluster_batch

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        """

        text_input_ids = list(map(lambda x: x['text_input_ids'], batch))
        text_attention_mask = list(map(lambda x: x['text_attention_mask'], batch))
        text_labels = list(map(lambda x: x['text_labels'], batch))
        svg_tensors = list(map(lambda x: x['svg_path'], batch))
        svg_end_token_id = list(map(lambda x: x['svg_end_token_id'], batch))

        if self.cluster_batch:
            # find longest sequence
            max_len = max(map(lambda x: x.shape[0], svg_tensors))
            max_len = min(max_len, self.max_seq_length)
        else:
            max_len = self.max_seq_length

        # pad according to max_len
        svg_tensors = list(map(lambda x: pad_tensor(x, max_len, 0, self.pad_token_id), svg_tensors))
        svg_tensors = torch.stack(svg_tensors, dim=0)

        # get padding mask
        if self.return_all_token_mask:
            padding_mask = ~(svg_tensors == self.pad_token_id)
        else:
            padding_mask = ~(svg_tensors == self.pad_token_id).all(dim=2, keepdim=True).squeeze()

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "svg_path": svg_tensors, 
            "text_labels": text_labels,
            "svg_padding_mask": padding_mask,
            "svg_end_token_id": svg_end_token_id
        }

    def __call__(self, batch):
        return self.pad_collate(batch)


class VQLLaMAData:
    def __init__(self, config, vq_svg_file, svg_begin_token, svg_end_token, tokenizer):  
        self.cfg = config
        self.tokenizer = tokenizer  
        content = auto_read_data(vq_svg_file) ## Load VQSVG data
        self.valid_data = content[:2000]
        self.train_data = content[2000:]
        self.svg_begin_token = svg_begin_token
        self.svg_end_token = svg_end_token

    @property
    def train_dataset(self) -> Dataset:
        return BasicDataset(
            content=self.train_data,
            min_path_nums=self.cfg.min_path_nums,
            max_path_nums=self.cfg.max_path_nums, 
            tokenizer=self.tokenizer,
            svg_begin_token = self.svg_begin_token,
            svg_end_token = self.svg_end_token,
            max_text_length=self.cfg.max_text_length,
            mode="train",
            cluster_batch=False
        )

    @property
    def valid_dataset(self) -> Dataset:
        return BasicDataset(
            content=self.valid_data,
            min_path_nums=self.cfg.min_path_nums,
            max_path_nums=self.cfg.max_path_nums, 
            tokenizer=self.tokenizer,
            svg_begin_token = self.svg_begin_token,
            svg_end_token = self.svg_end_token,
            max_text_length=self.cfg.max_text_length,
            mode="valid",
            cluster_batch=False
        )


    def predict_dataloader(self) -> DataLoader:
        pass
