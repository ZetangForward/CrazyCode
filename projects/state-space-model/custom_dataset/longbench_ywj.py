from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import yaml
import torch
import glob


class LongbenchDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.subtask =  kwargs["subtask"]
        self.max_seq_length = kwargs["max_seq_length"]          #
        self.cluster_batch = kwargs["cluster_batch"]            #
        self.filter_length(kwargs["testing_max_ctx"])           #
        
    def filter_length(self, max_ctx_length=12000):
        new_content = []
        print_c(f"begin to filter the context length | total {len(self.content)} instances", "yellow")
        for item in self.content:
            if item['ctx_length'] <= max_ctx_length:
                new_content.append(item)
        new_content = sorted(new_content, key=lambda x: x['ctx_length'], reverse=True)  # from long to short
        self.content = new_content
        print_c(f"filtering finished | total {len(self.content)} instances", "yellow")

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        passkey_context = item.pop('passkey_context')
        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            passkey_context,  
            return_tensors="pt",
        )

        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
       
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        res.update(item)

        return res
