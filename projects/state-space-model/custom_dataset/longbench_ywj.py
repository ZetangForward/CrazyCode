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
        self.config_path = kwargs["config_path"]
        self.subtask =  kwargs["subtask"]
        self.max_gen_len, self.prompt_format = self.get_task_config(self.subtask)
        
        self.max_seq_length = kwargs["max_seq_length"]          #
        self.cluster_batch = kwargs["cluster_batch"]            #
        # self.filter_length(kwargs["testing_max_ctx"])           #
    
    def get_task_config(self, subtask):
        with open (self.config_path+"longbench_config.yaml", encoding='utf-8') as f:
            data = yaml.safe_load(f)
        subtask = self.subtask
        max_len = data['dataset2maxlen'][subtask]
        prompt_format = data['dataset2prompt'][subtask]
        return max_len, prompt_format
    
    # def filter_length(self, max_ctx_length=12000):
    #     new_content = []
    #     print_c(f"begin to filter the context length | total {len(self.content)} instances", "yellow")
    #     for item in self.content:
    #         if item['ctx_length'] <= max_ctx_length:
    #             new_content.append(item)
    #     new_content = sorted(new_content, key=lambda x: x['ctx_length'], reverse=True)  # from long to short
    #     self.content = new_content
    #     print_c(f"filtering finished | total {len(self.content)} instances", "yellow")

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        prompt = self.prompt_format.format(**item)

        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            prompt,  
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
        real_length = attention_mask.size(-1)
       
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'answers': item.pop('answers'),
            'real_length': real_length,
            'max_generation_len': self.max_gen_len
        }
        return res
