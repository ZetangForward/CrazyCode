from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
import torch


class TextFillingDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super(TextFillingDataset).__init__()
        self.split = split
        self.content = content
        self.max_text_length = kwargs['max_text_length']
        self.tokenizer = tokenizer
        self.template = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: "
        
    def __getitem__(self, index):
        sample = self.content[index]
        s1 = sample["sentence1"]
        s2 = sample["sentence2"]
        s3 = sample["sentence3"]
        s4 = sample["sentence4"]
        s5 = sample["sentence5"]
        prompt = self.template.format(s1=s1, s2=s2, s3=s3, s5=s5)
        
        tokenized_prompt = self.tokenizer(
            prompt,  
            truncation=True, 
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        prompt_ids = tokenized_prompt.input_ids[0]
        prompt_mask = tokenized_prompt.attention_mask[0]
        prompt_sential = torch.empty_like(prompt_ids).fill_(self.tokenizer.pad_token_id)
        
        remain_length = self.max_text_length - prompt_ids.size(0)
        
        tokenized_mid = self.tokenizer(
            s4,  
            truncation=True, 
            padding="max_length",
            max_length=remain_length,
            return_tensors="pt",
        )
        label_ids = tokenized_mid.input_ids[0]
        label_attention_mask = tokenized_prompt.attention_mask[0]
        label_sentinel = label_ids
        
        input_ids = torch.concatenate([prompt_ids, label_ids], dim=0)
        tok_seq = torch.concatenate([prompt_sential, label_sentinel], dim=0)
        attention_mask = torch.concatenate([prompt_mask, label_attention_mask], dim=0)
        
        labels = torch.where(
            tok_seq != self.tokenizer.pad_token_id, tok_seq, -100
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.content)


class custom_datamodule(data_module):
    def __init__(self, file_path, tokenizer):
        super(custom_datamodule).__init__()
        self.tokenizer = tokenizer
        content = auto_read_data(file_path)
        self.kwargs = {
            "max_text_length": 512,
        }
        min_valid_num = min(1000, len(content)*0.1)
        self.valid_data = content[:min_valid_num]
        self.train_data = content[min_valid_num:]
        
    @property
    def train_dataset(self) -> Dataset:
        return BaseDataset(
            content=self.train_data, 
            tokenizer=self.tokenizer, 
            split="train",
            **self.kwargs,
        )
    
    @property
    def valid_dataset(self) -> Dataset:
        return BaseDataset(
            content=self.valid_data, 
            tokenizer=self.tokenizer, 
            split="train",
            **self.kwargs,
        )
    
    @property
    def test_dataset(self) -> Dataset:
        pass
    
    
if __name__ == "__main__":
    file_path = "/nvme/zecheng/data/roc_stories/ROCStories_winter2017.csv"
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    data_module = custom_datamodule(file_path, tokenizer)
    raw_data = data_module.content
    import pdb; pdb.set_trace()