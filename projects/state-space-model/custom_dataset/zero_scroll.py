from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob
from datasets import load_dataset


class AlpacaDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, max_seq_length=512, *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.max_text_length = max_seq_length
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template = "{instruction} {input} {output}"
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        sample = self.content[index]
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        
        prompt = self.template.format(instruction=instruction, input=input_text, output=output_text)
        
        tokenized_prompt = self.tokenizer(
            prompt,  
            truncation=True, 
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokenized_prompt.input_ids[0]
        attention_mask = tokenized_prompt.attention_mask[0]
        labels = torch.where(
            input_ids != self.tokenizer.pad_token_id, input_ids, -100
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ZeroScrolls(pl.LightningDataModule):

    datasets = [
        'gov_report',
        'summ_screen_fd',
        'qmsum',
        'qasper',
        'narrative_qa',
        'quality',
        'musique',
        'squality',
        'space_digest',
        'book_sum_sort'
    ]

    def __init__(self, cfg, tokenizer, max_input_length):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.prepare_data_per_node = True

    def trim_doc_keeping_suffix(self, tokenizer, tokenized_input_full, example, suffix_index, max_tokens, device):
        seperator_and_suffix = f"{example['truncation_seperator'].strip()}\n\n{example['input'][suffix_index:].strip()}\n"
        tokenized_seperator_and_suffix = tokenizer(seperator_and_suffix, return_tensors="pt").input_ids.to(device)
        tokenized_input_trimmed = tokenized_input_full[:, :max_tokens - tokenized_seperator_and_suffix.shape[1]]
        tokenized_input = torch.cat([tokenized_input_trimmed, tokenized_seperator_and_suffix], dim=1)
        return tokenized_input

    def process_model_input(self, tokenizer, example, max_tokens, device):
        tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
        if tokenized_input_full.shape[1] <= max_tokens:
            return tokenized_input_full
        seperator_and_query_text = example['truncation_seperator'] + example["input"][example['query_start_index']:]
        tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
        input_without_query = example['input'][:example['query_start_index']]
        tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
        tokenized_input_without_query = tokenized_input_without_query[:, :max_tokens - tokenized_seperator_and_query.shape[1]]
        tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
        return tokenized_input
    
    def setup(self, stage: str = 'predict') -> None:
        all_testing_data = dict()
        print_c("processing data ...", "magenta")
        for dataset in self.datasets:
            print_c(f"processing split {dataset}", "magenta")
            all_testing_data[dataset] = []
            local_data_path = os.path.join(self.cfg.data_path, dataset) # we save the data in local path
            data = load_dataset(local_data_path, split='test')
            for i, example in enumerate(data):
                model_input = self.process_model_input(self.tokenizer, example, self.max_input_length, 'cpu')
                all_testing_data[dataset].append(model_input)
        
        import pdb; pdb.set_trace()
