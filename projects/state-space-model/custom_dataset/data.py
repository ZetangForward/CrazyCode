from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import glob
from datasets import load_dataset

class TextFillingDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", full_modeling=True, *args, **kwargs):
        super(TextFillingDataset).__init__()
        self.split = split
        self.content = content
        self.max_text_length = kwargs['max_text_length']
        self.tokenizer = tokenizer
        self.full_modeling = full_modeling
        self.template1 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: "
        self.template2 = "Beginning: {s1} {s2} {s3}\nEnding: {s5}\nMiddle: {s4}"
        
    def __getitem__(self, index):
        sample = self.content[index]
        s1 = sample["sentence1"]
        s2 = sample["sentence2"]
        s3 = sample["sentence3"]
        s4 = sample["sentence4"]
        s5 = sample["sentence5"]
        
        if not self.full_modeling:
            prompt = self.template1.format(s1=s1, s2=s2, s3=s3, s5=s5)
            
            tokenized_prompt = self.tokenizer(
                prompt,  
                truncation=True, 
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            prompt_ids = tokenized_prompt.input_ids[0]
            label_ids = self.tokenizer(s4, return_tensors="pt").input_ids[0]
            
            if self.split == "test":  # test mode
                return {
                    "input_ids": prompt_ids,
                    "labels": label_ids,
                }
            
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
        
        else:
            prompt = self.template2.format(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5)
            
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

    def __len__(self):
        return len(self.content)


class FindNeedle(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer, eval_path) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eval_path = eval_path
        self.ctx_len = cfg.ctx_len
        self.depth = cfg.depth
        self.needle = cfg.needle
        self.prepare_data_per_node = True

    def load_context(self, fpath, ctx_len=10000):
        context = ""
        for file in glob.glob(fpath):
            with open(file, 'r') as f: 
                context += f.read()
        LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
        context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
        return context

    def insert_needle(self, context, needle, depth):
        context = context.split(".")
        c_len = len(context)
        needle_place = int(depth * c_len)
        context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
        return context

    def setup(self, stage: str = 'predict') -> None:
        context = self.load_context(fpath=self.eval_path, ctx_len=self.ctx_len)
        context = self.insert_needle(context, self.needle, depth=self.depth)
        needle_idx = context.find("The best thing to do in San Francisco is")
        print("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
        print(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 
        import pdb; pdb.set_trace()


    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.context, batch_size=1, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
        )
       

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


class custom_datamodule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.dataset_kwargs = {
            "max_text_length": self.cfg.max_seq_length,
        }
        
    def setup(self, stage: str = 'fit') -> None:
        self.test_dataset = None
        if self.cfg.inference_mode:
            self.test_data = auto_read_data(self.cfg.test_data_path)
            self.test_dataset = TextFillingDataset(
                content=self.test_data, 
                tokenizer=self.tokenizer, 
                full_modeling=False,
                split="test",
                **self.dataset_kwargs,
            )
        else:
            content = auto_read_data(self.cfg.file_path)
            min_valid_num = min(1000, len(content)*0.1)
            self.valid_data = content[:min_valid_num]
            self.train_data = content[min_valid_num:]
            
            self.train_dataset = TextFillingDataset(
                content=self.train_data, 
                tokenizer=self.tokenizer, 
                split="train",
                **self.dataset_kwargs,
            )
            
            self.valid_dataset = TextFillingDataset(
                content=self.valid_data, 
                tokenizer=self.tokenizer, 
                split="valid",
                **self.dataset_kwargs,
            )
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
            print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')

            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=True, shuffle=True, 
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset, batch_size=1, 
                num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            )
        return None
    
    
if __name__ == "__main__":
    file_path = "/nvme/zecheng/data/roc_stories/ROCStories_winter2017.csv"
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    data_module = custom_datamodule(file_path, tokenizer)
    raw_data = data_module.content
    import pdb; pdb.set_trace()