
import sys
import json
import re 
import random
import torch  
import torch.nn as nn 
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch.utils.data import DataLoader, Dataset 
from modelzipper.tutils import *


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

        if self.svg_token is not None:  # utilize svg_token as the end of the text
            text_input_ids = text_input_ids[:-1]
            text_attention_mask = text_attention_mask[:-1]
            text_labels = text_labels[:-1]

        return {
            "input_prompt_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_labels,
            "svg_quantised": svg_quantised,
        }


class VQDataCollator:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self):
        ...

    

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

    
    
    def extract_numerical(self, path_data, numerical_token):  
        ## match all numericals 
        number_pattern = r"-?\d+\.?\d*"  
        numbers = re.findall(number_pattern, path_data)  
        numbers = [float(item) for item in numbers]
        ## replace all matched numericals with numerical_token ("[NUM]")  
        replaced_data = re.sub(number_pattern, numerical_token, path_data) 
        replaced_data = re.sub(' +', ' ', replaced_data) 
        return numbers, replaced_data 
    
    def extract_c_segments(self, path_data):  
        ## find all the control path in svg paths
        c_pattern = r"c[^A-Za-z]*?(?=[A-Za-z])"  
        c_segments = re.findall(c_pattern, path_data)  
        if len(c_segments) == 1:  # only one control path, usually complex
            return [self.extract_consecutive_numbers(c_segments[0], 0.5)]
        return c_segments 
    
    def __getitem__(self, index):
        data = self.content[index]
        # svg_path = data["compress_path"].split("#Begin:")[-1].strip()
        svg_path_with_prompt = data["compress_path"]
        
        ## Step 1: extract numericals from svg paths 
        ## and replace the numericals with numerical_token
        extracted_numericals, replaced_paths = self.extract_numerical(
            svg_path_with_prompt, self.numerical_token)
        
        ## Step 2: encode the replaced svg paths
        if self.numerical_mode:
            seq_inputs = self.tokenizer(
                replaced_paths, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            text_input_ids = seq_inputs.input_ids[0]
            text_attention_mask = seq_inputs.attention_mask[0]
            text_labels = torch.where(
                text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100
            )
        
        return {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_labels,
            "numerical_values": extracted_numericals,
        }
        
    @classmethod
    def custom_datacollator(cls, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate examples for supervised fine-tuning."""
        
        batch_input_ids, batch_attn_mask, batch_label = [], [], []
        batch_numerical_input_ids = []
        max_numerical_nums = max([len(item["numerical_values"]) for item in instances])
        
        for ins in instances:
            batch_input_ids.append(ins["input_ids"])
            batch_attn_mask.append(ins["attention_mask"])
            batch_label.append(ins["labels"])
            
            ## process numerical values
            ### Step 1: convert to float tensor
            numerical_values = torch.FloatTensor(ins["numerical_values"])
            ### Step 2: pad to the same length
            numerical_values = torch.cat(  
                [numerical_values, torch.full((max_numerical_nums - len(numerical_values),), 300)]  
            )  

            batch_numerical_input_ids.append(numerical_values)
            
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attn_mask = torch.stack(batch_attn_mask, dim=0)
        batch_label = torch.stack(batch_label, dim=0)
        batch_numerical_input_ids = torch.stack(batch_numerical_input_ids, dim=0)
        
        return {
            "batch_input_ids": batch_input_ids,
            "batch_numerical_input_ids": batch_numerical_input_ids,
            "batch_attention_mask": batch_attn_mask,
            "batch_labels": batch_label,
        }
