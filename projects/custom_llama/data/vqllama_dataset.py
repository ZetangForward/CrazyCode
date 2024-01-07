
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
        pad_size[dim] = pad - vec.size(dim)
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
    def __init__(self, svg_pad_token_id=0, max_svg_length=1024, return_all_token_mask=False):
        self.svg_pad_token_id = svg_pad_token_id
        self.max_svg_length = max_svg_length
        self.return_all_token_mask = return_all_token_mask

    def pad_collate(self, batch):
        input_prompt_ids = list(map(lambda x: x['input_prompt_ids'], batch))
        attention_mask = list(map(lambda x: x['attention_mask'], batch))
        labels = list(map(lambda x: x['labels'], batch))
        svg_quantised = list(map(lambda x: x['svg_quantised'], batch))

        ## combine the text inputs
        input_prompt_ids = torch.stack(input_prompt_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)

        ## pad the vq svg quantised
        svg_quantised = list(map(lambda x: pad_tensor(x, self.max_svg_length, 0, self.svg_pad_token_id), svg_quantised))
        svg_quantised = torch.stack(svg_quantised, dim=0)

        ## obtain padding mask
        # get padding mask
        if self.return_all_token_mask:
            svg_padding_mask = ~(svg_quantised == self.svg_pad_token_id)
        else:
            svg_padding_mask = ~(svg_quantised == self.svg_pad_token_id).all(dim=2, keepdim=True).squeeze()

        return {
            "input_prompt_ids": input_prompt_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "svg_quantised": svg_quantised,
            "padding_mask": svg_padding_mask,
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
