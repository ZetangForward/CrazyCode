from modelzipper.tutils import *
from torch.utils.data import Dataset
import torch


class LongLoRA(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]
        if self.cluster_batch:
            self.cluster_batch_fn()

    def cluster_batch_fn(self):
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(self.content, key=lambda x: len(x['instruction'].split()), reverse=True)
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        item = self.content[index]

        # Add a bos token at the beginning of instruct_input_ids
        instruct_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(item["instruction"] + " ", return_tensors="pt")["input_ids"][0].tolist()
        output_ids = self.tokenizer(item["output"], return_tensors="pt")["input_ids"][0].tolist()

        total_length = len(instruct_input_ids) + len(output_ids)

        # Check if the total length exceeds max_seq_length
        if total_length > self.max_seq_length:
            # Calculate the excess length
            # Calculate the excess length based on the sum of instruct_input_ids and output_ids lengths
            excess_length = total_length - self.max_seq_length

            # Reduce the length of instruct_input_ids and output_ids based on their lengths ratio
            total_ids_length = len(instruct_input_ids) + len(output_ids)
            instruct_input_ids = instruct_input_ids[:int(len(instruct_input_ids) - len(instruct_input_ids) / total_ids_length * excess_length)]
            output_ids = output_ids[:int(len(output_ids) - len(output_ids) / total_ids_length * excess_length)]

        # Check if the total length is still more than max_seq_length due to rounding, if so, remove one more token from output_ids
        if len(instruct_input_ids) + len(output_ids) > self.max_seq_length:
            output_ids = output_ids[:-1]

        # Combine instruct_input_ids and output_ids
        input_ids = instruct_input_ids + output_ids

        # If the total length is less than max_seq_length, pad the sequence and create an attention mask
        if len(input_ids) < self.max_seq_length:
            # Create a padding mask
            attention_mask = [1] * len(input_ids) + [0] * (self.max_seq_length - len(input_ids))

            # Pad the sequence
            input_ids = torch.nn.functional.pad(input=torch.tensor(input_ids), pad=(0, self.max_seq_length - len(input_ids)), mode='constant', value=self.tokenizer.pad_token_id)
        else:
            # If no padding is needed, the attention mask is simply a list of ones
            attention_mask = [1] * len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids)
        
        # print(attention_mask.shape)
        # print(label.size())

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }