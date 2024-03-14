from modelzipper.tutils import *
from torch.utils.data import Dataset
import glob


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
        tmp = [item['instruction'] + ' ' + item['output'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        import pdb; pdb.set_trace()
        instruct_length = len(self.tokenizer(item["instruction"], return_tensors="pt")["input_ids"][0])
        output_length = len(self.tokenizer(item["output"], return_tensors="pt")["input_ids"][0])


        import pdb; pdb.set_trace()

        sequence = item["instruction"] + ' ' + item["output"]
        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            sequence,  
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
       
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return res
