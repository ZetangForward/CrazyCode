import torch
from torch.utils.data import Dataset
from modelzipper.tutils import *
import datasets

class Slimpajama(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]

        if self.cluster_batch: 
            print_c("Requires clustering batch, begin to process", "yellow")
            bt = time.time()
            self.cluster_batch_fn()
            print_c(f"Clustering batch finished, time elapsed: {time.time()-bt}", "yellow")

    def cluster_batch_fn(self):
        tmp = [item['source'] + ' ' + item['target'] for item in self.content]
        # tok_tmp = [self.tokenizer(item, return_tensors="pt") for item in tmp]
        sorted_tok_tmp = sorted(tmp, key=lambda x: len(x.split()))
        self.content = sorted_tok_tmp

    @classmethod
    def preprocess_data(cls, content, tokenizer, max_seq_length):
        all_process_data = []
        import pdb; pdb.set_trace()
        for item in content:
            src, tgt = item['source'], item['target']
            str_format = src + " " + tgt
            tokenized_sequence = tokenizer(
                str_format,  
                truncation=True, 
                padding="max_length",
                max_length=max_seq_length,
                return_tensors="pt",
            )
            input_ids = tokenized_sequence.input_ids[0]
            attention_mask = tokenized_sequence.attention_mask[0]
            labels = torch.where(
                input_ids != tokenizer.pad_token_id, input_ids, -100
            )
            all_process_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })


    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        if not self.cluster_batch:
            sample = self.content[index]
            src, tgt = sample['source'], sample['target']
            str_format = src + " " + tgt
        else: # after clustering batch, already in id format
            str_format = self.content[index]

        tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
            str_format,  
            truncation=True, 
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokenized_sequence.input_ids[0]
        attention_mask = tokenized_sequence.attention_mask[0]
        labels = torch.where(
            input_ids != self.tokenizer.pad_token_id, input_ids, -100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    dataset_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/data/slim_pajama_chunk1/data"
    data = datasets.load_dataset("/aifs4su/ziliwang/txw/InternLM/zecheng/data/slim_pajama_chunk1")
    max_seq_length = 4000  
    tokenizer = AutoTokenizer.from_pretrained("/aifs4su/ziliwang/txw/InternLM/zecheng/hf_models/mamba-370m-hf")

    processed_dataset = Slimpajama.preprocess_data(data, tokenizer, max_seq_length)
