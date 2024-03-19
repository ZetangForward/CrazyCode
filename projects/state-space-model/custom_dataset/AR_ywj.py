from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import random
import numpy as np
import numpy as np
import torch
import glob



class MQARDataset(Dataset):
    def __init__(self, content=None, tokenizer=None, split="train", *args, **kwargs):
        super().__init__()
        self.split = split
        self.content = content
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs["max_seq_length"]
        self.cluster_batch = kwargs["cluster_batch"]

    @classmethod
    def build_dataset(cls, vocab_size, num_examples, input_seq_len, num_kv_pairs, power_a, tokenizer, random_non_queries=True , random_seed=42):
        
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        context_size = num_kv_pairs * 2
        # import pdb;pdb.set_trace()
        # create keys so that each key is present exactly once in each example
        key_vocab_size = vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, vocab_size)

        keys_unshuffled = np.tile(key_choices, (num_examples, 1))
        keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

        values_unshuffled = np.tile(value_choices, (num_examples, 1))
        values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

        # create sequences
        kvs = np.zeros((num_examples, context_size), dtype=np.int64)
        kvs[:, 0::2] = keys
        kvs[:, 1::2] = values

        # compute power law
        space = (input_seq_len - context_size) // 2
        p = power_a * np.arange(1, space + 1) ** (power_a-1)    # 幂律分布
        p = p / p.sum()

        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

        # queries and answers
        queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
        examples = np.concatenate([
            kvs, 
            queries
        ], axis=1)

        labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
        
        # replace all the 0 with random values
        if random_non_queries:                          # 随机插入非 key-value值
            inputs[inputs == 0] = torch.randint(vocab_size+1, 51200, size=inputs.shape)[inputs == 0]

        all_test_data = []

        for i in range(inputs.size(0)):  
            input_list = inputs[i].to(torch.int32)
            # label_idx = torch.nonzero(labels[i] != -100).flatten().to(torch.int32)
            # label_value = torch.index_select(labels[i], 0, label_idx).to(torch.int32)
            
            # data_dict = {'input': input_list, 'label_idx': label_idx, 'label_value': label_value}
            label_list = labels[i].to(torch.int32)
            data_dict = {'input': input_list, 'label': label_list}
            
            all_test_data.append(data_dict)
        
        return all_test_data
    
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index) -> Any:
        item = self.content[index]
        input_ids = item.pop('input')
        # tokenized_sequence = self.tokenizer(  # re-tokenize to get attention mask
        #     input,  
        #     return_tensors="pt",
        # )

        # attention_mask = tokenized_sequence.attention_mask[0]
        attention_mask = torch.ones(input_ids.shape,dtype=input_ids.dtype)
       
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        res.update(item)

        return res

if __name__ == '__main__':
    train_len_num = [512, 1024, 2048, 4096, 8192] 
    train_kv_num = [32, 64, 128, 256, 512]
    for i in range(0,len(train_kv_num)):
        input_seq_len = train_len_num[i]
        number_kv_pairs = train_kv_num[i]
        test_data = MQARDataset.build_dataset(
            vocab_size=8192, 
            input_seq_len=input_seq_len,
            num_kv_pairs=number_kv_pairs,
            num_examples=100000,
            power_a=0.01,
            tokenizer=None,
            )
        data_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/data/MQAR/" + "train_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
        auto_save_data(test_data,data_path)
        
    for input_seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
        for number_kv_pairs in [32, 64, 128, 256, 512, 1024]:
            try:
                test_data = MQARDataset.build_dataset(
                    vocab_size=8192, 
                    input_seq_len=input_seq_len,
                    num_kv_pairs=number_kv_pairs,
                    num_examples=3000,
                    power_a=0.01,
                    tokenizer=None,
                    )
                data_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/data/MQAR/" + "test_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
                auto_save_data(test_data,data_path)
            except:
                print(input_seq_len,number_kv_pairs,"save+-failed")