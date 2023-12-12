from torch.utils.data import Dataset, DataLoader
from ..tutils import *

class BaseDataset(Dataset):
    def __init__(self, file_name, tokenizer=None, split="train", *args, **args):
        super(BaseData).__init__()
        self.content = auto_read_data(file_name)
        self.split = split
        
    def __getitem__(self, index):
        sample = self.content[index]
        return sample
    
    def __len__(self):
        return len(self.content)
    
    @classmethod
    def collect_fn(cls, batch_input):
        return batch_input
        


    
        
        
    