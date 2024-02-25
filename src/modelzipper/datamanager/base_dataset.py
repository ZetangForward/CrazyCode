from torch.utils.data import Dataset, DataLoader
from ..tutils import *

class BaseDataset(Dataset):
    def __init__(self, file_name=None, tokenizer=None, split="train", *args, **kwargs):
        super(BaseDataset).__init__()
        self.split = split
        
    def __getitem__(self, index):
        sample = self.content[index]
        return sample
    
    def __len__(self):
        return len(self.content)


class datamodule:
    def __init__(self) -> None:
        pass
    
    @property
    def train_dataset(self) -> Dataset:
        pass
    
    @property
    def valid_dataset(self) -> Dataset:
        pass
    
    @property
    def test_dataset(self) -> Dataset:
        pass
    
    
        
    