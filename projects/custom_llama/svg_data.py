from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch   
from torch.utils.data import DataLoader, Dataset  
from pathlib import Path  
from typing import List, Optional, Sequence, Union, Any, Callable, Dict, Tuple  
import torch.distributed as dist
from modelzipper.tutils import *
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import hydra  
import pytorch_lightning as L

class BasicDataset(Dataset):
    def __init__(self, dataset, max_path_nums=150, mode="train", pad_token_id=-1):
        super().__init__()
        self.dataset = dataset
        self.max_path_nums = max_path_nums
        self.mode = mode
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def custom_datacollator(batch):
        return batch
    

class SvgDataModule(L.LightningDataModule):
    
    def __init__(self, config, transform=None):
        self.cfg = config.dataset        
        self.transform = transform

    def prepare_data(self) -> None:
        # dataset processing operations here
        return None
    
    def setup(self, stage: str) -> None:
        self.svg_files = auto_read_data(self.cfg.train_data_path)
        self.train_file = self.svg_files[:-500]
        self.valid_file = self.svg_files[-500:]

        self.train_dataset = BasicDataset(
            self.train_file, max_path_nums=self.cfg.max_path_nums, 
            mode='train', pad_token_id=self.cfg.pad_token_id
        )
        self.valid_dataset = BasicDataset(
            self.valid_file, max_path_nums=self.cfg.max_path_nums, 
            mode='valid', pad_token_id=self.cfg.pad_token_id
        )    
        self.create_data_loaders()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.nworkers, sampler=self.train_sampler, 
            pin_memory=False, drop_last=True, shuffle=True, 
            collate_fn=BasicDataset.custom_datacollator,
        )
    
    def valid_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.nworkers, sampler=self.valid_sampler, 
            pin_memory=False, drop_last=True, shuffle=False,
            collate_fn=BasicDataset.custom_datacollator
        )



@hydra.main(config_path='.', config_name='config')
def main(config):
    # Setup dataset
    data_processor = SvgDatasetProcesser(config)
    train_dataset = data_processor.train_dataset
    train_dataloader = data_processor.train_loader
    valid_dataloader = data_processor.valid_loader
    print_c(len(train_dataloader))
    print_c(train_dataset[0])
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
