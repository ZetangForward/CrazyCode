from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch   
from torch.utils.data import DataLoader, Dataset  
from pathlib import Path  
from typing import List, Optional, Sequence, Union, Any, Callable, Dict, Tuple  
from modelzipper.tutils import *
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

EDGE = torch.tensor([  # after convert function
    [    0,    0,    0,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  199],
    [    1,    4,  199,    0,    0,    0,    0,  199,  199],
    [    1,  199,  199,    0,    0,    0,    0,  199,    4],
    [    1,  199,    4,    0,    0,    0,    0,    4,    4],
    [    1,    4,    4,    0,    0,    0,    0,    4,  104],
    [    1,    4,  104,    0,    0,    0,    0,    4,  104],
])


class BasicDataset(Dataset):
    def __init__(self, dataset, max_path_nums=150, mode="train", pad_token_id=0, num_bins = 9, vocab_size=200, return_all_token_mask=False, remove_redundant_col=False, cluster_batch_length=False):
        super().__init__()
        self.dataset = dataset
        self.max_path_nums = max_path_nums
        self.mode = mode
        self.pad_token_id = pad_token_id
        self.num_bins = num_bins
        self.vocab_size = vocab_size
        self.return_all_token_mask = return_all_token_mask
        self.remove_redundant_col = remove_redundant_col
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        keywords, sample = item['keywords'], item['mesh_data']
        sample = torch.clamp(sample, min=0, max=self.vocab_size)
        if sample[:7].equal(EDGE):
            sample = sample[7:]
        
        if self.remove_redundant_col:  # remove 2nd and 3rd column
            sample = torch.cat([sample[:, :1], sample[:, 3:]], dim=1)   
        
        sample = self.custom_command(sample)
        
        return sample.long()

        if len(sample) < self.max_path_nums:
            sample = torch.cat([sample, torch.empty(self.max_path_nums - len(sample), self.num_bins).fill_(self.pad_token_id)])
        else:
            sample = sample[:self.max_path_nums]
        
        
        

        if self.return_all_token_mask:
            padding_mask = ~(sample == self.pad_token_id)
        else:
            padding_mask = ~(sample == self.pad_token_id).all(dim=1, keepdim=True).squeeze()
        return {
            "svg_path": sample.long(), 
            "padding_mask": padding_mask,
        }

    def custom_command(self, svg_tensor):
        col1 = svg_tensor[:, 0]
        col1[col1 == 1] = 100
        col1[col1 == 2] = 200
        svg_tensor[:, 0] = col1
        return svg_tensor


def pad_tensor(vec, pad, dim, pad_token_id):
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


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, cluster_batch_length=False, max_seq_length=150, pad_token_id=0, return_all_token_mask=False):
        """
        args:
            cluster_batch_length - if True, cluster batch by length
            max_seq_length - max sequence length
            pad_token_id - padding token id
        """

        self.cluster_batch_length = cluster_batch_length
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.return_all_token_mask = return_all_token_mask
    

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        import pdb; pdb.set_trace()
        if self.cluster_batch_length:
            # find longest sequence
            max_len = max(map(lambda x: x[0].shape[-1], batch))
            max_len = min(max_len, self.max_seq_length)
            # pad according to max_len
            
            batch = list(map(lambda items: (pad_tensor(items[0], pad=max_len, dim=-1), items[1]), batch))

        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
    

class SvgDataModule(pl.LightningDataModule):
    def __init__(self, config, transform=None):
        super().__init__()
        self.cfg = config       
        self.transform = transform
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # dataset processing operations here
        return None
    
    def setup(self, stage: str = 'fit') -> None:
        self.test_dataset = None
        if self.cfg.inference_mode:
            self.test_file = auto_read_data(self.cfg.test_data_path)
            self.test_dataset = BasicDataset(
                self.test_file, max_path_nums=self.cfg.max_path_nums, 
                mode='test', pad_token_id=self.cfg.pad_token_id,
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col
            )
        else:
            self.svg_files = auto_read_data(self.cfg.train_data_path)
            self.train_file = self.svg_files[:-2000]
            self.valid_file = self.svg_files[-2000:]

            self.train_dataset = BasicDataset(
                self.train_file, max_path_nums=self.cfg.max_path_nums, 
                mode='train', pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col,
            )
            self.valid_dataset = BasicDataset(
                self.valid_file, max_path_nums=self.cfg.max_path_nums, 
                mode='valid', pad_token_id=self.cfg.pad_token_id,
                return_all_token_mask=self.cfg.return_all_token_mask,
                remove_redundant_col=self.cfg.remove_redundant_col
            )    
        

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=True, shuffle=True, 
            collate_fn=PadCollate(
                cluster_batch_length=self.cfg.cluster_batch_length, 
                max_seq_length=self.cfg.max_path_nums, 
                pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask
            ),
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.valid_dataset, batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            collate_fn=PadCollate(
                cluster_batch_length=self.cfg.cluster_batch_length, 
                max_seq_length=self.cfg.max_path_nums, 
                pad_token_id=self.cfg.pad_token_id, 
                return_all_token_mask=self.cfg.return_all_token_mask
            ),
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataloader is not None:
            return DataLoader(
                self.test_dataset, batch_size=self.cfg.val_batch_size, 
                num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            )
        return None