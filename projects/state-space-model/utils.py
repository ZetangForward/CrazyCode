import torch
import os
import sys
import lightning.pytorch as pl
import hydra
import importlib
from torch import optim, Tensor 
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from custom_mamba.position_mamba import LongContextMamba
from modelzipper.tutils import *
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from transformers import AutoTokenizer, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer


def get_model_tokenizer(root_dir, model_config):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)

    if "gpt" in model_path.lower():
        model = GPTNeoForCausalLM.from_pretrained(
            model_path, use_cache=False, torch_dtype=torch.bfloat16
        ).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif "mamba" in model_path.lower():
        model = LongContextMamba.from_pretrained(
            model_path, use_position=model_config.use_position,
            dtype=torch.bfloat16, device="cuda", strict=False
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        ).to('cuda')
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    if "gpt" in tokenizer_path or "llama" in tokenizer_path:
        # tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
    
    print_c("model and tokenzier already loaded ~")

    return model, tokenizer


class CustomDatamodule(pl.LightningDataModule):

    def __init__(self, cfg, root_dir, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.dataset_kwargs = {
            "max_seq_length": self.cfg.max_seq_length,
            "cluster_batch": self.cfg.cluster_batch,            
        }
        self.dataset_kwargs.update(self.cfg.other_cfgs)
        
    def setup(self, stage: str = 'fit') -> None:
        train_data, valid_data, test_data = None, None, None
         # import Dataset Class
        dataset_module = importlib.import_module(self.cfg.module)
        CustomDataset = getattr(dataset_module, self.cfg.class_name)
        
        # prepare dataset
        if self.cfg.inference_mode:  # whether in inference mode
            if "needle" in self.cfg.data_path.lower():  # sanity check passkey search data
                if self.cfg.processed_data_path is None:  # preporcess the passkey_search data on-the-fly
                    processed_data = CustomDataset.build_dataset(
                        fpath=os.path.join(self.root_dir, self.cfg.data_path), 
                        key=self.cfg.key,
                        value=self.cfg.value,
                        ctx_len=self.cfg.max_seq_length,
                        tokenizer=self.tokenizer,
                    )
                    auto_save_data(...)  # auto save processed data fn
                    raise NotImplementedError
            
            if self.cfg.processed_data_path is not None:
                test_data = auto_read_data(self.cfg.processed_data_path)
            else:
                test_data = auto_read_data(self.cfg.test_data_path)
            
            self.test_dataset = CustomDataset(
                content=test_data, 
                tokenizer=self.tokenizer, 
                split="test",
                **self.dataset_kwargs,
            )
        else:
            if self.cfg.processed_data_path is not None:
                # check if is a directory
                processed_data_path = os.path.join(self.root_dir, self.cfg.processed_data_path)
                if os.path.isdir(processed_data_path):
                    for split in self.cfg.split:  # TODO: support multiple splits
                        if "train" in split:
                            train_data = auto_read_data(os.path.join(processed_data_path, split))
                        elif "valid" in split:
                            valid_data = auto_read_data(os.path.join(processed_data_path, split))
                        else:
                            raise NotImplementedError(f"split {split} is not supported")
                else:
                    content = auto_read_data(processed_data_path)
                    min_valid_num = min(1000, len(content)*0.1)
                    valid_data = content[:min_valid_num]
                    train_data = content[min_valid_num:]

        # check data initialization     
        assert train_data is not None, "train data is None"
        try:
            assert valid_data is not None, "valid data is None"
        except:
            pass

        # init dataset
        self.train_dataset = CustomDataset(
            content=train_data, 
            tokenizer=self.tokenizer, 
            split="train",
            **self.dataset_kwargs,
        )
        print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
        
        if valid_data is not None:
            self.valid_dataset = CustomDataset(
                content=valid_data, 
                tokenizer=self.tokenizer, 
                split="valid",
                **self.dataset_kwargs,
            )
            print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')

            
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.train_batch_size, 
            num_workers=self.cfg.nworkers, 
            pin_memory=self.cfg.pin_memory, 
            drop_last=True, 
            shuffle=False if self.cfg.cluster_batch else True, 
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset, 
                batch_size=self.cfg.val_batch_size, 
                num_workers=self.cfg.nworkers, 
                pin_memory=self.cfg.pin_memory, 
                drop_last=False, 
                shuffle=False,
            )
        return None
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset, batch_size=1, 
                num_workers=self.cfg.nworkers, pin_memory=self.cfg.pin_memory, drop_last=False, shuffle=False,
            )
        return None
    
