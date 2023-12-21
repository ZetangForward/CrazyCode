import os  
import sys  
import torch   
import pytorch_lightning as pl
import re
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim, Tensor  
from torchvision import transforms, utils as vutils, datasets as dsets  
from torch.utils.data import DataLoader, Dataset  
from pathlib import Path  
from typing import List, Optional, Sequence, Union, Any, Callable, Dict, Tuple  
from pytorch_lightning import Trainer 
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.distributed as dist
import hydra  
import random
from my_imp import EncodecModel
import librosa
import pandas as pd

class CustomExperiment(pl.LightningModule):

    def __init__(self, model, config, train_data_len) -> None:
        super(CustomExperiment, self).__init__()

        self.model = model
        self.config = config
        self.max_iter = config.experiment.max_epoch * train_data_len
        self.warmup_iter = config.lr_scheduler.warmup_epoch * train_data_len

        self.params = {
            "LR": 0.005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.000025,
            "manual_seed": 1265,
        }

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()
        numerical_inputs = batch['batch_numerical_input_ids']
        padding_mask = batch["padding_mask"]

        outputs = self.forward(numerical_inputs)
        
        all_loss = self.model.loss_function(kld_weight=self.kld_weight, padding_mask=padding_mask, *outputs)
        
        loss, batch_kld_loss, batch_recon_loss = all_loss['loss'], all_loss['KLD'], all_loss['Reconstruction_Loss']
        
        train_loss = {'loss': loss, 'Reconstruction_Loss':batch_recon_loss, 'KLD':batch_kld_loss}

        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        numerical_inputs = batch['batch_numerical_input_ids']
        padding_mask = batch["padding_mask"]

        outputs = self.forward(numerical_inputs)
        
        all_loss = self.model.loss_function(kld_weight=self.kld_weight, padding_mask=padding_mask, *outputs)
        
        loss, batch_kld_loss, batch_recon_loss = all_loss['loss'], all_loss['KLD'], all_loss['Reconstruction_Loss']
        
        val_loss = {'loss': loss, 'Reconstruction_Loss':batch_recon_loss, 'KLD':batch_kld_loss}

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        pass


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            [
                {
                    'params': self.model.parameters(), 
                    'lr': self.config.optimization.lr
                }
            ], 
            betas=(0.5, 0.9)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.warmup_iter, 
            num_training_steps=self.max_iter,
        ) 

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            self.audio_files = pd.read_csv(config.datasets.train_csv_path,on_bad_lines='skip')
        elif mode == 'test':
            self.audio_files = pd.read_csv(config.datasets.test_csv_path,on_bad_lines='skip',)
        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)  

    def __getitem__(self, idx):
        # waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # """you can preprocess the waveform's sample rate to save time and memory"""
        # if sample_rate != self.sample_rate:
        #     waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        waveform,sample_rate = librosa.load(self.audio_files.iloc[idx, :].values[0],sr=self.sample_rate)
        waveform = torch.as_tensor(waveform).unsqueeze(0)
 
        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1) # random start point
                waveform = waveform[:, start:start+self.tensor_cut] # cut tensor
                return waveform, sample_rate
            else:
                return waveform, sample_rate


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors



@hydra.main(config_path='encodec/confg', config_name='config')
def main(config):
    print(config)


     # set train dataset
    trainset = CustomAudioDataset(config=config)
    testset = CustomAudioDataset(config=config,mode='test')
    
    train_sampler, test_sampler = None, None

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler, 
        shuffle=(train_sampler is None), collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler, 
        shuffle=False, collate_fn=collate_fn,
        pin_memory=config.datasets.pin_memory)

    # set encodec model and discriminator model
    model = EncodecModel._get_model(
                config.model.target_bandwidths, 
                config.model.sample_rate, 
                config.model.channels,
                causal=False, model_norm='time_group_norm', 
                audio_normalize=config.model.audio_normalize,
                segment=None, name='my_encodec',
                ratios=config.model.ratios)

    tb_logger = TensorBoardLogger(
        save_dir=config.experiment.model_save_dir, 
        name=f"{config.experiment.exp_name}"
    )

    experiment = CustomExperiment(
        model, config, train_data_len=len(trainloader)
    )

    runner = Trainer(
        default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=5, 
                dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
                monitor= "val_loss",
                filename="pure_numerical_vae-{epoch:02d}",
                save_last= True),
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config.experiment.max_epoch,
        devices=config.experiment.device_num,
        gradient_clip_val=1.5
    )

    # print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, train_dataloaders=trainloader, val_dataloaders=testloader)

    exit()


if __name__ == '__main__':
    main()

    exit()


    num_workers = 0
    device_num = 1
    num_bins = 8
    train_file = "/zecheng/svg/icon-shop/meta_data/offline_750_train_v2.jsonl"
    valid_file = "/zecheng/svg/icon-shop/meta_data/offline_750_valid_v2.jsonl"
    
    model_save_dir = "/zecheng/svg_model_hub/vanilla_vae"
    exp_name = "vanilla_vae"
    
    batch_size = 256
    
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(beijing_tz) 
     
    time_string = now.strftime("%Y%m%d") 
    tb_logger = TensorBoardLogger(save_dir=model_save_dir, name=f"{exp_name}_{time_string}")
    
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    
    model = EncodecModel(in_channels=num_bins, latent_dim=4096)
    
    train_dataset = MyDataset(svg_file=train_file, split="train")
    valid_dataset = MyDataset(svg_file=valid_file, split="valid")
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=MyDataset.custom_datacollator,
        )
    
    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=MyDataset.custom_datacollator,
        )
    
    num_epochs = 100
    num_training_steps = len(train_dataloader) * num_epochs  
    warmup_steps = num_training_steps // 10  # 10% of training steps as warmup
    
    experiment = VAEXperiment(vae_model=model, kld_weight=0.000025, warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    
    runner = Trainer(
        default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=5, 
                dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
                monitor= "val_loss",
                filename="pure_numerical_vae-{epoch:02d}",
                save_last= True),
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=num_epochs,
        devices=device_num,
        gradient_clip_val=1.5
    )

    # print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)