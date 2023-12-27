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
import librosa
import pandas as pd
# from vqvae import VQVAE
from svg_data import SvgDataModule
from modelzipper.tutils import *
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from models.vqvae import VQVAE
from models.utils import *

class Experiment(pl.LightningModule):

    def __init__(self, model, config, state="train") -> None:
        super(Experiment, self).__init__()

        self.model = model
        if state == "train":
            self.model.train()
        else:
            self.model.eval()

        self.cfg = config

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        output, loss_w, metrics = self.forward(batch)
        self.log_dict(metrics, sync_dist=True)
        return loss_w


    def validation_step(self, batch, batch_idx):
        output, loss_w, metrics = self.forward(batch)
        self.log_dict(metrics, sync_dist=True)
        return loss_w


    def configure_optimizers(self):
        betas = (self.cfg.experiment.beta1, self.cfg.experiment.beta2)
        optimizer = FusedAdam(
            self.model.parameters(), 
            lr=self.cfg.experiment.lr,
            weight_decay=self.cfg.experiment.weight_decay, 
            betas=betas, 
            eps=self.cfg.experiment.eps
        )

        def lr_lambda(step):
            return self.cfg.experiment.lr_scale * (self.cfg.experiment.lr_gamma ** (step // self.cfg.experiment.lr_decay)) * min(1.0, step / self.cfg.experiment.lr_warmup)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


@hydra.main(config_path='.', config_name='config')
def main(config):

    # set training dataset
    data_module = SvgDataModule(config.dataset)

    tb_logger = TensorBoardLogger(
        save_dir=config.experiment.model_save_dir, 
        name=f"{config.experiment.exp_name}"
    )

    block_kwargs = dict(
        width=config.vqvae_conv_block.width, depth=config.vqvae_conv_block.depth, m_conv=config.vqvae_conv_block.m_conv,
        dilation_growth_rate=config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )

    vqvae = VQVAE(config, multipliers=None, **block_kwargs)

    experiment = Experiment(vqvae, config)

    trainer = Trainer(
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
        gradient_clip_val=1.5,
        enable_model_summary=True,
    )

    # print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(experiment, datamodule=data_module, fast_dev_run=True, num_sanity_val_steps=2)  # for debugging
    # runner.fit(experiment, datamodule=data_module)  # for training




if __name__ == '__main__':
    main()

    exit()





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



# @hydra.main(config_path='.', config_name='config')
# def main(config):

#     # set training dataset
#     svgdatamodule = SvgDataModule(config)

#     import pdb; pdb.set_trace()

#     # set encodec model and discriminator model
#     vqvae = VQVAE(config)
    
    


#     tb_logger = TensorBoardLogger(
#         save_dir=config.experiment.model_save_dir, 
#         name=f"{config.experiment.exp_name}"
#     )

#     experiment = CustomExperiment(
#         model, config, train_data_len=len(trainloader)
#     )

#     runner = Trainer(
#         default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
#         logger=tb_logger,
#         callbacks=[
#             LearningRateMonitor(),
#             ModelCheckpoint(
#                 save_top_k=5, 
#                 dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
#                 monitor= "val_loss",
#                 filename="pure_numerical_vae-{epoch:02d}",
#                 save_last= True),
#         ],
#         strategy=DDPStrategy(find_unused_parameters=False),
#         max_epochs=config.experiment.max_epoch,
#         devices=config.experiment.device_num,
#         gradient_clip_val=1.5
#     )

#     # print(f"======= Training {config['model_params']['name']} =======")
#     runner.fit(experiment, train_dataloaders=trainloader, val_dataloaders=testloader)

#     exit()




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