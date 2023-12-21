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
from my_imp import EncodecModel
 
class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 kld_weight: float = 0.000025,
                 warmup_steps: int = None,
                 num_training_steps: int = None) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.curr_device = None
        self.hold_graph = False
        self.kld_weight = kld_weight
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
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

        optims = []
        scheds = []

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                # scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma = self.params['scheduler_gamma'])
                
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps) 
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return {
                    "optimizer": optims[0],
                    "lr_scheduler": scheds[0]    
                }
        except:
            return optims


class MyDataset(Dataset):
    def __init__(self, svg_file, max_seq_len=512, split="train"):
        self.split = split
        ## Load SVG data
        with open(svg_file, "r") as f2:
            self.content = [json.loads(line) for line in f2]
        self.max_seq_len = max_seq_len
        
    def extract_numerical(self, path_data):  
        ## match all numericals 
        number_pattern = r"-?\d+\.?\d*"  
        numbers = re.findall(number_pattern, path_data)  
        numbers = [float(item) for item in numbers]
        return numbers 
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        data = self.content[index]
        svg_path_with_prompt = data["compress_path"]
        # extract all the numerical values
        extracted_numericals = self.extract_numerical(svg_path_with_prompt)
        # extracted_numericals = [min(item, 200) for item in extracted_numericals]
        # extracted_numericals = [item / 205 if item > -1 else -1 for item in extracted_numericals]
        # extracted_numericals = [math.log(item + 1) if item > -1 else -1 for item in extracted_numericals]  
        return extracted_numericals
    
    @classmethod  
    def custom_datacollator(cls, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  
        """Collate examples for supervised fine-tuning."""  
        
        padding_mask, batch_numerical_input_ids = [], []   
        max_numerical_nums = 512  
    
        for ins in instances:    
            ## process numerical values    
            ### Step 1: convert to float tensor    
            numerical_values = torch.FloatTensor(ins)
            ori_len = numerical_values.shape[0]
            ### Step 2: truncate to max length and pad to the longest length    
            numerical_values = numerical_values[:max_numerical_nums]    
            ori_len = min(max_numerical_nums, ori_len)
            numerical_values = torch.cat([numerical_values[:ori_len], torch.full((max_numerical_nums - ori_len,),255)])  # utilize 255 for padding token
            ### Step 3: create padding mask    
            padding_mask.append([1]*ori_len + [0]*(max_numerical_nums - ori_len))  
            
            batch_numerical_input_ids.append(numerical_values)    
    
        batch_numerical_input_ids = torch.stack(batch_numerical_input_ids, dim=0)    
        padding_mask = torch.tensor(padding_mask, dtype=torch.long)    
    
        return {    
            "batch_numerical_input_ids": batch_numerical_input_ids,    
            "padding_mask": padding_mask,    
        }  



@hydra.main(config_path='encodec/confg', config_name='config')
def main(config):
    print(config)

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