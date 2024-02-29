import torch
import os
import sys
sys.path.append(os.getcwd())
from transformers import AutoTokenizer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import hydra
from custom_dataset.data import custom_datamodule
from modelzipper.tutils import *
from torch import optim, Tensor 
from custom_mamba.position_mamba import PositionMamba

class Experiment(pl.LightningModule):

    def __init__(self, model, config, tokenizer=None, state="train") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.cfg = config
        self.exp_cfg = config.experiment
        self.tokenizer = tokenizer
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch.pop("input_ids")
        lm_logits = self.forward(input_ids).logits
        
        labels = batch.pop("labels")
        labels = labels.to(lm_logits.device)
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.pop("input_ids")
        lm_logits = self.forward(input_ids).logits
        
        labels = batch.pop("labels")
        labels = labels.to(lm_logits.device)
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        self.log("val_lm_loss", lm_loss, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        # init optimizer
        if self.cfg.optimizer.optimizer_type.lower() == "adamw":
            optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.cfg.optimizer.lr,
            )
        else: # implement with adam as default 
            betas = (self.exp_cfg.beta_1, self.exp_cfg.beta_2)
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.exp_cfg.peak_lr,
                weight_decay=self.exp_cfg.weight_decay, 
                betas=betas, 
                eps=self.exp_cfg.eps
            )
        
        # init lr scheduler
        if self.cfg.lr_scheduler.scheduler_type == "get_cosine_schedule_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.cfg.lr_scheduler.warmup_steps,
                num_training_steps=self.exp_cfg.num_training_steps,
            )
        else:
            def get_scheduler(optimizer, num_training_steps, warmup_steps, peak_lr, last_lr):
                
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        return current_step / warmup_steps
                    progress = (current_step - warmup_steps) / (num_training_steps - warmup_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = (last_lr + (peak_lr - last_lr) * cosine_decay)
                    return lr / peak_lr
                
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            scheduler = get_scheduler(
                optimizer, 
                self.exp_cfg.num_training_steps, 
                self.exp_cfg.warmup_steps, 
                self.exp_cfg.peak_lr, 
                self.exp_cfg.last_lr
            )

        lr_scheduler = {
            'scheduler': scheduler,
            'name': f"{self.cfg.lr_scheduler.scheduler_type}",
            'interval': 'step',  # Ensure learning rate updates per step
            'frequency': 1,  # Optional: If you want to make sure it updates every step
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


@hydra.main(config_path='../configs/', config_name='train_mamba', version_base='1.1')
def main(config):
    pl.seed_everything(config.experiment.seed, workers=True)
    
    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(config.model, config.tokenizer)
    
    # load data
    data_module = custom_datamodule(config.dataset, tokenizer)
    
    # load experiment
    experiment = Experiment(model, config, tokenizer=tokenizer, state="train")
    
    # init logger
    tb_logger = TensorBoardLogger(
        save_dir=config.experiment.model_save_dir, 
        name=f"{config.experiment.task}",
        version=config.experiment.version
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
        logger=tb_logger,
        callbacks=[
            lr_monitor,
            ModelCheckpoint(
                save_top_k=5, 
                dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
                monitor="val_lm_loss",
                filename=f"mamba-{config.experiment.task}"+"-{epoch:02d}",
                save_last=True,
                mode='min',
            ),
        ],
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_steps=config.experiment.num_training_steps,
        devices=config.experiment.device_num,
        gradient_clip_val=1,
        enable_model_summary=True,
        num_sanity_val_steps=20,
        # fast_dev_run=5 # for debugging
    )

    trainer.fit(experiment, datamodule=data_module)

def get_model_tokenizer(model_config, tokenizer_config):
    model = PositionMamba.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16, device="cuda", strict=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_name_or_path)
    if "gpt-neo" in tokenizer_config.tokenizer_name_or_path:
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


if __name__ == '__main__':
    main()








