import torch
import os
import sys
sys.path.append(os.getcwd())
import pytorch_lightning as pl
import hydra
import importlib
from torch import optim, Tensor 
from transformers import AutoTokenizer, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from custom_mamba.position_mamba import PositionMamba
from modelzipper.tutils import *
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

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
        
    def setup(self, stage: str = 'fit') -> None:
        train_data, valid_data, test_data = None, None, None
        
        # prepare dataset
        if self.cfg.inference_mode:  # whether in inference mode
            self.test_data = auto_read_data(self.cfg.test_data_path)
            self.test_dataset = CustomDataset(
                content=self.test_data, 
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

        # import Dataset Class
        dataset_module = importlib.import_module(self.cfg.module)
        CustomDataset = getattr(dataset_module, self.cfg.class_name)

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
    

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="train") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.tokenizer = tokenizer
        self.cfg = config
        self.platform_cfg = config.platform

        if state == "train":
            self.loss_fct = torch.nn.CrossEntropyLoss()

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

        lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        ppl = torch.exp(lm_loss)
        
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("train_ppl", ppl, sync_dist=True, prog_bar=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.pop("input_ids")
        lm_logits = self.forward(input_ids).logits
        labels = batch.pop("labels")
        labels = labels.to(lm_logits.device)
        
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        ppl = torch.exp(lm_loss)
        
        self.log("valid_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("valid_ppl", ppl, sync_dist=True, prog_bar=True)


    def configure_optimizers(self):
        # init optimizer
        if self.cfg.optimizer.optimizer_type.lower() == "adamw":
            optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.cfg.optimizer.lr,
            )
        else: # implement with adam as default 
            betas = (self.cfg.experiment.beta_1, self.cfg.experiment.beta_2)
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.cfg.experiment.peak_lr,
                weight_decay=self.cfg.experiment.weight_decay, 
                betas=betas, 
                eps=self.cfg.experiment.eps
            )
        
        # init lr scheduler
        if self.cfg.lr_scheduler.scheduler_type == "get_cosine_schedule_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.cfg.lr_scheduler.warmup_steps,
                num_training_steps=self.cfg.experiment.num_training_steps,
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
                self.cfg.experiment.num_training_steps, 
                self.cfg.experiment.warmup_steps, 
                self.cfg.experiment.peak_lr, 
                self.cfg.experiment.last_lr
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


class TransformerExperiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="train") -> None:
        super(TransformerExperiment, self).__init__()
        self.model = model
        self.model.train()
        self.tokenizer = tokenizer
        self.cfg = config
        self.platform_cfg = config.platform

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)


    def training_step(self, batch, batch_idx):
        lm_loss = self.model(**batch).loss
        ppl = torch.exp(lm_loss)
        
        self.log("train_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("train_ppl", ppl, sync_dist=True, prog_bar=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        lm_loss = self.model(**batch).loss
        ppl = torch.exp(lm_loss)
        
        self.log("valid_lm_loss", lm_loss, sync_dist=True, prog_bar=True)
        self.log("valid_ppl", ppl, sync_dist=True, prog_bar=True)
        return lm_loss


    def configure_optimizers(self):
        # init optimizer
        if self.cfg.optimizer.optimizer_type.lower() == "adamw":
            optimizer = transformers.AdamW(  # transformers.AdamW
                self.model.parameters(), 
                lr=self.cfg.optimizer.lr,
            )
        else: # implement with adam as default 
            betas = (self.cfg.experiment.beta_1, self.cfg.experiment.beta_2)
            optimizer = optim.Adam(   # optim.Adam
                self.model.parameters(),
                lr=self.cfg.experiment.peak_lr,
                weight_decay=self.cfg.experiment.weight_decay, 
                betas=betas, 
                eps=self.cfg.experiment.eps
            )
        
        # init lr scheduler
        if self.cfg.lr_scheduler.scheduler_type == "get_cosine_schedule_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.cfg.lr_scheduler.warmup_steps,
                num_training_steps=self.cfg.experiment.num_training_steps,
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
                self.cfg.experiment.num_training_steps, 
                self.cfg.experiment.warmup_steps, 
                self.cfg.experiment.peak_lr, 
                self.cfg.experiment.last_lr
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


def get_model_tokenizer(root_dir, model_config):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)

    if "gpt" in model_path.lower():
        model = GPTNeoForCausalLM.from_pretrained(
            model_path, use_cache=False, torch_dtype=torch.bfloat16
        ).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif "mamba" in model_path.lower():
        model = PositionMamba.from_pretrained(
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
    
    return model, tokenizer


@hydra.main(config_path='../configs/', config_name='train_mamba', version_base='1.1')
def main(config):

    # print_c(f"Conduct Experiment: {config.exp_task} | Model: {config.model} | State: {config.state} | Platform: {config.platform}", "magenta")
    print_c(OmegaConf.to_yaml(config), "yellow")
    
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.exp_path
    data_root_dir = config.platform.dataset_path

    pl.seed_everything(config.experiment.seed, workers=True)
    
    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(model_root_dir, config.model)
    
    # load data
    data_module = CustomDatamodule(config.task.dataset, data_root_dir, tokenizer)
    
    # load experiment
    experiment = TransformerExperiment(model, config, tokenizer=tokenizer, state="train")
    
    # init logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(save_root_dir, config.experiment.model_save_dir), 
        name=f"{config.exp_task}",
        version=config.JOB_ID
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_monitor = ModelCheckpoint(
        save_top_k=config.experiment.save_top_k, 
        dirpath =os.path.join(tb_logger.log_dir, "checkpoints"), 
        monitor=config.experiment.monitor_metric,
        filename=f"mamba-{config.exp_task}"+"-{epoch:02d}",
        save_last=True,
        mode='min',
        save_weights_only=True, # only save state dict
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(tb_logger.log_dir , "checkpoints"),
        logger=tb_logger,
        callbacks=[lr_monitor, ckpt_monitor],
        check_val_every_n_epoch=1 if data_module.val_dataloader is not None else 1000000,  # set a large number if no validation set
        # strategy=DDPStrategy(find_unused_parameters=False),
        strategy="deepspeed_stage_2",
        precision="bf16-mixed",
        max_steps=config.experiment.num_training_steps,
        devices=config.experiment.device_num,
        gradient_clip_val=1,
        enable_model_summary=True,
        num_sanity_val_steps=20,
        fast_dev_run=5 # for debugging
    )

    trainer.fit(experiment, datamodule=data_module)


if __name__ == '__main__':
    main()








