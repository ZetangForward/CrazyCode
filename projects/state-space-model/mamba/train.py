from modelzipper.tutils import *
from modelzipper.datamanager.base_dataset import datamodule
from transformers import AutoTokenizer
import torch
import argparse
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import pytorch_lightning as pl
import hydra
from data import custom_datamodule
from modelzipper.tutils import *
from torch import optim, Tensor 


class Experiment(pl.LightningModule):

    def __init__(self, model, config, state="train") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.train()
        self.cfg = config
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
        betas = (self.cfg.experiment.beta_1, self.cfg.experiment.beta_2)
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


@hydra.main(config_path='./configs/', config_name='train_mamba', version_base='1.1')
def main(config):
    pl.seed_everything(config.seed, workers=True)
    
    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(config.model, config.tokenizer)
    
    # load data
    data_module = custom_datamodule(config.dataset)
    
    
    

def get_model_tokenizer(model_config, tokenizer_config):
    model = MambaLMHeadModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_name_or_path)
    return model, tokenizer


    



if __name__ == '__main__':
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-2.8b")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./data/ultrachat_small.jsonl")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)







