import os  
import torch   
import pytorch_lightning as pl
from torch import optim, Tensor  
from torchvision import transforms, utils as vutils, datasets as dsets  
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra  
from svg_data import SvgDataModule
from modelzipper.tutils import *
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
        return self.model(input['svg_path'], input['padding_mask'], **kwargs)

    def training_step(self, batch, batch_idx):
        output, loss_w, metrics = self.forward(batch)
        self.log("total_loss", loss_w, prog_bar=True)
        self.log_dict(metrics, sync_dist=True)
        return loss_w


    def validation_step(self, batch, batch_idx):
        _, loss_w, _ = self.forward(batch)
        self.log_dict({"val_loss": loss_w.item()}, sync_dist=True)
        return loss_w


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output, loss_w, metrics = self.forward(batch)
        self.log_dict(metrics, sync_dist=True)
        return loss_w


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


@hydra.main(config_path='./configs', config_name='config_test')
def main(config):

    # set training dataset
    data_module = SvgDataModule(config.dataset)

    block_kwargs = dict(
        width=config.vqvae_conv_block.width, depth=config.vqvae_conv_block.depth, m_conv=config.vqvae_conv_block.m_conv,
        dilation_growth_rate=config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )

    vqvae = VQVAE(config, multipliers=None, **block_kwargs)

    experiment = Experiment(vqvae, config)
    ckp_path = os.path.join(config.experiment.model_save_dir, config.experiment.exp_name)
    experiment.load_from_checkpoint(config.experiment.model_save_dir)

    trainer = pl.Trainer(
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
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=config.experiment.max_epoch,
        devices=config.experiment.device_num,
        gradient_clip_val=1.5,
        enable_model_summary=True,
        # fast_dev_run=True, num_sanity_val_steps=2  # for debugging
    )

    # print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(experiment, datamodule=data_module)

if __name__ == '__main__':
    main()