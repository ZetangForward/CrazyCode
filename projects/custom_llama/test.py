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



def postprocess(x):
    """
    x: batch_size x seq_len x 9
    """
    # first remove the 1, 2 columns
    m_x = torch.cat((x[:, :, :1], x[:, :, 3:]), dim=2)
    # find the right command value
    m_x[:, :, 0] = torch.round(m_x[:, :, 0] / 100) * 100
    # clip all the value to max bins 
    m_x = torch.clamp(m_x, 0, 200)
    # process the M and L path
    m_x[:, :, 1:5][m_x[:, :, 0] != 200] = 0
    # add to extra column to satisfy the 9 columns
    x_0_y_0 = torch.zeros((m_x.size(0), m_x.size(1), 2), dtype=m_x.dtype)
    x_0_y_0[:, 1:, 0] = m_x[:, :-1, -2]  # x_3 of the previous row
    x_0_y_0[:, 1:, 1] = m_x[:, :-1, -1]  # y_3 of the previous row
    full_x = torch.cat((m_x[:, :, :1], x_0_y_0, m_x[:, :, 1:]), dim=2)

    return full_x



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
        
    
    def denormalize_func(self, normalized_tensor, min_val=0, max_val=200):
        tensor = (normalized_tensor + 1) / 2
        tensor = tensor * (max_val - min_val) + min_val
        tensor = torch.round(tensor).long()
        return tensor


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
        output, _, metrics = self.forward(batch)
        output = self.denormalize_func(output)
        import pdb; pdb.set_trace()
        post_process_output = postprocess(output)
        import pdb; pdb.set_trace()

        return {
            "raw_predict": output,
            "p_predict": post_process_output,
            "golden": batch['svg_path'], 
            "metrics": metrics,
        }


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


@hydra.main(config_path='./configs/experiment', config_name='config_test')
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
    # experiment = Experiment.load_from_checkpoint(config.experiment.ckeckpoint_path)

    trainer = pl.Trainer(devices=config.experiment.device_num)

    # print(f"======= Training {config['model_params']['name']} =======")
    predictions = trainer.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        ckpt_path=config.experiment.ckeckpoint_path
    )
    
    import pdb; pdb.set_trace()

    save_path = os.path.join(config.experiment.prediction_save_path, "predictions.pkl")
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}")

if __name__ == '__main__':
    main()