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
    x_0_y_0 = torch.zeros((m_x.size(0), m_x.size(1), 2), dtype=m_x.dtype, device=m_x.device)
    x_0_y_0[:, 1:, 0] = m_x[:, :-1, -2]  # x_3 of the previous row
    x_0_y_0[:, 1:, 1] = m_x[:, :-1, -1]  # y_3 of the previous row
    full_x = torch.cat((m_x[:, :, :1], x_0_y_0, m_x[:, :, 1:]), dim=2)

    # replace the command value to 0, 1, 2
    full_x[:, :, 0][full_x[:, :, 0] == 100] = 1
    full_x[:, :, 0][full_x[:, :, 0] == 200] = 2

    return full_x


def merge_dicts(dict_list):
    merge_res = {k: [] for k in dict_list[0].keys()} 
    for key in merge_res.keys():
        items = [d[key] for d in dict_list if key in d]
        if items and isinstance(items[0], torch.Tensor):
            merge_res[key] = torch.cat(items, dim=0).cpu()
        elif items and isinstance(items[0], List):
            num_tensors = len(items[0])
            tensor_lists = [[] for _ in range(num_tensors)]
            for sublist in items:
                for i, tensor in enumerate(sublist):
                    tensor_lists[i].append(tensor)
            merge_res[key] = [torch.cat(t, dim=0).cpu() for t in tensor_lists]
        else:
            raise ValueError(f'Unsupported data type for merge: {type(items[0])}')
    return merge_res


class Experiment(pl.LightningModule):

    def __init__(self, model, config, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.return_all_quantized_res = True
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

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        outputs, _, _ = self.forward(batch, return_all_quantized_res=True)
        output = outputs[self.cfg.experiment.compress_level - 1]
        output = self.denormalize_func(output)
        post_process_output = postprocess(output)
        golden = batch['svg_path']
        golden[:, :, 0][golden[:, :, 0] == 100] = 1
        golden[:, :, 0][golden[:, :, 0] == 200] = 2
        standard_test_reconstruct = {
            "raw_predict": output,
            "p_predict": post_process_output,
            "golden": golden,
        }

        if self.return_all_quantized_res:
            zs, xs_quantised = self.model.encode(batch['svg_path'], start_level=0, end_level=None)
            standard_test_reconstruct.update({
                "zs": zs[self.cfg.experiment.compress_level - 1],
                "xs_quantised": xs_quantised[self.cfg.experiment.compress_level - 1]
            })

        return standard_test_reconstruct
    

@hydra.main(config_path='./configs/experiment', config_name='config_test', version_base='1.1')
def main(config):
    print_c(f"compress_level: {config.experiment.compress_level}", "magenta")
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
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    m_predictions = merge_dicts(predictions)
    save_path = os.path.join(config.experiment.prediction_save_path, f"compress_level_{config.experiment.compress_level}_predictions.pkl")
    auto_save_data(m_predictions, save_path)
    print_c(f"save predictions to {save_path}")

if __name__ == '__main__':
    main()