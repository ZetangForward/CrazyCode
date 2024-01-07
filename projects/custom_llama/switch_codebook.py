"""
Scripts for:
1. get intermediate quantized results
2. get the final quantized results given the predicted quantized results
"""

import os  
import torch   
import pytorch_lightning as pl
from torch import Tensor  
import hydra  
from data.svg_data import SvgDataModule
from modelzipper.tutils import *
from models.vqvae import VQVAE
from models.utils import *


def postprocess(x, padding_mask=None, path_interpolation=True):
    """
    postprocess the generated results

    x: batch_size x seq_len x 9
    padding_mask: batch_size x seq_len
    path_interpolation: whether to interpolate the path
    """
    dtype = x.dtype
    if path_interpolation:  
        # conduct path interpolation
        # return List[Tensor]
        batch_size = x.size(0)
        x = remove_padding(x, padding_mask)  # remove the padding
        full_x = []
        for i in range(batch_size):
            current_path = []
            last_x3, last_y3 = None, None
            seq_len = x[i].size(0)
            for j in range(seq_len):
                row = x[i][j]
                cmd = 100 * torch.round(row[0] / 100).item()
                cmd = 1 if cmd == 100 else 2 if cmd == 200 else 0
                x0, y0, x1, y1, x2, y2, x3, y3 = map(lambda coord: min(max(coord, 0), 200), row[1:].tolist())
                if last_x3 is not None and (last_x3 != x0 or last_y3 != y0):
                    # if the current row's start point is not the same as the previous row's end point
                    current_path.append([0, last_x3, last_y3, 0, 0, 0, 0, x0, y0])
                if cmd in [0, 100]:
                    # if the current row is M or L, set control point to 0
                    x1, y1, x2, y2 = 0, 0, 0, 0
                current_path.append([cmd, x0, y0, x1, y1, x2, y2, x3, y3])
                last_x3, last_y3 = x3, y3  # update the last end point
            full_x.append(torch.tensor(current_path, dtype=dtype))
    
    else:  # no path interpolation
        if x.size(-1) == 9:
            # first remove the 1, 2 columns
            m_x = torch.cat((x[:, :, :1], x[:, :, 3:]), dim=2)
        else:
            m_x = x
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
        # remove the padding
        full_x = remove_padding(full_x, padding_mask)

    return full_x


def remove_padding(x, padding_mask):
    """
    x: (batch_size x) seq_len num_bins
    padding: seq_len x num_bins

    if batch, return List[Tensor]
    if tensor, return Tensor
    """
    if x.ndim == 2:  # seq_len x num_bins
        return x[:padding_mask.sum(), :]
    elif x.ndim == 3:  # batch_size x seq_len x num_bins
        res = []
        for i in range(x.size(0)):
            res.append(x[i, :padding_mask[i].sum(), :])
        return res
    
    
def sanint_check_golden(x):
    """
    x: batch_size x seq_len x (7, 9)
    """
    # replace the command value to 0, 1, 2
    x[:, :, 0][x[:, :, 0] == 100] = 1
    x[:, :, 0][x[:, :, 0] == 200] = 2
    if x.size(-1) == 9:
        return x
    elif x.size(-1) == 7:
        # add two columns
        x_0_y_0 = torch.zeros((x.size(0), x.size(1), 2), dtype=x.dtype, device=x.device)
        x_0_y_0[:, 1:, 0] = x[:, :-1, -2]  # x_3 of the previous row
        x_0_y_0[:, 1:, 1] = x[:, :-1, -1]  # y_3 of the previous row
        full_x = torch.cat((x[:, :, :1], x_0_y_0, x[:, :, 1:]), dim=2)
    return full_x


def merge_dicts(dict_list, device='cpu'):
    merge_res = {k: [] for k in dict_list[0].keys()} 
    for key in merge_res.keys():
        print_c(f"begin to merge {key}", "magenta")
        items = [d[key] for d in dict_list if key in d]
        # process items
        if items and isinstance(items[0], torch.Tensor):
            tmp_tensors = []
            flag = False
            for sublist in items:
                if isinstance(sublist, torch.Tensor):
                    flag = True
                    tmp_tensors.append(sublist)
                elif isinstance(sublist, list):
                    tmp_tensors.extend(sublist)
            if flag:
                tmp_tensors = torch.cat(tmp_tensors, dim=0).to(device)
            else:
                tmp_tensors = [tmp.cpu() for tmp in tmp_tensors]
            merge_res[key] = tmp_tensors  # each row is a tensor

        elif items and isinstance(items[0], List):
            tmp_lists = []
            for sublist in items:
                tmp_lists.extend(sublist)
            tmp_lists = [tmp.to(device) for tmp in tmp_lists]
            merge_res[key] = tmp_lists  # each row is a tensor
        else:
            raise ValueError(f'Unsupported data type for merge: {type(items[0])}')
    return merge_res


class ObtainSVGCode(pl.LightningModule):
    def __init__(self, model, config, state="eval") -> None:
        super(ObtainSVGCode, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.return_all_quantized_res = config.experiment.return_all_quantized_res
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
        svg_tensors, padding_mask = batch['svg_path'], batch['padding_mask']
        zs, xs_quantised = self.model.encode(svg_tensors, start_level=0, end_level=1) # just get the first compressed level

        standard_results = {
            "svg_tensors": svg_tensors.squeeze(),
            "padding_mask": padding_mask.squeeze(),
            "keys": batch['keywords'][0],
            "xs_quantised": xs_quantised[0].squeeze(),  # just return the first compressed level
            "zs": zs[0].squeeze(), # just return the first compressed level
        }

        return standard_results


@hydra.main(config_path='./configs/experiment', config_name='config_switch_codebook', version_base='1.1')
def main(config):
    print_c(f"compress_level: {config.experiment.compress_level}", "magenta")
    # set training dataset

    # sanity check
    snap_id = config.experiment.snap_id
    dataset_id = int(os.path.basename(config.dataset.test_data_path).split('.')[0].split('_')[-1])
    assert snap_id == dataset_id, f"dataset_id: {dataset_id} != snap_id: {snap_id}"
    
    # load snap datasets
    data_module = SvgDataModule(config.dataset)

    block_kwargs = dict(
        width=config.vqvae_conv_block.width, 
        depth=config.vqvae_conv_block.depth, 
        m_conv=config.vqvae_conv_block.m_conv,
        dilation_growth_rate=config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )

    vqvae = VQVAE(config, multipliers=None, **block_kwargs)
    experiment = ObtainSVGCode(vqvae, config)

    tester = pl.Trainer(devices=config.experiment.device_num)

    predictions = tester.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        ckpt_path=config.experiment.ckeckpoint_path
    )

    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    # m_predictions = merge_dicts(predictions) # don't merge dicts

    save_path = os.path.join(config.experiment.prediction_save_path, f"inference_full_data_compress_{config.experiment.compress_level}_snaps_{snap_id}.pkl")
    b_t = time.time()
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()