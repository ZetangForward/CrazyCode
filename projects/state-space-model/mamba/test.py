import os  
import torch   
import pytorch_lightning as pl
from torch import Tensor  
import hydra  
from data import custom_datamodule
from modelzipper.tutils import *


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
                    tmp_tensors.append(sublist)
                elif isinstance(sublist, list):
                    tmp_tensors.extend(sublist)
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


class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
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


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        outputs, _, _ = self.model.generate(batch['input_ids'], max_length=2000, temperature=0.9, top_p=0.7, eos_token_id=self.tokenizer.eos_token_id)
        output = outputs[self.cfg.experiment.compress_level - 1]
        post_process_output1 = postprocess(output, batch['padding_mask'], True)  # path interpolation
        post_process_output2 = postprocess(output, batch['padding_mask'], False)  # path interpolation
        golden = sanint_check_golden(batch['svg_path'])
        
        standard_test_reconstruct = {
            "raw_predict": output,
            "p_predict1": post_process_output1,
            "p_predict2": post_process_output2,
            "golden": golden,
        }

        if self.return_all_quantized_res:
            zs = self.model.encode(batch['svg_path'], start_level=0, end_level=None)
            standard_test_reconstruct.update({
                "zs": zs[self.cfg.experiment.compress_level - 1],
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

    tester = pl.Trainer(devices=config.experiment.device_num)

    predictions = tester.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        ckpt_path=config.experiment.ckeckpoint_path
    )
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    m_predictions = merge_dicts(predictions)
    save_path = os.path.join(config.experiment.prediction_save_path, f"compress_level_{config.experiment.compress_level}_predictions.pkl")
    b_t = time.time()
    auto_save_data(m_predictions, save_path)
    print_c(f"save predictions to {save_path}, total time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()