import random
import os
import transformers
import sys
sys.path.append("/workspace/zecheng/modelzipper/projects/custom_llama")
from dataclasses import dataclass, field
from transformers import Trainer
from modelzipper.tutils import *
from data.vqllama_dataset import VQDataCollator, VQLLaMAData
from models.vqvae import VQVAE
from data.svg_data import *
import pytorch_lightning as pl

VQVAE_CONFIG_PATH = "/workspace/zecheng/modelzipper/projects/custom_llama/configs/deepspeed/vqvae_config.yaml"
DATA_PATH = "/zecheng2/svg/icon-shop/test_data_snaps/test_mesh_data_svg_convert_p.pkl"

content = auto_read_data(DATA_PATH)
dataset = BasicDataset(dataset=content)
vqvae_config = load_yaml_config(VQVAE_CONFIG_PATH)

block_kwargs = dict(
        width=vqvae_config.vqvae_conv_block.width, 
        depth=vqvae_config.vqvae_conv_block.depth, 
        m_conv=vqvae_config.vqvae_conv_block.m_conv,
        dilation_growth_rate=vqvae_config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=vqvae_config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=vqvae_config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )

class PluginVQVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

vqvae = VQVAE(vqvae_config, multipliers=None, **block_kwargs)
plugin_vqvae = PluginVQVAE(vqvae)
checkpoint = torch.load(vqvae_config.ckpt_path)  # load vqvae ckpt
plugin_vqvae.load_state_dict(checkpoint['state_dict'])
plugin_vqvae.eval()
plugin_vqvae.cpu()


def cal_compress_padding_mask(x):

    # 确保长度是偶数，如果是奇数，可以添加一个值以配合压缩逻辑
    if len(x) % 2 != 0:
        x = torch.cat((x, torch.tensor([False])))

    # 压缩mask
    # 使用.view(-1, 2)将原始mask分为两列，然后使用.any(dim=1)检查每对是否有任何True值
    x = x.view(-1, 2).any(dim=1)
    
    return x


sample = dataset[0]['svg_path']
max_seq_len = 512
padded_sample = torch.concatenate([sample, torch.zeros(max_seq_len - sample.shape[0], 9)])
padding_mask = ~(padded_sample == 0).all(dim=1, keepdim=True).squeeze()
compress_padding_mask = cal_compress_padding_mask(padding_mask)
svg_token_ids, _ = plugin_vqvae.model.encode(padded_sample.unsqueeze(0), start_level=0, end_level=1)
svg_token_ids = svg_token_ids[0]  # 这里是不加padding mask的svg token ids

remain_svg_token_ids = svg_token_ids[:, :compress_padding_mask.sum()] # 这里是加入padding mask的svg token ids

postprocess_output = plugin_vqvae.model.decode(svg_token_ids, 0, 1, padding_mask, True, True)


