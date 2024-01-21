import random
import os
import transformers
import sys
sys.path.append("/workspace/zecheng/modelzipper/projects/custom_llama")
from dataclasses import dataclass, field
from transformers import Trainer
from modelzipper.tutils import *
from data.vqseq2seq_dataset import OfflineBasicDataset
from models.vqvae import VQVAE, postprocess
from data.svg_data import *
import pytorch_lightning as pl
from utils.visualize_svg import convert_svg
import transformers
from tqdm import trange
from PIL import Image

FILE_PATH = "/zecheng2/svg/icon-shop/test_data_snaps/test_data_all_seq_with_mesh.pkl"

VQVAE_CONFIG_PATH = "/workspace/zecheng/modelzipper/projects/custom_llama/configs/deepspeed/vqvae_config_v2.yaml"
DATA_PATH = "/zecheng2/svg/icon-shop/test_data_snaps/test_data_all_seq_with_mesh.pkl"

tokenizer = transformers.AutoTokenizer.from_pretrained("/zecheng2/model_hub/flan-t5-xl")

content = auto_read_data(DATA_PATH)
dataset = OfflineBasicDataset(content=content, tokenizer=tokenizer, mode='test')
vqvae_config = load_yaml_config(VQVAE_CONFIG_PATH)

block_kwargs = dict(
        width=vqvae_config.vqvae_conv_block.width, 
        depth=vqvae_config.vqvae_conv_block.depth, 
        m_conv=vqvae_config.vqvae_conv_block.m_conv,
        dilation_growth_rate=vqvae_config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=vqvae_config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=vqvae_config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )

def add_background(image_obj=None, save_suffix="b", raw_image_size_w=None, raw_image_size_h=None):
    image = image_obj
   
    sub_image_w = raw_image_size_w if raw_image_size_w is not None else image.size[0]
    sub_image_h = raw_image_size_h if raw_image_size_h is not None else image.size[1]

    new_image_size = (sub_image_w, sub_image_h)
    background_image = Image.new('RGB', new_image_size)

    background_image.paste(image, (0, 0))

    return background_image

class PluginVQVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

vqvae = VQVAE(vqvae_config, multipliers=None, **block_kwargs)
plugin_vqvae = PluginVQVAE(vqvae)
checkpoint = torch.load(vqvae_config.ckpt_path)  # load vqvae ckpt
plugin_vqvae.load_state_dict(checkpoint['state_dict'])
plugin_vqvae.eval()
plugin_vqvae.cuda()
plugin_vqvae.model.half()

vq_test = []
for i in trange(200):
    
    try:
        sample = dataset[i]
    except:
        continue
    
    keys = tokenizer.decode(dataset[i]['text_input_ids'], skip_special_tokens=True)
    cur_save_case = {"keys": keys}
    zs = dataset[i]['svg_tensors'][1:]
    cur_save_case['zs_len'] = len(zs)
    with torch.no_grad():
        PI_RES = plugin_vqvae.model.decode(zs.unsqueeze(0).cuda(), 0, 1, padding_mask=None, path_interpolation=True, return_postprocess=True)[0]
        PC_RES = plugin_vqvae.model.decode(zs.unsqueeze(0).cuda(), 0, 1, padding_mask=None, path_interpolation=False, return_postprocess=True)[0]
        
        cur_save_case['pi_res_len'] = PI_RES.size(0)
        cur_save_case['pc_res_len'] = PC_RES.size(0)
        cur_save_case['gt_res_len'] = dataset[i]['mesh_data'].size(0)
        
        PI_RES_IMAGE_PATH = os.path.join("/zecheng2/evaluation/test_vq/version_8/image", f"PI_{i}.png")
        PC_RES_IMAGE_PATH = os.path.join("/zecheng2/evaluation/test_vq/version_8/image", f"PC_{i}.png")
        GT_IMAGE_PATH = os.path.join("/zecheng2/evaluation/test_vq/version_8/image", f"GT_{i}.png")
        
        PI_RES_image, PI_RES_str = convert_svg(PI_RES, True, PI_RES_IMAGE_PATH)
        PC_RES_image, PC_RES_str = convert_svg(PC_RES, True, PC_RES_IMAGE_PATH)
        GOLDEN_image, GT_str = convert_svg(dataset[i]['mesh_data'], True, GT_IMAGE_PATH)
        
        cur_save_case['pi_res_str'] = PI_RES_image.numericalize(n=200).to_str()
        cur_save_case['pc_res_str'] = PC_RES_image.numericalize(n=200).to_str()
        cur_save_case['gt_str'] = GOLDEN_image.numericalize(n=200).to_str()
        
        # PI_RES_image_b = add_background(PI_RES_image)
        # PC_RES_image_b = add_background(PC_RES_image)
        # GT_RES_image_b = add_background(GT_IMAGE_PATH)
        
        # PI_RES_image.save_png(PI_RES_IMAGE_PATH)
        # PC_RES_image.save_png(PC_RES_IMAGE_PATH)
        # GOLDEN_image.save_png(GT_IMAGE_PATH)
        
        cur_save_case['PI_RES_image_path'] = PI_RES_IMAGE_PATH
        cur_save_case['PC_RES_image_path'] = PC_RES_IMAGE_PATH
        cur_save_case['GT_image_path'] = GT_IMAGE_PATH
        
        vq_test.append(cur_save_case)
    
auto_save_data(vq_test, "/zecheng2/evaluation/test_vq/version_8/vq_test.pkl")
        