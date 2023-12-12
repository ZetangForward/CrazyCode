import json
import os
import random 
import time
import math
import pickle
import yaml
import types
import torch
import pdb
import transformers
import argparse
import re
import gc
import fire
import torch.nn as nn
import matplotlib.pyplot as plt  
from termcolor import colored  
from typing import List, Dict
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, TopKLogitsWarper, TemperatureLogitsWarper, TopPLogitsWarper, LogitsProcessorList



def print_c(s, c='green', *args, **kwargs):
    """
    灰色：'grey'
    红色：'red'
    绿色：'green'
    黄色：'yellow'
    蓝色：'blue'
    洋红色：'magenta'
    青色：'cyan'
    白色：'white'
    
    高亮：'on_<color>'（例如 'on_red' 会使用红色背景）
    加粗：'bold'
    下划线：'underline'
    闪烁：'blink'
    反转：'reverse'
    隐藏：'concealed'
    
    e.g., print(colored('Hello, World!', 'green', 'on_red', attrs=['blink']))
    """
    attributes = kwargs.pop('attrs', [])
    kwargs.pop('color', None)  
    # Pass 'attrs' as a keyword argument to 'colored'
    print(colored(s, color=c, attrs=attributes))


def top_k_top_p_sampling(logits: torch.FloatTensor, top_k: int = 0, top_p: float = 1.0, temperature: float = 0.7, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, num_samples = 1):
    next_token_scores = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, temperature=temperature, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)
    
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=num_samples).squeeze(1)
    
    return sampled_tokens
    

def top_k_top_p_filtering(logits: torch.FloatTensor, top_k: int = 0, top_p: float = 1.0, temperature: float = 0.7, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    """ Warning: This is modified from transformers.generation_utils.py
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    if 0 <= top_p <= 1.0:
        logits_warper = LogitsProcessorList(
            [
                TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep),
                TemperatureLogitsWarper(temperature),
            ]
        )
        logits = logits_warper(None, logits)
        
    return logits
    

def auto_load_hf_casual_models(model_name_or_path, torch_type=None, device="cpu", **kwargs):
    """
    torch_type = torch.float16
    """
    print_c("automatically load hf casual inference models", "green")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if 'llama' in model_name_or_path.lower() or 'alpaca' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower() or 'StableBeluga' in model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(model_name_or_path,device_map=device, torch_dtype=torch_type)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        
    if 'gpt' in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch_type, device_map=device)
        
    if 't5' in model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch_type, device_map=device)
    print()
    return model, tokenizer


def load_yaml_config(config_path):  
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.

    """
    
    def dict_to_simplenamespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_simplenamespace(value)
            return types.SimpleNamespace(**d)
        elif isinstance(d, list):
            return [dict_to_simplenamespace(item) for item in d]
        else:
            return d
    
    print_c("load config files from {}".format(config_path))
    with open(config_path, 'r') as config_file:  
        try:  
            config = yaml.safe_load(config_file)  
            config = dict_to_simplenamespace(config)
        except yaml.YAMLError as exc:  
            print(exc)  
            return None  
    print_c("config loaded successfully!")
    print_c("config: {}".format(config), "green", "underline")
    print()
    return config


def count_png_files(directory, file_type=".png"):  
    """
    Quick count the number of png files in a directory
    """
    len_ = len([f for f in os.listdir(directory) if f.endswith(file_type)])
    print_c(f"Total {len_} {file_type} files in {directory}")
    return len_


def random_sample_from_file(file_path, num_samples=10, output_file=None):
    '''
    Random sample from a file
    '''
    assert os.path.exists(file_path), f"{file_path} not exist!"
    content = load_jsonl(file_path)
    res = random.sample(content, num_samples)
    save_file(res, output_file)
    return res


def split_file(file_path: json, output_dir, num_snaps=3):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print_c(f"{output_dir} not exist! --> Create output dir {output_dir}")
    
    content = load_jsonl(file_path)
    
    snap_length = len(content) // num_snaps + 1

    new_content = []
    for i in range(num_snaps):
        new_content.append(content[i*snap_length:(i+1)*snap_length])
    
    origin_file_name = os.path.basename(file_path).split(".")[0]
    for i, item in enumerate(new_content):
        save_file(item, os.path.join(output_dir, f"{origin_file_name}_{i}.jsonl"))
        
    print_c(f"Split file successfully into {num_snaps} parts! Check in {output_dir}")


def count_words(s: str):
    '''
    Count words in a string
    '''
    return len(s.split())   


def save_image(image, output_file=None):
    '''
    save images to output_file
    '''
    image.save(output_file)


def visualize_batch_images(batch_images, ncols=6, nrows=6, subplot_size=2, output_file=None):
    '''
    Visualize a batch of images
    '''
    
    images = batch_images  
    n = len(images)  # 图像的数量  
    assert n == ncols * nrows, f"None match images: {n} != {ncols * nrows}"
    
    # 计算figure的宽度和高度  
    fig_width = subplot_size * ncols  
    fig_height = subplot_size * nrows  
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height)) 
    
    for i, (index, item) in enumerate(batch_images.items()):  
        title, img = item[0], item[1]
        ax = axs[i // ncols, i % ncols]  
        ax.imshow(img)  
        if title is not None:
            ax.set_title(title)  # 设置子图标题  
        ax.axis('off')  # 不显示坐标轴
        
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()  



def load_jsonl(file_path, return_format="list"):
    if return_format == "list":
        with open(file_path, "r") as f:
            res = [json.loads(item) for item in f]
        return res
    else:
        pass
    print_c("jsonl file loaded successfully!")


def save_file(lst: List, file_path):
    """
    Save a list of items to a file.
    Automatically detect the file type by the suffix of the file_path.

    Args:
        lst (List): The list of items to be saved.
        file_path (str): The path to the file.
        //* Support file types
            - jsonl
            - pkl
            - txt
        *//
        
    Raises:
        ValueError: If the file type is not supported.
    """
    
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print_c(f"{data_dir} not exist! --> Create data dir {data_dir}")
    suffix_ = file_path.split(".")[-1]
    
    if suffix_ == "jsonl":
        with open(file_path, "w") as f:
            for item in lst:
                json.dump(item, f)
                f.write("\n")
        print_c("jsonl file saved successfully!")
        
    elif suffix_ == "pkl":
        with open(file_path, "wb") as f:
            pickle.dump(lst, f)
        print_c("pkl file saved successfully!")
        
    elif suffix_ == "txt":
        with open(file_path, "w") as f:
            for item in lst:
                f.write(item + "\n")
        print_c("txt file saved successfully!")
    else:
        raise ValueError(f"file_type {suffix_} not supported!")
    
    print_c(f"Save file to {file_path} | len: {len(lst)}")


def sample_dict_items(dict_, n=3):
    print_c(f"sample {n} items from dict", 'green')
    cnt = 0
    for key, value in dict_.items():  
        print(f'Key: {key}, Value: {value}')  
        cnt += 1
        if cnt == n:
            break


def filter_jsonl_lst(lst: List[Dict], kws: List[str]=None):
    """
    Filter a list of dictionaries based on a list of keywords.

    Args:
        lst (List[Dict]): The list of dictionaries to be filtered.
        kws (List[str], optional): The list of keywords to filter the dictionaries. Defaults to None.

    Returns:
        List[Dict]: The filtered list of dictionaries.
    """
    if kws is None:
        res = lst
        print_c("Warning: no filtering, return directly!")
    else:
        res = [dict([(k, item.get(k)) for k in kws]) for item in lst]
    return res


def merge_dicts(dict1: Dict, dict2: Dict, key: str=None):
    '''
    Merge two dicts with the same key value
    '''
    pass