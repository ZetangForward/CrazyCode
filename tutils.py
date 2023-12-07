import json
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration,AutoModelForCausalLM,LlamaForCausalLM, LlamaTokenizer
from termcolor import colored  
from typing import List, Dict
import matplotlib.pyplot as plt  
import random 
import time
import math
import pickle
import yaml
import types


print(colored('CrazyCode aleady loaded, status: >>> ready <<<', 'green'))  


def print_c(s, c='green'):
    print(colored(s, color=c))


def auto_load_hf_casual_models(model_name_or_path, device="auto"):
    print_c("automatically load hf casual inference models", "green")
    if 'llama' in model_name_or_path.lower() or 'alpaca' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower() or 'StableBeluga' in model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(model_name_or_path,device_map=device, torch_dtype=torch.bfloat16)
    if 'gpt' in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16,device_map=device)
    if 't5' in model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16,device_map=device)
    tokenizer = Autokenizer.from_pretrained(model_name_or_path)
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
    print_c("config: {}".format(config), "purple")
    
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
    save_jsonl(res, output_file)
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
        save_jsonl(item, os.path.join(output_dir, f"{origin_file_name}_{i}.jsonl"))
        
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
        raise ValueError(f"file_type {file_type} not supported!")
    
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

