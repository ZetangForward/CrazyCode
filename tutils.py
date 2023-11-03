import json
import os
from termcolor import colored  
from typing import List, Dict
import matplotlib.pyplot as plt  
import random 
import time
import math

print(colored('Load CrazyCode -- Road is under your feet, ZetangForward', 'green'))  


def count_words(s: str):
    '''
    Count words in a string
    '''
    return len(s.split())   

def print_c(s, c='green'):
    print(colored(s, color=c))


def visualize_batch_images(batch_images, ncols=6, nrows=6, subplot_size=2, output_file=None):
    '''
    Visualize a batch of images
    '''
    
    images = batch_images  
    n = len(images)  # 图像的数量  
    assert n == ncols * nrows, f"None match images: {n} != {ncols * nrows}"
    # ncols = 5  # 设置每行显示的图像数量  
    # nrows = n // ncols + (n % ncols > 0)  # 计算行数 
    
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
        plt.savefig(output_file)
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


def save_jsonl(lst: List[Dict], file_path):
    with open(file_path, "w") as f:
        for item in lst:
            json.dump(item, f)
            f.write("\n")
    print_c("jsonl file saved successfully!")


def sample_dict_items(dict_, n=3):
    print_c(f"sample {n} items from dict", 'green')
    cnt = 0
    for key, value in dict_.items():  
        print(f'Key: {key}, Value: {value}')  
        cnt += 1
        if cnt == n:
            break


def filter_jsonl_lst(lst: List[Dict], kws: List[str]=None):
    '''
    
    '''
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