import json
import os
from termcolor import colored  
from typing import List, Dict

print(colored('Load CrazyCode -- Road is under your feet, ZetangForward', 'red'))  

def print_c(s, color='green'):
    print(colored(s, color=color))


def load_jsonl(file_path, return_format="list"):
    if return_format == "list":
        with open(file_path, "r") as f:
            res = [json.loads(item) for item in f]
        return res
    else:
        pass

def sample_dict_items(dict_, n=3):
    print_c(f"sample {n} items from dict", 'green')
    cnt = 0
    for key, value in dict_.items():  
        print(f'Key: {key}, Value: {value}')  
        cnt += 1
        if cnt == n:
            break

def filter_jsonl_lst(lst: List[Dict], kws=None):
    if kws is None:
        res = lst
        print_c("Warning: no filtering, return directly!")
    else:
        res = [dict([(k, item.get(k)) for k in kws]) for item in lst]
    return res