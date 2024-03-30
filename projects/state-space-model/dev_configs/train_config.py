
from modelzipper.tutils import *
from optimizer_config import Optimizer
from platform_config import platform_configs
from task_configs import task_configs
from model_configs import model_configs

def merge_configs(default, custom):
    final_config = default.copy()
    for key, value in custom.items():
        if isinstance(value, dict):
            # 假设所有自定义配置都是字典，这样我们就可以递归地合并
            node = final_config.setdefault(key, {})
            final_config[key] = merge_configs(node, value)
        else:
            final_config[key] = value
    return final_config

# define default configs





class DotDict(dict):
    """一个可以通过点访问的字典。"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value



default_config = {
    "platform": "amax_a100",
    "mark": 1,
    "state": "train",
    "exp_task": "simplepajama",
    "model_name": "mamba-1_4b",
    "optimizer": None, 

    # ... 其他默认设置
}

# 假设以下文件专门用于不同平台、任务和模型的特定配置
# platform_configs.py, task_configs.py, model_configs.py

