import os
import sys
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
sys.path.append(current_file_dir)
import argparse
from modelzipper.tutils import *
from model_config import ModelConfig
from lr_optimizer_config import OptimizerConfig, LR_Scheduler_Config
from platform_config import PlatformConfig
from task_config import TaskConfig


def merge_configs(default, custom):
    final_config = default.copy()
    for key, value in custom.items():
        if isinstance(value, dict):
            node = final_config.setdefault(key, {})
            final_config[key] = merge_configs(node, value)
        else:
            final_config[key] = value
    return final_config


def merge_dotdicts(d1, d2):
    merged = d1.copy()  # 做一个d1的浅拷贝
    for k, v in d2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            # 如果两个DotDict都有相同的key，并且对应的value也是字典，递归合并
            merged[k] = merge_dotdicts(DotDict(merged[k]), v)
        else:
            # 如果不存在冲突或者其中一个value不是字典，直接更新值
            merged[k] = v
    return DotDict(merged)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __repr__(self):
        def recursive_repr(dct, level=0):
            indent = '  ' * level
            lines = []
            for key, value in dct.items():
                if isinstance(value, DotDict):
                    lines.append(f"{indent}{key}:")
                    lines.extend(recursive_repr(value, level + 1))
                else:
                    lines.append(f"{indent}{key}: {value}")
            return lines
        
        str_lines = recursive_repr(self)
        return '\n'.join(str_lines)


class WrapConfigs:

    def __init__(
            self, 
            model_name_or_path, 
            model_configs,
            opt_name,
            opt_configs, 
            lr_scheduler_name, 
            lr_scheduler_configs, 
            platform_name, 
            data_name,
            task_configs,
        ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_configs = model_configs
        self.opt_name = opt_name
        self.opt_configs = opt_configs
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_configs = lr_scheduler_configs
        self.platform_name = platform_name
        self.data_name = data_name
        self.task_configs = task_configs
        self.all_configs = self.set_all_configs()

    def set_all_configs(self):
        model_config = ModelConfig(self.model_name_or_path, **self.model_configs)
        optimizer_config = OptimizerConfig(self.opt_name, **self.opt_configs)
        lr_scheduler_config = LR_Scheduler_Config(self.lr_scheduler_name, **self.lr_scheduler_configs)
        platform_config = PlatformConfig(self.platform_name)
        task_config = TaskConfig(self.data_name, **self.task_configs)

        default_config = {
            "model": model_config.cfg,
            "optimizer": optimizer_config.cfg,
            "lr_scheduler": lr_scheduler_config.cfg,
            "platform": platform_config.cfg,
            "task": task_config.cfg,
        }

        return DotDict(default_config)


def parse_args():
    parser = argparse.ArgumentParser(description='Process some hyperparameters.')

    # Configs of Model Hyper-parameters
    parser.add_argument('--model_name_or_path', '-mn', type=str, default='mamba-370m-k8', 
                        help='Model name or path', 
                        choices=[
                            'mamba-370m-hf', 'mamba-1_4b-hf', 
                            'mamba-370m-k8', 'mamba-370m-k16', 
                            'mamba-370m-k32', 'mamba-370m-k64', 
                            'mamba-370m-km', 'tiny_mamba', 
                            'tiny_mamba-k8', 'tiny_mamba-k16',
                            'tiny_mamba-k32', 'tiny_mamba-k64'
                            ],
                        )
    parser.add_argument('--tokenizer_name_or_path', type=str, default=None, 
                        help='Tokenizer path. If not set, will use the model_name_or_path')
    parser.add_argument('--ckpt_path', type=str, default=None, 
                        help='ckpt path for model after training')
    parser.add_argument('--use_relative_position',action='store_true',
                        help='whether to use relative position embeddings')
    parser.add_argument('--use_abs_position',action='store_true',
                        help='whether to use absolute position embeddings')
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help='if use_abs_position, set the max_position_embeddings')

    # Configs of Optimizer Hyper-parameters
    parser.add_argument('--opt_name', type=str, default='adawm', 
                        help='optimizer name')
    parser.add_argument('--max_training_steps', type=int, default=None,
                        help='set training steps')
    parser.add_argument('--warmup_step', type=int, default=None,
                        help='set warmup steps')

    # Configs of Lr_Scheduler Hyper-parameters
    parser.add_argument('--lr_scheduler_type', type=str, default='get_cosine_schedule_with_warmup', 
                        help='lr scheduler name')

    # Configs of Platform Hyper-parameters
    parser.add_argument('--platform_name', '-pn', type=str, default='amax_a100',
                        required=True, help='define platform name')

    # Configs of Task Hyper-parameters
    parser.add_argument('--data_name', '-dn', type=str, default='passkey_search',
                        help='define task name')
    parser.add_argument('--processed_data_path', '-pdp', type=str, default=None,
                        help='define preprocess data path')
    parser.add_argument('--num_examples', type=int, default=3000,
                        help='define the number of dataset (for building dataset)')
    parser.add_argument('--input_seq_len', type=int, default=512,
                        help='len of input sequence')  
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='max training epoches')
    parser.add_argument('--num_kv_pairs', type=int, default=32, 
                        help='number of insert key-value pairs')
    parser.add_argument('--test_power_a', type=float, default=0.01,
                        help='power_a of MQAR dataset, for building datset')
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='validation batch size')
    parser.add_argument('--inference_mode', action='store_true')
        

    # Configs of Training Hyper-parameters
    parser.add_argument('--experiment_name', '-en', type=str, default=None,
                        required=True, help='mark for the experiment (connect with the experiment name)')
    parser.add_argument('--version', '-v', type=int, default=1,
                        help='version of the experiments, if not set, default is 1')
    parser.add_argument('--state', type=str, default='train', choices=['train', 'eval'],
                        help='define the state of the experiment')
    parser.add_argument('--accumulate_grad_batches', '-agb', type=int, default=1,
                        help='accumulate_grad_batches')
    parser.add_argument('--save_top_k', type=int, default=2,
                        help='save top k model ckpts')
    parser.add_argument('--every_n_train_steps', type=int, default=None,
                        help='save ckpt every n train steps')
    parser.add_argument('--monitor_metric', type=str, default='train_lm_loss',
                        help='monitor metric for save best model')
    parser.add_argument('--use_deepspeed', action='store_true', 
                        help='Enable to use DeepSpeed optimization.')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable to activate debug mode.')
    parser.add_argument('--hf_trainer', action='store_true', 
                        help='Enable to use Hugging Face Trainer.')
    parser.add_argument('--low_rank_train', action='store_true', 
                        help='Enable to use low rank training approach.')
    parser.add_argument('--device_num', type=int, default=1, 
                        help='Set the number of devices to use.')
    parser.add_argument('--node_num', type=int, default=1, 
                        help='Set the number of nodes for distributed training.')

    args = parser.parse_args()

    return args


def get_final_configs(args):
    model_args = {
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "ckpt_path": args.ckpt_path,
        "use_relative_position": args.use_relative_position,
        "use_abs_position": args.use_abs_position,
        "max_position_embeddings": args.max_position_embeddings,
    }
    opt_args = {
        "train_step": args.max_training_steps,
        "warmup_step": args.warmup_step,
    }
    lr_scheduler_args = {
        "train_step": args.max_training_steps,
        "warmup_step": args.warmup_step,
    }
    task_args = {
        "processed_data_path": args.processed_data_path,
        "inference_mode": args.inference_mode,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
    }
    basic_configs = WrapConfigs(
        args.model_name_or_path,
        model_args, 
        args.opt_name, 
        opt_args,
        args.lr_scheduler_type,
        lr_scheduler_args,
        args.platform_name, 
        args.data_name,
        task_args,
    ).all_configs

    train_configs = DotDict(
        {
            "experiment": {
                "save_top_k": args.save_top_k,
                "every_n_train_steps": args.every_n_train_steps,
                "experiment_name": args.experiment_name,
                "version": args.version,
                "state": args.state,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "use_deepspeed": args.use_deepspeed,
                "debug": args.debug,
                "hf_trainer": args.hf_trainer,
                "low_rank_train": args.low_rank_train,
                "device_num": args.device_num,
                "node_num": args.node_num,
                "seed": 42,
                "max_epochs": args.max_epochs,
                "monitor_metric": args.monitor_metric,
            }
        }
    )
    final_configs = merge_dotdicts(basic_configs, train_configs) # merge all configs
    return final_configs


