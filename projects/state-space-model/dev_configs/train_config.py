import os
import sys
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
sys.path.append(current_file_dir)
import argparse
from modelzipper.tutils import *
from . import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some hyperparameters.')

    # Configs of Model Hyper-parameters
    parser.add_argument('--model_name_or_path', '-mn', type=str, default='mamba-370m-k8', 
                        help='Model name or path', 
                        choices=[
                            'mamba-370m-hf', 'mamba-1_4b-hf', 
                            'mamba-370m-k8', 'mamba-370m-k16', 
                            'mamba-370m-k32', 'mamba-370m-k64', 
                            'mamba-370m-km'],
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
    parser.add_argument('--train_step', type=int, default=20000,
                        help='set training steps')
    parser.add_argument('--warmup_step', type=int, default=2000,
                        help='set warmup steps')

    # Configs of Lr_Scheduler Hyper-parameters
    parser.add_argument('--scheduler_type', type=str, default='get_cosine_schedule_with_warmup', 
                        help='optimizer name')

    # Configs of Platform Hyper-parameters
    parser.add_argument('--platform_name', '-pn', type=str, default='amax_a100',
                        required=True, help='define platform name')

    # Configs of Task Hyper-parameters
    parser.add_argument('--data_name', '-dn', type=str, default='slimpajama',
                        help='define task name')

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
    parser.add_argument('--every_n_train_steps', type=int, default=2000,
                        help='save ckpt every n train steps')
    parser.add_argument('--monitor_metric', type=str, default='loss',
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
        "train_step": args.train_step,
        "warmup_step": args.warmup_step,
    }
    task_args = {}
    basic_configs = WrapConfigs(
        args.model_name_or_path,
        model_args, 
        args.opt_name, 
        opt_args,
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
            }
        }
    )
    final_configs = merge_dotdicts(basic_configs, train_configs) # merge all configs
    return final_configs


