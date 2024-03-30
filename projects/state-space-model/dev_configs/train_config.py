import os
import sys
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
sys.path.append(current_file_dir)
import argparse
from modelzipper.tutils import *
from . import *


parser = argparse.ArgumentParser(description='Process some hyperparameters.')

# Configs of Model Hyper-parameters
parser.add_argument('--model_name_or_path', type=str, default='mamba-370m-k8', 
                    required=True, help='Model name or path')
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
                    required=True, help='optimizer name')
parser.add_argument('--train_step', type=int, default=20000,
                    help='set training steps')
parser.add_argument('--warmup_step', type=int, default=2000,
                    help='set warmup steps')

# Configs of Lr_Scheduler Hyper-parameters
parser.add_argument('--scheduler_type', type=str, default='get_cosine_schedule_with_warmup', 
                    required=True, help='optimizer name')

# Configs of Platform Hyper-parameters
parser.add_argument('--platform_name', type=str, default='amax_a100',
                    required=True, help='define platform name')

# Configs of Task Hyper-parameters
parser.add_argument('--data_name', type=str, default='simplepajama',
                    required=True, help='define task name')

# Configs of Training Hyper-parameters
parser.add_argument('--experiment_name', '-en', type=str, default=None,
                    required=True, help='mark for the experiment (connect with the experiment name)')
parser.add_argument('--version', '-v', type=int, default=1,
                    help='version of the experiments, if not set, default is 1')
parser.add_argument('--state', type=str, default='train', choices=['train', 'eval'],
                    help='define the state of the experiment')
parser.add_argument('--accumulate_grad_batches', '-agb', type=int, default=1,
                    help='accumulate_grad_batches')
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


