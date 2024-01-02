git pull && x=($(python -c "import torch;import os;x = ['151'];l = ','.join([f'GCRHYPC{e}' for e in x]).split(',');print(l[0], len(l), torch.cuda.device_count(), l.index(os.environ['HOSTNAME']))")) && export NCCL_DEBUG=INFO && python -m torch.distributed.launch --nproc_per_node ${x[2]} --nnodes ${x[1]} --node_rank ${x[3]} --master_addr ${x[0]} --master_port 9504 train_pl_v2.py 

x = ['151'];l = ','.join([f'GCRHYPC{e}' for e in x])
