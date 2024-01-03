x=($(python -c "import torch;import os;x = list(range(16));l = ','.join(['worker-{}'.format(e) for e in x]).split(',');print(l[0], len(l), torch.cuda.device_count(), l.index(os.environ['HOSTNAME']))")) 

export NCCL_DEBUG=INFO 

python -m torch.distributed.launch \
    --nproc-per-node ${x[2]} \
    --nnodes ${x[1]} \
    --node-rank ${x[3]} \
    --master-addr ${x[0]} \
    --master-port 9504 \
    train_pl_v2.py;
