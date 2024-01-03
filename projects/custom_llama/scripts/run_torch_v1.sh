x=($(python -c "import torch;import os;x = ['106','117','154','146','150','122','124','145','126','138','139','137','151','105','136','142'];l = ','.join([f'GCRHYPC{e}' for e in x]).split(',');print(l[0], len(l), torch.cuda.device_count(), l.index(os.environ['HOSTNAME']))"))

export NCCL_DEBUG=INFO 

echo "NCCL_DEBUG: ${NCCL_DEBUG}"
echo ${x[0]} ${x[1]} ${x[2]} ${x[3]}

python -m torch.distributed.launch \
    --nproc-per-node ${x[2]} \
    --nnodes ${x[1]} \
    --node-rank ${x[3]} \
    --master-addr ${x[0]} \
    --master-port 9504 \
    train_pl_v2.py;
