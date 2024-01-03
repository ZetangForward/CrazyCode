# x=($(python -c "import torch;import os;x = ['106','117','154','146','150','122','124','145','126','138','139','137','151','105','136','142'];l = ','.join([f'GCRHYPC{e}' for e in x]).split(',');print(l[0], len(l), torch.cuda.device_count(), l.index(os.environ['HOSTNAME']))"))

x=($(python -c "import torch;import os;x = ['106','117'];l = ','.join([f'GCRHYPC{e}' for e in x]).split(',');print(l[0], len(l), torch.cuda.device_count(), l.index(os.environ['HOSTNAME']))"))

export NCCL_DEBUG=INFO 
# export NCCL_IB_DISABLE=1
# export NCCL_IBEXT_DISABLE=1 


echo "NCCL_DEBUG: ${NCCL_DEBUG}"
echo ${x[0]} ${x[1]} ${x[2]} ${x[3]}

python -m torch.distributed.launch \
    --nproc_per_node ${x[2]} \
    --nnodes ${x[1]} \
    --node_rank ${x[3]} \
    --master_addr ${x[0]} \
    --master_port 9404 \
    --use_env \
    train_pl_v2.py;
