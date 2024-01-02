torchrun \
    --nnodes=2 \
    --nproc-per-node=4 \
    --max-restarts=3 \
    --rdzv_id=111 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.184.185.106:4331 \
    train_pl_v2.py;