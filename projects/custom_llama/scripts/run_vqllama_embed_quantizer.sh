torchrun \
    --nnodes=16 \
    --nproc-per-node=4 \
    --max-restarts=3 \
    --rdzv_id=111 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=worker-0:4331 \
    train_pl_v2.py;