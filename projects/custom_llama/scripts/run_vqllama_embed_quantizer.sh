torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --max-restarts=3 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=worker-0:29400 \
    train_pl_v2.py;