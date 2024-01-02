torchrun \
    --nnodes=16 \
    --nproc-per-node=4 \
    --max-restarts=3 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=worker0:36722 \
    train_pl_v2.py;