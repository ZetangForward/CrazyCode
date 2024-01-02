torchrun \
    --nnodes=16 \
    --nproc-per-node=4 \
    --rdzv_id=223 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=worker-0:2234 \
    train_pl_v2.py;