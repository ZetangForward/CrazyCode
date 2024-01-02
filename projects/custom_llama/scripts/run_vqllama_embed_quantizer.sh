torchrun \
    --nnodes=1:4 \
    --nproc-per-node=4 \
    --max-restarts=3 \
    --rdzv_id=543 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.184.185.106:4321 \
    train_pl_v2.py;