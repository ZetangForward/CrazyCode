torchrun \
    --nnodes=0:15 \
    --nproc-per-node=4 \
    --max-restarts=3 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --master_addr=worker0 \
    --master_port=36722 \
    --rdzv_endpoint=worker0:36722 \
    train_pl_v2.py;