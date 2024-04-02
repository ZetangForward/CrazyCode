export CUDA_VISIBLE_DEVICES=1

torchrun --nnode=1 --nproc_per_node=1 src/train_dev2.py \
    -mn mamba-370m-km \
    -pn amax_a100 \
    -en test \
    -dn longalpaca \
    --node_num 1 \
    --device_num 1 \
    --state train \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --max_epochs 50 \
    --input_seq_len 4096;


