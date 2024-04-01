export CUDA_VISIBLE_DEVICES=0

torchrun --nnode=1 --nproc_per_node=1 src/train_dev2.py \
    -mn tiny_mamba-k8 \
    -pn langchao \
    -en tinymamba-k8-4096-256 \
    -dn MQAR_ywj \
    --node_num 1 \
    --device_num 1 \
    --state train \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --max_epochs 50 \
    --input_seq_len 4096 \
    --num_kv_pairs 256 \
    --processed_data_path "MQAR/train_C8192_N4096_D256.pkl";


