export CUDA_VISIBLE_DEVICES=1

python src/train_dev2.py \
    -mn tiny_mamba \
    -pn amax_a100 \
    -en test \
    -dn MQAR_ywj \
    --state train \
    --input_seq_len 4096 \
    --num_kv_pairs 256 \
    --processed_data_path "/nvme/zecheng/data/MQAR/train_C8192_N4096_D256.pkl" \
    --accumulate_grad_batches 8;


