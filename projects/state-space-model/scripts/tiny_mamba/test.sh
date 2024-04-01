export CUDA_VISIBLE_DEVICES=1

python src/train_dev2.py \
    -mn tiny_mamba-k8 \
    -pn amax_a100 \
    -en test \
    -dn MQAR_ywj \
    --state train \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --input_seq_len 4096 \
    --num_kv_pairs 256 \
    --processed_data_path "/nvme/zecheng/data/MQAR/train_C8192_N4096_D256.pkl";


