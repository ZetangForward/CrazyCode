version=$1
ckpt=$2

export CUDA_VISIBLE_DEVICES=4

python test_vq_seq2seq.py \
    --version ${version} \
    --ckpt ${ckpt} \
    --tokenier_config_path "/zecheng2/model_hub/flan-t5-xl" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_data_long_seq.pkl" \
    --save_dir "/zecheng2/vqllama/test_vq_seq2seq" \
    --max_generate_length 64 \
    --predict_batch_size 24 \
    --model_max_length 512 \
    --inference_nums 2000 \
    --dataloader_num_workers 0 \
    --fp16 False \
    --vqvae_config_path "configs/deepspeed/vqvae_config_v2.yaml" \
    --do_sample False \
    --top_p 0.9 \
    --top_k 40 \
    --num_beams 1 \
    --temperature 0.8 \
    --decode_golden True;