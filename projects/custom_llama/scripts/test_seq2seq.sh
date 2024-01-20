snap_id=$1
ckpt=$2
save_dir="/zecheng2/vqllama/test_vq_seq2seq/test_flat_t5"


export CUDA_VISIBLE_DEVICES=6

python test_vq_seq2seq.py \
    --snap_id ${snap_id} \
    --ckpt ${ckpt} \
    --tokenier_config_path "/zecheng2/model_hub/flan-t5-xl" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_data_long_seq_with_mesh.pkl" \
    --save_dir ${save_dir} \
    --max_generate_length 512 \
    --predict_batch_size 8 \
    --model_max_length 512 \
    --inference_nums 48 \
    --dataloader_num_workers 0 \
    --fp16 False \
    --vqvae_config_path "configs/deepspeed/vqvae_config_v2.yaml" \
    --do_sample False \
    --top_p 0.9 \
    --top_k 40 \
    --num_beams 1 \
    --temperature 0.8 \
    --decode_golden True;