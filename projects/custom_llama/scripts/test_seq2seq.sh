ckpt="/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-8100"
save_dir="/zecheng2/vqllama/test_vq_seq2seq/test_flat_t5/epoch_8100_topp/"
snap_id=0
export CUDA_VISIBLE_DEVICES=0

python test_vq_seq2seq.py \
    --ckpt ${ckpt} \
    --snap_id 0 \
    --tokenier_config_path "/zecheng2/model_hub/flan-t5-xl" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/split_snaps_v3/long_test_split_${snap_id}.pkl" \
    --save_dir ${save_dir} \
    --max_generate_length 512 \
    --predict_batch_size 4 \
    --model_max_length 512 \
    --inference_nums -1 \
    --dataloader_num_workers 0 \
    --fp16 False \
    --vqvae_config_path "configs/deepspeed/vqvae_config_v2.yaml" \
    --do_sample True \
    --top_p 0.9 \
    --top_k 40 \
    --num_beams 4 \
    --temperature 0.7 \
    --decode_golden True \
    --do_raster False \
    --do_inference True;