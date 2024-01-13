version=$1
ckpt=$2

python test_vqllama.py \
    --version ${version} \
    --ckpt ${ckpt} \
    --tokenier_config_path "/zecheng2/model_hub/Llama-2-7b-hf" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_mesh_data_svg_convert_p.pkl" \
    --save_dir "/zecheng2/vqllama/test_vqllama" \
    --max_generate_length 1500 \
    --predict_batch_size 16 \
    --model_max_length 1024 \
    --dataloader_num_workers 0 \
    --dataloader_num_workers 64 \
    --fp16 False \
    --vqvae_config_path "configs/deepspeed/vqvae_config.yaml" \
    --do_sample True \
    --top_p 0.9 \
    --top_k 40 \
    --num_beams 1 \
    --temperature 0.8;