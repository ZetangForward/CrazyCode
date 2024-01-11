version=$1
epoch=$2


python train_vqllama.py \
    --version ${version} \
    --epoch ${epoch} \
    --tokenier_config_path "/zecheng2/model_hub/Llama-2-7b-hf" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_merged.pkl" \
    --save_image_dir "/zecheng2/vqllama/vqllama_llama/version_1" \
    --output_dir "/zecheng2/vqllama/vqllama_llama/version_1" \
    --max_generate_length 1500 \
    --predict_batch_size 16 \
    --nworkers 0 \
    --dataloader_num_workers 64 \
    --fp16 False \
    --vqvae_config_path "configs/deepspeed/vqvae_config.yaml" \
    --do_sample True \
    --top_p 0.9 \
    --top_k 40 \
    --num_beams 1 \
    --temperature 0.8;