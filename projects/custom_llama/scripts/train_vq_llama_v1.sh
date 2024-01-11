deepspeed --num_gpus 16 \
    --num_nodes 4 \
    --master_addr worker-0 \
    --master_port 7529 \            
    --hostfile configs/machine/hostfile_v64_sxm4 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/model_hub/Llama-2-7b-hf" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_merged.pkl" \
    --output_dir "/zecheng2/vqllama/vqllama_llama/version_1" \
    --num_train_epochs 100 \
    --model_max_length 1500 \
    --per_device_train_batch_size 36 \
    --per_device_eval_batch_size 36 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 80 \
    --save_total_limit 10 \
    --learning_rate 3e-6 \
    --warmup_steps 20 \
    --logging_steps 1 \
    --dataloader_num_workers 48 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


