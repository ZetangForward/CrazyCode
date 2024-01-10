deepspeed --num_gpus 8 \
    --num_nodes 1 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/model_hub/Llama-2-7b-hf" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_mesh_data_svg_convert_p.pkl" \
    --output_dir "/zecheng2/vqllama/vqllama_llama/version_0" \
    --num_train_epochs 40 \
    --model_max_length 1500 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 8 \
    --learning_rate 3e-6 \
    --warmup_steps 20 \
    --logging_steps 1 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


