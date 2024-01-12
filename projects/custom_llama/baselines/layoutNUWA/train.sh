deepspeed --num_gpus 8 \
    --num_nodes 1 \
    layoutNUWA/train.py \
    --model_name_or_path "/zecheng2/model_hub/open_llama_3b_v2" \
    --data_path "/zecheng2/svg/icon-shop/meta_data" \
    --output_dir "/zecheng2/vqllama/baselines/layoutNUWA" \
    --num_train_epochs 40 \
    --model_max_length 1500 \
    --per_device_train_batch_size 18 \
    --per_device_eval_batch_size 18 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 8 \
    --learning_rate 3e-6 \
    --warmup_steps 20 \
    --logging_steps 1 \
    --dataloader_num_workers 20 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed "/workspace/zecheng/modelzipper/projects/custom_llama/configs/deepspeed/stage3.json" \
    --fp16 True \
    --remove_unused_columns False;


