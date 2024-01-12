deepspeed --num_gpus 8 \
    --num_nodes 1 \
    layoutNUWA/train.py \
    --model_name_or_path "/zecheng2/vqllama/baselines/layoutNUWA/checkpoint-350" \
    --resume_from_checkpoint "/zecheng2/vqllama/baselines/layoutNUWA/checkpoint-350" \
    --data_path "/zecheng2/svg/icon-shop/meta_data" \
    --output_dir "/zecheng2/vqllama/baselines/layoutNUWA" \
    --num_train_epochs 40 \
    --model_max_length 1500 \
    --per_device_train_batch_size 18 \
    --per_device_eval_batch_size 18 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --greater_is_better False \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --save_total_limit 5 \
    --learning_rate 3e-6 \
    --warmup_steps 20 \
    --logging_steps 10 \
    --dataloader_num_workers 20 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed "/workspace/zecheng/modelzipper/projects/custom_llama/configs/deepspeed/stage3.json" \
    --fp16 True \
    --remove_unused_columns False;


