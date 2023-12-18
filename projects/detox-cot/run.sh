deepspeed --num_gpus 8 \
    --num_nodes 1 \
    --hostfile config/hostfile_24 \
    peft_model.py \
    --deepspeed config/ds_config_zero2.json \
    --cf config/llama7b.yaml \
    --output_dir /zecheng/detox-cot/llama2 \
    --save_strategy "epoch" \
    --num_train_epochs 8 \
    --remove_unused_columns False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 3e-6 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing False \
    --fp16 True;