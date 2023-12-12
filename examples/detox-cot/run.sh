deepspeed --num_gpus 8 \
    --num_nodes 1 \
    peft_model.py \
    --cf config/llama7b.yaml \
    --output_dir /zecheng/detox-cot/llama2 \
    --save_strategy "epoch" \
    -—num_train_epochs 15 \
    --remove_unused_columns False \
    --per_device_train_batch_size 4 \
    --deepspeed config/ds_config.json