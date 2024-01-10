deepspeed --num_gpus 16 \
    --num_nodes 4 \
    --master_addr worker-0 \
    --master_port 7329 \
    --hostfile configs/hostfile_v128 \
    train_svg_offline_v1.py \
    --model_name_or_path "/zecheng/model_hub/CodeLlama-7b-hf" \
    --data_path ""/zecheng2/svg/icon-shop/pkl_data/full_data.pkl"" \
    --output_dir "/zecheng/svg_model_hub/Iconshop_CodeLlama-7b" \
    --num_train_epochs 40 \
    --model_max_length 1500 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
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
    --deepspeed configs/deepspeed_config_2.json \
    --fp16 True \
    --remove_unused_columns False;


