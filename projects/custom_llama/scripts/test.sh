OUTPUT_DIR="/zecheng2/vqllama/vqllama_openllama/test"

deepspeed --num_gpus 1 \
    --num_nodes 1 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/model_hub/open_llama_3b_v2" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_7.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 100 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 80 \
    --greater_is_better False \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --save_steps 80 \
    --save_total_limit 10 \
    --learning_rate 3e-5 \
    --warmup_steps 20 \
    --logging_steps 5 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


