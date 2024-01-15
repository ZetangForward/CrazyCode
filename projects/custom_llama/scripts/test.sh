OUTPUT_DIR="/zecheng2/vqllama/vqllama_openllama/version_3_aug"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 4 \
    --num_nodes 1 \
    --master_addr worker-2 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/vqllama/vqllama_openllama/version_3/checkpoint-2100" \
    --resume_from_checkpoint "/zecheng2/vqllama/vqllama_openllama/version_3/checkpoint-2100" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer_testset/version_8/epoch_84/inference_full_data_compress_1_snaps_0.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --greater_is_better False \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 3e-6 \
    --warmup_steps 3 \
    --logging_steps 10 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --freezen_llm True \
    --add_eval False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


