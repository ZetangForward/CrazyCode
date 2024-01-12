OUTPUT_DIR="/zecheng2/vqllama/vqllama_openllama/version_1"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 16 \
    --num_nodes 4 \
    --master_addr worker-0 \
    --master_port 6529 \
    --hostfile configs/machine/hostfile_v64_sxm4 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/vqllama/vqllama_openllama/version_1/checkpoint-700" \
    --resume_from_checkpoint "/zecheng2/vqllama/vqllama_openllama/version_1/checkpoint-700" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_merged.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 1024 \
    --per_device_train_batch_size 78 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --greater_is_better False \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 3e-5 \
    --warmup_steps 20 \
    --logging_steps 10 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


