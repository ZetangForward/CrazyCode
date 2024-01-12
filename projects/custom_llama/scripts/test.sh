VERSION=1
OUTPUT_DIR="/zecheng2/vqllama/vqllama_llama/version_${VERSION}"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 8 \
    --num_nodes 1 \
    --hostfile configs/machine/hostfile_v64_sxm4 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/vqllama/vqllama_llama/version_1/checkpoint-160" \
    --resume_from_checkpoint "/zecheng2/vqllama/vqllama_llama/version_1/checkpoint-160" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_merged.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 100 \
    --model_max_length 1500 \
    --per_device_train_batch_size 68 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --save_steps 5 \
    --save_total_limit 10 \
    --learning_rate 3e-5 \
    --warmup_steps 20 \
    --logging_steps 5 \
    --dataloader_num_workers 32 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


