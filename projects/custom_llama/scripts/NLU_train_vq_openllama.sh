OUTPUT_DIR="/zecheng2/vqllama/vqllama_openllama/understanding"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 8 \
    --num_nodes 1 \
    train_vqllama_understanding.py \
    --model_name_or_path "/zecheng2/model_hub/open_llama_3b_v2" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer/version_8/epoch_84/inference_full_data_compress_1_snaps_7.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 60 \
    --model_max_length 1500 \
    --per_device_train_batch_size 48 \
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
    --dataloader_num_workers 0 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 True \
    --remove_unused_columns False;


