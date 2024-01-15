export CUDA_VISIBLE_DEVICES=2

OUTPUT_DIR="/zecheng2/vqllama/vqllama_openllama/test"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 1 \
    --num_nodes 1 \
    train_vqllama.py \
    --model_name_or_path "/zecheng2/model_hub/open_llama_3b_v2" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_mesh_data_svg_convert_p.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --greater_is_better False \
    --save_strategy "steps" \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --eval_steps 300 \
    --save_steps 300 \
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
    --remove_unused_columns False \
    --freezen_llm True \
    --add_eval False \
    --config_path "configs/deepspeed/vqvae_config.yaml";


