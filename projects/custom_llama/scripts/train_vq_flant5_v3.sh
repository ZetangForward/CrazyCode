OUTPUT_DIR="/zecheng2/vqllama/vqllama_flant5/version_aug"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 4 \
    --num_nodes 16 \
    --master_addr worker-0 \
    --master_port 6668 \
    --hostfile configs/machine/hostfile_v64 \
    train_vq_seq2seq.py \
    --model_name_or_path "/zecheng2/model_hub/flan-t5-xl" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/helpme.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 50 \
    --model_max_length 512 \
    --per_device_train_batch_size 54 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --greater_is_better False \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --warmup_steps 3 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 False \
    --remove_unused_columns False \
    --freezen_llm True \
    --config_path "configs/deepspeed/vqvae_config_v2.yaml";
