OUTPUT_DIR="/zecheng2/vqllama/vqllama_flant5/version_aug_v7"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 16 \
    --num_nodes 1 \
    --master_addr worker-2 \
    train_vq_seq2seq_aug.py \
    --model_name_or_path "/zecheng2/vqllama/test_vq_seq2seq/test_flat_t5/epoch_8100" \
    --resume_from_checkpoint "/zecheng2/vqllama/test_vq_seq2seq/test_flat_t5/epoch_8100" \
    --data_path "/zecheng2/svg/icon-shop/test_data_snaps/test_data_all_seq_with_mesh.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 10 \
    --model_max_length 512 \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --warmup_steps 60 \
    --logging_steps 1 \
    --dataloader_num_workers 12 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 False \
    --remove_unused_columns False \
    --freezen_llm True \
    --config_path "configs/deepspeed/vqvae_config_v2.yaml";
