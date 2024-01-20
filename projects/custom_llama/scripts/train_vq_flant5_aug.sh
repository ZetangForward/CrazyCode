OUTPUT_DIR="/zecheng2/vqllama/vqllama_flant5/version_aug_v7"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 16 \
    --num_nodes 4 \
    --master_addr worker-0 \
    --master_port 6668 \
    --hostfile configs/machine/hostfile_v64_sxm4 \
    train_vq_seq2seq_aug.py \
    --model_name_or_path "/zecheng2/vqllama/vqllama_flant5/version_aug_v6/checkpoint-484" \
    --resume_from_checkpoint "/zecheng2/vqllama/vqllama_flant5/version_aug_v6/checkpoint-484" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer_testset/version_12/epoch_37/aug_stage2_pro_data.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 512 \
    --per_device_train_batch_size 54 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --greater_is_better False \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --load_best_model_at_end True \
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
