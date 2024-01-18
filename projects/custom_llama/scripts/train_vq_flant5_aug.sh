OUTPUT_DIR="/zecheng2/vqllama/vqllama_flant5/version_aug"

mkdir -p ${OUTPUT_DIR}

deepspeed --num_gpus 8 \
    --num_nodes 3 \
    --master_addr worker-0 \
    --master_port 6668 \
    --hostfile configs/machine/hostfile_v24 \
    train_vq_seq2seq_aug.py \
    --model_name_or_path "/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-3000" \
    --resume_from_checkpoint "/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-3000" \
    --data_path "/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/test_vqllama_quantizer_testset/version_12/epoch_37/augment_stage2_data.pkl" \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --model_max_length 512 \
    --per_device_train_batch_size 54 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --warmup_steps 20 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed/stage3.json \
    --fp16 False \
    --remove_unused_columns False \
    --freezen_llm True \
    --config_path "configs/deepspeed/vqvae_config_v2.yaml";
