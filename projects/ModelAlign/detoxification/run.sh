CONFIG=configs/accelerate_configs/single_gpu.yaml
MODEL_SAVE_PATH=/home/amax/zecheng/ckpts/gpt-neo-1.3B

export CUDA_VISIBLE_DEVICES=2,3,4,5
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${CONFIG} detoxification/example.py \
    --log_with wandb \
    --batch_size 32 \
    --model_save_path ${MODEL_SAVE_PATH} \
    --model_name "/nvme/hf_models/gpt-neo-1.3B" \
    --dataset_name "/home/amax/zecheng/data/real-toxicity-prompts" \
    --toxicity_model_id "/nvme/hf_models/roberta-hate-speech-dynabench-r4-target";