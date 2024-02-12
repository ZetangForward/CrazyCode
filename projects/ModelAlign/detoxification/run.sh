CONFIG=configs/accelerate_configs/single_gpu.yaml
MODEL_SAVE_PATH=/home/amax/zecheng/models/detoxification/gpt-j-6b
export CUDA_VISIBLE_DEVICES=2,3

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${CONFIG} detoxification/example.py \
    --log_with tensorboard \
    --model_save_path ${MODEL_SAVE_PATH} \
    --model_name "/nvme/hf_models/gpt-j-6b" \
    --dataset_name "/home/amax/zecheng/data/real-toxicity-prompts";