CONFIG=configs/accelerate_configs/single_gpu.yaml
MODEL_SAVE_PATH=/home/amax/zecheng/models/Detoxification

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ${CONFIG} Detoxification/example.py \
    --log_with tensorboard \
    --model_save_path ${MODEL_SAVE_PATH} \
    --model_name "/nvme/hf_models/gpt-j-6b";