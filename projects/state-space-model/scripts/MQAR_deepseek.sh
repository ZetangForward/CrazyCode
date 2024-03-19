#!/bin/bash
model_name=deepseek-1_3b
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=langchao
task=AR_ywj

nproc_per_node=$num_devices
device_num=$num_devices
input_seq_len=$1
num_kv_pairs=$2

mark=${input_seq_len}_${num_kv_pairs}

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

torchrun --nnode=1 --nproc_per_node=$nproc_per_node --master_port 6781  mamba/train.py \
    mark=${mark} \
    model=$model_name \
    model_name=$model_name \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.low_rank_train=False \
    experiment.device_num=$device_num \
    experiment.use_deepspeed=False \
    experiment.num_training_steps=40000 \
    experiment.warmup_steps=4000 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=1\
    task.dataset.max_seq_length=10000 \
    task.dataset.input_seq_len=${input_seq_len} \
    task.dataset.num_kv_pairs=${num_kv_pairs} \
    task.dataset.inference_mode=False \
    task.dataset.processed_data_path=MQAR/train/train_C8192_N${input_seq_len}_D${num_kv_pairs}.pkl \
    task.dataset.nworkers=4 \
    
    

