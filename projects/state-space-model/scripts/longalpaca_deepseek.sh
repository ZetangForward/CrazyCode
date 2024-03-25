#!/bin/bash
model_name=deepseek-1_3b
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=$1
task=longalpaca

nproc_per_node=$num_devices
device_num=$num_devices

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

torchrun --nnode=1 --nproc_per_node=$nproc_per_node --master_port 6789  src/train.py \
    mark=deepseek-1_3b \
    model=$model_name \
    model_name=$model_name \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.low_rank_train=False \
    experiment.device_num=$device_num \
    experiment.use_deepspeed=True \
    experiment.accumulate_grad_batches=12 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=1 \
    task.dataset.max_seq_length=14000 \
    task.dataset.nworkers=8 \
    
    

