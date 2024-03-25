#!/bin/bash
model_name=mamba_370m_big_kernel
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=$1
task=slimpajama

nproc_per_node=$num_devices
device_num=$num_devices

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

CUDA_LAUNCH_BLOCKING=1 torchrun --nnode=1 --nproc_per_node=$nproc_per_node --master_port 6789  src/train.py \
    mark=$model_name-k16 \
    model=$model_name \
    model_name=$model_name-k16 \
    model.conv1d_configs.kernel_size=16 \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.hf_trainer=True \
    experiment.low_rank_train=False \
    experiment.device_num=$device_num \
    experiment.use_deepspeed=True \
    experiment.accumulate_grad_batches=5 \
    experiment.save_top_k=5 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=12 \
    task.dataset.max_seq_length=2048 \
    task.dataset.nworkers=12 \
    
    

