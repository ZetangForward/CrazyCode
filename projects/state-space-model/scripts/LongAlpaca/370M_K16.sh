#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
model_name=mamba_370m_big_kernel_16
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=langchao
task=longalpaca
mark=370m-k16
nproc_per_node=$num_devices
device_num=$num_devices

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

CUDA_LAUNCH_BLOCKING=1 torchrun --nnode=1 --nproc_per_node=$nproc_per_node --master_port 17952  src/train.py \
    mark=$mark \
    model_name=$mark \
    model=$model_name \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.hf_trainer=True \
    experiment.low_rank_train=False \
    experiment.device_num=$device_num \
    experiment.use_deepspeed=False \
    experiment.accumulate_grad_batches=32 \
    experiment.save_top_k=1 \
    optimizer.num_training_steps=1000 \
    experiment.every_n_train_steps=200 \
    optimizer.warmup_steps=100 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=2 \
    task.dataset.max_seq_length=4096 \
    task.dataset.nworkers=4 \
    
    

