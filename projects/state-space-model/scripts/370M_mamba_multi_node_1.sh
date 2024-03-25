#!/bin/bash
model_name=mamba_370m_multi
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=$1
task=slimpajama

nproc_per_node=$num_devices
device_num=$num_devices

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr dgx-071 \
    --master_port 6668 \
    src/train.py \
    mark=$model_name-multi \
    model=$model_name \
    model_name=$model_name-multi \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.hf_trainer=True \
    experiment.low_rank_train=False \
    experiment.device_num=8 \
    experiment.use_deepspeed=True \
    experiment.accumulate_grad_batches=10 \
    experiment.save_top_k=5 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=4 \
    task.dataset.max_seq_length=2048 \
    task.dataset.nworkers=4 \
    
    

