#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_name=mamba_370m_multi
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=$1
task=longalpaca

nproc_per_node=$num_devices
device_num=$num_devices

echo "Number of devices: $num_devices"

echo "Available GPU device IDs: $CUDA_VISIBLE_DEVICES"

torchrun --nnode=1 --nproc_per_node=$nproc_per_node --master_port 6948  src/train.py \
    mark=longalpaca \
    model=$model_name \
    model_name=$model_name \
    task=$task \
    exp_task=$task \
    platform=$platform \
    experiment.debug=False \
    experiment.low_rank_train=False \
    experiment.use_deepspeed=True \
    experiment.device_num=$device_num \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=1 \
    task.dataset.max_seq_length=5120 \
    task.dataset.nworkers=4 \
    optimizer.num_training_steps=10000 \
    optimizer.warmup_steps=1000 \
    experiment.accumulate_grad_batches=1 \
    task.dataset.cluster_batch=False \
    task.dataset.train_batch_size=1 \
    model.use_custom_module=True \
    model.ckpt_path=/public/home/ljt/tzc/ckpt/mamba_370m_slimpajama_1Btoken/mamba-multi-pretraining/checkpoints/model.bin;


    

