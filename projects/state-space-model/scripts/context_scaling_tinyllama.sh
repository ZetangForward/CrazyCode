#!/bin/bash
model_name=tiny_llama
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

nproc_per_node=$num_devices
device_num=$num_devices

torchrun --nnode=1 --nproc_per_node=$nproc_per_node mamba/train.py \
    experiment.hf_trainer=True \
    JOB_ID=2 \
    model=$model_name \
    model_name=$model_name \
    platform=amax_a100 \
    experiment.debug=True \
    experiment.device_num=$device_num
