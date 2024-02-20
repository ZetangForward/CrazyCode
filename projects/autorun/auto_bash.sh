#!/bin/bash

# 初始化GPU设备计数器
device=0

# 遍历层（5到40之间的奇数）
for layer in $(seq 1 2 20); do
    # 检查GPU显存占用
    while true; do
        memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $device)

        # 检查显存是否超过10000MB
        if [ "$memory_usage" -le 10000 ]; then
            break
        else
            echo "Waiting for GPU $device memory to be less than 10000MB (current usage: $memory_usage MB)"
            sleep 60  # 每60秒检查一次
        fi
    done

    # 创建日志文件目录（如果尚不存在）
    mkdir -p logs/gpt2-6B
    mkdir -p /nvme/wpz/model_interpretability/Kowledge_edit/EasyEdit/results/gptj-6B/layer_${layer}

    # 使用nohup运行程序并将输出重定向到日志文件
    CUDA_VISIBLE_DEVICES=$device nohup python run_knowedit_llama2.py \
        --editing_method=ROME \
        --edit_layer ${layer} \
        --hparams_dir=../hparams/ROME/gpt-j-6B.yaml \
        --data_dir=/nvme/wpz/model_interpretability/Kowledge_edit/EasyEdit/data/KnowEdit/benchmark/wiki_counterfact/test_cf.json \
        --datatype='counterfact' \
        --metrics_save_dir=/nvme/wpz/model_interpretability/Kowledge_edit/EasyEdit/results/gptj-6B/layer_${layer} \
        > logs/gpt2-6B/${layer}.log 2>&1 &

    echo "Started training for layer $layer on GPU $device. Output is being logged to logs/gpt2-6B/${layer}.log."

    # 在启动下一个进程前等待60秒
    sleep 60

    # 更新GPU设备计数器（在0-5之间循环）
    device=$(( (device + 1) % 6 ))
done

# layer=5
# python run_knowedit_llama2.py \
#     --editing_method=ROME \
#     --edit_layer ${layer} \
#     --hparams_dir=../hparams/ROME/gpt-j-6B.yaml \
#     --data_dir=/nvme/wpz/model_interpretability/Kowledge_edit/EasyEdit/data/KnowEdit/benchmark/wiki_counterfact/test_cf.json \
#     --datatype='counterfact' \
#     --metrics_save_dir=/nvme/wpz/model_interpretability/Kowledge_edit/EasyEdit/results/gptj-6B/layer_${layer}