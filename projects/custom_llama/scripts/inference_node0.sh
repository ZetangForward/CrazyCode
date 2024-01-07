LOG_OUTPUT="/workspace/zecheng/modelzipper/projects/custom_llama/Logs"

mkdir -p ${LOG_OUTPUT}

for i in {0..1}; do  
    device=$((i - 0))
    CUDA_VISIBLE_DEVICES=${device} python switch_codebook.py SNAP_ID=${i}  > ${LOG_OUTPUT}/inference_${i}.log 2>&1 &  
done 
