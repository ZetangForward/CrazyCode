LOG_OUTPUT="/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/Logs"

for i in {16..23}; do  
    device=$((i - 16))
    CUDA_VISIBLE_DEVICES=${device} python switch_codebook.py SNAP_ID=${i}  > ${LOG_OUTPUT}/inference_${i}.log 2>&1 &  
done 
