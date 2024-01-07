LOG_OUTPUT="/zecheng2/svg/icon-shop/pkl_data/efficient_inference_full_data/Logs"

for i in {8..15}; do  
    device=$((i - 8))
    CUDA_VISIBLE_DEVICES=${device} python switch_codebook.py SNAP_ID=${i}  > ${LOG_OUTPUT}/inference_${i}.log 2>&1 &  
done 
