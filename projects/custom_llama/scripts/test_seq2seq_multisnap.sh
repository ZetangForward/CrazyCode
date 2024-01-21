snap_id=$1
ckpt="/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-8100"
save_dir="/zecheng2/vqllama/test_vq_seq2seq/test_flat_t5/epoch_8100_bs"
LOG_OUTPUT="/workspace/zecheng/modelzipper/projects/custom_llama/Logs/multisnap_inference"

mkdir -p ${LOG_OUTPUT}
mkdir -p ${save_dir}


for i in {1..7}; do  
    CUDA_VISIBLE_DEVICES=$i python test_vq_seq2seq.py \
        --snap_id ${i} \
        --ckpt ${ckpt} \
        --tokenier_config_path "/zecheng2/model_hub/flan-t5-xl" \
        --data_path "/zecheng2/svg/icon-shop/test_data_snaps/split_snaps_v2/long_test_split_${i}.pkl" \
        --save_dir ${save_dir} \
        --max_generate_length 512 \
        --predict_batch_size 4 \
        --model_max_length 512 \
        --inference_nums -1 \
        --dataloader_num_workers 0 \
        --fp16 False \
        --vqvae_config_path "configs/deepspeed/vqvae_config_v2.yaml" \
        --do_sample False \
        --top_p 0.9 \
        --top_k 40 \
        --do_inference True \
        --do_raster False \
        --num_beams 4 \
        --temperature 0.7 \
        --decode_golden True > ${LOG_OUTPUT}/inference_${i}.log 2>&1 &  
done 