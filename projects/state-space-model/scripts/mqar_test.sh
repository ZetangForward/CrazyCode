# export CUDA_VISIBLE_DEVICES=0

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

model_name=$1
# mkdir -p /public/home/ljt/tzc/data/evaluation/AR_ywj/${model_name}/
MODEL_N=$2
MODEL_D=$3
data_n_list="512 1024 2048 4096 8192"
for DATA_N in $data_n_list
do
    data_d_list="32 48 64 96 128 192 256"
    for DATA_D in $data_d_list
    do
        JOB_ID=N${MODEL_N}_D${MODEL_D}-N${DATA_N}_D${DATA_D}

        python mamba/test.py \
            model=$model_name \
            model_name=$model_name \
            model.ckpt_path=/aifs4su/ziliwang/txw/InternLM/zecheng/ckpt/AR_ywj/$model_name/version_${MODEL_N}${MODEL_D}/checkpoints/last.ckpt \
            model.load_model_state_dict=True \
            platform=h_800 \
            task='AR_ywj' \
            exp_task='AR_ywj' \
            job_id=$JOB_ID \
            experiment.device_num=$num_devices \
            task.dataset.processed_data_path=MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl \
            task.dataset.input_seq_len=$DATA_N \
            task.dataset.num_kv_pairs=$DATA_D;

        wait
        echo $JOB_ID >> /home/tianxiangwu/zecheng/evaluation/AR_ywj/AR_ywj/${model_name}/eval.txt
        python evaluate/evaluator.py \
            --task AR_ywj \
            --fpath /home/tianxiangwu/zecheng/evaluation/AR_ywj/${model_name}/version_$JOB_ID/results/predictions.pkl \
            --save_evaluation_path /home/tianxiangwu/zecheng/evaluation/AR_ywj/${model_name}/eval.txt;
    done
done
