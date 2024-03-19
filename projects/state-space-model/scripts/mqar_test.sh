# export CUDA_VISIBLE_DEVICES=0

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

model_name=deepseek-1_3b
mkdir -p /public/home/ljt/tzc/data/evaluation/AR_ywj/${model_name}/
MODEL_N=$1
MODEL_D=$2
data_n_list="512 1024 2048 4096"
for DATA_N in $data_n_list
do
    data_d_list="32 48 64 96 128 192 256"
    for DATA_D in $data_d_list
    do
        JOB_ID=N${MODEL_N}_D${MODEL_D}-N${DATA_N}_D${DATA_D}

        python mamba/test.py \
            model=$model_name \
            model_name=$model_name \
            model.ckpt_path=/public/home/ljt/tzc/ckpt/AR_ywj/version_MQAR_C8192_N${MODEL_N}_D${MODEL_D}/checkpoints/last.ckpt \
            model.load_model_state_dict=True \
            platform=langchao \
            task='AR_ywj' \
            exp_task='AR_ywj' \
            job_id=$JOB_ID \
            experiment.device_num=$num_devices \
            task.dataset.processed_data_path=MQAR/test/test_C8192_N${DATA_N}_D${DATA_D}.pkl \
            task.dataset.input_seq_len=$DATA_N \
            task.dataset.num_kv_pairs=$DATA_D;

        wait

        python evaluate/evaluator.py \
            --task AR_ywj \
            --fpath /public/home/ljt/tzc/data/evaluation/AR_ywj/${model_name}/version_$JOB_ID/results/predictions.pkl \
            --save_evaluation_path /public/home/ljt/tzc/data/evaluation/AR_ywj/${model_name}/version_$JOB_ID/results/eval.txt;
    done
done
