export CUDA_VISIBLE_DEVICES=0

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

mkdir -p /public/home/ljt/tzc/evaluation/AR_ywj/mamba-1_4b
MODEL_N=1024
MODEL_D=64
DATA_N=512
DATA_D=48

JOB_ID=N${MODEL_N}_D${MODEL_D}-N${DATA_N}_D${DATA_D}

# JOB_ID=N4096_D256-N4096_D128
python mamba/test.py \
    model.ckpt_path=/public/home/ljt/tzc/ckpt/AR_ywj-mamba-1_4b/version_MQAR_C8192_N${MODEL_N}_D${MODEL_D}/AR_ywj/MQAR_C8192_N${MODEL_N}_D${MODEL_D}/checkpoints/last.ckpt \
    model.load_model_state_dict=True \
    platform=langchao \
    task='AR_ywj' \
    exp_task='AR_ywj' \
    job_id=$JOB_ID \
    experiment.device_num=$num_devices \
    task.dataset.processed_data_path=MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl \
    task.dataset.input_seq_len=$DATA_N \
    task.dataset.num_kv_pairs=$DATA_D;
