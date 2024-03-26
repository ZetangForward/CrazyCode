export CUDA_VISIBLE_DEVICES=$1

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$2
model_name=mamba_370m_big_kernel_$id
platform=amax_a100
task=AR_ywj

declare -A model_pairs=(
    [512]=32
    [1024]=64
    [2048]=128
    [4096]=256
)

declare -A kv_pair_list=(
    [512]="32 48 64 96 128"
    [1024]="32 48 64 96 128 192 256"
    [2048]="32 48 64 96 128 192 256 384 512"
    [4096]="32 48 64 96 128 192 256 384 512 768 1024"
    # [8192]="32 48 64 96 128 192 256 384 512 768 1024"
)

for MODEL_N in "${!model_pairs[@]}"; do
    MODEL_D=${model_pairs[$MODEL_N]}
    for DATA_N in "${!kv_pair_list[@]}"; do
        DATA_D_LSIT=${kv_pair_list[$DATA_N]}
        for DATA_D in $DATA_D_LSIT; do
            mark=N${MODEL_N}_D${MODEL_D}-N${DATA_N}_D${DATA_D}
            model_path=/nvme/zecheng/ckpt/h_800/ckpt/AR_ywj/${model_name}_${MODEL_N}_${MODEL_D}/checkpoints/last.ckpt
            python src/test.py \
                mark=$mark \
                model=$model_name \
                model_name=$model_name \
                model.ckpt_path=$model_path \
                model.load_model_state_dict=True \
                platform=$platform \
                task=$task \
                exp_task=$task \
                task.dataset.processed_data_path=MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl \
                task.dataset.input_seq_len=$DATA_N \
                task.dataset.num_kv_pairs=$DATA_D \
                experiment.device_num=$num_devices \
                experiment.results_save_dir=$task/${model_name}/${mark}/results \
                > /nvme/zecheng/modelzipper/projects/state-space-model/scripts/log/${model_name}_test.log 2>&1 & 
            wait
        done
    done
done

                # 

                # # wait
                # if [ -f "$DATA_PATH" ] && [ -f "$PREDICTION_PATH" ] ; then
                #     echo $JOB_ID >> /home/tianxiangwu/zecheng/evaluation/AR_ywj/${model_name}/eval.txt
                #     python evaluate/evaluator.py \
                #     --task AR_ywj \
                #     --fpath /home/tianxiangwu/zecheng/evaluation/AR_ywj/${model_name}/version_$JOB_ID/results/predictions.pkl \
                #     --save_evaluation_path /home/tianxiangwu/zecheng/evaluation/AR_ywj/${model_name}/eval.txt;
        
                # else
                #     echo NOT_FOUND $DATA_PATH
                # fi