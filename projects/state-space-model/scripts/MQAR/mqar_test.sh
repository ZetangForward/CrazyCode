export CUDA_VISIBLE_DEVICES=$1

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$2
model_name=mamba-370m-k$id
platform=langchao 
task=AR_ywj

declare -A model_pairs=(
    # [512]=32
    [1024]=64
    # [2048]=128
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
            model_path=/public/home/ljt/tzc/ckpt/AR_ywj/mamba_370m_big_kernel_${id}_${MODEL_N}_${MODEL_D}/checkpoints/last.ckpt
            python src/test_dev2.py \
                -mn $model_name \
                -en $mark \
                -pn $platform \
                -dn MQAR_ywj \
                --state eval \
                --inference_mode True \
                --ckpt_path $model_path \
                --input_seq_len $DATA_N \
                --num_kv_pairs $DATA_D\
                --processed_data_path MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl;
            wait
        done
    done
done

            #   \
            #     > /nvme/zecheng/modelzipper/projects/state-space-model/scripts/log/${model_name}_test.log 2>&1 & 

# export CUDA_VISIBLE_DEVICES=$1
# num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# id=$2
# model_name=mamba-370m-k$id
# platform=langchao 
# task=AR_ywj
# DATA_N=1024
# DATA_D=64
# mark=N${MODEL_N}_D${MODEL_D}-N${DATA_N}_D${DATA_D}
# model_path=/public/home/ljt/tzc/ckpt/AR_ywj/mamba_370m_big_kernel_8_1024_64/checkpoints/last.ckpt
# # model_path=/nvme/zecheng/ckpt/AR_ywj-mamba-1_4b/version_MQAR_C8192_N1024_D64/AR_ywj/MQAR_C8192_N1024_D64/checkpoints/last.ckpt
# python src/test_dev2.py \
#     -mn $model_name \
#     -en $mark \
#     -pn $platform \
#     -dn MQAR_ywj \
#     --state eval \
#     --inference_mode True \
#     --ckpt_path $model_path \
#     --input_seq_len $DATA_N \
#     --num_kv_pairs $DATA_D\
#     --processed_data_path MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl \


#     # \
#     # > /nvme/zecheng/modelzipper/projects/state-space-model/scripts/log/${model_name}_test.log 2>&1 & 
# # load_model_state_dict
    