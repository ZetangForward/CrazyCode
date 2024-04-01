export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$2

model_name=mamba-370m-k$id
platform=langchao
task=longbench_ywj
model_path=/public/home/ljt/tzc/ckpt/longalpaca/370m-k${id}/checkpoints/last.ckpt

python src/test.py \
    -mn $model_name \
    -en $mark \
    -pn $platform \
    -dn $task \
    --state eval \
    --inference_mode \
    --ckpt_path $model_path \
    --processed_data_path MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl;
    mark=${mark} \
    model=$model \
    model_name=$model \
    model.ckpt_path=/nvme/zecheng/ckpt/h_800/ckpt/longalpaca/mamba_370m_big_kernel_$id/checkpoints/last.ckpt \
    model.load_model_state_dict=True \
    task=${task} \
    exp_task=${task} \
    platform=${platform} \
    experiment.device_num=${num_devices} \
    experiment.results_save_dir=longbench_ywj/$model/results 

    # >/nvme/zecheng/modelzipper/projects/state-space-model/scripts/log/${model}_${mark}.log 2>&1 &

# wait 

# python evaluate/evaluator.py \
#     --task longbench_ywj \
#     --fpath /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/ \
#     --save_evaluation_path /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/;
   

    
    