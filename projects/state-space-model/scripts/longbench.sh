# export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$1
model=mamba_370m_big_kernel_$id
platform=amax_a100
task=longbench_ywj
mark=${model}


python src/test.py \
    mark=${mark} \
    model=$model \
    model_name=$model \
    model.ckpt_path=/nvme/zecheng/ckpt/h_800/ckpt/longalpaca/${model}/checkpoints/last.ckpt \
    model.load_model_state_dict=True \
    task=${task} \
    exp_task=${task} \
    platform=${platform} \
    experiment.device_num=${num_devices} \
    experiment.results_save_dir=longbench_ywj/$model/results;
    # >/public/home/ljt/tzc/modelzipper/projects/state-space-model/scripts/${model}_${job_id}.log 2>&1 &

# wait 

# python evaluate/evaluator.py \
#     --task longbench_ywj \
#     --fpath /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/ \
#     --save_evaluation_path /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/;
   

    
    