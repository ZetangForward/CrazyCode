# export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
job_id=0
model=$1
# model=deepseek-1_3b

# python mamba/test.py \
#     job_id=${job_id} \
#     model=$model \
#     model_name=$model \
#     model.ckpt_path=/aifs4su/ziliwang/txw/InternLM/zecheng/ckpt/longalpaca/mamba-1_4b/checkpoints/last.ckpt \
#     model.load_model_state_dict=True \
#     task='longbench_ywj' \
#     exp_task='longbench_ywj' \
#     platform=h_800 \
#     experiment.device_num=${num_devices} \
#     experiment.results_save_dir=longbench_ywj/$model/results;
#     # >/public/home/ljt/tzc/modelzipper/projects/state-space-model/scripts/${model}_${job_id}.log 2>&1 &

# wait 

python evaluate/evaluator.py \
    --task longbench_ywj \
    --fpath /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/ \
    --save_evaluation_path /home/tianxiangwu/zecheng/evaluation/longbench_ywj/$model/results/;
   

    
    