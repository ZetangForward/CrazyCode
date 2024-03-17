export CUDA_VISIBLE_DEVICES=7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
job_id=3
model=deepseek-1_3b_instruct
# model=deepseek-1_3b

# python mamba/test.py \
#     job_id=${job_id} \
#     model=$model \
#     model_name=$model \
#     task='longbench_ywj' \
#     exp_task='longbench_ywj' \
#     platform=langchao \
#     experiment.device_num=${num_devices} \
#     >/public/home/ljt/tzc/modelzipper/projects/state-space-model/scripts/${model}_${job_id}.log 2>&1 &

# wait 

python evaluate/evaluator.py \
    --task longbench_ywj \
    --fpath /public/home/ljt/tzc/data/evaluation/longbench_ywj/$model/ \
    --save_evaluation_path /public/home/ljt/tzc/data/evaluation/longbench_ywj/$model/;
   

    
    