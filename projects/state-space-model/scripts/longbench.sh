export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
job_id=$1
model=deepseek-1_3b
# model=deepseek-1_3b

python mamba/test.py \
    job_id=${job_id} \
    model=$model \
    model_name=$model \
    model.ckpt_path=/public/home/ljt/tzc/ckpt/longalpaca/deepseek-1_3b/checkpoints/last.ckpt \
    model.load_model_state_dict=True \
    task='longbench_ywj' \
    exp_task='longbench_ywj' \
    platform=langchao \
    experiment.device_num=${num_devices} \
    experiment.results_save_dir=longbench_ywj/$model/results \
    >/public/home/ljt/tzc/modelzipper/projects/state-space-model/scripts/${model}_${job_id}.log 2>&1 &

wait 

python evaluate/evaluator.py \
    --task longbench_ywj \
    --fpath /public/home/ljt/tzc/data/evaluation/longbench_ywj/$model/results/ \
    --save_evaluation_path /public/home/ljt/tzc/data/evaluation/longbench_ywj/$model/results/;
   

    
    