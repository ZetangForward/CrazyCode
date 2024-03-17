export CUDA_VISIBLE_DEVICES=7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
job_id=3

# python mamba/test.py \
#     job_id=${job_id} \
#     model.ckpt_path=/public/home/ljt/tzc/ckpt/simplepajama/longalpaca_1/checkpoints/last.ckpt/model.bin \
#     model.load_model_state_dict=True \
#     task='longbench_ywj' \
#     exp_task='longbench_ywj' \
#     platform=langchao \
#     experiment.device_num=${num_devices} \
#     >/public/home/ljt/tzc/modelzipper/projects/state-space-model/scripts/${job_id}.log 2>&1 &
   
python evaluate/evaluator.py \
    --task longbench_ywj \
    --fpath /public/home/ljt/tzc/evaluation/longbench_ywj/mamba-1_4b/ \
    --save_evaluation_path /public/home/ljt/tzc/evaluation/longbench_ywj/mamba-1_4b/;
   

    
    