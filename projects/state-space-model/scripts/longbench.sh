export CUDA_VISIBLE_DEVICES=0


python mamba/test.py \
    model.ckpt_path=/public/home/ljt/tzc/ckpt/simplepajama/longalpaca_1/checkpoints/last.ckpt/model.bin \
    model.load_model_state_dict=True \
    task='longbench_ywj' \
    exp_task='longbench_ywj' \
    platform=langchao 
    
    
    
    \
    # job_id=$JOB_ID \
    # task.dataset.processed_data_path=MQAR/test_C8192_N${DATA_N}_D${DATA_D}.pkl \
    # task.dataset.input_seq_len=$DATA_N \
    # task.dataset.num_kv_pairs=$DATA_D;

# wait

# python evaluate/evaluator.py \
#     --task AR_ywj \
#     --fpath /nvme/zecheng/evaluation/AR_ywj/mamba-1_4b/version_$JOB_ID/results/predictions.pkl \
#     --save_evaluation_path /nvme/zecheng/evaluation/AR_ywj/mamba-1_4b/version_$JOB_ID/results/eval.txt;
