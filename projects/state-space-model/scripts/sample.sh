export CUDA_VISIBLE_DEVICES=7

# python mamba/test.py \
#     model.ckpt_path=/nvme/zecheng/ckpt/AR_ywj-mamba-1_4b/version_MQAR_C8192_N4096_D256/AR_ywj/MQAR_C8192_N4096_D256/checkpoints/last.ckpt \
#     model.load_model_state_dict=True \
#     task='AR_ywj' \
#     exp_task='AR_ywj';


python evaluate/evaluator.py \
    --task AR_ywj \
    --fpath /nvme/zecheng/evaluation/AR_ywj/mamba-1_4b/version_N4096_D256/results/predictions.pkl \
    --save_evaluation_path /nvme/zecheng/evaluation/AR_ywj/mamba-1_4b/version_N4096_D256/results/eval.txt \
