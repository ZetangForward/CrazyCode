export CUDA_VISIBLE_DEVICES=4,5,6,7 

python mamba/train.py experiment.low_rank_train=True experiment.device_num=4 exp_task=longalpaca task=longalpaca model.