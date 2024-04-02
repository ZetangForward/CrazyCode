export CUDA_VISIBLE_DEVICES=7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=64

model_name=mamba-370m-k$id
platform=langchao
task=longbench_ywj
model_path=/public/home/ljt/tzc/ckpt/longalpaca/370m-k${id}/checkpoints/last.ckpt
mark=alpaca

python src/test_dev2.py \
    -mn $model_name \
    -en $mark \
    -pn $platform \
    -dn $task \
    --state eval \
    --inference_mode \
    --ckpt_path $model_path;
    