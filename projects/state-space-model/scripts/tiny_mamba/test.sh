export CUDA_VISIBLE_DEVICES=1

python src/test_dev2.py \
-mn tiny_mamba \
-pn amax_a100 \
-en test \
--data_name "passkey_search";