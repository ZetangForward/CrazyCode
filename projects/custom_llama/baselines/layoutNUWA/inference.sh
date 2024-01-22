# define model
MODEL_NAME="codellama7b"
CHECKPOINT="/zecheng2/vqllama/baselines/layoutNUWA/checkpoint-9280"

# define data
DATA_PATH="/zecheng2/svg/icon-shop/meta_data/test_snaps"
FILE_NAME="offline_750_valid_v2_sampled_1000_2.jsonl"

# define output
OUTPUT_PATH="/zecheng2/svg/icon-shop/meta_data/test_generation_results"

python test.py --file_path ${DATA_PATH}/${FILE_NAME} --base_model ${CHECKPOINT} --output_file ${OUTPUT_PATH}/${CHECKPOINT}_${FILE_NAME} --max_new_tokens 2048