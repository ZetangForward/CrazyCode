# define model
MODEL_NAME="codellama7b"
MODEL_DIR="/zecheng2/svg_model_hub/CodeLlama-7b-offline_v3/checkpoint-386"
FILE_PATH="/zecheng2/svg/icon-shop/test_data_snaps/offline_750_valid_v2_0.jsonl"
OUTPUT_DIR="/zecheng/svg/icon-shop/meta_data/test_generation_results"



python generate_svg.py --file_path ${FILE_NAME} --base_model ${MODEL_DIR} --output_file ${OUTPUT_PATH}/${CHECKPOINT}_${FILE_NAME} --max_new_tokens 2048