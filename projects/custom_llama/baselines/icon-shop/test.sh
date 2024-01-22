# OUTPUT_DIR="/zecheng/svg/icon-shop/meta_data/length_filtering/rendered_images"
# FILE_NAME="checkpoint-2000_len_300_random50cases.jsonl"
INPUT_DIR="/zecheng2/svg/icon-shop/meta_data/test_generation_results"
OUTPUT_DIR="/zecheng2/svg/icon-shop/meta_data/length_filtering/rendered_images"
FILE_NAME="checkpoint-2360_offline_750_valid_v2_sampled_1000_0.jsonl"

python /workspace/zecheng/SUWA/svg_dataset/batch_visualize.py \
    -f ${INPUT_DIR}/${FILE_NAME} \
    -nc 10 -nr 10 \
    -od ${OUTPUT_DIR} \
    -ac -mt codellama2360