import torch
from diffusers import StableDiffusionPipeline
import sys
sys.path.append('/workspace/zecheng/modelzipper/projects/custom_llama')
from data.vqseq2seq_dataset import VQDataCollator, VQSeq2SeqData
import transformers
from modelzipper.tutils import *
from tqdm import tqdm


pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")


MODEL_NAME_OR_PATH = "/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-8100"
    
flant5_tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    model_max_length=64,
    padding_side="right",
    use_fast=True,
)

# config 
flant5config = transformers.AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
flant5config.frozen_llm = False
flant5config.max_text_length = 64
flant5config.min_path_nums = 4
flant5config.max_path_nums = 512
flant5config.use_cache = False
flant5config.predict_batch_size = 1
flant5config.dataloader_num_workers = 0

svg_data_module = VQSeq2SeqData(
        flant5config, 
        "/zecheng2/svg/icon-shop/test_data_snaps/test_data_all_seq_with_mesh.pkl", 
        tokenizer=flant5_tokenizer, 
        offline_mode=True,
        mode="test",
        svg_begin_token = None,
        inferece_nums=2000,
    )

# predict_datasets = svg_data_module.predict_dataset
predict_datasets = [r"electronic, equipment, device, chip"]

pipeline.set_progress_bar_config(leave=False)

# SAVE_DIR = "/zecheng2/vqllama/baselines/stablediffusion/"
SAVE_DIR = "/workspace/zecheng/modelzipper/projects/custom_llama/baselines/stablediffusion"
PROMPT = "Please generate a image in the icon format for me. here is the keywords: {keywords}"

with tqdm(total=len(predict_datasets)) as pbar:
    for i, data in enumerate(predict_datasets):
        # keywords = flant5_tokenizer.decode(data['text_input_ids'], skip_special_tokens=True)
        text_prompt = PROMPT.format(keywords=data)
        image = pipeline(text_prompt).images[0]
        file_path = os.path.join(SAVE_DIR, f"chip.png")
        image.save(file_path)
        pbar.update(1)
        
        import pdb; pdb.set_trace()