import torch
from diffusers import StableDiffusionPipeline
import sys
sys.path.append('/workspace/zecheng/modelzipper/projects')
from data.vqlseq2seq_dataset import VQDataCollator, VQSeq2SeqData
import transformers
from modelzipper.tutils import *



pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


MODEL_NAME_OR_PATH = "/zecheng2/vqllama/vqllama_flant5/version_1/checkpoint-1200"
    
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
        "/zecheng2/svg/icon-shop/test_data_snaps/test_mesh_data_svg_convert_p.pkl", 
        tokenizer=flant5_tokenizer, 
        offline_mode=False,
        mode="test",
        svg_begin_token = None,
        inferece_nums=2000,
    )

predict_dataset = svg_data_module.predict_dataset

for data in predict_dataset:
    
    keywords = data['keywords']
    
    image = pipe(keywords).images[0]

    import pdb; pdb.set_trace()