import random
import os
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm, trange
from torch import Tensor
from modelzipper.tutils import *
from models.vqllama import VQSVGLlama
from data.vqllama_dataset import VQLLaMAData
from models.vqvae import VQVAE
from train_vqllama import smart_tokenizer_and_embedding_resize
from utils.visualize_svg import sanint_check_svg_tensor, convert_svg, merge_images


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SVG_BEGIN_TOKEN = "<SVG>"


@dataclass
class TestConfig:
    vqvae_config_path: str = field(default=None)
    tokenier_config_path: str = field(default=None)
    version: int = field(default=None)
    epoch: int = field(default=None)
    data_path: str = field(default=None)
    predict_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=0)
    max_generate_length: int = field(default=1024)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.9)
    top_k: int = field(default=40)
    num_beams: int = field(default=1)
    temperature: float = field(default=0.8)


class PluginVQVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def predict_loop(model, vqvae, dataloader, tokenizer, max_generate_length=1024, **kwargs) -> List[Tensor]:
    
    res = []
    with tqdm(desc="Predicting", total=len(dataloader)) as pbar:
        for batch_ in dataloader:
            cur_batch_res = []
            text_input_ids = batch_.get("text_input_ids")
            text_attention_mask = batch_.get("text_attention_mask")
            golden_svg_path = batch_.get("svg_path")
            
            text_input_ids = text_input_ids.to(model.device) if text_input_ids is not None else None
            text_attention_mask = text_attention_mask.to(model.device) if text_attention_mask is not None else None
            golden_svg_path = golden_svg_path.to(model.device) if golden_svg_path is not None else None
            
            with torch.no_grad():
                _, post_processed_ids = model.generate(  # List[Tensor]
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    max_generate_length=max_generate_length,
                    **kwargs
                )
                
                for i, svg_token_ids in enumerate(post_processed_ids):
                    decoded_svg_path = vqvae.decode(  # L x l_bins ?
                        zs=svg_token_ids, start_level=0, start_level=1, padding_mask=None, path_interpolation=True, return_postprocess=True)[0]
                    
                    text = tokenizer.decode(text_input_ids[i], skip_special_tokens=True)
                    
                    cur_batch_res.append(  # move to the CPU menory
                        dict(
                            golden_svg_path = golden_svg_path.cpu(),
                            generated_svg_path = decoded_svg_path.cpu(),
                            text = text,
                            svg_token_ids = svg_token_ids.cpu(),
                        )
                    )
                    
            res.extend(cur_batch_res)
            pbar.update(1)
    return res
                    
                    
def post_process(res: List[Dict], save_dir=None, generate_big_map=True, add_background=False, save_intermediate_results=False) -> None:
    
    assert save_dir is not None, "save_dir must be specified!"
    SINGLE_IMAGE_SAVED_DIR = auto_mkdir(os.path.join(save_dir, "rendered_single_image")) # save single image
    SVG_PATH_SAVED_PATH = os.path.join(save_dir, "svg_paths.jsonl") # save svg path
    
    keys = ['generated_svg_path', 'golden_svg_path', 'text', 'svg_token_ids']
    str_paths = []
    all_image_paths = []
    
    for i in trange(len(res)):
        generated_svg_path = res[i]['generated_svg_path']
        golden_svg_path = res[i]['golden_svg_path']
        text = res[i]['text']

        predict = sanint_check_svg_tensor(generated_svg_path).squeeze(0)
        p_svg, p_svg_str = convert_svg(predict, True)
        golden = sanint_check_svg_tensor(golden_svg_path).squeeze(0)
        g_svg, g_svg_str = convert_svg(golden, True)

        str_paths.append({
            "text": text,
            "p_svg_str": p_svg_str,
            "g_svg_str": g_svg_str,
        })
        
        p_svg.save_png(os.path.join(SINGLE_IMAGE_SAVED_DIR, f"{i}_p_svg.png"))
        all_image_paths.append(os.path.join(SINGLE_IMAGE_SAVED_DIR, f"{i}_p_svg.png")))
    
    auto_save_data(str_paths, SVG_PATH_SAVED_PATH)
    
    if generate_big_map:
        print_c("begin to generate big map", "magenta")
        BIG_MAP_SAVED_DIR = auto_mkdir(os.path.join(save_dir, "rendered_big_map"))
        p_svg_images = merge_images(
            folder_path=SINGLE_IMAGE_SAVED_DIR, 
            image_suffix='p_svg.png', 
            num_images=500, 
            save_dir=BIG_MAP_SAVED_DIR
        )
        
    if add_background:
        print_c(f"add background to {len(all_image_paths)} images", "magenta")
        for i in trange(len(all_image_paths)):
            image_path = all_image_paths[i]
            if "_b.png" in image_path:
                continue
            add_background(image_path=image_path)
            
    if save_intermediate_results:
        raise NotImplementedError("save_intermediate_results is not implemented yet!")
        

def test():
    parser = transformers.HfArgumentParser((TestConfig))
    test_args = parser.parse_args_into_dataclasses()
    
    # parsing vqvae_config:
    vqvae_config = load_yaml_config(test_args.vqvae_config_path)

    # parsing trained model path
    SAVE_DIR="/zecheng2/vqllama/vqllama_llama"
    MODEL_NAME_OR_PATH = os.path.join(SAVE_DIR, f"version_{test_args.version}/checkpoint-{test_args.epoch}")
    
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_args.tokenier_config_path,
        model_max_length=test_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    # model config 
    llamaconfig = transformers.LlamaConfig.from_pretrained(MODEL_NAME_OR_PATH)
    llamaconfig.frozen_llm = False
    llamaconfig.max_text_length = 64
    llamaconfig.svg_token_dims = 4096
    llamaconfig.min_path_nums = 4
    llamaconfig.max_path_nums = 512
    
    svg_data_module = VQLLaMAData(
        llamaconfig, 
        test_args.data_path, 
        svg_begin_token=DEFAULT_SVG_BEGIN_TOKEN, 
        tokenizer=llama_tokenizer, 
        offline_mode=False,
        mode="test"
    )
    
    predict_dataloader = svg_data_module.predict_dataloader()

    svgllama = VQSVGLlama.from_pretrained(
       MODEL_NAME_OR_PATH, 
        config=llamaconfig, 
        codebook_size=vqvae_config.vqvae.l_bins,
    )
    
    # add new tokens and resize embedding & LM head
    added_tokens = {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN,
        "additional_special_tokens": [DEFAULT_SVG_BEGIN_TOKEN],
    }
    smart_tokenizer_and_embedding_resize(
        added_tokens, llama_tokenizer, svgllama
    )

    svg_begin_token_id = llama_tokenizer.convert_tokens_to_ids(DEFAULT_SVG_BEGIN_TOKEN)
    svgllama.add_svg_begin_token_id(svg_begin_token_id)
    svgllama.set_tokenizer(llama_tokenizer)

    # init VQVAE
    block_kwargs = dict(
        width=vqvae_config.vqvae_conv_block.width, 
        depth=vqvae_config.vqvae_conv_block.depth, 
        m_conv=vqvae_config.vqvae_conv_block.m_conv,
        dilation_growth_rate=vqvae_config.vqvae_conv_block.dilation_growth_rate,
        dilation_cycle=vqvae_config.vqvae_conv_block.dilation_cycle,
        reverse_decoder_dilation=vqvae_config.vqvae_conv_block.vqvae_reverse_decoder_dilation
    )
    vqvae = VQVAE(vqvae_config, multipliers=None, **block_kwargs)
    plugin_vqvae = PluginVQVAE(vqvae)
    checkpoint = torch.load(vqvae_config.ckpt_path)  # load vqvae ckpt
    plugin_vqvae.load_state_dict(checkpoint['state_dict'])
    print_c("VQVAE loaded!", "green")
    vqvae = plugin_vqvae.model
    
    svgllama.eval().cuda()
    vqvae.eval().cuda()
    
    if test_args.fp16:
        svgllama = svgllama.half()
        vqvae = vqvae.half()
        
    sampling_strategy = dict(
        do_sample=test_args.do_sample,
        temperature=test_args.temperature,
        top_p=test_args.top_p,
        top_k=test_args.top_k,
        num_beams=test_args.num_beams,
    )
    
    predicted_results = predict_loop(
        model=svgllama, 
        vqvae=vqvae,
        dataloader=predict_dataloader, 
        tokenizer=llama_tokenizer,
        max_generate_length=test_args.max_generate_length,
        **sampling_strategy,
    )
    
    post_process(
        predicted_results, 
        save_dir=test_args.save_image_dir, 
        generate_big_map=True, 
        add_background=False, 
        save_intermediate_results=False
    )
    
   


if __name__ == "__main__":
    test()