import random
import os
import transformers
from dataclasses import dataclass, field
from transformers import Trainer
from modelzipper.tutils import *
from models.vqllama import VQSVGLlama
from data.vqllama_dataset import VQDataCollator, VQLLaMAData
from models.vqvae import VQVAE

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
    model_name_or_path: str = field(default=None)
    data_path: str = field(default=None)




class PluginVQVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

def train():
    parser = transformers.HfArgumentParser((TestConfig))
    test_args = parser.parse_args_into_dataclasses()
    
    # parsing vqvae_config:
    vqvae_config = load_yaml_config(test_args.vqvae_config_path)

    # config 
    llamaconfig = transformers.LlamaConfig.from_pretrained(test_args.model_name_or_path)
    llamaconfig.frozen_llm = False
    llamaconfig.max_text_length = 64
    llamaconfig.svg_token_dims = 4096
    llamaconfig.min_path_nums = 4
    llamaconfig.max_path_nums = 512
    
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_args.tokenier_config_path,
        model_max_length=test_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    svg_data_module = VQLLaMAData(
        llamaconfig, 
        test_args.data_path, 
        svg_begin_token=DEFAULT_SVG_BEGIN_TOKEN, 
        tokenizer=llama_tokenizer, 
        offline_mode=False,
        mode="test"
    )

    data_collator = VQDataCollator(
        svg_pad_token_h=llamaconfig.svg_token_dims, 
        max_svg_length=llamaconfig.max_path_nums,
        offline_mode=True,
        return_all_token_mask=True, # for offline setting
    )
    
    data_module = dict(
        train_dataset=svg_data_module.train_dataset, 
        eval_dataset=svg_data_module.valid_dataset, 
        data_collator=data_collator
    )

    svgllama = VQSVGLlama.from_pretrained(
        test_args.model_name_or_path, 
        config=llamaconfig, 
        codebook_size=vqvae_config.vqvae.l_bins,
        cache_dir=training_args.cache_dir
    )

    if "llama" in test_args.model_name_or_path.lower():
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
    # offline inference version
    vqvae = VQVAE(vqvae_config, multipliers=None, **block_kwargs)
    plugin_vqvae = PluginVQVAE(vqvae)
    checkpoint = torch.load(vqvae_config.ckpt_path)  # load vqvae ckpt
    plugin_vqvae.load_state_dict(checkpoint['state_dict'])
    print_c("VQVAE loaded!", "green")
    svgllama.init_vqvae(plugin_vqvae)
    

    # # init optimizer
    # if svgllama.model_parallel:
    #     all_params = [param for module in svgllama.modules() for param in module.parameters()]
    # else:
    #     all_params = svgllama.parameters()
    
    # trainable_params = [p for p in all_params if p.requires_grad]
    # optimizer = torch.optim.AdamW(trainable_params, lr=training_args.learning_rate)

    # # init lr scheduler
    # lr_scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=training_args.warmup_steps,
    #     num_training_steps=training_args.max_steps,
    # )

    trainer = CustomTrainier(model=svgllama, tokenizer=llama_tokenizer, args=training_args, **data_module)
    
    svgllama.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()