import random
import os
import transformers
from dataclasses import dataclass, field
from transformers import Trainer
from modelzipper.tutils import *
from models.vqllama_understanding import VQSVGLlamaUnderstanding
from data.vqllama_dataset import UnderstandingDataCollator, VQLLaMAData

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SVG_BEGIN_TOKEN = "<SVG>"

@dataclass
class VQVAEConfig:
    config_path: str = field(default=None)
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    vq_svg_pad_file: str = field(default=None, metadata={"help": "Path to the vq svg pad file."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mask_ratio: float = field(default=0.5)
    n_mask: int = field(default=4)
    hybrid: str = field(default="keywords")  # description, hybrid
    is_augment: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_token_num = model.resize_token_embeddings(len(tokenizer))
    print(f"Adding {new_token_num} tokens to the pretrained dict")
    
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
        
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss
        

class PluginVQVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, VQVAEConfig))
    model_args, data_args, training_args, vqvae_args = parser.parse_args_into_dataclasses()

    # config 
    llamaconfig = transformers.LlamaConfig.from_pretrained(model_args.model_name_or_path)
    llamaconfig.frozen_llm = False
    llamaconfig.max_text_length = 256
    llamaconfig.min_path_nums = 4
    llamaconfig.max_path_nums = 1024
    
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    llama_tokenizer.add_eos_token = True

    svgllama = VQSVGLlamaUnderstanding.from_pretrained(
        model_args.model_name_or_path, 
        config=llamaconfig, 
        cache_dir=training_args.cache_dir,
        codebook_size=8192,
    )

    if "llama" in model_args.model_name_or_path.lower():
        # add new tokens and resize embedding & LM head
        added_tokens = {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
        smart_tokenizer_and_embedding_resize(
            added_tokens, llama_tokenizer, svgllama
        )
    
    svg_data_module = VQLLaMAData(
        llamaconfig, 
        data_args.data_path, 
        svg_begin_token=DEFAULT_SVG_BEGIN_TOKEN, 
        tokenizer=llama_tokenizer, 
        offline_mode=True,
        task="understanding",
    )

    data_collator = UnderstandingDataCollator()

    data_module = dict(
        train_dataset=svg_data_module.train_dataset, 
        eval_dataset=svg_data_module.valid_dataset, 
        data_collator=data_collator
    )


    svgllama.set_tokenizer(llama_tokenizer)


    # Tell Trainer not to attempt DataParallel
    svgllama.is_parallelizable = True
    svgllama.model_parallel = True

    trainer = CustomTrainier(model=svgllama, tokenizer=llama_tokenizer, args=training_args, **data_module)
    
    svgllama.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()