import sys
sys.path.append("/workspace/zecheng/layout-generation/custom_llama_x")
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
from hybird_llama import SvgLlama, SvgTokenizer, SvghybirdDataset
import torch
import torch.distributed
import transformers
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_EOS_TOKEN = "<s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SVG_TOKEN = "<SVG>"
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

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
    text_width: int = field(default=64)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch_text_input_ids, batch_text_attn_mask, batch_text_label = [], [], []
        batch_svg_input_ids, batch_svg_attn_mask, batch_svg_label = [], [], []
        
        for ins in instances:
            batch_text_input_ids.append(ins["input_ids"]["text_inputs"])
            batch_svg_input_ids.append(ins["input_ids"]["svg_inputs"])
            batch_text_attn_mask.append(ins["attention_mask"]["text_attention_mask"])
            batch_svg_attn_mask.append(ins["attention_mask"]["svg_path_attention_mask"])
            batch_text_label.append(ins["labels"]["text_labels"])
            batch_svg_label.append(ins["labels"]["svg_labels"])
            
        batch_text_input_ids = torch.stack(batch_text_input_ids, dim=0)
        batch_svg_input_ids = torch.stack(batch_svg_input_ids, dim=0)
        batch_text_attn_mask = torch.stack(batch_text_attn_mask, dim=0)
        batch_svg_attn_mask = torch.stack(batch_svg_attn_mask, dim=0)
        batch_text_label = torch.stack(batch_text_label, dim=0)
        batch_svg_label = torch.stack(batch_svg_label, dim=0)

        sample = {
            "batch_input_ids": {
                "batch_text_input_ids": batch_text_input_ids,
                "batch_svg_input_ids": batch_svg_input_ids,
            },
            "batch_attention_mask": {
                "batch_text_attn_mask": batch_text_attn_mask,
                "batch_svg_attn_mask": batch_svg_attn_mask,
            },
            "batch_labels": {
                "batch_text_label": batch_text_label,
                "batch_svg_label": batch_svg_label,
            }
        }
        
        return sample
        

class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, svg_tokenizer=None, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
        self.svg_tokenizer = svg_tokenizer
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("batch_input_ids")
        batch_attention_mask = inputs.get("batch_attention_mask")
        batch_labels = inputs.get("batch_labels")
        
        outputs = model(
            input_ids=input_ids,
            attention_masks=batch_attention_mask,
            labels=batch_labels,
        )

        loss = outputs.loss
        total_loss = loss["text loss"] + 2 * loss["svg loss"]

        return (total_loss, outputs) if return_outputs else total_loss 


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


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # config 
    llamaconfig = transformers.LlamaConfig.from_pretrained(
        model_args.model_name_or_path
    )
    
    svg_tokenizer = SvgTokenizer.from_pretrained(  
        "/zecheng/svg_model_hub/custom_config",  
        vocab_file="/zecheng/svg_model_hub/custom_config/vocab_v2.txt"  
    )  
    
    llamaconfig.svg_vocab_size = svg_tokenizer.vocab_size
    llamaconfig.text_width = 64
    llamaconfig.frozen_llm = False
    
    llama_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # svgllama = SvgLlama(llamaconfig)
    svgllama = SvgLlama.from_pretrained(
        model_args.model_name_or_path, 
        config=llamaconfig, 
        text_tokenizer=llama_tokenizer,
        cache_dir=training_args.cache_dir
    )
    
    if "llama" in model_args.model_name_or_path.lower():
        # add new tokens and resize embedding & LM head
        added_tokens = {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
            "additional_special_tokens": [DEFAULT_SVG_TOKEN],
        }

        smart_tokenizer_and_embedding_resize(
            added_tokens, llama_tokenizer, svgllama
        )
        
        # llama_tokenizer.pad_token_id = llama_tokenizer.unk_token_id
    
    train_file = os.path.join(data_args.data_path, "offline_500_train.jsonl")
    val_file = os.path.join(data_args.data_path, "offline_500_valid.jsonl")
    
    train_dataset = SvghybirdDataset(training_args, train_file, svg_tokenizer=svg_tokenizer, model_tokenizer=llama_tokenizer)
    val_dataset = SvghybirdDataset(training_args, val_file, svg_tokenizer=svg_tokenizer, model_tokenizer=llama_tokenizer)


    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=llama_tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    svgllama.is_parallelizable = True
    svgllama.model_parallel = True

    trainer = CustomTrainier(model=svgllama, tokenizer=llama_tokenizer, args=training_args, **data_module)
    svgllama.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()