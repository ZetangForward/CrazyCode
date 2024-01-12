import sys
sys.path.append("/workspace/zecheng/modelzipper/projects/custom_llama/baselines")
from custom_dataset import *
import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

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
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch_input_ids, batch_attn_mask, batch_label = [], [], []
        
        for ins in instances:
            batch_input_ids.append(ins["input_ids"])
            batch_attn_mask.append(ins["attention_mask"])
            batch_label.append(ins["labels"])
            
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attn_mask = torch.stack(batch_attn_mask, dim=0)
        batch_label = torch.stack(batch_label, dim=0)
        
        return {
            "batch_input_ids": batch_input_ids,
            "batch_attn_mask": batch_attn_mask,
            "batch_labels": batch_label,
        }
        
@dataclass
class DataCollatorForCM3(object):
    """Collate examples for CM3 training objective."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch_input_ids, batch_attn_mask, batch_label = [], [], []
        
        for ins in instances:
            batch_input_ids.append(ins["input_ids"])
            batch_attn_mask.append(ins["attention_mask"])
            batch_label.append(ins["labels"])
            
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attn_mask = torch.stack(batch_attn_mask, dim=0)
        batch_label = torch.stack(batch_label, dim=0)
        
        return {
            "batch_input_ids": batch_input_ids,
            "batch_attn_mask": batch_attn_mask,
            "batch_labels": batch_label,
        }

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
        input_ids = inputs.get("batch_input_ids")
        batch_attention_mask = inputs.get("batch_attention_mask")
        batch_labels = inputs.get("batch_labels")
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )
        
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss 

              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    model.config.pad_token_id = 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.pad_token_id = tokenizer.unk_token_id

    train_file = os.path.join(data_args.data_path, "offline_750_train_v2.jsonl")
    val_file = os.path.join(data_args.data_path, "offline_750_valid_v2.jsonl")
    
    train_dataset = OfflineDataset(training_args, train_file, tokenizer)
    val_dataset = OfflineDataset(training_args, val_file, tokenizer)

    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = CustomTrainier(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()