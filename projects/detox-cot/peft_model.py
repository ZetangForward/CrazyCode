
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig
from dataclasses import field, dataclass
from modelzipper import *  # modelzipper will load all the necessary modules
from modelzipper.datamanager import *

# Template for vanilla alpaca-lora
LLAMA_TEMPLATE_V1 = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    # "prompt_no_input": "\"{instruction}\" ",
    "prompt_no_input": "",
    "response_split": "### Response:"    
}


class BaseData(BaseDataset):
    
    def __init__(self, file, tokenizer=None, max_seq_length=None, split="train"):
        super(BaseData, self).__init__()
        
        self.content = auto_read_data(file)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __getitem__(self, index):
        sample = self.content[index]
        instruction = sample.get("instruction", "")
        input_ = sample.get("input", "")
        output = sample.get("output", "")
        
        # create input text
        if len(input_) != 0:
            sample_ipt = LLAMA_TEMPLATE_V1["prompt_input"].format(instruction=instruction, input=input_)
        else:
            sample_ipt = LLAMA_TEMPLATE_V1["prompt_no_input"].format(instruction=instruction)
        
        ipt_text = sample_ipt + "" + output

        seq_inputs = self.tokenizer(
            ipt_text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        text_input_ids = seq_inputs.input_ids[0]
        text_attention_mask = seq_inputs.attention_mask[0]
        text_labels = torch.where(text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100)

        return {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_labels,
        }

    @classmethod
    def custom_datacollator(cls, instances) -> Dict[str, torch.Tensor]:
        """Collate examples for supervised fine-tuning."""
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
            "batch_attention_mask": batch_attn_mask,
            "batch_labels": batch_label,
        }


class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset=None, tokenizer=None, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
        
    def compute_loss(self, model, inputs, return_outputs=False):
        text_inputs = inputs.get("batch_input_ids")
        batch_attention_mask = inputs.get("batch_attention_mask")
        batch_labels = inputs.get("batch_labels")
        
        outputs = model(
            input_ids=text_inputs,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
        )
        
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

@dataclass
class CustomArguments:
    cf: str = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    
def main(cf: str = None):
    assert cf is not None, "Please specify a config file --cf config_path"

    cfg = load_yaml_config(cf)

    parser = transformers.HfArgumentParser((TrainingArguments, CustomArguments))
    hf_args, _ = parser.parse_args_into_dataclasses()

    if cfg.load_tuned_model:
        peft_config = LoraConfig.from_pretrained(cfg.model_name_or_path)
        peft_config.base_model_name_or_path = cfg.model_name_or_path
    else:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj","v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = LlamaForCausalLM.from_pretrained(cfg.model_name_or_path)
    model.config.pad_token_id = 0
    # model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id

    train_dataset = BaseData(cfg.data_path, tokenizer, cfg.model_max_length, "train")

    model.is_parallelizable = True
    model.model_parallel = True

    trainer = CustomTrainier(
        model,
        args=hf_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.custom_datacollator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(hf_args.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
