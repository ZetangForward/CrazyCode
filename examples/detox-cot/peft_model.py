from modelzipper import *  # modelzipper will load all the necessary modules
from modelzipper.datamanager import *
from peft import PeftModel, LoraConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig
from accelerate import Accelerator
from dataclasses import field, dataclass

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
        
        ipt_text = sample_ipt + " " + label

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

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        


    
def main(cf: str = None):
    assert cf is not None, "Please specify a config file --cf config_path"
    
    accelerator = Accelerator()

    cfg = load_yaml_config(cf)

    parser = transformers.HfArgumentParser((TrainingArguments, CustomArguments))
    hf_args = parser.parse_args_into_dataclasses()[0]

    if cfg.load_tuned_model:
        peft_config = LoraConfig.from_pretrained(cfg.model_name_or_path)
        peft_config.base_model_name_or_path = cfg.model_name_or_path
    else:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = LlamaForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = 0
    model = PeftModel(model, peft_config)
    model.print_trainable_parameters()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        model_max_length=cfg.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # import pdb; pdb.set_trace()

    train_dataset = BaseData(cfg.data_path, tokenizer, cfg.model_max_length, "train")

    trainer = CustomTrainier(
        model,
        args=hf_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(hf_args.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
