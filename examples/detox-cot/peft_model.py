from modelzipper import *  # modelzipper will load all the necessary modules
from peft import PeftModel, LoraConfig
import logging
logging.basicConfig(level=logging.INFO)


def train():

    accelerator = Accelerator()
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    
    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 0

    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2

    # tokenizer.padding_side = "right"  ## Allow batched inference
    max_seq_length = {
        "max_enc_length": 512,   
        "max_dec_length": 512,   
    }
    
    tokenizer_args = {
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }

    device_map = "auto"
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    # lora  hyperparams
    
    
    dataset = BaseData(args.data_paths[0], tokenizer, tokenizer_args, max_seq_length, "train")
    
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collect_fn)
    trainer = Trainer(
        model,
        args=args,
        data_collator=dataset.collect_fn,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


def main(cf: str = None):
    assert cf is not None, "Please specify a config file --cf config_path"

    cfg = load_yaml_config(cf)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = PeftModel(model, peft_config)
    model.print_trainable_parameters()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
