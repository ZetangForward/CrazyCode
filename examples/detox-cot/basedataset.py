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
    
    def __init__(self, file, tokenizer=None, tokenizer_args=None, max_seq_length=None, split="train"):
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

    
        
        
    