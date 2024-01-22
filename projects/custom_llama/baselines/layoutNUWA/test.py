import sys
import os
import fire
import torch
import transformers
import json
import re
from transformers import GenerationConfig
from tqdm import *
from modelzipper.tutils import *

def convert_svg_path(s):
    
    def truncate_path(path):  
        segments = re.findall(r'[MLC]\s+\d+\s+\d+', path)  
        truncated_path = ''  
        for segment in segments:  
            if segment.startswith('C'):  
                numbers = re.findall(r'\d+', segment)  
                if len(numbers) == 6:  # Only keep C segments with 6 numbers  
                    truncated_path += segment + ' '  
            else:  
                truncated_path += segment + ' '  
        return truncated_path.rstrip()  # Remove trailing whitespace 
    
    # first check the </svg> tag:
    if "</svg>" in s:
        path = s.split("</svg>")[0].split("<svg")[-1].strip()
    else:
        path = s.split("<svg>")[-1].strip()
        
    total_seq = []
    # truncate all the sub paths together
    matches = re.findall(r'<path d="(.*?)">', s)  
    for match in matches:
        total_seq.append(match)
    
    # saint check for the last path
    last_path = path.split("<path d=")[-1].strip()
    if ">" not in last_path:  # incomplete path
        complete_last_path = truncate_path(last_path)
        total_seq.append(complete_last_path)
    
    full_seq_str = " ".join(total_seq)
    
    template = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0.0 0.0 200.0 200.0\" height=\"200px\" width=\"200px\"><path fill=\"black\" fill-opacity=\"1.0\" filling=\"0\" d=\"{svg_path} \"></path></svg>"
    full_seq_str = template.format(svg_path=full_seq_str)
    
    return full_seq_str


def main(
    file_path: str = None,
    base_model: str = "/zecheng/model_hub/Llama-2-7b-multinode/checkpoint-1800",
    output_file: str=None,
    max_new_tokens: int=2048,
    num_beams: int=4,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    
    assert file_path, (
        "Please specify a --file_path, e.g. --file_path='/path/to/json_file'"
    )
    
    assert output_file, (
        "Please specify a --output_dir, e.g. --output_dir='/path/to/output_dir/output_file'"
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
        )
    
    model.half()
    model.eval()
    
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    def evaluate(
        instruction,
        temperature=0.6,
        top_p=0.9,
        num_beams=4,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        **kwargs,
    ):
        prompt = instruction
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )
        
        with torch.no_grad():
            generation_output = model.generate(
                use_cache=True,
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        return generation_output.sequences
    
    # with open(file_path, "r") as f:
    #     content = [json.loads(line) for line in f]
    
    content = auto_read_data(file_path)
    
    with open(output_file, "w") as f:
        with tqdm(total=len(content)) as pbar:
            for samples in content:
                prompt = "Keywords: " + ', '.join(samples.get("keys")) + " #Begin:"
                output = evaluate(prompt, num_beams=num_beams, max_new_tokens=max_new_tokens)
                processed_res = []
                for s in output:
                    processed_res.append(convert_svg_path(tokenizer.decode(s)))
                
                tmp = {
                    "processed_res": processed_res[0],
                    "prompt": prompt,
                    "keywords": samples.get("keys"),
                }
                pbar.update(1)
                f.write(json.dumps(tmp) + "\n")
                f.flush()
            
    print("generation done !!")
            
        
if __name__ == "__main__":
    fire.Fire(main)