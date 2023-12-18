
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_path = "/nvme/hf_models/llama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 7},
    )

model.half()
model.eval()

# for llama model
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

def evaluate(
    instruction,
    temperature=0.4,
    top_p=0.9,
    top_k=60,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = instruction
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda:7')
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        **kwargs,
    )
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output


if __name__ == "__main__":
    ipt = "You are the not smart one for trying to make sense of it all."
    print(evaluate(ipt))

    print("second time")

    ipt = "In fact, garbage is already flooding"
    print(evaluate(ipt))