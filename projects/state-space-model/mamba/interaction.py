import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default='cuda', help='Device to run the model on')
    parser.add_argument("--tokenizer", type=str, default='/nvme/hf_models/EleutherAI/gpt-neox-20b')
    parser.add_argument("--model", type=str, default="/nvme/hf_models/mamba-1.4b")
    parser.add_argument("--ckpt", type=str, default='/nvme/zecheng/ckpt/mamba_alpaca/version_1/checkpoints/mamba-mamba_alpaca-epoch=36.ckpt', help='Model to use')
    parser.add_argument("--share", action="store_true", default=False, help="share your instance publicly through gradio")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)
    return args

class custom_model(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model

if __name__ == "__main__":
    args = get_args()

    device = args.device
    eos = "<|endoftext|>"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    model = custom_model(model)
    
    state_dict = torch.load(args.ckpt, map_location=device)['state_dict']
    model.load_state_dict(state_dict)

    model = model.model
    
    def chat_with_mamba(
        user_message,
        history: list[list[str]],
        temperature: float = 0.9,
        top_p: float = 0.7,
        max_length: int = 64,
    ):
        # history_dict: list[dict[str, str]] = []
        # for user_m, assistant_m in history:
        #     history_dict.append(dict(role="user", content=user_m))
        #     history_dict.append(dict(role="assistant", content=assistant_m))
        # history_dict.append(dict(role="user", content=user_message))

        # input_ids = tokenizer.apply_chat_template(
        #     history_dict, return_tensors="pt", add_generation_prompt=True
        # ).to(device)

        input_ids = tokenizer(user_message, return_tensors="pt",).to(device).input_ids

        out = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        
        # assistant_message = (
        #     decoded[0].split("<|assistant|>\n")[-1].replace(eos, "")
        # )
        return decoded_text
    

    demo = gr.ChatInterface(
        fn=chat_with_mamba,
        # examples=[
        #     "Explain what is state space model",
        #     "Nice to meet you!",
        #     "'Mamba is way better than ChatGPT.' Is this statement correct?",
        # ],
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="temperature"),
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.7, label="top_p"),
            gr.Number(value=256, label="max_length"),
        ],
        title="Mamba Chat",
    )

    demo.launch(server_port=args.port, share=args.share)