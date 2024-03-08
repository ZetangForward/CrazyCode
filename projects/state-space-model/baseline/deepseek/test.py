import os
import sys
sys.path.append(os.getcwd())
import torch   
import hydra 
import lightning.pytorch as pl
from modelzipper.tutils import *
from model import LlamaForCausalLM, LlamaModel
from utils import get_model_tokenizer, CustomDatamodule

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        output = self.model(
            batch['input_ids'], 
            # max_new_tokens=64,
            # eos_token_id=self.tokenizer.eos_token_id,
            # use_cache=True,
            output_attentions=True,
            return_dict=True,
            depth = batch['depth'].cpu().item(),
            ctx_length = batch['ctx_length'].cpu().item()
        )

        # label = batch['labels'][0]
        standard_test_reconstruct = None
        # standard_test_reconstruct = {
        #     "prediction": self.tokenizer.decode(output[0]),
        #     # "golden": self.tokenizer.decode(label),
        # }
        
        return standard_test_reconstruct
    

@hydra.main(config_path='../../configs', config_name='test_config', version_base='1.1')
def main(config):
    
    print_c(OmegaConf.to_yaml(config), "yellow")

    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.result_path
    data_root_dir = config.platform.dataset_path

    # load model and tokenizer
    model = LlamaModel.from_pretrained(os.path.join(model_root_dir, config.model.model_name_or_path), torch_dtype=torch.bfloat16).to('cuda:6')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root_dir, config.model.tokenizer_name_or_path))
        
    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    
    # load testing data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()

    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if config.model.load_model_state_dict else None
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    save_path = f"{config.experiment.results_save_dir}/predictions.jsonl"
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()