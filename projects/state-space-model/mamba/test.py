import os  
import sys
sys.path.append(os.getcwd())
import torch   
import pytorch_lightning as pl
import hydra  
from custom_dataset import *
from custom_dataset.zero_scroll import *
from modelzipper.tutils import *
from custom_mamba.position_mamba import PositionMamba


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
        
    def denormalize_func(self, normalized_tensor, min_val=0, max_val=200):
        tensor = (normalized_tensor + 1) / 2
        tensor = tensor * (max_val - min_val) + min_val
        tensor = torch.round(tensor).long()
        return tensor

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model.generate(batch['input_ids'], max_length=self.cfg.experiment.max_seq_length, temperature=0.9, top_p=0.7, eos_token_id=self.tokenizer.eos_token_id)
        import pdb; pdb.set_trace() 
        subset = batch['subset'][0]
        print_c("one sample generation ending")
       
        standard_test_reconstruct = {
            "prediction": self.tokenizer.decode(output[0]),
        }
        
        return standard_test_reconstruct


@hydra.main(config_path='../configs', config_name='test_mamba', version_base='1.1')
def main(config):
    
    print_c(f"Experiment: {config.experiment.test_task}", "magenta")
    
    # load model and tokenizer
    model = PositionMamba.from_pretrained(
        config.model.model_name_or_path, 
        use_position=config.model.use_position,
        dtype=torch.bfloat16, 
        device="cuda", 
        strict=False
    )

    if config.model.load_model_state_dict:
        state_dict = torch.load(config.model.ckpt_path, map_location='cuda')
        model.load_state_dict(state_dict, strict=True)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if "gpt-neo" in config.tokenizer.tokenizer_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    
    # load data
    if config.experiment.test_task.lower() == "insert_needle":
        data_module = FindNeedle(config.dataset, tokenizer, config.dataset.data_path)
    elif config.experiment.test_task.lower() == "zero_scroll":
        data_module = ZeroScrolls(config.dataset, tokenizer)
    else:
        data_module = custom_datamodule(config.dataset, tokenizer)

    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()

    predictions = tester.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if not config.model.load_model_state_dict else None
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    save_path = f"{config.experiment.results_save_dir}/predictions.jsonl"
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()