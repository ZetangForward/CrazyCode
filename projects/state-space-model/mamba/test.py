import os  
import sys
sys.path.append(os.getcwd())
import torch   
import hydra  
import lightning.pytorch as pl
from modelzipper.tutils import *
from utils import get_model_tokenizer, CustomDatamodule

class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        if hasattr(config.task, "inference_cfg"):  # what to save for task setting
            for key in config.task.inference_cfg:
                if isinstance(key, int):
                    key = str(key)
                setattr(self, key, config.task.inference_cfg[key])

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        input_ids = batch.pop("input_ids")
        
        output = self.model.generate(
            input_ids, 
            max_length=input_ids.size(-1) + self.cfg.task.other_cfgs.max_generation_length,
            min_length=input_ids.size(-1) + 10, 
            eos_token_id=self.tokenizer.eos_token_id, 
        )
        final_res = {}
        final_res['predictions'] = output[0]
        
        if self.save_keys is not None:
            for key in self.save_keys:
                if key in batch:
                    value = batch[key]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    final_res[key] = value
        return final_res


@hydra.main(config_path='../configs', config_name='test_config', version_base='1.1')
def main(config):
    print_c(OmegaConf.to_yaml(config), "yellow")
 
    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.result_path
    data_root_dir = config.platform.dataset_path

    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(model_root_dir, config.model)
    
    # load testing data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    # if config.model.load_model_state_dict:
    #     state_dict = torch.load(
    #         os.path.join(config.platform.hf_model_path, config.model.ckpt_path), 
    #         map_location='cuda'
    #     )
    #     model.load_state_dict(state_dict, strict=True)

    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    
    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()
    
    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if config.model.load_model_state_dict else None
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")
    save_path = os.path.join(save_root_dir, f"{config.experiment.results_save_dir}/predictions.pkl")
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()