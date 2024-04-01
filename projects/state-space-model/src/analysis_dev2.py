import os  
import sys
sys.path.append(os.getcwd())
import torch   
import hydra  
import lightning.pytorch as pl
from modelzipper.tutils import *
from utils import get_model_tokenizer, CustomDatamodule
from evaluate.evaluator import Evaluator
from dev_configs.config import parse_args, get_final_configs

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

        depth = batch.get('depth').item()
        ctx_length = batch.get('before_insert_context_length').item()
        bos_pos, eos_pos = batch.get('bos_pos'), batch.get('eos_pos')

        
        if ctx_length % 1000 != 0:
            pass

        extra_kwargs = {
            "ctx_length": ctx_length,
            "depth": depth,
            "save_dir": "/nvme/zecheng/modelzipper/projects/state-space-model/analysis/inner_state2",
            "bos_pos": bos_pos, 
            "eos_pos": eos_pos,
        }
        
        output = self.model.generate(
            input_ids, 
            max_length=input_ids.size(-1) + self.cfg.task.other_cfgs.max_generation_length,
            min_length=input_ids.size(-1) + 10, 
            eos_token_id=self.tokenizer.eos_token_id, 
            extra_kwargs=extra_kwargs,
        )
        
        batch['predictions'] = output.squeeze(0)[input_ids.size(1):]
        batch['depth'] = depth
        batch['ctx_length'] = ctx_length
        batch['bos_pos'] = bos_pos
        batch['eos_pos'] = eos_pos
        
        return batch

       

def main(config):

    model_root_dir = config.platform.hf_model_path
    save_root_dir = config.platform.result_path
    data_root_dir = config.platform.dataset_path

    # if use_custom_module 
    use_custom_module = False
    if hasattr(config.model, "use_custom_module"):
        use_custom_module = config.model.use_custom_module

    model, tokenizer = get_model_tokenizer(
        model_root_dir, config.model, use_custom_module=use_custom_module,
    )

    # load data module
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)

    # init tester
    tester = pl.Trainer(devices=config.experiment.device_num)

    # predict the results
    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
    )

    import pdb; pdb.set_trace()

    
if __name__ == '__main__':

    args = parse_args()
    config = get_final_configs(args)
    print_c(config, 'yellow')

    main(config)
