import os  
import sys
sys.path.append(os.getcwd())
import torch   
import pytorch_lightning as pl
import hydra  
from custom_dataset import *
from custom_dataset.zero_scroll import *
from modelzipper.tutils import *
from utils import get_model_tokenizer, CustomDatamodule
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analysis_cov1d_kernel(module):
    weights = module.weight.data.cpu().numpy()
    for i, weight in enumerate(weights):
        plt.plot(weight[0], label=f'Conv Kernel {i}')
    plt.title('Convolution Kernels Weights')
    plt.xlabel('Kernel Size')
    plt.legend()
    plt.show()

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
        input_ids = batch.get("input_ids")
        output = self.model.generate(input_ids, max_length=self.cfg.task.other_cfgs.max_generation_length, eos_token_id=self.tokenizer.eos_token_id)
        batch['predictions'] = output
        return batch


@hydra.main(config_path='../configs', config_name='mamba_test', version_base='1.1')
def main(config):
    
    print_c(OmegaConf.to_yaml(config), "yellow")
    
    model_root_dir = config.platform.hf_model_path
    data_root_dir = config.platform.dataset_path

    # load model and tokenizer
    model, tokenizer = get_model_tokenizer(model_root_dir, config.model)
    
    # load testing data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    if config.model.load_model_state_dict:
        state_dict = torch.load(
            os.path.join(config.platform.hf_model_path, config.model.ckpt_path), 
            map_location='cuda'
        )
        model.load_state_dict(state_dict, strict=True)

    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    tester = pl.Trainer(devices=config.experiment.device_num)
    
    #########################
    ## Analysis function Lens
    #########################
    # register hook to get the output of the last layer
    conv_outputs = []

    def conv_hook_fn(module, input, output):
        conv_outputs.append(output.clone().detach())

    # register hook to get the output of the last layer
    hook = model.backbone.layers[-1].mixer.conv1d.register_forward_hook(conv_hook_fn)

    b_t = time.time()
    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if not config.model.load_model_state_dict else None
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")
    import pdb; pdb.set_trace()
    save_path = os.path.join(config.platform.result_path, f"{config.experiment.results_save_dir}/predictions.pkl")
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()