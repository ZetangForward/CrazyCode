import os  
import sys
sys.path.append(os.getcwd())
import torch   
import pytorch_lightning as pl
import hydra  
from custom_dataset.data import custom_datamodule
from modelzipper.tutils import *
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def remove_padding(x, padding_mask):
    """
    x: (batch_size x) seq_len num_bins
    padding: seq_len x num_bins

    if batch, return List[Tensor]
    if tensor, return Tensor
    """
    if x.ndim == 2:  # seq_len x num_bins
        return x[:padding_mask.sum(), :]
    elif x.ndim == 3:  # batch_size x seq_len x num_bins
        res = []
        for i in range(x.size(0)):
            res.append(x[i, :padding_mask[i].sum(), :])
        return res
    
    
def sanint_check_golden(x):
    """
    x: batch_size x seq_len x (7, 9)
    """
    # replace the command value to 0, 1, 2
    x[:, :, 0][x[:, :, 0] == 100] = 1
    x[:, :, 0][x[:, :, 0] == 200] = 2
    if x.size(-1) == 9:
        return x
    elif x.size(-1) == 7:
        # add two columns
        x_0_y_0 = torch.zeros((x.size(0), x.size(1), 2), dtype=x.dtype, device=x.device)
        x_0_y_0[:, 1:, 0] = x[:, :-1, -2]  # x_3 of the previous row
        x_0_y_0[:, 1:, 1] = x[:, :-1, -1]  # y_3 of the previous row
        full_x = torch.cat((x[:, :, :1], x_0_y_0, x[:, :, 1:]), dim=2)
    return full_x


def merge_dicts(dict_list, device='cpu'):
    merge_res = {k: [] for k in dict_list[0].keys()} 
    for key in merge_res.keys():
        print_c(f"begin to merge {key}", "magenta")
        items = [d[key] for d in dict_list if key in d]
        # process items
        if items and isinstance(items[0], torch.Tensor):
            tmp_tensors = []
            flag = False
            for sublist in items:
                if isinstance(sublist, torch.Tensor):
                    tmp_tensors.append(sublist)
                elif isinstance(sublist, list):
                    tmp_tensors.extend(sublist)
            tmp_tensors = [tmp.cpu() for tmp in tmp_tensors]
            merge_res[key] = tmp_tensors  # each row is a tensor

        elif items and isinstance(items[0], List):
            tmp_lists = []
            for sublist in items:
                tmp_lists.extend(sublist)
            tmp_lists = [tmp.to(device) for tmp in tmp_lists]
            merge_res[key] = tmp_lists  # each row is a tensor
        else:
            raise ValueError(f'Unsupported data type for merge: {type(items[0])}')
    return merge_res


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
        output = self.model.generate(batch['input_ids'], max_length=64, temperature=0.9, top_p=0.7, eos_token_id=self.tokenizer.eos_token_id)
        
        label = batch['labels'][0]
        
        standard_test_reconstruct = {
            "prediction": self.tokenizer.decode(output[0]),
            "golden": self.tokenizer.decode(label),
        }
        
        return standard_test_reconstruct
    

@hydra.main(config_path='../configs', config_name='test_mamba', version_base='1.1')
def main(config):
    
    print_c(f"Experiment: {config.experiment.task}", "magenta")
    
    # load model and tokenizer
    model = MambaLMHeadModel.from_pretrained(config.model.model_name_or_path, dtype=torch.bfloat16, device="cuda")
    state_dict = torch.load(config.model.ckpt_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=True)


    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_name_or_path)
    if "gpt-neo" in config.tokenizer.tokenizer_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        
    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    
    # load data
    data_module = custom_datamodule(config.dataset, tokenizer)

    tester = pl.Trainer(devices=config.experiment.device_num)

    b_t = time.time()

    predictions = tester.predict(
        experiment, 
        datamodule=data_module,
        return_predictions=True,
        # ckpt_path=config.model.ckpt_path  # second pass for safety
    )
    
    print_c(f"======= prediction end, begin to post process and save =======", "magenta")

    save_path = f"{config.experiment.results_save_dir}/predictions.jsonl"
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()