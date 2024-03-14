import torch
import os
# import pytorch_lightning as pl
import lightning.pytorch as pl
import importlib
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from custom_mamba.custom_mamba_v2 import CustomMambaForCausalLM
from transformers import MambaForCausalLM, AutoTokenizer, GPTNeoForCausalLM, LlamaForCausalLM, LlamaTokenizer
from transformers import MambaConfig
from modelzipper.tutils import *
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

def get_model_tokenizer_simple(root_dir, tokenizer_name_or_path=None, model_name_or_path=None):
    tokenizer, model = None, None
    if tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_dir, tokenizer_name_or_path))
    if model_name_or_path is not None:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(root_dir, model_name_or_path))

    return tokenizer, model


def load_big_kernel_mamba(model_path, use_relative_position=False):  # TODO: add more args
    raw_config = MambaConfig.from_pretrained(model_path)
    raw_config.expand = 4
    raw_config.use_relative_position = use_relative_position
    raw_config.use_abs_position = False
    raw_config.max_position_embeddings = 9012
    model = CustomMambaForCausalLM(raw_config, dtype=torch.bfloat16, device="cuda")
    state_dict = torch.load("/nvme/hf_models/mamba-1.4b/pytorch_model.bin", map_location="cuda")
    import pdb; pdb.set_trace()
    model._load_from_state_dict(state_dict, dtype=torch.bfloat16)

    return model


def get_low_rank_model_tokenizer(root_dir, model_config, use_custom_module=False):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)

    lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    # elif "mamba" in model_path.lower():
    model = CustomMambaForCausalLM.from_pretrained(
        model_path, use_relative_position=model_config.use_relative_position,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 假设 `model` 是你的模型实例
    for param in model.parameters():
        param.requires_grad = False

    peft_model = get_peft_model(model, lora_config, mixed=True)

    peft_model.print_trainable_parameters()

    return peft_model, tokenizer

def get_model_tokenizer(root_dir, model_config, use_custom_module=False):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)

    if use_custom_module:  # custom model
        model = load_big_kernel_mamba(
            model_path, use_relative_position=model_config.use_relative_position,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return model, tokenizer
    
    if "gpt" in model_path.lower():
        model = GPTNeoForCausalLM.from_pretrained(
            model_path, use_cache=False, torch_dtype=torch.bfloat16
        ).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif "mamba" in model_path.lower():
        model = CustomMambaForCausalLM.from_pretrained(
            model_path, use_relative_position=model_config.use_relative_position,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif "llama" or "deepseek" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        ).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    if "gpt" in tokenizer_path or "llama" in tokenizer_path:
        # tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
    
    print_c("model and tokenzier already loaded ~")

    return model, tokenizer


class CustomDatamodule(pl.LightningDataModule):

    def __init__(self, cfg, root_dir, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        self.dataset_kwargs = {
            "max_seq_length": self.cfg.dataset.max_seq_length,
            "cluster_batch": self.cfg.dataset.cluster_batch,            
        }
        
        if self.cfg.other_cfgs is not None:
            self.dataset_kwargs.update(self.cfg.other_cfgs)
    
    def load_data_with_root_dir(self, fpath):
        '''
        read data with root dir
        '''
        if not self.root_dir in fpath:
            fpath = os.path.join(self.root_dir, fpath)
        if 'hf' in fpath:
            return load_from_disk(fpath)
        return auto_read_data(fpath)

    def setup(self, stage: str = 'fit') -> None:
        train_data, valid_data, test_data = None, None, None
         # import Dataset Class
        dataset_module = importlib.import_module(self.cfg.dataset.module)
        CustomDataset = getattr(dataset_module, self.cfg.dataset.class_name)
        
        # prepare dataset
        if self.cfg.dataset.inference_mode:  # whether in inference mode
            if "needle" in self.cfg.dataset.data_path.lower():  # sanity check passkey search data
                if self.cfg.dataset.processed_data_path is None:  # preporcess the passkey_search data on-the-fly
                    processed_data = CustomDataset.build_dataset(
                        fpath=os.path.join(self.root_dir, self.cfg.dataset.data_path), 
                        key=self.cfg.dataset.key,
                        value=self.cfg.dataset.value,
                        ctx_len=self.cfg.dataset.max_seq_length,
                        tokenizer=self.tokenizer,
                    )
                    auto_save_data(...)  # auto save processed data fn
                    raise NotImplementedError
            
            if self.cfg.dataset.processed_data_path is not None:
                test_data = self.load_data_with_root_dir(self.cfg.dataset.processed_data_path)
            else:
                test_data = self.load_data_with_root_dir(self.cfg.dataset.test_data_path)
            
            self.test_dataset = CustomDataset(
                content=test_data, 
                tokenizer=self.tokenizer, 
                split="test",
                **self.dataset_kwargs,
            )
        else:
            if self.cfg.dataset.processed_data_path is not None:
                # check if is a directory
                processed_data_path = os.path.join(self.root_dir, self.cfg.dataset.processed_data_path)
                
                if os.path.isdir(processed_data_path):
                    for split in self.cfg.dataset.split:  # TODO: support multiple splits
                        if "train" in split:
                            train_data = self.load_data_with_root_dir(os.path.join(processed_data_path, split))
                        elif "valid" in split:
                            valid_data = self.load_data_with_root_dir(os.path.join(processed_data_path, split))
                        else:
                            raise NotImplementedError(f"split {split} is not supported")
                else:
                    content = self.load_data_with_root_dir(processed_data_path)
                    min_valid_num = min(1000, len(content)*0.1)
                    valid_data = content[:min_valid_num]
                    train_data = content[min_valid_num:]
            else:
                # check if is a directory
                data_path = os.path.join(self.root_dir, self.cfg.dataset.data_path)
                if not os.path.isdir(data_path):
                    train_data = auto_read_data(data_path)

                elif "hf" in self.cfg.dataset.data_path.lower():
                    if "pajama" in self.cfg.dataset.data_path.lower():
                        all_data = self.load_data_with_root_dir(self.cfg.dataset.data_path)
                        import pdb; pdb.set_trace()  
                        ...
                else:
                    raise NotImplementedError(f"split {self.cfg.dataset.data_path} is not supported")

        # further process data with stage
        if stage == "fit":  # training mode
            # check data initialization  
            assert train_data is not None, f"train data should not be None during {stage} stage"
            try:
                assert valid_data is not None, f"valid data is None during {stage} stage"
            except:
                pass
            
            # init dataset
            self.train_dataset = CustomDataset(
                content=train_data, 
                tokenizer=self.tokenizer, 
                split="train",
                **self.dataset_kwargs,
            )
            print_c(f"num of train samples: {len(self.train_dataset)}", color='magenta')
            
            if valid_data is not None: 
                self.valid_dataset = CustomDataset(
                    content=valid_data, 
                    tokenizer=self.tokenizer, 
                    split="valid",
                    **self.dataset_kwargs,
                )
                print_c(f"num of valid samples: {len(self.valid_dataset)}", color='magenta')
        
        else: # prediction mode
            assert test_data is not None, f"test data should not be None during {stage} stage"

            # init dataset
            self.test_dataset = CustomDataset(
                content=test_data, 
                tokenizer=self.tokenizer, 
                split="test",
                **self.dataset_kwargs,
            )
            
            print_c(f"num of testing samples: {len(self.test_dataset)}", color='magenta')
            

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.dataset.train_batch_size, 
            num_workers=self.cfg.dataset.nworkers, 
            pin_memory=self.cfg.dataset.pin_memory, 
            drop_last=True, 
            shuffle=False if self.cfg.dataset.cluster_batch else True, 
        )
        

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset, 
                batch_size=self.cfg.dataset.val_batch_size, 
                num_workers=self.cfg.dataset.nworkers, 
                pin_memory=self.cfg.dataset.pin_memory, 
                drop_last=False, 
                shuffle=False,
            )
        return None
    

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_dataset is not None, "test dataset should not be None"
        predict_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=self.cfg.dataset.nworkers, 
            pin_memory=self.cfg.dataset.pin_memory, 
            drop_last=False, 
            shuffle=False,
        )
        return predict_loader
