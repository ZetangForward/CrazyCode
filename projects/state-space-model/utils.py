import torch
import os
# import pytorch_lightning as pl
import lightning.pytorch as pl
import importlib
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, GPTNeoForCausalLM, LlamaForCausalLM
from transformers import MambaConfig, MambaForCausalLM
from modelzipper.tutils import *
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from custom_mamba.custom_mamba_analysis import LongContextMambaAna
import custom_mamba.custom_mamba_v2
import custom_mamba.custom_mamba_v3
from custom_mamba.custom_mamba_v2 import CustomMambaForCausalLM

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


def get_model_tokenizer(root_dir, model_config, use_custom_module=False, analysis=False):
    model_path = os.path.join(root_dir, model_config.model_name_or_path)
    tokenizer_path = os.path.join(root_dir, model_config.tokenizer_name_or_path)
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}'
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # load model
    if analysis: # analysis
        model = LongContextMambaAna.from_pretrained(
            "/nvme/hf_models/mamba-1.4b", use_relative_position=model_config.use_relative_position,
            dtype=torch.bfloat16, device="cuda"
        )
    

    elif use_custom_module:  # custom model just for mamba now
        # import pdb;pdb.set_trace()
        log_c("use_custom_module")
        config = MambaConfig.from_pretrained(model_path)
        if model_config.ckpt_path is not None and "multi" in model_config.ckpt_path.lower():
            log_c("use_v3")
            model = custom_mamba.custom_mamba_v3.CustomMambaForCausalLM(
                config, 
                use_relative_position=model_config.use_relative_position,
                max_position_embeddings=model_config.max_position_embeddings,
                use_abs_position=model_config.use_abs_position,
                custom_conv1d_configs=model_config.conv1d_configs,
            ).to(device)
        else:
            log_c("use_v2")
            model = custom_mamba.custom_mamba_v2.CustomMambaForCausalLM(
                config, 
                use_relative_position=model_config.use_relative_position,
                max_position_embeddings=model_config.max_position_embeddings,
                use_abs_position=model_config.use_abs_position,
                custom_conv1d_configs=model_config.conv1d_configs,
            ).to(device)
            
        if hasattr(model_config, "ckpt_path") and model_config.ckpt_path is not None:
            log_c("load_mamba_ckpt")
            model.custom_from_pretrained(
                model_config.ckpt_path, 
                dtype=torch.bfloat16,
                is_from_pytorch_lightning=True,
            )
        else:
            log_c("don't_load_mamba_ckpt")
    
    else:  # load hf model
        # import pdb;pdb.set_trace()
        if "gpt" in model_path.lower():
            model = GPTNeoForCausalLM.from_pretrained(
                model_path, use_cache=False, torch_dtype=torch.bfloat16
            ).to(device)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
 
        elif "mamba" in model_path.lower():
            raw_config = MambaConfig.from_pretrained(model_path)
            model = CustomMambaForCausalLM(raw_config).to(device)

            # model = CustomMambaForCausalLM.from_pretrained(
            #     model_path, torch_dtype=torch.bfloat16
            # ).to(device)

        elif "llama" or "deepseek" in model_path.lower():
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            ).to(device)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print_c("model and tokenzier already loaded ~", "red")

    return model, tokenizer


class EmptyDataset(Dataset):

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


class CustomDatamodule(pl.LightningDataModule):

    def __init__(self, cfg, root_dir, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.prepare_data_per_node = True
        # import pdb;pdb.set_trace()
        print(self.cfg.dataset)
        self.dataset_kwargs = {
            "max_seq_length": self.cfg.dataset.max_seq_length,
            "cluster_batch": self.cfg.dataset.cluster_batch,           
        }
        
        if "longbench" in self.cfg.dataset.module.lower() and \
              self.cfg.dataset.subtask is not None:
            self.dataset_kwargs.update({"subtask": self.cfg.dataset.subtask})
            self.dataset_kwargs.update({"config_path": os.path.join(self.root_dir, self.cfg.dataset.data_path)})
        
        if self.cfg.other_cfgs is not None:
            self.dataset_kwargs.update(self.cfg.other_cfgs)
    
    def load_data_with_root_dir(self, fpath, type='custom'):
        '''
        read data with root dir
        '''
        if not self.root_dir in fpath:
            fpath = os.path.join(self.root_dir, fpath)
        if type == 'hf':
            return load_from_disk(fpath)['train']
        return auto_read_data(fpath)

    def setup(self, stage: str = 'fit') -> None:
        train_data, valid_data, test_data = None, None, None
         # import Dataset Class
        dataset_module = importlib.import_module(self.cfg.dataset.module)
        CustomDataset = getattr(dataset_module, self.cfg.dataset.class_name)
        # prepare dataset
        if self.cfg.dataset.inference_mode:  # whether in inference mode
            if "needle" in self.cfg.dataset.data_path.lower():  # sanity check passkey search data
                processed_data_path = os.path.join(self.root_dir, self.cfg.dataset.processed_data_path)
                if self.cfg.dataset.processed_data_path is None or not os.path.exists(processed_data_path):  # preporcess the passkey_search data on-the-fly
                    processed_data = CustomDataset.build_dataset(
                        fpath=os.path.join(self.root_dir, self.cfg.dataset.data_path), 
                        key=self.cfg.dataset.key,
                        value=self.cfg.dataset.value,
                        ctx_len=self.cfg.dataset.max_seq_length,
                        tokenizer=self.tokenizer,
                    )
                    auto_save_data(processed_data, processed_data_path)  # auto save processed data fn
                    log_c("Processed data has been saved\nPlease re-start the program", color="yellow")
                    exit()
            
            if "ar" in self.cfg.dataset.module.lower():
                if self.cfg.dataset.processed_data_path is None:
                    processed_data = CustomDataset.build_dataset(
                        vocab_size=self.cfg.dataset.vocab_size, 
                        input_seq_len=self.cfg.dataset.input_seq_len,
                        num_kv_pairs=self.cfg.dataset.num_kv_pairs,
                        num_examples=self.cfg.dataset.num_examples,
                        power_a=self.cfg.dataset.test_power_a,
                        tokenizer=self.tokenizer,
                    )
                    # data_path = "/opt/data/private/zecheng/data/MQAR/" + "test_C8192_N"+str(self.cfg.dataset.input_seq_len) + "_D"+str(self.cfg.dataset.num_kv_pairs)+".pkl"
                    # auto_save_data(test_data,data_path)
                # import pdb;pdb.set_trace()
                # for input_seq_len in [1024, 2048]:
                #     for number_kv_pairs in [256]:
                #         test_data = CustomDataset.build_dataset(
                #             vocab_size=self.cfg.dataset.vocab_size, 
                #             input_seq_len=input_seq_len,
                #             num_kv_pairs=number_kv_pairs,
                #             num_examples=3000,
                #             power_a=self.cfg.dataset.test_power_a,
                #             tokenizer=self.tokenizer,
                #         )
                #         data_path = "/nvme/zecheng/data/MQAR/" + "test_C8192_N"+str(input_seq_len) + "_D"+str(number_kv_pairs)+".pkl"
                #         auto_save_data(test_data,data_path)


            if "longbench" in self.cfg.dataset.module.lower():
                data_path = self.cfg.dataset.data_path
                if self.cfg.dataset.subtask is not None:
                    data_path = data_path + self.cfg.dataset.subtask
                # if self.cfg.dataset.e:
                #     data_path = data_path + "_e.jsonl"
                # else:
                data_path = data_path + ".jsonl"
                test_data = self.load_data_with_root_dir(data_path)
            
            else:
                try:
                    test_data = self.load_data_with_root_dir(self.cfg.dataset.processed_data_path)
                except:
                    test_data = self.load_data_with_root_dir(self.cfg.dataset.test_data_path)
                
           
                    # auto_save_data(test_data,"/opt/data/private/zecheng/data/MQAR/MQAR.pkl")
                    # auto save processed data fn
                    # raise NotImplementedError
            
            # if self.cfg.dataset.processed_data_path is not None:
            #     test_data = self.load_data_with_root_dir(self.cfg.dataset.processed_data_path)
            # else:
            #     test_data = self.load_data_with_root_dir(self.cfg.dataset.test_data_path)
            # import pdb;pdb.set_trace()
            self.test_dataset = CustomDataset(
                content=test_data, 
                tokenizer=self.tokenizer, 
                split="test",
                **self.dataset_kwargs,
            )

        else:
            if self.cfg.dataset.processed_data_path is not None and self.cfg.dataset.processed_data_path != "":
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

                    # train_data = self.load_data_with_root_dir(processed_data_path)
            else:
                # check if is a directory
                data_path = os.path.join(self.root_dir, self.cfg.dataset.data_path)
                if hasattr(self.cfg.dataset, "type"):
                    # import pdb;pdb.set_trace()
                    if "hf" in self.cfg.dataset.type.lower() or "huggingface" in self.cfg.dataset.type.lower():  # huggingface dataset
                        train_data = self.load_data_with_root_dir(self.cfg.dataset.data_path, type='hf')
                    else:
                        try:
                            train_data = auto_read_data(data_path)
                        except:
                            raise NotImplementedError(f"{self.cfg.dataset.type} is not support")
                elif not os.path.isdir(data_path):  # custom dataset
                    train_data = auto_read_data(data_path)
                else:
                    raise NotImplementedError(f"split {self.cfg.dataset.data_path} is not supported")

        # further process data with stage
        if stage == "fit":  # training mode
            # check data & initialization
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
            else:
                self.valid_dataset = EmptyDataset()

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
        if isinstance(self.valid_dataset, EmptyDataset):
            return DataLoader(self.valid_dataset)
            
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.cfg.dataset.val_batch_size, 
            num_workers=self.cfg.dataset.nworkers, 
            pin_memory=self.cfg.dataset.pin_memory, 
            drop_last=False, 
            shuffle=False,
        )


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
