from modelzipper.tutils import *
import re

class TaskConfig:
    def __init__(
            self, data_name, data_path=None, processed_data_path=None,
            module=None, class_name=None, nworkers=0, max_seq_length=4096, 
            train_batch_size=1,val_batch_size=1, inference_mode=False, 
            pin_memory=False, cluster_batch=False, **other_cfgs
        ) -> None:
        """ 
        default config can only contain:
            - data_path
            - processed_data_path
            - module
            - class_name
            - nworkers
            - max_seq_length
            - train_batch_size
            - val_batch_size
            - inference_mode
            - pin_memory
            - cluster_batch
        
        other_cfgs should be a dict, which can contains specific configs for the task
        """
        self.data_name = data_name
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.module = module
        self.class_name = class_name
        self.nworkers = nworkers
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.inference_mode = inference_mode
        self.pin_memory = pin_memory
        self.cluster_batch = cluster_batch
        self.other_cfgs = other_cfgs
        
        self.cfg = self.return_config(
            data_name, processed_data_path,
            train_batch_size, val_batch_size,
            inference_mode,
        )

    def return_config(
        self, data_name, processed_data_path, 
        train_batch_size, val_batch_size, inference_mode,
    ):
        if "mqar" in data_name.lower():
            pattern = r"N(\d+)_D(\d+)"
            match = re.search(pattern, processed_data_path)
            input_seq_len = int(match.group(1))  # N
            num_kv_pairs = int(match.group(2))   # D

            return TaskConfig.mqar_config(
                inference_mode = inference_mode,
                train_batch_size = train_batch_size if not inference_mode else val_batch_size,
                processed_data_path = processed_data_path,
                num_examples = 3000 if inference_mode else 100000,
                input_seq_len = input_seq_len,
                num_kv_pairs = num_kv_pairs,
                test_power_a = 0.01,
            )
        
        elif "longbench" in data_name.lower():
            pass
            
        elif "passkey" in data_name.lower():
            return TaskConfig.passkey_config()

        elif "longalpaca" in data_name.lower():
            return TaskConfig.longalpaca_config()
        
        elif "slimpajama" in data_name.lower():
            return TaskConfig.slimpajama_config()
    

    @classmethod
    def mqar_config(
        cls, processed_data_path, inference_mode, 
        num_examples, input_seq_len, num_kv_pairs, test_power_a,
        train_batch_size
    ):
        mqar_confg = {
            'dataset': {
                "data_name": "MQAR",
                "data_path": None, 
                "processed_data_path": processed_data_path,
                "module": 'custom_dataset.AR_ywj', 
                "class_name": 'MQARDataset',
                "nworkers": 4,
                "max_seq_length": 5000,
                "train_batch_size": train_batch_size,
                "val_batch_size": 1,
                "inference_mode": inference_mode,
                "pin_memory": False,
                "cluster_batch": False,
                "vocab_size": 8192,
                "num_examples": num_examples,
                "input_seq_len": input_seq_len,
                "num_kv_pairs": num_kv_pairs,
                "test_power_a": test_power_a
            },
            "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": 128000,
            },
        }
        return mqar_confg


    @classmethod
    def passkey_config(cls):
        passkey_config = {
            "dataset": {
                "data_name": "PasskeySearch",
                "data_path": "needle/PaulGrahamEssays/*.txt",
                "processed_data_path": "needle/processed_data/128k_500_insert_ids.pkl",
                "module": 'custom_dataset.passkey_search',
                "class_name": 'PasskeySearchDataset',
                "nworkers": 4,
                "max_seq_length": 128000,
                "val_batch_size": 1,
                "inference_mode": True,
                "pin_memory": False,
                "cluster_batch": False,
                "depth": 0.5,
                "key": "The best thing to do in San Francisco is",
                "value": "eat a sandwich and sit in Dolores Park on a sunny day.",
            },
            "other_cfgs": {
                "max_generation_length": 48,
                "testing_max_ctx": 128000,
            },
            "inference_cfg": {
                "save_keys": ['depth', 'ctx_length', 'real_length']
            }
        }
        return passkey_config
    
    
    @classmethod
    def longalpaca_config(cls):
        longalpaca_config = {
            "dataset": {
                "data_path": "LongAlpaca-12k/LongAlpaca-12k.json",
                "processed_data_path": None,
                "max_seq_length": 3000,
                "module": 'custom_dataset.longlora',
                "class_name": 'LongLoRA',  
                "nworkers": 4,
                "train_batch_size": 1,
                "val_batch_size": 1,
                "pin_memory": False,
                "inference_mode": False,
                "cluster_batch": True  
            },
            "other_cfgs": None,
        }
        return longalpaca_config
    
    
    @classmethod
    def slimpajama_config(cls):
        slimpajama_config = {
            "dataset": {
                "data_path": "slimpajama-per-source-length-upsample-gpt-hf",
                "module": 'custom_dataset.simplepajama',
                "processed_data_path": None,
                "class_name": 'SimplepajamaDataset',
                "max_seq_length": 4200,
                "nworkers": 2,
                "type": "hf",
                "train_batch_size": 1,
                "val_batch_size": 1,
                "pin_memory": False,
                "inference_mode": False,
                "cluster_batch": True,
                "require_process": False,
            },
            "other_cfgs": None,
        }
        return slimpajama_config
    
    
            




    

