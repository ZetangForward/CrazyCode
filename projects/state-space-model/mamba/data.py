from modelzipper.datamanager import *
from modelzipper.tutils import *
from transformers import AutoTokenizer


class custom_datamodule(data_module):
    def __init__(self, file_path, tokenizer):
        super(custom_datamodule).__init__()
        self.tokenizer = tokenizer
        self.content = auto_read_data(file_path)
        
    @property
    def train_dataset(self) -> Dataset:
        pass
    
    @property
    def valid_dataset(self) -> Dataset:
        pass
    
    @property
    def test_dataset(self) -> Dataset:
        pass
    
    
if __name__ == "__main__":
    file_path = "/nvme/zecheng/data/roc_stories/ROCStories_winter2017.csv"
    tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/gpt-neo-1.3B")
    data_module = custom_datamodule(file_path, tokenizer)
    raw_data = data_module.content
    import pdb; pdb.set_trace()