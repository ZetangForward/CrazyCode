from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import random
import math
import torch
import re

GLOBAL_TEMPLATE = {
    "CM3-keywords": {
        "instance": "{left} {mask} {right}", 
        "full_seq": "Please generate SVG paths according to the keywords: {instruction} ##Here is the template: {in_seq} ##Begin to generate: {out_seq}", 
    },
    "CM3-description": {
        "instance": "{left} {mask} {right}", 
        "full_seq": "{instruction} ##Here is the template: {in_seq} ##Begin to generate: {out_seq}", 
    }
}

class OfflineDataset(Dataset):  
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        with open(file, "r") as f:
            self.content = [json.loads(line) for line in f]
  
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        # description, svg_path = self.content[index]["caption"], self.content[index]["str_path"]
        # svg_file, keywords = self.content[index]["svg_file"], self.content[index]["keywords"]
        # if isinstance(keywords, list):
        #     key_words = ", ".join(keywords)
        # else:
        #     key_words = keywords
        seq_modeling = self.content[index]["compress_path"]
        seq_inputs = self.tokenizer(
            seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        seq_input_ids = seq_inputs.input_ids[0]
        seq_attention_mask = seq_inputs.attention_mask[0]
        seq_labels = torch.where(
            seq_input_ids != self.tokenizer.pad_token_id, seq_input_ids, -100
        )
        return {
            "input_ids": seq_input_ids,
            "attention_mask": seq_attention_mask,
            "labels": seq_labels,
        }
        
        
class IconshopDataset(Dataset):  
    def __init__(self, args, file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        self.template = "Keywords: {keywords} #Begin: {svg_path}"
        with open(file, "r") as f:
            self.content = [json.loads(line) for line in f]
  
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        # description, svg_path = self.content[index]["caption"], self.content[index]["str_path"]
        # svg_file, keywords = self.content[index]["svg_file"], self.content[index]["keywords"]
        # if isinstance(keywords, list):
        #     key_words = ", ".join(keywords)
        # else:
        #     key_words = keywords
        seq_modeling = self.content[index]["compress_path"]
        svg_path = seq_modeling.split("#Begin to generate:")[-1].strip()
        keywords = ", ".join(self.content[index]["keywords"])
        input_seq = self.template.format(keywords=keywords, svg_path=svg_path)
        
        
        seq_inputs = self.tokenizer(
            input_seq, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        seq_input_ids = seq_inputs.input_ids[0]
        seq_attention_mask = seq_inputs.attention_mask[0]
        seq_labels = torch.where(
            seq_input_ids != self.tokenizer.pad_token_id, seq_input_ids, -100
        )
        return {
            "input_ids": seq_input_ids,
            "attention_mask": seq_attention_mask,
            "labels": seq_labels,
        }

        
class OnlineDataset(Dataset):  
    def __init__(self, args, svg_file, tokenizer: PreTrainedTokenizer):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        self.mask_ratio = args.mask_ratio
        self.n_mask = 4
        with open(svg_file, "r") as f2:
            self.content = [json.loads(line) for line in f2]
        self.hybrid = args.hybrid if args.hybrid is not None else False
        if self.hybrid == "keywords":
            self.tmplate_format = ["CM3-keywords"]
        elif self.hybrid == "description":  
            self.tmplate_format = ["CM3-description"]
        elif self.hybrid == "hybrid":  
            self.tmplate_format = ["CM3-keywords", "CM3-description"]
        else:
            raise ValueError("You must define --hybird from [keywords, description, hybrid]")
        
    def __len__(self):
        return len(self.content)
    
    def extract_consecutive_numbers(self, path_data, percentage):  
        # 匹配数字（包括负号和小数点）  
        number_pattern = r"-?\d+\.?\d*"  
        numbers = re.findall(number_pattern, path_data)  
    
        # 计算要提取的数字数量  
        num_to_extract = int(len(numbers) * percentage)  
    
        # 计算连续提取的起始索引  
        start_index = len(numbers) - num_to_extract  
    
        # 提取指定范围的数字  
        extracted_numbers = numbers[start_index:]  
    
        # 将提取的数字连接成字符串  
        extracted_numbers_str = " ".join(extracted_numbers)  
        return extracted_numbers_str 
    
    
    def extract_c_segments(self, path_data):  
        ## find all the control path in svg paths
        c_pattern = r"c[^A-Za-z]*?(?=[A-Za-z])"  
        c_segments = re.findall(c_pattern, path_data)  
        if len(c_segments) == 1:  # only one control path, usually complex
            return [self.extract_consecutive_numbers(c_segments[0], 0.5)]
        return c_segments 
    
    
    def build_cm3_hybrid_online(self, item, keywords_template=None, des_template=None, mask_portion=0.5, n_mask=1):
        """
        item = {"str_path": svg path, "caption": svg caption, "svg_file": data path}
        """
        description, svg_path = item["caption"], item["str_path"]
        svg_file, keywords = item["svg_file"], item["keywords"]
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        # if IsHybrid:
        #     des_portion, keywords_portion = 0.4, 0.6
        
        # extract all <path> tags  
        path_tags = re.findall(r'<path.*?\/>', svg_path)  
        
        ### step1. determine how many paths to mask (1~s)
        if n_mask == 1:
            selected_paths = random.sample(path_tags, 1)  
        else: 
            max_mask_num = min(n_mask, len(path_tags))
            select_idxs = random.sample([i for i in range(len(path_tags))], max_mask_num)
            selected_paths = [path_tags[i] for i in select_idxs] 
        
        ### step2. determine how much portion to mask (currently full mask)
        origin_path = svg_path # init mask_data
        concated_paths = []
        cnt_masks = 0  # count mask numbers
        for mask_part in selected_paths:
            if mask_portion != 0:
                all_control_paths = self.extract_c_segments(mask_part)    
                mask_parts = random.sample(all_control_paths, math.ceil(len(all_control_paths)*mask_portion))
                concated_paths.extend(mask_parts)
                # import pdb; pdb.set_trace()
                for mask_tmp in mask_parts:
                    cnt_masks += 1
                    origin_path = origin_path.replace(mask_tmp, f'<MASK_{cnt_masks}> ') 
            else: # mask a full path
                origin_path = origin_path.replace(mask_part, '<MASK> ') 
        recovered_data = ""
        for recover_span_idx, recover_span in enumerate(concated_paths):
            recovered_data = recovered_data + f"<MASK_{recover_span_idx+1}> " + recover_span
        if torch.random() <= 0.6:  # 60% for keywords 
            concate_path = keywords_template.format(instruction=keywords, in_seq=origin_path, out_seq=recovered_data)
        else:
            concate_path = des_template.format(instruction=description, in_seq=origin_path, out_seq=recovered_data)
            
        return {"source": concate_path, "svg_file": svg_file, "target": concate_path, "raw_path": svg_path}
    
    def build_cm3_keywords_online(self, item, template=None, mask_portion=0.5, n_mask=1, keywords=None):
        """
        item = {"str_path": svg path, "caption": svg caption, "svg_file": data path}
        """
        caption, svg_path = item["caption"], item["str_path"]
        svg_file, keywords = item["svg_file"], item["keywords"]
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        # key_words = svg_file.split("/")[-1].split("-")[0].replace("_", " ")    
        # extract all <path> tags  
        path_tags = re.findall(r'<path.*?\/>', svg_path)  
        
        ### step1. determine how many paths to mask (1~s)
        if n_mask == 1:
            selected_paths = random.sample(path_tags, 1)  
        else: 
            max_mask_num = min(n_mask, len(path_tags))
            select_idxs = random.sample([i for i in range(len(path_tags))], max_mask_num)
            selected_paths = [path_tags[i] for i in select_idxs] 
        
        ### step2. determine how much portion to mask (currently full mask)
        origin_path = svg_path # init mask_data
        concated_paths = []
        cnt_masks = 0  # count mask numbers
        for mask_part in selected_paths:
            if mask_portion != 0:
                all_control_paths = self.extract_c_segments(mask_part)    
                mask_parts = random.sample(all_control_paths, math.ceil(len(all_control_paths)*mask_portion))
                concated_paths.extend(mask_parts)
                # import pdb; pdb.set_trace()
                for mask_tmp in mask_parts:
                    cnt_masks += 1
                    origin_path = origin_path.replace(mask_tmp, f'<MASK_{cnt_masks}> ') 
            else: # mask a full path
                origin_path = origin_path.replace(mask_part, '<MASK> ') 
        recovered_data = ""
        for recover_span_idx, recover_span in enumerate(concated_paths):
            recovered_data = recovered_data + f"<MASK_{recover_span_idx+1}> " + recover_span
        concate_path = template.format(instruction=keywords, in_seq=origin_path, out_seq=recovered_data)
        return {"source": concate_path, "svg_file": svg_file, "target": concate_path, "raw_path": svg_path}

    
    def __getitem__(self, index):
        data = self.content[index]
        mask_portion = random.uniform(self.mask_ratio, 1)
        
        if self.hybrid == "hybrid":
            item = self.build_cm3_hybrid_online(
                data, 
                keywords_template=GLOBAL_TEMPLATE[self.tmplate_format[0]]["full_seq"],
                des_template=GLOBAL_TEMPLATE[self.tmplate_format[1]]["full_seq"],
                mask_portion=mask_portion, 
                n_mask=self.n_mask
            )
        else:
            item = self.build_cm3_keywords_online(
                data, 
                GLOBAL_TEMPLATE[self.tmplate_format[0]]["full_seq"], 
                mask_portion=mask_portion, 
                n_mask=self.n_mask
            )
        seq_modeling = item["source"]
        
        seq_inputs = self.tokenizer(
            seq_modeling, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        seq_input_ids = seq_inputs.input_ids[0]
        seq_attention_mask = seq_inputs.attention_mask[0]
        seq_labels = torch.where(
            seq_input_ids != self.tokenizer.pad_token_id, seq_input_ids, -100
        )
        
        return {
            "input_ids": seq_input_ids,
            "attention_mask": seq_attention_mask,
            "labels": seq_labels,
        }
        

# class TmpClass:
#     model_max_length = 3500
#     mask_ratio = 0.5

# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     args = TmpClass()
#     tokenizer = AutoTokenizer.from_pretrained("/zecheng/model_hub/Llama-2-7b-hf")
#     tokenizer.pad_token_id = tokenizer.unk_token_id
#     IGNORE_INDEX = -100
#     DEFAULT_PAD_TOKEN = "[PAD]"
#     DEFAULT_EOS_TOKEN = "</s>"
#     DEFAULT_BOS_TOKEN = "</s>"
#     DEFAULT_UNK_TOKEN = "</s>"
#     tokenizer.add_special_tokens(
#             {
#                 "eos_token": DEFAULT_EOS_TOKEN,
#                 "bos_token": DEFAULT_BOS_TOKEN,
#                 "unk_token": DEFAULT_UNK_TOKEN,
#             }
#         )
#     svg_file = "/zecheng/svg/filtered_svg/simplified_svgs/valid_online.jsonl"
#     dataset = OnlineDataset(args, svg_file, tokenizer)
    
#     print(dataset[0])