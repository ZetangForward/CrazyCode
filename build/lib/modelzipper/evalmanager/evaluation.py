"""
Version 1.0.0, Evaluate the visual results
"""

import torch
import clip
from PIL import Image
from typing import List, Dict
import requests
import matplotlib.pyplot as plt
import transformers
from transformers import CLIPProcessor, CLIPModel

def cal_text2image_similarity(text: List[str], image: Image, processor: CLIPProcessor, model: CLIPModel):
    image = Image.open(image)
    text_inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    return logits_per_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")




