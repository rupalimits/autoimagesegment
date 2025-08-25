# Web links Handler
import requests
# Backend
import torch
# Image Processing
from PIL import Image
from IPython.display import display
# Transformer and Pretrained Model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
# Managing loading processsing
from tqdm import tqdm

# from image_captioning.utilities.utilities_common import tokenizer

# Assign available GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ViT Encoder - Decoder Model
# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
model = VisionEncoderDecoderModel.from_pretrained("../image_captioning/VIT_large_gpt2/checkpoint-1500").to(device)


# Corresponding ViT Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Image processor
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

import urllib.parse as parse
import os
# Verify url
def check_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

# Load an image
def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    
    
    
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)

    # Preprocessing the Image
    img = image_processor(image, return_tensors="pt").to(device)

    # Generating captions
    output = model.generate(**img)

    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(caption)
    return caption


def generate_caption(image):
    return get_caption(model, image_processor, tokenizer, image)
    
# Loading URLs
url = "https://images.pexels.com/photos/101667/pexels-photo-101667.jpeg?auto=compress&cs=tinysrgb&w=600"
urlNew = "https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg?auto=compress&cs=tinysrgb&w=600"
#
# # Display Image
# display(load_image(url))
#
# # Display Caption
# get_caption(model, image_processor, tokenizer, url)
# get_caption(model, image_processor, tokenizer, urlNew)
# generate_caption(url)
generate_caption(urlNew)
# generate_caption("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIUit3zC5d6XLTRgu6cSy0PQ2xQsQIacWIT5v3O5TmbxonNYgcvVmuTuNVo7W7b2zfH2I&usqp=CAU")