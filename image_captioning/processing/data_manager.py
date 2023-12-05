import sys
import os
import datasets
import numpy as np
from PIL import Image
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from config.core import config


transforms = transforms.Compose(
    [
        transforms.Resize(224), 
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=0.5, 
        #     std=0.5
        # )
   ]
)
      

# def load_dataset(*, file_name: str) -> None:
#     # preprocess the image

#     # extract the image features

#     # load the captions

#     # tokenize the captions 

#     # clean the text

#     # Tokenizing captions and creating word-to-index mapping

#     # divide the data into train, test and validation

# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(420, 420))
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     image = image_processor(img, return_tensors="pt").to(device)
#     return image

# def extract_image_features(model, image_path):
#     img = preprocess_image(image_path)
#     features = model.predict(img, verbose=0)
#     return features