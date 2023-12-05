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
# from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

# from processing.data_manager import transforms

import config

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yml"

DATASET_DIR = ROOT / "dataset"
TRAINED_MODEL_DIR = ROOT / "trained_models"
CAPTIONS_DIR = DATASET_DIR / "captions.txt"
IMAGES_DIR = DATASET_DIR / "Images"


class ImgDataset(Dataset):
    def __init__(self,df,root_dir,tokenizer,feature_extractor, transform = None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50
    def __len__(self,):
        return len(self.df)
    def __getitem__(self,idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir , image)
        img = Image.open(img_path).convert("RGB")
        
        img_array = np.array(img).astype(np.float32)

        # Check if the image is already in the desired range
        if img_array.min() < 0 or img_array.max() > 1:
            # Rescale from [-1, 1] to [0, 1]
            img_array = (img_array + 1) / 2
            img_array = np.clip(img_array, 0, 1)  # Ensure values are within [0, 1]

        # Rescale from [-1, 1] to [0, 1]
        # rescaled_array = (img_array + 1) / 2
        # rescaled_array = np.clip(rescaled_array, 0, 1)  # Ensure values are within [0, 1]
        rescaled_pil_img = Image.fromarray((img_array * 255).astype(np.uint8), "RGB")

        if self.transform is not None:
            rescaled_pil_img= self.transform(rescaled_pil_img)
        
        # print("hulalalalalalalalalalaallalalalalalalal : ", type(rescaled_pil_img))

        pixel_values = self.feature_extractor(rescaled_pil_img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,
                                 padding='max_length',
                                 max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding



# class Mapper(BaseEstimator, TransformerMixin):
#     """
#     Ordinal categorical variable mapper:
#     Treat column as Ordinal categorical variable, and assign values accordingly
#     """

#     def __init__(self, variable:str, mappings:dict):

#         if not isinstance(variable, str):
#             raise ValueError("variable name should be a string")

#         self.variable = variable
#         self.mappings = mappings

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         X[self.variable] = X[self.variable].map(self.mappings).astype(int)

#         return X