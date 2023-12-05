import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from processing.data_manager import transforms
from processing.features import ImgDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator

# from pipeline import image_captioning_pipe

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "dataset"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
CAPTIONS_DIR = DATASET_DIR / "captions.txt"
IMAGES_DIR = DATASET_DIR / "Images"

# # Assign available GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
#     return outputs

# AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.unk_token

def run_training(tokenizer, feature_extractor):
    """
    Traing the image reading model and generate captions.
    """
     
    df=  pd.read_csv(CAPTIONS_DIR)
    train_df , temp_df = train_test_split(df , test_size = 0.2, random_state = 42)

    val_df, test_df = train_test_split(temp_df, test_size = 0.5, random_state = 42)

    train_dataset = ImgDataset(train_df, root_dir = IMAGES_DIR,tokenizer=tokenizer,feature_extractor = feature_extractor ,transform = transforms)
    val_dataset = ImgDataset(val_df , root_dir = IMAGES_DIR,tokenizer=tokenizer,feature_extractor = feature_extractor , transform  = transforms)
    
    # print(train_dataset[0])
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_df))
    return train_dataset, val_dataset, test_df
    # data = load_dataset(file_name = config.app_config.training_data_file)

    #  # divide train and test
    # X_train, X_test, y_train, y_test = train_test_split(
        
    #     data[config.model_config.features],     # predictors
    #     data[config.model_config.target],       # target
    #     test_size = config.model_config.test_size,
    #     random_state=config.model_config.random_state,   # set the random seed here for reproducibility
    # )

    # # Pipeline fitting
    # image_captioning.fit(X_train, y_train)
    # #y_pred = bikeshare_pipe.predict(X_test)

    # # Calculate the score/error
    # #print("R2 score:", r2_score(y_test, y_pred).round(2))
    # #print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # # persist trained model
    # save_pipeline(pipeline_to_persist = image_captioning_pipe)
    
# if __name__ == "__main__":
#     run_training()