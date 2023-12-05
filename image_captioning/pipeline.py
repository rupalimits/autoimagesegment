import datasets
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator
from sklearn.pipeline import Pipeline
from processing.features import ImgDataset
from config.core import config
from train_pipeline import run_training

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "dataset"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
CAPTIONS_DIR = DATASET_DIR / "captions.txt"
IMAGES_DIR = DATASET_DIR / "Images"

print("encoder : ", config.model_config.ENCODER)

# Assign available GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

# build the metrics
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


feature_extractor = ViTFeatureExtractor.from_pretrained(config.model_config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.model_config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

# build the datasets
train_dataset, val_dataset, test_df = run_training(tokenizer, feature_extractor)
# train_dataset = (train_dataset - np.min(train_dataset)) / (np.max(train_dataset) - np.min(train_dataset))
# train_dataset = np.clip(train_dataset, 0, 1)
# val_dataset = (val_dataset - np.min(val_dataset)) / (np.max(val_dataset) - np.min(val_dataset))
# val_dataset = np.clip(val_dataset, 0, 1)
# print(train_dataset[0])
# print(len(train_dataset))
# print(len(val_dataset))
# print(len(test_df))

# Model Initialization
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.model_config.ENCODER, config.model_config.DECODER)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# print(model)

# Model training arguments initialization
training_args = Seq2SeqTrainingArguments(
    output_dir='trained_model/VIT_large_gpt2',
    per_device_train_batch_size=config.model_config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.model_config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    learning_rate = 5e-5,
    max_steps=1500, # delete for full training
    num_train_epochs = config.model_config.EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
)

# initiate training
trainer = Seq2SeqTrainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()

# save the trained model
trainer.save_model('trained_model/VIT_large_gpt2')

img =  Image.open("IMAGES_DIR/1001773457_577c3a7d70.jpg").convert("RGB")
img
generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])
print('\033[96m' +generated_caption[:85]+ '\033[0m')