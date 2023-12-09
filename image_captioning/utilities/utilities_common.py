
import os
import torch
import evaluate
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from image_captioning.config.core import *
from transformers import AutoTokenizer, ViTFeatureExtractor

def build_inputs_with_special_tokens(self, token_ids_0):
    """
    This is a function to build special tokens while tokenizing the captions.
    :param self:
    :param token_ids_0:
    :return:
    """
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

# Add bos/eos tokens to each token
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
# load the GPT2 tokenizer using the AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.lmodel_config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

# load feature extractor using ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained(config.lmodel_config.ENCODER)


def compute_metrics(eval_pred):
    """
    Compute the following metrics:
    1. Rouge-1 : https://huggingface.co/spaces/evaluate-metric/rouge
    2. BLEU : https://huggingface.co/spaces/evaluate-metric/bleu
    3. BERTScore : https://huggingface.co/spaces/evaluate-metric/bertscore
    4. METEOR : https://huggingface.co/spaces/evaluate-metric/meteor
    Note: the metrics BLEU and METEOR specific files have been downloaded from
            https://github.com/huggingface/datasets/tree/main/metrics
    :param tokenizer:
    :param eval_pred:
    :return: dict_metrics
    """
    dict_metrics = {"rouge2": [evaluate.load("rouge")],
                    # "bleu": [evaluate.load(METRICS_DIR / 'blue')],
                    "bertscore": [evaluate.load("bertscore")],
                    # "meteor": [evaluate.load(METRICS_DIR / 'meteor')]
                    }

    labels_ids = eval_pred.label_ids
    pred_ids = eval_pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # calculating various metrics
    rouge_output = dict_metrics["rouge2"][0].compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
    dict_metrics["rouge2"].append({"rouge2_score": rouge_output['rouge2'], })
    # dict_metrics["rouge2"].append({"rouge2_precision": round(rouge_output.precision, 4),  "rouge2_recall": round(rouge_output.recall, 4), "rouge2_fmeasure": round(rouge_output.fmeasure, 4),})

    # bleu_output = dict_metrics["blue"][0].compute(predictions=pred_str, references=label_str)
    # dict_metrics["blue"].append({"bleu_score": bleu_output['bleu'],})

    bertscore_output = dict_metrics["bertscore"][0].compute(predictions=pred_str, references=label_str, lang="en")
    dict_metrics["bertscore"].append({"bertscore_precision": bertscore_output['precision'], "bertscore_recall": bertscore_output['recall'], "bertscore_f1": bertscore_output['f1']})

    # meteor_output = dict_metrics["meteor"][0].compute(predictions=pred_str, references=label_str)
    # dict_metrics["meteor"].append({"meteor_score": meteor_output['meteor'],})

    return dict_metrics


"""
The Transformations used are
    1. Resizing the image to (224,224)
    2. Normalizing the image
    3. Converting the image to Tensor
"""
img_transforms = transforms.Compose(
    [
        transforms.Resize(config.lmodel_config.IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=config.lmodel_config.MEAN, std=config.lmodel_config.STD)
    ]
)


class ImgDataset(Dataset):
    """
    The dataset is created using the following steps
        1. We read the image using the Image function of PIL library
        2. The image is transformed using the img_transformer defined above
        3. The transformed image is passed through the feature extractor to extract the pixel values from the image
        4. The captions are loaded from the dataframe
        5. The captions are tokenized
        6. The tokenized captions are padded to max length
        7. The images and tokenized captions are returned
    """
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = 50

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")

        img_array = np.array(img).astype(np.float32)

        # Check if the image is already in the desired range
        if img_array.min() < 0 or img_array.max() > 1:
            # Rescale from [-1, 1] to [0, 1]
            img_array = (img_array + 1) / 2
            img_array = np.clip(img_array, 0, 1)  # Ensure values are within [0, 1]
        rescaled_pil_img = Image.fromarray((img_array * 255).astype(np.uint8), "RGB")

        if self.transform is not None:
            rescaled_pil_img = self.transform(rescaled_pil_img)
        pixel_values = self.feature_extractor(rescaled_pil_img, return_tensors="pt").pixel_values

        # if self.transform is not None:
        #     img = self.transform(img)
        # pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,
                                  padding='max_length',
                                  max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding
