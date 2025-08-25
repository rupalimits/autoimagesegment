# Automatic Image Captioning with ViT + GPT-2

This project implements an end-to-end image captioning pipeline by integrating a **Vision Transformer (ViT)** for visual feature extraction with **GPT-2** for sequence generation. The model is trained on the **Flickr8k dataset** and achieves a strong BLEU-2 score. The pipeline is **containerized using Docker**, deployed on **AWS ECS**, and monitored in real-time with **Prometheus and Grafana**.

---

## Objective

- Generate meaningful and grammatically correct captions for images using a hybrid ViT-GPT2 architecture.
- Achieve high performance on standard image captioning benchmarks (e.g., BLEU scores).
- Deploy the pipeline in a scalable and observable production environment.

---

## Model Architecture

- **Encoder**: Pretrained Vision Transformer (ViT) extracts visual tokens from input images.
- **Decoder**: GPT-2 (fine-tuned) generates captions based on visual tokens from ViT.
- **Token Fusion**: ViT embeddings are linearly projected and prepended to GPT-2’s text input.
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: AdamW

---

## Dataset

- **Flickr8k**: A dataset containing 8,000 images, each annotated with 5 different captions.
- **Source**: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Preprocessing**:
  - Images resized and normalized for ViT
  - Captions tokenized using GPT-2 tokenizer
  - Filtered captions > 20 tokens

---

## Evaluation Metrics

| Metric     | Score    |
|------------|----------|
| BLEU-1     | 94%      |
| **BLEU-2** | **87%**  |
| BLEU-3     | 75%      |
| BLEU-4     | 63%      |

> BLEU scores were computed on the validation set using `nltk.translate.bleu_score`.

---

## Deployment Stack

| Component     | Tech Used            |
|---------------|----------------------|
| Containerization | Docker             |
| Orchestration   | AWS ECS + Fargate   |
| Monitoring      | Prometheus + Grafana |
| Logging         | AWS CloudWatch      |
| Model Serving   | FastAPI             |

---

## Sample Output

| Input Image              | Generated Caption               |
|--------------------------|---------------------------------|
| ![Screenshot 2025-02-28 at 11 19 01 PM](https://github.com/user-attachments/assets/90a028e4-2bfc-457b-9cec-347e63aed1a0)     | "a girl climbing stairs"|


---

## Technologies Used

- Python 3.9
- PyTorch
- Hugging Face Transformers
- OpenCV
- FastAPI
- Docker, ECS, CloudWatch
- Prometheus + Grafana

---

## 🚀 Highlights

- Hybrid encoder-decoder architecture using state-of-the-art models
- Achieved **87% BLEU-2**, reflecting meaningful short-phrase caption generation
- Fully containerized and deployed with auto-scaling support
- Monitoring integrated for live inference metrics and uptime stats

---

## 🔭 Future Work

- Integrate CLIP for improved vision-text alignment
- Extend to larger datasets like Flickr30k or MS-COCO
- Add user feedback collection via a frontend interface

---

To make it easy to use, I made the solution as a CLI application that can be run directly with "python main.py", enabling end-to-end training or inference from the command line.

For training the model on the flickr8k dataset, follow these steps -
cd ImageCaptioning\
python3 -m venv .venv\
source .venv/bin/activate

pip install -r requirements/requirements.txt

cd image_captioning\
python3 -m main.py

In case you face an error at this step related to punkt, download the punkt from this URL and place it in the home directory of your system (or in the venv folder), following this directory structure -

mkdir nltk_data && cd nltk_data\
mkdir tokenizers && cd tokenizers\
Inside this directory structure, put the punkt folder here (which can be downloaded from the following link)\
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip


After training the model, you will see a folder getting generated named VIT_large_gpt2 which will contain the trained model. Now to test the model, I have created a test file.

If you are standing on image_captioning, come to the test folder from the parent folder and run the main file inside the tests folder.\
cd tests/\
python3 main.py
