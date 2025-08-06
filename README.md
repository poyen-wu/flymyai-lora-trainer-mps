# LoRA Training for Qwen-Image

This project allows you to train Low-Rank Adaptation (LoRA) layers for a Qwen/Qwen-Image.

## 🚀 Features

- LoRA-based fine-tuning for efficient training.
- Compatible with Hugging Face `diffusers`.
- Easily configurable via YAML.

---

## 📦 Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/FlyMyAI/qwen-image-lora-trainer
   cd qwen-image-lora-trainer
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the latest `diffusers` from GitHub:
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

---

## 🏁 Start Training

To begin training with your configuration file (e.g., `train_lora.yaml`), run:

```bash
accelerate launch train.py --config ./train_configs/train_lora.yaml
```

Make sure `train_lora.yaml` is correctly set up with paths to your dataset, model, output directory, and other parameters.
