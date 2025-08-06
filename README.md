# LoRA Training for Qwen-Image

An open-source implementation for training LoRA (Low-Rank Adaptation) layers for Qwen/Qwen-Image models by [FlyMy.AI](https://flymy.ai).

## 🌟 About FlyMy.AI

[FlyMy.AI](https://flymy.ai) is an innovative platform for image generation and AI work. We specialize in creating cutting-edge solutions for generative AI and actively support the open-source community.

**🔗 Useful Links:**
- 🌐 [Official Website](https://flymy.ai)
- 📚 [Documentation](https://docs.flymy.ai)
- 💬 [Community](https://community.flymy.ai)

## 🚀 Features

- LoRA-based fine-tuning for efficient training
- Compatible with Hugging Face `diffusers`
- Easy configuration via YAML
- Open-source implementation for LoRA training

## ⚠️ Project Status

**🚧 Under Development:** We are actively working on improving the code and adding test coverage. The project is in the refinement stage but ready for use.

**📋 Development Plans:**
- ✅ Code stability improvements
- ✅ Adding comprehensive tests
- ✅ Performance optimization
- ✅ Documentation expansion

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

---

## 🤝 Support

If you have questions or suggestions, join our community:
- 🌐 [FlyMy.AI](https://flymy.ai)
- 📧 [Support](mailto:support@flymy.ai)

**⭐ Don't forget to star the repository if you like it!**
