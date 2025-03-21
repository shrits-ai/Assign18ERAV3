# 🧠 Phi-2 QLoRA 

Welcome to the Phi-2 QLoRA Adapter project! This repository contains resources for fine-tuning Microsoft Phi-2 with QLoRA (Quantized Low-Rank Adapter) and deploying a chatbot interface on Hugging Face Spaces.

The goal of this project is to create an efficient conversational model using low-rank adapters, which drastically reduce the need for large computational resources, and to provide an easy-to-use interface where users can interact with the fine-tuned model.

## 🚀 Model Details
- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- **Adapter**: Trained on [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **Adapter Type**: QLoRA (Quantized Low-Rank Adapter)
- **Dataset**: OpenAssistant/oasst1
- **Quantization**: 4-bit using bitsandbytes
- **Training Framework**: Transformers + PEFT
- **Generate Interface**: Gradio app on Hugging Face Spaces
- **Fine-Tuned Adapter Checkpoints**: Available on Hugging Face Hub

## 🛠️ Training Process
The QLoRA adapter was trained in Google Colab with the following steps:
- Loaded Phi-2 in 4-bit quantization using `BitsAndBytesConfig`.
- Fine-tuned with PEFT's LoRA on OpenAssistant dataset.
- Saved adapter weights as `adapter_model.bin` and `adapter_config.json`.

## ✨ Features
- Runs on GPU in Hugging Face Spaces.
- Optimized for low latency with bfloat16 precision.
- Supports **chat history** for more interactive conversations.

## 🎯 Live Demo
👉 [Try it on Hugging Face Spaces!](https://huggingface.co/spaces/Shriti09/MicrosoftPhiQloraExample)

## ⚡ Inference Example
Ask the chatbot anything about AI, technology, or more!

## 💻 How It Works
- `app.py`: Gradio interface for chat with history.
- `requirements.txt`: Python dependencies.
