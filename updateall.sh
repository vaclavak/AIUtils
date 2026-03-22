#!/bin/bash

# ---------------------------
# RunPod LoRA Environment Setup
# ---------------------------

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python3, venv, pip, git, build tools..."
sudo apt install -y python3 python3-venv python3-pip git build-essential

echo "Creating virtual environment 'lora_env'..."
python3 -m venv lora_env

echo "Activating virtual environment..."
source lora_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Removing old PEFT if exists..."
pip uninstall -y peft

echo "Installing GPU-ready ML packages..."
pip install --upgrade torch transformers bitsandbytes datasets accelerate

echo "Installing latest PEFT from GitHub..."
pip install --upgrade git+https://github.com/huggingface/peft.git

echo "Installing unsloth"
pip install unsloth

echo "Setup complete!"
echo "Activate your environment with: source lora_env/bin/activate"

echo "Checking GPU availability..."
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"