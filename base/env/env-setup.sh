#!/usr/bin/env bash
set -e

############################################################
# Conda CPU Environment Setup Script (Python 3.12)
# Creates env "cpu" and installs Jupyter + PyTorch CPU
############################################################

ENV_NAME="cpu"
PYTHON_VERSION="3.12"

echo "=================================================="
echo " 🐍 Setting up Conda Environment: $ENV_NAME"
echo "=================================================="

# 检查 conda 是否存在
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda is not installed or not in PATH."
  exit 1
fi

echo "[INFO] Configuring conda channels..."

conda config --remove-key channels 2>/dev/null || true
conda config --add channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

echo "[INFO] Cleaning conda cache..."
conda clean -a -y

echo "[INFO] Creating environment: $ENV_NAME (Python $PYTHON_VERSION)..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "[INFO] Initializing conda for bash..."
SHELL_NAME=$(basename "$SHELL")
echo "[INFO] Current shell: $SHELL_NAME"

conda init bash

# 重新加载 shell 配置
source ~/.bashrc

echo "[INFO] Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "[INFO] Installing Jupyter Notebook..."
conda install -n "$ENV_NAME" notebook -y

echo "[INFO] Installing PyTorch (CPU version)..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cpu

echo
echo "=================================================="
echo " ✅ Environment setup completed successfully!"
echo "=================================================="
echo "Environment Name : $ENV_NAME"
echo "Python Version   : $PYTHON_VERSION"
echo "=================================================="