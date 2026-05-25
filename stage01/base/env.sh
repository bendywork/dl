#!/usr/bin/env bash
set -e

############################################################
# Conda CPU Environment Setup Script (Python 3.12)
# Compatible with GitHub Codespaces (non-interactive shell)
############################################################

ENV_NAME="cpu"
PYTHON_VERSION="3.12"

echo "=================================================="
echo " Setting up Conda Environment: $ENV_NAME"
echo "=================================================="

# ── 1. 检查 conda ──────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda is not installed or not in PATH."
    exit 1
fi

# ── 2. 找到 conda 根目录，直接 source conda.sh ─────────
# Codespaces 里 conda init 改写的 .bashrc 在子shell中不会自动执行
# 所以手动 source，让 conda activate 在当前进程可用
CONDA_BASE=$(conda info --base)
echo "[INFO] Conda base: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 3. 配置 channels ───────────────────────────────────
echo "[INFO] Configuring conda channels..."
conda config --remove-key channels 2>/dev/null || true
conda config --add channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict

# ── 4. 清理缓存 ────────────────────────────────────────
echo "[INFO] Cleaning conda cache..."
conda clean -a -y

# ── 5. 创建环境（已存在则跳过）────────────────────────
if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "[INFO] Environment '$ENV_NAME' already exists, skipping creation."
else
    echo "[INFO] Creating environment: $ENV_NAME (Python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# ── 6. 激活环境 ────────────────────────────────────────
echo "[INFO] Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# ── 7. 安装 Jupyter ────────────────────────────────────
echo "[INFO] Installing Jupyter Notebook..."
conda install -n "$ENV_NAME" notebook -y

# ── 8. 安装 PyTorch CPU ────────────────────────────────
echo "[INFO] Installing PyTorch (CPU version)..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 9. 写入 .bashrc，让后续终端默认激活该环境 ──────────
echo "[INFO] Registering conda init in ~/.bashrc..."
conda init bash

if ! grep -q "conda activate $ENV_NAME" ~/.bashrc; then
    echo "conda activate $ENV_NAME" >> ~/.bashrc
    echo "[INFO] Auto-activate '$ENV_NAME' added to ~/.bashrc"
fi

# ── 10. 验证 ───────────────────────────────────────────
echo ""
echo "=================================================="
echo " Environment setup completed successfully!"
echo "=================================================="
echo "Environment : $ENV_NAME"
echo "Python      : $(python --version)"
echo "Torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "Jupyter     : $(jupyter --version 2>&1 | head -1)"
echo "=================================================="