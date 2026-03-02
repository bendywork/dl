#!/usr/bin/env bash
set -e

############################################################
# Auto generate requirements.txt using pipreqs
# and install missing dependencies
#
# Usage:
#   ./auto_requirements.sh /path/to/project
############################################################

TARGET_DIR=${1:-.}
OUTPUT_FILE="requirements.txt"

echo "=================================================="
echo " 📦 Auto Generating requirements.txt (pipreqs)"
echo "=================================================="
echo "[INFO] Target directory: $TARGET_DIR"

if [ ! -d "$TARGET_DIR" ]; then
  echo "[ERROR] Directory does not exist."
  exit 1
fi

# 确保 pipreqs 已安装
if ! command -v pipreqs >/dev/null 2>&1; then
  echo "[INFO] Installing pipreqs..."
  pip install pipreqs
fi

echo "[INFO] Removing old requirements.txt (if exists)"
rm -f "$OUTPUT_FILE"

echo "[INFO] Generating requirements.txt from notebooks..."

# 生成 requirements.txt
pipreqs "$TARGET_DIR" \
  --force \
  --encoding=utf-8 \
  --mode no-pin

echo
echo "================ Generated requirements.txt ================"
cat "$OUTPUT_FILE"
echo "============================================================"

echo
echo "=================================================="
echo " 🚀 Installing missing dependencies..."
echo "=================================================="

pip install --upgrade pip >/dev/null

if [ -s "$OUTPUT_FILE" ]; then
  pip install -r "$OUTPUT_FILE"
else
  echo "[INFO] No dependencies detected."
fi

echo
echo "=================================================="
echo " ✅ Done!"
echo "=================================================="