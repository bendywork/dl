#!/bin/bash

echo "请选择要启动的工具："
echo "1) Claude"
echo "2) Codex"
read -p "输入选项 (1/2): " choice

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export OPENAI_API_KEY=tp-ccux4zawmz1amfvqjy2p67qegn3zni2mfm41w5svg7rscb8s
export OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
export ANTHROPIC_AUTH_TOKEN=tp-ccux4zawmz1amfvqjy2p67qegn3zni2mfm41w5svg7rscb8s
export ANTHROPIC_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic

case $choice in
  1)
    export ANTHROPIC_MODEL=MiMo-V2.5-Pro
    echo "启动 Claude..."
    claude --dangerously-skip-permissions
    ;;
  2)
    export OPENAI_MODEL=glm-5.2
    echo "启动 Codex..."
    codex --full-auto
    ;;
  *)
    echo "无效选项，请输入 1 或 2"
    exit 1
    ;;
esac