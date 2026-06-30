@echo off
chcp 65001 >nul

echo 请选择要启动的工具：
echo 1) Claude
echo 2) Codex
set /p choice="输入选项 (1/2): "

@REM 如果有代理，就填你的本地代理地址
set OPENAI_API_KEY=sk-8wgI8BjczfIV4L1VC0xqYaXPC7kS0vv7m6DXhiqHnjoVd2gS
set OPENAI_BASE_URL=https://sinsy.eu.cc/v1
set ANTHROPIC_API_KEY=sk-8wgI8BjczfIV4L1VC0xqYaXPC7kS0vv7m6DXhiqHnjoVd2gS
set ANTHROPIC_BASE_URL=https://sinsy.eu.cc/

if "%choice%"=="1" (
    set ANTHROPIC_MODEL=claude-opus-4-7-max
    echo 启动 Claude...
    claude --dangerously-skip-permissions
) else if "%choice%"=="2" (
    echo 启动 Codex...
    codex --model gpt-5.5
) else (
    echo 无效选项，请输入 1 或 2
    exit /b 1
)
