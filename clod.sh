#!/bin/bash

echo "请选择要启动的工具："
echo "1) Claude"
echo "2) Codex"
read -p "输入选项 (1/2): " choice

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export OPENAI_API_KEY=sk-0c82e598f89a09a71970ff6312449e9290bf205847df93f807d7a28f2f5783e1
export OPENAI_BASE_URL=http://49.12.223.172:9090/v1
export ANTHROPIC_AUTH_TOKEN=sk-0c82e598f89a09a71970ff6312449e9290bf205847df93f807d7a28f2f5783e1
export ANTHROPIC_BASE_URL=http://49.12.223.172:9090

case $choice in
  1)
    unset ANTHROPIC_API_KEY
    export ANTHROPIC_MODEL=claude-sonnet-4-6
    echo "启动 Claude (模型: claude-sonnet-4-6)..."
    claude --dangerously-skip-permissions
    ;;
  2)
    CODEX_CONFIG="$HOME/.codex/config.toml"
    CODEX_BACKUP="$HOME/.codex/config.toml.bak"
    CODEX_AUTH="$HOME/.codex/auth.json"
    CODEX_AUTH_BACKUP="$HOME/.codex/auth.json.bak"

    cp "$CODEX_CONFIG" "$CODEX_BACKUP"
    cp "$CODEX_AUTH" "$CODEX_AUTH_BACKUP"
    trap 'mv "$CODEX_BACKUP" "$CODEX_CONFIG"; mv "$CODEX_AUTH_BACKUP" "$CODEX_AUTH"; echo "已还原原始配置"' EXIT

    cat > "$CODEX_CONFIG" << EOF
model = "gpt-5.5"
model_provider = "custom"
model_reasoning_effort = "xhigh"
disable_response_storage = true
approvals_reviewer = "user"
plan_mode_reasoning_effort = "xhigh"

[model_providers.custom]
name = "custom"
base_url = "$OPENAI_BASE_URL"
api_key = "$OPENAI_API_KEY"
wire_api = "responses"

[projects."/Volumes/Data/project/javaScript/nextjs/bendy-faka-nextjs"]
trust_level = "trusted"

[projects."/Volumes/Data/project/python/lc"]
trust_level = "trusted"

[projects."/Volumes/Data/project/javaScript/new-api"]
trust_level = "trusted"

[notice]
hide_full_access_warning = true

[notice.model_migrations]
gpt-5-codex = "gpt-5.3-codex"
"gpt-5.3-codex" = "gpt-5.4"

[tui.model_availability_nux]
"gpt-5.5" = 2
EOF

    echo "配置已更新"

    cat > "$CODEX_AUTH" << AUTHJSON
{
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}
AUTHJSON

    echo "auth.json 已更新"
    export OPENAI_MODEL=gpt-5.5
    echo "启动 Codex (模型: gpt-5.5)..."
    codex --full-auto
    ;;
  *)
    echo "无效选项，请输入 1 或 2"
    exit 1
    ;;
esac