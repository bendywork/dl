# Claude Code 本地安装清单

> 记录日期：2026-05-21
> 环境信息：macOS Darwin 22.6.0 / Node v24.3.0 / Claude Code v2.1.146

---

## 一、内置 Skills（系统提供）

以下 Skills 由 Claude Code 系统内置，通过 `/skill-name` 触发使用：

| Skill 名称 | 触发方式 | 功能说明 |
|-----------|---------|---------|
| **update-config** | `/update-config` | 配置 Claude Code 的 settings.json，包括权限、环境变量、hooks 等 |
| **keybindings-help** | `/keybindings-help` | 自定义键盘快捷键，修改 `~/.claude/keybindings.json` |
| **simplify** | `/simplify` | 审查变更代码的复用性、质量和效率，并修复问题 |
| **fewer-permission-prompts** | `/fewer-permission-prompts` | 扫描常用只读工具调用，自动添加权限白名单减少确认弹窗 |
| **loop** | `/loop` | 按间隔循环执行提示词或斜杠命令 |
| **schedule** | `/schedule` | 创建/管理远程定时任务（cron 调度或一次性定时） |
| **claude-api** | `/claude-api` | 构建/调试 Claude API / Anthropic SDK 应用，处理缓存、工具使用等 |
| **design** | `/design` | 从 `~/.claude/design-md/` 读取设计系统文件，用于生成一致 UI |
| **init** | `/init` | 初始化项目 CLAUDE.md 文件 |
| **review** | `/review` | 审查 Pull Request |
| **security-review** | `/security-review` | 对当前分支待合并变更进行安全审查 |

---

## 二、自定义 Slash Commands（用户配置）

| 命令名称 | 文件路径 | 功能说明 |
|---------|---------|---------|
| **design** | `~/.claude/commands/design.md` | 从 `~/.claude/design-md/` 读取指定品牌设计系统，复制到项目根目录作为 DESIGN.md 使用 |

---

## 三、MCP Servers

| MCP Server | 安装方式 | 版本 | 状态 | 说明 |
|-----------|---------|------|------|------|
| **chrome-devtools-mcp** | npm 全局安装 | 0.23.0（本地）/ 1.0.1（最新） | 未在 Claude Code 中注册配置 | Chrome DevTools MCP 服务，提供 CDP 协议连接，用于浏览器自动化调试 |

> 注意：`chrome-devtools-mcp` 已全局安装但未通过 `claude mcp add` 注册到 Claude Code 中，需执行配置后才能在会话中使用。

---

## 四、全局 npm 包（与 AI 编码相关）

| 包名 | 版本 | 说明 |
|------|------|------|
| @anthropic-ai/claude-code | 2.1.146 | Claude Code CLI 主程序 |
| @openai/codex | 0.130.0 | OpenAI Codex CLI |
| chrome-devtools-mcp | 0.23.0 | Chrome DevTools MCP Server |
| cloakbrowser | 0.3.28 | 隐身浏览器（49项指纹修改） |
| codebuff | 1.0.644 | CodeBuff 工具 |
| freebuff | 0.0.88 | FreeBuff 工具 |
| @vscode/vsce | 3.7.1 | VS Code 扩展打包工具 |

---

## 五、DESIGN.md 设计系统库

位置：`~/.claude/design-md/`，共 **72 个品牌**设计系统文件。

| # | 品牌 | 说明 |
|---|------|------|
| 1 | airbnb | 旅行平台，温暖插画风格 |
| 2 | airtable | 低代码数据库，彩色区块 |
| 3 | apple | 极简产品展示 |
| 4 | binance | 加密货币交易所 |
| 5 | bmw | 汽车品牌，深色高端 |
| 6 | bmw-m | BMW M 系列，运动性能感 |
| 7 | bugatti | 超级跑车，奢华深色 |
| 8 | cal | 开源日程管理 |
| 9 | claude | Anthropic AI 助手，暖色珊瑚 |
| 10 | clay | 3D 黏土风格设计系统 |
| 11 | clickhouse | 分析数据库，黄色技术文档风 |
| 12 | cohere | 企业 AI 平台，渐变色彩 |
| 13 | coinbase | 加密货币平台 |
| 14 | composio | 工具集成平台 |
| 15 | cursor | AI 代码编辑器，深色渐变 |
| 16 | elevenlabs | AI 语音平台，暗色电影风 |
| 17 | expo | React Native 平台 |
| 18 | ferrari | 跑车品牌，红色激情 |
| 19 | figma | 设计工具，多彩协作 |
| 20 | framer | 动效设计平台 |
| 21 | hashicorp | 基础设施自动化，黑白企业风 |
| 22 | hp | 消费电子目录，电光蓝 |
| 23 | ibm | 企业科技，蓝色商务 |
| 24 | intercom | 客户消息，友好蓝色 |
| 25 | kraken | 加密货币交易所 |
| 26 | lamborghini | 超级跑车，棱角锋利 |
| 27 | linear.app | 项目管理，极简紫色 |
| 28 | lovable | AI 全栈构建，渐变趣味 |
| 29 | mastercard | 支付品牌 |
| 30 | meta | 社交平台 |
| 31 | minimax | AI 模型，深色霓虹 |
| 32 | mintlify | 文档平台，绿色阅读优化 |
| 33 | miro | 协作白板 |
| 34 | mistral.ai | 开源 LLM，法式极简紫色 |
| 35 | mongodb | 文档数据库，绿叶品牌 |
| 36 | nike | 运动品牌 |
| 37 | notion | 全能工作区，温暖极简 |
| 38 | nvidia | GPU/AI 计算 |
| 39 | ollama | 本地 LLM，终端单色 |
| 40 | opencode.ai | AI 编码平台，深色开发者风 |
| 41 | pinterest | 图片社交 |
| 42 | playstation | 游戏主机 |
| 43 | posthog | 产品分析，趣味暗色 |
| 44 | raycast | 效率启动器，深色渐变 |
| 45 | renault | 汽车品牌 |
| 46 | replicate | ML 模型 API，白色代码风 |
| 47 | resend | 邮件 API，暗色等宽 |
| 48 | revolut | 金融科技 |
| 49 | runwayml | AI 创意工具，电影编辑风 |
| 50 | sanity | 内容平台，珊瑚红强调 |
| 51 | sentry | 错误监控，暗色粉紫 |
| 52 | shopify | 电商平台 |
| 53 | slack | 团队协作 |
| 54 | spacex | 航天科技 |
| 55 | spotify | 音乐流媒体 |
| 56 | starbucks | 咖啡品牌 |
| 57 | stripe | 支付平台，渐变条纹 |
| 58 | supabase | 开源 Firebase，翡翠深色 |
| 59 | superhuman | 邮件客户端，紫光暗色 |
| 60 | tesla | 电动汽车 |
| 61 | theverge | 科技媒体 |
| 62 | together.ai | 开源 AI 基础设施 |
| 63 | uber | 出行平台 |
| 64 | vercel | 前端部署，黑白精准 Geist 字体 |
| 65 | vodafone | 电信品牌 |
| 66 | voltagent | AI Agent 框架，虚空黑翡翠绿 |
| 67 | warp | 现代终端，暗色 IDE 风 |
| 68 | webflow | 无代码建站 |
| 69 | wired | 科技杂志 |
| 70 | wise | 跨境金融 |
| 71 | x.ai | Elon Musk AI，极简未来 |
| 72 | zapier | 自动化平台，温暖橙色 |

---

## 六、使用方式

### Skills 调用
```
/simplify          # 审查代码质量
/design            # 使用设计系统
/review            # 审查 PR
/security-review   # 安全审查
/init              # 初始化 CLAUDE.md
```

### MCP 配置（待完成）
```bash
# 注册 chrome-devtools-mcp 到 Claude Code
claude mcp add chrome-devtools -- chrome-devtools-mcp
```

### 安装新设计系统
```bash
# 安装单个品牌到项目
npx getdesign@latest add <brand>

# 指定输出路径
npx getdesign@latest add <brand> --out ./DESIGN.md

# 查看所有可用品牌
npx getdesign@latest list
```
