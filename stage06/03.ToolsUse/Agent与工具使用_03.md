# Agent 与工具使用

## 一、什么是 Agent

### 定义

Agent = LLM + 工具调用 + 自主规划 + 环境交互

```
传统 LLM：问题 → 回答（纯文本，一次性）
Agent：    问题 → 思考 → 调用工具 → 观察结果 → 再思考 → ... → 最终回答
```

### Agent 的核心能力

| 能力 | 说明 |
|------|------|
| **工具调用** (Tool Use / Function Calling) | 模型决定调用哪个函数、传什么参数 |
| **规划** (Planning) | 复杂任务拆成多步，按顺序执行 |
| **记忆** (Memory) | 记住上下文、之前的结果 |
| **反思** (Reflection) | 观察执行结果，判断是否正确，必要时重新来过 |

---

## 二、函数调用 (Function Calling)

### 工作流程

```
Step 1: 注册工具定义给模型
  {
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "parameters": { "city": "string" }
  }

Step 2: 用户输入 → 模型决定是否调用工具
  用户："北京今天天气怎么样？"
  模型：→ 决定调用 get_weather(city="北京")

Step 3: 程序执行函数，返回结果给模型
  返回：{"temperature": 25, "weather": "晴"}

Step 4: 模型基于结果生成最终回答
  "北京今天晴天，温度 25°C。"
```

### 关键能力

模型需要学会：
1. **何时调用**：该不该用工具
2. **选哪个工具**：多个工具中选最合适的
3. **参数提取**：从用户自然语言中提取结构化参数
4. **结果理解**：正确解读工具返回的结果
5. **多轮调用**：一次不够再来一次
6. **非调用时正常回复**：闲聊时不会乱调函数

---

## 三、MCP (Model Context Protocol)

Anthropic 2024 年底推出的**开放协议**，解决 Agent-工具连接层的标准化问题。

### 解决的问题

```
之前：每个工具需要单独写适配代码
  Claude → 自写代码 → Slack API
  Claude → 自写代码 → GitHub API
  Claude → 自写代码 → Google Drive API
  每个都要手工集成，N 个工具 = N 套代码

MCP：
  Claude → MCP Client → MCP Server (Slack)    → Slack API
                       → MCP Server (GitHub)   → GitHub API
                       → MCP Server (GDrive)   → Google Drive API
  统一协议，社区共建 Server，即插即用
```

### 架构

```
Host (Claude Desktop / IDE / 你的应用)
  │
  ├── MCP Client (协议层)
  │     │
  │     ├── MCP Server A (本地，访问文件系统)
  │     ├── MCP Server B (本地，访问数据库)
  │     └── MCP Server C (远程，调用外部 API)
  │
  └── LLM (推理 + 决策)
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Resources** | Server 暴露的数据（文件、数据库记录），模型可以读取 |
| **Tools** | Server 暴露的可调用函数 |
| **Prompts** | Server 提供的 prompt 模板 |
| **Sampling** | Server 可以反向请求 LLM 生成内容 |

### 2026 现状

- Anthropic 官方大力推动，成为 Agent-工具连接的事实标准候选
- 社区 Server 数量快速增长（Github, Slack, Postgres, Filesystem, Puppeteer...）
- 类似协议：OpenAI 的 `functions` / Google 的 A2A (Agent-to-Agent)

---

## 四、Computer Use

### 概念

模型不是通过 API 操作工具，而是**像人一样操作电脑**——看屏幕截图 → 移动鼠标 → 点击 → 输入文字。

```
输入：当前屏幕截图 + 用户指令
输出：鼠标点击 (x,y) / 键盘输入 / 滚动
```

### 代表系统

| 系统 | 公司 | 特点 |
|------|------|------|
| **Claude Computer Use** | Anthropic | 看截图，输出鼠标键盘操作 |
| **OpenAI Operator** | OpenAI | 浏览器自动化 Agent |
| **UI-TARS** | 字节跳动 | 开源 GUI Agent，直接理解界面元素 |
| **Adept ACT-1/2** | Adept | Transformer 直接预测操作 |

### 优势与局限

| 优势 | 局限 |
|------|------|
| 无需 API 即可操作任何软件 | 速度慢（图像理解 + 逐歩操作） |
| 人类可用 = 模型可用 | 安全风险（误删数据、意外操作） |
| 跨应用泛化 | 精确点击容易出错 |

---

## 五、2026 Agent 全景

### 主流 Agent 框架

| 框架 | 特点 |
|------|------|
| **LangChain / LangGraph** | 最早的 Agent 编排框架，生态最大 |
| **AutoGen (Microsoft)** | 多 Agent 对话，Agent 间互相调用 |
| **CrewAI** | 角色分配，多 Agent 协作（CEO/工程师/测试...） |
| **Dify / Coze** | 低代码 Agent 搭建平台 |
| **SWARM (OpenAI)** | 轻量级 Agent 编排，强调最小抽象 |
| **MCP + Claude** | 协议标准化路线，去框架化 |

### Agent 分级 (Anthropic 定义)

| Level | 名称 | 能力 |
|-------|------|------|
| L1 | Tool Use | 函数调用，无自主规划 |
| L2 | Planning Agent | 多步拆解，顺序执行 |
| L3 | Self-Correcting Agent | 观察结果，修正错误，重新执行 |
| L4 | Multi-Agent | 多 Agent 协作，分工完成复杂任务 |
| L5 | Autonomous Agent | 长期自主运行，设定目标后无需人工干预 |

目前（2026）主流在 L2-L3 之间，L4 在探索期，L5 尚未达到。

### 关键趋势

1. **MCP 推动协议标准化**：工具接入从"手写代码"变成"装个 Server"
2. **Computer Use 走向实用**：自动化测试、RPA、无障碍辅助
3. **Agent → Agent 通信**：A2A 协议（Google）让不同公司的 Agent 互操作
4. **可信计算**：Agent 运行在飞地（TEE），操作可审计
5. **框架的终局**：MCP 这种协议层标准化可能让 Agent 框架本身不再必要
