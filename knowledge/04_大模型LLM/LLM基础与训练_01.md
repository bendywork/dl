# LLM 大模型基础与训练

## 📌 核心问题
> GPT/LLaMA 这类大模型是怎么训练出来的？预训练 → 微调 → 对齐，每步解决什么问题？

## 🌱 根源与动机

从 BERT 到 ChatGPT 的演进：
```
BERT (2018)          → 双向预训练，擅长理解
GPT-1/2 (2018/2019) → 单向预训练，擅长生成
GPT-3 (2020)        → 1750亿参数，In-Context Learning
InstructGPT (2022)  → RLHF 对齐，听指令
ChatGPT (2022)      → 对话版 InstructGPT
LLaMA (2023)        → 开源高效，推动微调生态
```

## 📐 理论推导

### 三阶段训练流程

**阶段 1：预训练（Pre-training）**
```
目标：预测下一个 token（自回归 Language Model）
数据：TB 级互联网文本
损失：Cross-Entropy（每个位置预测下一词）
结果：模型学会语言规律，但不会听指令
```

**阶段 2：有监督微调（SFT, Supervised Fine-Tuning）**
```
数据：(指令, 高质量回答) 对，人工标注几万~几十万条
目标：让模型学会"按指令回答"的格式
学习率：比预训练小 10~100 倍
```

**阶段 3：RLHF（Reinforcement Learning from Human Feedback）**
```
① 训练 Reward Model：人类对多条回答排序 → 学习"什么是好回答"
② PPO 强化学习：用 RM 的分数反馈优化 LLM，最大化人类偏好
结果：模型更安全、更有帮助、更诚实（HHH原则）
```

### 模型量化

| 方法 | 精度 | 内存占用 | 速度 | 质量损失 |
|------|------|---------|------|---------|
| FP32 | 全精度 | 4B/参数 | 基准 | 无 |
| FP16/BF16 | 半精度 | 2B/参数 | 快 ~2x | 极小 |
| INT8 | 8位整数 | 1B/参数 | 快 | 小 |
| INT4 (GPTQ/AWQ) | 4位整数 | 0.5B/参数 | 最快 | 中 |

**量化公式**：`x_quant = round(x / scale) + zero_point`

## 💡 关键理解

- **预训练是最贵的**：LLaMA-65B 训练消耗 100万 GPU 小时；SFT 只需几百 GPU 小时
- **Emergent Abilities**：模型超过某个规模阈值后，突然具备某些能力（如 Chain-of-Thought）
- **RLHF 的关键**：Reward Model 的质量上限了最终效果，"RLHF 的天花板是 RM"
- **Scaling Law**：模型性能随参数量、数据量、算力按幂律提升（Chinchilla 法则）

## 🔧 代码实现

对应资料：`knowledge/PDF课件/LLM_01_训练阶段.pdf`

相关工具：
- HuggingFace Transformers：模型加载和微调
- PEFT：LoRA/QLoRA 高效微调
- LLaMA-Factory：一站式微调框架

## ⚠️ 易错点与常见误解

1. **SFT 不是"教会模型新知识"**，知识在预训练阶段已经学完，SFT 只是调整行为格式
2. **量化会有精度损失**，INT4 在复杂推理任务上比 FP16 差 5~15%
3. **RLHF ≠ 万能**：仍会出现幻觉，模型"学会表现得好"而非"真的正确"
4. **BF16 比 FP16 更稳定**：动态范围更大，训练大模型首选 BF16

## 🔗 知识延伸

- [[LLM微调方法（LoRA/QLoRA）]] — 高效微调详细原理
- [[LLM模型量化]] — GPTQ、AWQ 量化实现
- [[Transformer原理与实现]] — LLM 的基础架构

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/LLM_01_训练阶段.pdf` ~ `LLM_04_微调数据构造.pdf`
