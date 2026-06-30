# LLM 大模型链路

## 逻辑链

```
BERT(Encoder预训练) + GPT(Decoder自回归) → 规模化(更多参数+数据)
    → GPT-2/3(涌现能力) → InstructGPT/ChatGPT(RLHF对齐)
    → Prompt Engineering → RAG(检索增强) → Agent(工具调用)
```

## 从 BERT/GPT 到 LLM

```
BERT ：双向 Encoder，理解型任务，Fine-tune 使用
GPT  ：单向 Decoder，生成型任务，Few-shot 使用
      ↓ 统一方向
自回归 Decoder-only 架构统治 LLM
      ↓
   Scaling Law：更大模型 → 涌现能力
      ↓
   问题：能力强了，但不会听话
      ↓
   RLHF：用人类偏好训练奖励模型 → PPO 对齐
```

## 关键知识点

| 主题 | 说明 |
|------|------|
| Scaling Law | 模型/数据/算力越大，性能可预测增长 |
| Emergence | 规模大到一定程度，突然出现新能力 |
| RLHF | SFT → Reward Model → PPO，让模型对齐人类偏好 |
| Prompt Engineering | 不用微调，靠精心设计的 prompt 调用模型能力 |
| RAG | 检索+生成，给 LLM 接外部知识库 |
| Agent | LLM + 工具调用 + 记忆 + 规划 |

## 知识文件（待完善）

> 当前 knowledge 中 LLM 专题内容正在建设中，现有覆盖：
> - BERT/GPT 预训练 → `05_Transformer与BERT/04_BERT模型预训练详解.md`
> - 文本生成 → `06_NLP应用/01_文本生成与评估指标.md`

## 向上连接

← `05_Attention与Transformer`：Transformer 是 LLM 的技术底座
← `06_NLP应用`：LLM 取代了很多传统 NLP 流水线

## 向实践落地

↓ `08_工程实践`：LoRA 微调、量化部署、推理优化
