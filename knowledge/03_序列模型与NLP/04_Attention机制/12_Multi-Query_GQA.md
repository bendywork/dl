# 03 · MQA 与 GQA（多查询 / 分组查询注意力）

MHA 在推理时有个严重瓶颈：每个 head 独立存 K 和 V，decode 时并行读取全部 head 的 KV cache → 显存带宽成为墙。MQA 和 GQA 通过**减少 K/V 的 head 数**来破这个瓶颈。

## 三种对比

```
MHA:   Q₁ Q₂ Q₃ Q₄     K₁ K₂ K₃ K₄     V₁ V₂ V₃ V₄    ← 8 head = 8 份 K/V
MQA:   Q₁ Q₂ Q₃ Q₄     K        V                      ← 8 head = 1 份 K/V（共享）
GQA:   Q₁ Q₂  Q₃ Q₄    K₁ K₂           V₁ V₂           ← 8 head = 2 份 K/V（分组共享）
```

## MQA（Multi-Query Attention）

**K 和 V 所有 head 共享同一份**，只有 Q 是多头。

| 优点 | 缺点 |
|------|------|
| KV cache 大小缩到 MHA 的 1/h（8 head = 1/8） | 表达能力损失 |
| decode 时读带宽减少 h 倍 | 长上下文质量可能下降 |

**代表**：**PaLM 540B**（2022）、**Falcon 180B**。推理速度显著快于 MHA。

## GQA（Grouped-Query Attention）

**把 head 分成 G 组，组内共享 K/V**。G=1 就是 MQA，G=h 就是 MHA。一般在 2~8 之间取折中值。

| | KV cache 大小 | 推理速度 | 质量 |
|------|------------|---------|------|
| MHA | 最大 | 慢 | 最好 |
| GQA (G=4) | MHA 的 1/4 | 快不少 | 接近 MHA |
| GQA (G=8) | MHA 的 1/8 | 快很多 | 略降 |
| MQA | MHA 的 1/h | 最快 | 最差 |

## 工业选型

| 模型 | 使用 | G 值 |
|------|------|------|
| **LLaMA 2 70B** | GQA | 8 |
| **LLaMA 3** | GQA | 8 |
| **GPT-4** | 未公开，推测 GQA | |
| **Falcon-180B** | MQA | 1 |
| **PaLM 540B** | MQA | 1 |
| **Mistral 7B** | GQA + Sliding Window | |

## 一句话总结

> MHA 推理时被显存带宽杀 → MQA 省 KV 但掉质量 → GQA 折中，LLaMA 2/3 全系标配。
