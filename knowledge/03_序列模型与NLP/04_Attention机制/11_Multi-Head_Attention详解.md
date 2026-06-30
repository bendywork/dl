# 02 · Multi-Head Attention（MHA，多头注意力）

是 Transformer 论文 (2017) 的核心创新之一。不是凭空创造新机制，而是把 Scaled Dot-Product Attention **并行跑多份**。

## 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

head_i = Attention(Q·W_i^Q,  K·W_i^K,  V·W_i^V)
```

投影矩阵把 d_model 维的 Q/K/V 映射到 d_k = d_model / h 维子空间，各头在低维空间独立算 Attention，最后拼接回去。

## 为什么需要多头

| 问题 | 答案 |
|------|------|
| 一个头不够吗？ | 一个头只能学到一种"关注模式" |
| 多个头解决什么？ | 不同的头关注不同语义关系 |

```
Head 1: 学到"主语 ↔ 谓语"的语法关系
Head 2: 学到"形容词 ↔ 名词"的修饰关系
Head 3: 学到"代词 → 指代对象"的共指关系
Head 4: 学到位置相邻关系（局部依赖）
Head 5: 学到远距离句型结构关系
...
Head 8: 学到标点/分隔符的结构作用
```

8 个头不是"8 个人各自看一遍"，而是**8 个低维子空间并行算**，总计算量跟 1 个大头差不多（d_model/h 减小抵消了 head 数量增加）。

## 工程细节

```
d_model = 512
h = 8
d_k = d_v = 64

总计算量: O(8 × n² × 64) = O(n² × 512)
         ≈ 1 个 512 维大头的计算量

但 8 个头能学到 8 种不同的关注模式 → 表达能力远大于单头
```

## 工业应用

| 模型 | 配置 |
|------|------|
| Transformer 原版 | h=8, d_model=512 |
| BERT-base | h=12, d_model=768 |
| GPT-3 | h=96 (175B), d_model=12288 |
| LLaMA-7B | h=32, d_model=4096 |

**MHA 是所有 Transformer 的默认配置**。后续变体（MQA/GQA）是在 MHA 基础上做 KV 共享以减少推理代价。

## MHA 的推理解析瓶颈

每个 head 都有独立的 K 和 V，推理时 decode 阶段需要把所有 head 的 KV 拼起来做 Attention → 显存带宽被打满：

```
每生成 1 个 token → 读所有 head 的完整 KV cache → 带宽瓶颈
```

这就是 MQA 和 GQA 要解决的问题。

---

## 完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # 四个投影矩阵：Q、K、V、输出
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: (batch, seq, d_model) → (batch, heads, seq, d_k)
        B, S, _ = x.shape
        x = x.view(B, S, self.h, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)

        # 1. 线性投影
        Q = self.split_heads(self.W_q(Q))  # (B, h, seq_q, d_k)
        K = self.split_heads(self.W_k(K))  # (B, h, seq_k, d_k)
        V = self.split_heads(self.W_v(V))  # (B, h, seq_k, d_k)

        # 2. Scaled Dot-Product Attention（各头并行）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)  # (B, h, seq_q, d_k)

        # 3. 拼接所有头
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, -1, self.d_model)  # (B, seq_q, d_model)

        # 4. 输出投影
        return self.W_o(context)


# 使用示例
d_model, num_heads, seq_len = 512, 8, 20
mha = MultiHeadAttention(d_model, num_heads)

x = torch.randn(2, seq_len, d_model)
out = mha(x, x, x)          # Self-Attention
print(out.shape)             # (2, 20, 512)
```

---

## MHA / MQA / GQA 对比

推理阶段 KV Cache 是显存瓶颈，衍生出两种变体：

| 方案 | Q 头数 | K/V 头数 | KV Cache 大小 | 代表模型 |
|------|--------|---------|-------------|---------|
| MHA（标准多头） | h | h | 100% | BERT、GPT-2 |
| MQA（多查询） | h | 1 | 1/h | PaLM、Falcon |
| GQA（分组查询） | h | g (1<g<h) | g/h | LLaMA2-70B、Mistral |

**示意图：**

```
MHA：Q1K1V1  Q2K2V2  Q3K3V3  Q4K4V4   ← 每头独立 KV
MQA：Q1      Q2      Q3      Q4
        ↘      ↓      ↙      ↗
            K1  V1                      ← 共享同一组 KV
GQA：Q1 Q2  |  Q3 Q4                  ← 每组共享一组 KV
         K1V1       K2V2
```

**GQA 代码片段：**

```python
# GQA：g 组，每组 h//g 个 Q 头共享 1 组 K/V
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.h = num_heads
        self.g = num_kv_heads          # KV 头数，g < h
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.g * self.d_k)
        self.W_v = nn.Linear(d_model, self.g * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
```

---

## 可视化理解

**8个头各自关注什么（论文实验结论）：**

```
头1：句法依赖（主语 ← 动词）
头2：指代消解（"它" → 指代的名词）
头3：位置关系（关注相邻词）
头4：长距离依赖（句首 ↔ 句尾）
头5：语义相关（近义词之间）
头6：标点边界（句子结构）
头7：局部上下文（窗口内词语）
头8：罕见词关注（低频词的语境）
```

用代码可视化 Attention 权重：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(weights, tokens, head_idx=0):
    """
    weights: (batch, heads, seq, seq)
    tokens:  词列表
    """
    attn = weights[0, head_idx].detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', vmin=0, vmax=1)
    plt.title(f'Head {head_idx} Attention Weights')
    plt.tight_layout()
    plt.show()
```

---

## 总结

| 问题 | Multi-Head Attention 的解法 |
|------|--------------------------|
| 单头只能关注一种关系 | 多头并行，每头学习不同子空间 |
| d_model 维度太大，单头点积方差大 | 拆成 h 份，每头 d_k = d_model/h |
| 推理 KV Cache 显存爆炸 | MQA/GQA 减少 KV 头数 |
| 长序列 n² 复杂度 | FlashAttention 分块计算 |

**核心公式回顾：**

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
```

**记忆口诀：**

> 多头 = 多视角看问题；拼接 = 汇总所有视角；W_O = 整合成统一表达。

Multi-Head Attention 是 Transformer 最核心的创新，理解它的四个矩阵（W_Q、W_K、W_V、W_O）和多头并行的设计，就掌握了现代大模型注意力机制的精髓。
