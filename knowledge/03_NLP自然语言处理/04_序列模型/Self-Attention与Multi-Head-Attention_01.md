# Self-Attention 与 Multi-Head Attention

## 📌 核心问题
> RNN 逐步串行处理序列，信息要一步步传递，远了就丢了。如何让序列中每个位置**直接**看到所有其他位置？

## 🌱 根源与动机

### RNN 的根本缺陷

```
"下周看涨" → 看"下" → 记住一点 → 看"周" → 记住一点 → 看"看" → 记住一点 → 看"涨"
```

到"涨"的时候，"下"的信息已经传了 3 步，逐步衰减。RNN 的核心问题是**信息必须逐步传递，无法直达**。

### Attention 的直觉

人在读一句话时，理解"涨"不是靠回忆前面每个字，而是**一眼看所有字，找出和"涨"最相关的**：

```
"下周看涨" 中，"涨"最关注"看"（看涨是固定搭配），不太关注"下"
```

这就是 Attention——**根据相关性，从一堆信息里挑出最相关的部分。**

## 📐 理论推导

### Attention 的三要素

| 角色 | 含义 | 类比 |
|------|------|------|
| **Q（Query）** | 我想知道什么 | 你的问题："考试考什么？" |
| **K（Key）** | 每条信息的标签 | 每句话的标签："这是闲聊"、"这是考点" |
| **V（Value）** | 每条信息的实际内容 | 每句话的具体文字 |

### 点积 = 相似度

两个向量点积越大，方向越一致，越相似：

```python
# 相似向量 → 点积大
a = [1.0, 0.8, 0.5]
b = [0.9, 0.7, 0.6]
a·b = 1.76

# 相反向量 → 点积为负
c = [-1.0, -0.8, -0.5]
a·c = -1.89
```

**Q·K 就是算"我的问题"和"你的标签"有多相关。**

### 为什么 K 需要转置？

"下周看涨"4 个字，每个字都有自己的 K 向量（1D），叠在一起变成矩阵：

```
K₁("下") = [0.2, 0.1, 0.3]    ← 一个向量
K₂("周") = [0.3, 0.5, 0.2]
K₃("看") = [0.1, 0.4, 0.6]
K₄("涨") = [0.8, 0.7, 0.5]

叠起来：K shape: (4, d_k)     ← 4个向量叠成矩阵
```

"涨"的 Q₄ 要和所有 K 算相似度：

```
逐个算：Q₄·K₁, Q₄·K₂, Q₄·K₃, Q₄·K₄   → 4次点积

一次算：Q₄ · K^T                         → 1次矩阵乘法
        (d_k,) · (d_k, 4) = (4,)          → 4个相似度一次出
```

**4次逐个点积等价于1次矩阵乘法，结果一样，矩阵乘法更快。**

### 完整 Self-Attention 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

每一步的 shape 变化（以 `d_model=128, d_k=32, seq_len=4` 为例）：

```
x:       (batch, 4, 128)
  ↓ W_q / W_k / W_v
q, k, v: (batch, 4, 32)
  ↓ q @ k^T
scores:  (batch, 4, 4)      ← 每个位置对每个位置的分数
  ↓ / √d_k
  ↓ softmax
weights: (batch, 4, 4)      ← 每个位置对每个位置的权重（概率）
  ↓ @ v
context: (batch, 4, 32)     ← 每个位置的新表示
```

### 除以 √d_k 的原因

d_k 越大，点积结果越大（向量维度多，加的项多），softmax 输入太大会饱和（输出接近 0 或 1），梯度几乎为 0，训练不动。除以 √d_k 把数值拉回合理范围。

## 💡 关键理解

### Self-Attention vs Cross-Attention

| | Q 来自 | K/V 来自 | 作用 |
|---|---|---|---|
| **Cross-Attention** | 源 A | 源 B | A 去 B 里找相关信息 |
| **Self-Attention** | 源 A | 源 A | A 内部自己找关系 |

Self-Attention 是 Attention 的特例——Q 和 K/V 同源。情感分类只需要 Self-Attention，Cross-Attention 是翻译场景用的。

### Multi-Head 的本质

一个 Self-Attention 只能学一种关系。Multi-Head 是做多个 Self-Attention，每个学不同类型的关系：

```
头1：可能学到语法关系（主谓宾）
头2：可能学到情感关系（看涨看跌）
头3：可能学到否定关系（不、没）
头4：可能学到位置关系（前后文）
```

### d_k 为什么 = d_model // n_heads？

```
d_model = 128, n_heads = 4
每个头 d_k = 128 // 4 = 32
4个头输出拼接：32 + 32 + 32 + 32 = 128 = d_model
```

**这么设是为了拼接后刚好等于 d_model，维度对得上。**

### 拼接后为什么还要 fc 层？

直接拼接是生硬的——4个头各干各的，互相不知道对方学了什么。fc 层让它们融合：

```
拼接：[头1结果 | 头2结果 | 头3结果 | 头4结果]
  ↓ fc 层
融合：4个头的信息混合在一起，得到最终表示
```

### d_model 的含义

d_model 是模型内部统一的向量维度，Transformer 里每一步的维度都是 d_model，从头到尾不变，这样才能像水管一样一层一层堆叠：

```
Embedding 输出:     (batch, seq_len, d_model)
Multi-Head 输出:    (batch, seq_len, d_model)
FFN 输出:           (batch, seq_len, d_model)
下一层输入:         (batch, seq_len, d_model)
```

命名来自 Transformer 原论文：`d_` 前缀 + 含义（d_model, d_k, d_ff...）

## 🔧 代码实现

### Self-Attention

```python
import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        q = self.W_q(x)  # (batch, seq_len, d_k)
        k = self.W_k(x)  # (batch, seq_len, d_k)
        v = self.W_v(x)  # (batch, seq_len, d_k)

        # 相似度 → softmax → 加权求和
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v  # (batch, seq_len, d_k)
        return context
```

### Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 升维 128→512
        self.relu = nn.ReLU()                  # 过滤
        self.fc2 = nn.Linear(d_ff, d_model)   # 降回来 512→128

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.fc1(x)       # (batch, seq_len, d_ff)
        x = self.relu(x)      # (batch, seq_len, d_ff)
        x = self.fc2(x)       # (batch, seq_len, d_model)
        return x
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        d_k = d_model // n_heads
        self.attentions = nn.ModuleList([
            SelfAttention(d_model, d_k) for _ in range(n_heads)
        ])
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 每个头各自跑 SelfAttention
        head_outputs = [attn(x) for attn in self.attentions]
        # 在最后一个维度拼接: (batch, seq_len, d_model)
        concat = torch.cat(head_outputs, dim=-1)
        # fc 融合
        return self.fc(concat)
```

## ⚠️ 易错点与常见误解

1. **以为 K 是矩阵**——每个位置的 K 是一维向量，叠在一起才是矩阵，为了批量计算
2. **忘记除 √d_k**——softmax 会饱和，梯度消失，训练不动
3. **用普通 list 存 SelfAttention**——PyTorch 看不到参数，不会优化，必须用 `nn.ModuleList`
4. **以为 Multi-Head 是串行的**——多个头是并行的，各自独立，最后拼接
5. **以为 Self-Attention 和 Cross-Attention 是完全不同的东西**——Self-Attention 只是 Q/K/V 同源的特例

## 🔗 知识延伸

- [[RNN四种输入输出结构]] — RNN 的串行处理方式，对比 Self-Attention 的并行
- [[LSTM为什么拆分输出与状态]] — LSTM 用门控缓解信息衰减，Self-Attention 直接消除这个问题
- Transformer 完整架构 = Self-Attention + FFN + LayerNorm + 残差连接 + 位置编码（后续文档）
- BERT = Transformer 编码器 + MLM 预训练

## 📚 参考资料

- Attention Is All You Need (Vaswani et al., 2017) — Transformer 原论文
- The Illustrated Transformer (Jay Alammar) — 可视化图解
