# Attention 机制全面分类详解

> 文档编号：配套 Attention 系列第4篇
> 覆盖范围：Bahdanau → Luong → Transformer → 稀疏/局部 Attention

---

## 一、背景：为什么需要 Attention

### 1.1 RNN/LSTM 的信息瓶颈问题

RNN/LSTM 在处理序列时，所有历史信息都被压缩到固定维度的隐状态 h_t 中。
序列越长，早期 token 的信息在传递过程中不断被稀释，梯度也越来越难以回传。

```
输入序列:  x1 → x2 → x3 → ... → xT
隐状态:    h1 → h2 → h3 → ... → hT  (固定维度，信息瓶颈)
```

**核心问题：**
- 长距离依赖难以捕捉（梯度消失）
- 无法并行计算（逐步递推）
- 所有位置共享同一个 context vector

### 1.2 Seq2Seq 的 context vector 局限

经典 Seq2Seq（Sutskever 2014）用 Encoder 最后隐状态作为唯一 context：

```
Encoder: x1...xT → c (单一向量)
Decoder: c → y1, y2, ..., yT'
```

**问题：**
- c 必须记住整个输入序列，长句表达能力严重不足
- 解码每一步都用同一个 c，没有"聚焦"能力
- 实验证明：句子超过20词后翻译质量急剧下降


---

## 二、Attention 的核心公式

Attention 本质是：**根据查询 Q，从键值对 (K, V) 中加权提取信息**。

### 2.1 三步骤

```
步骤1：计算 score（相关性分数）
  score(Q, K_i) = 某种相似度函数

步骤2：softmax 归一化
  α_i = softmax(score_i) = exp(score_i) / Σ exp(score_j)

步骤3：加权求和
  context = Σ α_i · V_i
```

### 2.2 通用公式

```
Attention(Q, K, V) = softmax( score(Q, K) ) · V
```

其中：
- **Q (Query)**：当前解码器隐状态，表示"我想查什么"
- **K (Key)**：编码器各位置表示，表示"我能提供什么索引"
- **V (Value)**：编码器各位置的实际内容，表示"实际信息"
- **α (attention weight)**：注意力分布，表示"关注程度"

### 2.3 不同 score 函数对比

| 名称 | 公式 | 提出者 |
|------|------|--------|
| 加性（Additive） | v^T · tanh(W1·Q + W2·K) | Bahdanau 2015 |
| 点积（Dot-Product） | Q · K^T | Luong 2015 |
| 缩放点积 | Q · K^T / √d_k | Vaswani 2017 |
| 双线性（General） | Q · W · K^T | Luong 2015 |


---

## 三、主要分类详解

---

### 3.1 Additive Attention（加性注意力 / Bahdanau Attention）

**提出：** Bahdanau et al., 2015《Neural Machine Translation by Jointly Learning to Align and Translate》

#### 原理

解码时每一步动态查询编码器所有位置，用一个小型前馈网络计算相关性分数。

#### 公式

```
score(s_t, h_i) = v_a^T · tanh(W_a · s_t + U_a · h_i)

α_ti = softmax(score(s_t, h_i))

c_t = Σ_i α_ti · h_i
```

- s_t：解码器 t 时刻隐状态
- h_i：编码器第 i 个位置隐状态
- W_a, U_a, v_a：可学习参数

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 无缩放问题，数值稳定 | 参数量多（需 W_a, U_a, v_a） |
| 对长序列效果好 | 计算量 O(n·d)，略慢于点积 |
| 最早解决信息瓶颈 | 不适合 Q/K 维度很高的场景 |

#### 适用场景
机器翻译（早期 NMT）、语音识别、序列生成任务


#### PyTorch 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """Bahdanau Additive Attention"""
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(key_dim,   hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        query:  (batch, query_dim)           -- 解码器隐状态
        keys:   (batch, seq_len, key_dim)    -- 编码器输出
        values: (batch, seq_len, value_dim)  -- 通常等于 keys
        """
        # query 扩展维度以便广播: (batch, 1, hidden)
        q = self.W_query(query).unsqueeze(1)
        # keys 映射:              (batch, seq_len, hidden)
        k = self.W_key(keys)
        # score:                  (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)          # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), values)  # (batch, 1, value_dim)
        return context.squeeze(1), weights


# 示例运行
batch, seq_len, enc_dim, dec_dim = 2, 10, 256, 256
query  = torch.randn(batch, dec_dim)
keys   = torch.randn(batch, seq_len, enc_dim)
attn   = AdditiveAttention(dec_dim, enc_dim, 128)
ctx, w = attn(query, keys, keys)
print("context:", ctx.shape)   # (2, 256)
print("weights:", w.shape)     # (2, 10)
```


---

### 3.2 Dot-Product Attention（点积注意力）

**提出：** Luong et al., 2015《Effective Approaches to Attention-based Neural Machine Translation》

#### 原理

直接用 Q 和 K 的内积衡量相关性，无需额外参数，计算极其高效。

#### 公式

```
score(Q, K) = Q · K^T
α = softmax(Q · K^T)
output = α · V
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 无额外参数，极简 | 维度高时内积值大，softmax 梯度消失 |
| 速度快，可矩阵化 | 没有缩放，数值不稳定 |

#### 适用场景
维度较小（≤64）的场景、快速原型

#### PyTorch 代码示例

```python
class DotProductAttention(nn.Module):
    """Luong Dot-Product Attention"""
    def forward(self, query, keys, values, mask=None):
        """
        query:  (batch, 1, d)
        keys:   (batch, seq_len, d)
        values: (batch, seq_len, d)
        """
        # scores: (batch, 1, seq_len)
        scores = torch.bmm(query, keys.transpose(1, 2))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, values)   # (batch, 1, d)
        return context, weights

# 示例
query  = torch.randn(2, 1, 64)
keys   = torch.randn(2, 10, 64)
attn   = DotProductAttention()
ctx, w = attn(query, keys, keys)
print(ctx.shape, w.shape)   # (2,1,64)  (2,1,10)
```

