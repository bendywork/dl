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

---

### 3.3 Scaled Dot-Product Attention（缩放点积注意力）

**提出：** Vaswani et al., 2017《Attention Is All You Need》

#### 原理

在点积基础上除以 √d_k，将内积方差从 d_k 归一化为 1，防止 softmax 进入饱和区。

#### 公式

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 数值稳定，梯度健康 | 仍是 O(n²) 复杂度 |
| 无额外参数 | 长序列显存爆炸 |
| 可完全并行化 | — |

#### PyTorch 代码示例

```python
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights

# 示例
Q = torch.randn(2, 8, 10, 64)   # (batch, heads, seq, d_k)
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)
attn = ScaledDotProductAttention()
out, w = attn(Q, K, V)
print(out.shape)  # (2, 8, 10, 64)
```

---

### 3.4 Self-Attention（自注意力）

**核心思想：** Q、K、V 全部来自同一序列，让序列中每个位置都能关注自身所有位置。

#### 公式

```
Q = X · W_Q,  K = X · W_K,  V = X · W_V   ← 同一个输入 X
Self-Attention(X) = softmax( XW_Q(XW_K)ᵀ / √d_k ) · XW_V
```

#### 三种使用场景

| 场景 | 说明 | 代表 |
|------|------|------|
| Encoder Self-Attention | 双向，可看全文 | BERT |
| Decoder Self-Attention | 单向（Causal Mask），只看历史 | GPT |
| Cross-Attention | Q来自Decoder，K/V来自Encoder | 翻译模型 |

#### 与 RNN 的对比

```
RNN：信息通过时间步逐步传递，长距离依赖靠梯度反传
Self-Attention：任意两个位置直接交互，路径长度恒为 O(1)
```

| 对比项 | RNN | Self-Attention |
|--------|-----|---------------|
| 长距离依赖 | 难（梯度消失） | 易（直接交互） |
| 并行计算 | 不可并行 | 完全并行 |
| 复杂度 | O(n·d²) | O(n²·d) |
| 序列长度限制 | 无（但效果差） | 受显存限制 |

#### 代码

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x, causal=False):
        seq_len = x.size(1)
        mask = None
        if causal:
            # 上三角 mask，防止看未来
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return out
```

---

### 3.5 Cross-Attention（交叉注意力）

**核心思想：** Q 来自一个序列，K/V 来自另一个序列，实现两个序列之间的信息交互。

#### 公式

```
Q = X_dec · W_Q          ← 来自 Decoder
K = X_enc · W_K          ← 来自 Encoder
V = X_enc · W_V          ← 来自 Encoder
Cross-Attention = softmax( Q·Kᵀ / √d_k ) · V
```

#### 典型应用

| 场景 | Q 来源 | K/V 来源 |
|------|--------|---------|
| 机器翻译 | 目标语言 | 源语言 |
| 图文生成 | 文本 | 图像特征 |
| 语音识别 | 文本解码 | 音频编码 |

---

### 3.6 Local Attention（局部注意力）

**动机：** 全局 Self-Attention 是 O(n²)，长文本无法承受。局部 Attention 每个位置只关注固定窗口 w 内的邻居，复杂度降为 O(n·w)。

#### 两种变体

```
滑动窗口 Attention（Longformer）：
每个 token 关注左右 w/2 个邻居

扩张窗口 Attention（Longformer）：
跳跃式采样，w 不变但感受野更大
```

#### 与全局 Attention 组合使用

```
普通 token：Local Attention（窗口=512）
特殊 token（[CLS]、问题 token）：Global Attention（全文）
```

| 方案 | 复杂度 | 效果 | 代表模型 |
|------|--------|------|---------|
| 全局 Attention | O(n²) | 最好 | BERT、GPT |
| 局部窗口 | O(n·w) | 稍弱 | Longformer |
| 稀疏 Attention | O(n√n) | 接近全局 | BigBird |

---

### 3.7 Flash Attention（IO 感知注意力）

**动机：** 标准 Attention 的瓶颈不是计算量而是显存带宽——score 矩阵写入/读回 HBM 太慢。

#### 核心思想

分块（Tiling）计算，score 矩阵从不完整写到 HBM，只在 SRAM（片上高速缓存）内完成 softmax + 加权求和：

```
标准 Attention：
  compute QKᵀ → 写 HBM → 读回 → softmax → 写 HBM → 读回 → ×V
  显存：O(n²)，带宽：O(n²)

Flash Attention：
  分块计算，在 SRAM 内完成所有操作，结果才写 HBM
  显存：O(n)，带宽：O(n)，速度：2-4x
```

#### 版本演进

| 版本 | 改进 | 加速比 |
|------|------|--------|
| FlashAttention v1 (2022) | 分块 + 在线 softmax | 2-4x |
| FlashAttention v2 (2023) | 并行度优化，减少非矩阵乘操作 | 4-8x |
| FlashAttention v3 (2024) | 针对 H100 异步流水线 | 10x+ |

#### PyTorch 调用

```python
# PyTorch 2.0+ 内置，自动选用 Flash Attention（H100/A100）
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

---

### 3.8 RoPE Attention（旋转位置编码注意力）

**动机：** 绝对位置编码无法外推到训练长度以外；RoPE 把相对位置关系编码进 Q/K 的旋转中。

#### 核心公式

```
q_m = R_m · q,  k_n = R_n · k
q_m · k_n = q · Rᵀ_m · R_n · k = q · R_{n-m} · k
```

内积只依赖相对位置 (n-m)，与绝对位置无关，天然支持长度外推。

#### 代表模型

| 模型 | 位置编码 |
|------|---------|
| BERT | 可学习绝对位置 |
| GPT-2 | 可学习绝对位置 |
| LLaMA / Qwen | RoPE |
| Mistral | RoPE + 滑动窗口 |

---

## 四、全面对比总结

### 4.1 按计算方式分类

| 类型 | 公式核心 | 复杂度 | 代表 |
|------|---------|--------|------|
| 加法 Attention | tanh(W[q;k]) | O(n·d) | Bahdanau |
| 点积 Attention | q·k | O(n·d) | Luong |
| 缩放点积 | q·k/√d_k | O(n·d) | Transformer |
| 多头 | 多组缩放点积拼接 | O(n²·d) | BERT/GPT |
| 局部窗口 | 窗口内缩放点积 | O(n·w·d) | Longformer |
| 稀疏 | 选择性点积 | O(n√n·d) | BigBird |
| Flash | 分块缩放点积 | O(n²·d)计算/O(n)显存 | GPT-4 |

### 4.2 按使用场景分类

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 短文本理解（≤512） | MHA | 效果最好，复杂度可接受 |
| 长文本理解（>4k） | Local + Global | 显存友好 |
| 自回归生成 | Causal MHA + GQA | KV Cache 效率高 |
| 序列到序列（翻译） | Encoder MHA + Cross-Attention | 跨序列信息交互 |
| 超长序列（>100k） | Flash Attention v2+ | 显存 O(n) |

### 4.3 选型决策树

```
你的序列长度是多少？
├── ≤ 512
│   └── 用标准 Multi-Head Attention
├── 512 ~ 4096
│   └── 用 Multi-Head Attention + Flash Attention
├── 4096 ~ 32k
│   ├── 理解任务 → Longformer 滑动窗口
│   └── 生成任务 → GQA + Flash Attention
└── > 32k
    └── Flash Attention v2 + RoPE 外推

你的硬件是什么？
├── A100 / H100 → Flash Attention v2/v3
├── 消费级 GPU（显存<24G）→ GQA + 4bit量化
└── CPU → 加法 Attention（无需矩阵乘加速）
```

### 4.4 演进时间线

```
2015  Bahdanau Attention（加法，NMT）
2015  Luong Attention（点积/拼接，NMT）
2017  Scaled Dot-Product + Multi-Head（Transformer）
2018  Self-Attention 大规模应用（BERT/GPT）
2020  Sparse Attention（BigBird/Longformer，长文本）
2022  Flash Attention v1（IO 感知，显存革命）
2023  GQA（LLaMA2，KV Cache 压缩）
2023  Flash Attention v2（并行优化）
2024  Flash Attention v3（H100 异步流水线）
```

---

## 五、总结一句话

> **Attention 的本质：用 Q 查询 K 得到权重，用权重对 V 加权求和。** 所有变体都在优化这个过程的计算效率、显存占用或表达能力。

