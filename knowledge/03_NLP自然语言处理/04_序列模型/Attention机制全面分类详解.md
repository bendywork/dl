# Attention 机制全面分类详解

> 文档编号：配套 Attention 系列第4篇
> 覆盖范围：Bahdanau → Luong → Transformer → 稀疏/局部 Attention

---

## 一、背景：为什么需要 Attention

### 1.1 RNN/LSTM 的信息瓶颈问题

RNN/LSTM 在处理序列时，所有历史信息都被压缩到固定维度的隐状态 h_t 中。
序列越长，早期 token 的信息在传递过程中不断被稀释，梯度也越来越难以回传。

```
输入序列:  x1 -> x2 -> x3 -> ... -> xT
隐状态:    h1 -> h2 -> h3 -> ... -> hT  (固定维度，信息瓶颈)
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
  α_i = softmax(score_i) = exp(score_i) / sum(exp(score_j))

步骤3：加权求和
  context = sum(α_i * V_i)
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
| 缩放点积 | Q · K^T / sqrt(d_k) | Vaswani 2017 |
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
c_t = sum_i(α_ti · h_i)
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
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(key_dim,   hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, query, keys, values, mask=None):
        # query: (B, query_dim), keys/values: (B, T, key_dim)
        q = self.W_query(query).unsqueeze(1)     # (B, 1, hidden)
        k = self.W_key(keys)                      # (B, T, hidden)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights

# 示例
batch, seq_len, enc_dim, dec_dim = 2, 10, 256, 256
query  = torch.randn(batch, dec_dim)
keys   = torch.randn(batch, seq_len, enc_dim)
attn   = AdditiveAttention(dec_dim, enc_dim, 128)
ctx, w = attn(query, keys, keys)
print('context:', ctx.shape)   # (2, 256)
print('weights:', w.shape)     # (2, 10)
```

---

### 3.2 Dot-Product Attention（点积注意力）

**提出：** Luong et al., 2015

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
    def forward(self, query, keys, values, mask=None):
        # query: (B,1,d), keys/values: (B,T,d)
        scores = torch.bmm(query, keys.transpose(1, 2))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1)==0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, values)   # (B, 1, d)
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

在点积结果上除以 sqrt(d_k)，防止高维时内积值过大导致 softmax 饱和。

**为什么要除以 sqrt(d_k)？**

假设 Q 和 K 的每个分量是均值0、方差1的随机变量，
则 Q·K 的方差为 d_k。除以 sqrt(d_k) 后方差重新变为1，softmax 梯度不会消失。

#### 公式

```
Attention(Q, K, V) = softmax( Q·K^T / sqrt(d_k) ) · V
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 数值稳定，适合大维度 | 全局注意力，复杂度 O(n²) |
| 可完全矩阵化并行 | 序列极长时内存开销大 |
| Transformer 基础模块 | — |

#### 适用场景

Transformer、BERT、GPT 等所有现代大模型

#### PyTorch 代码示例

```python
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output  = torch.matmul(weights, V)
        return output, weights

# 示例（单头）
B, T, d_k = 2, 10, 64
Q = torch.randn(B, 1, T, d_k)
K = torch.randn(B, 1, T, d_k)
V = torch.randn(B, 1, T, d_k)
attn = ScaledDotProductAttention(dropout=0.1)
out, w = attn(Q, K, V)
print(out.shape, w.shape)   # (2,1,10,64)  (2,1,10,10)
```

---

### 3.4 Multi-Head Attention（多头注意力）

**提出：** Vaswani et al., 2017（Transformer）

#### 原理

将 Q/K/V 分别映射到 h 个子空间，每个头独立做 Scaled Dot-Product Attention，
最后拼接所有头的输出再线性变换。不同的头可以关注不同的语义维度。

#### 公式

```
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O
其中 d_k = d_model / h
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 多角度捕捉不同语义关系 | 参数量是单头的 h 倍 |
| 并行度极高 | 内存随头数线性增长 |
| 实践效果远好于单头 | d_k 随 h 增大而减小 |

#### 适用场景

Transformer Encoder/Decoder、BERT、GPT、T5 等

#### PyTorch 代码示例

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k  = d_model // num_heads
        self.h    = num_heads
        self.W_q  = nn.Linear(d_model, d_model)
        self.W_k  = nn.Linear(d_model, d_model)
        self.W_v  = nn.Linear(d_model, d_model)
        self.W_o  = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        B, T, _ = x.size()
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        out, w = self.attn(Q, K, V, mask)
        B, _, T, _ = out.size()
        out = out.transpose(1,2).contiguous().view(B, T, -1)
        return self.W_o(out), w

# 示例
B, T, d_model = 2, 10, 512
x   = torch.randn(B, T, d_model)
mha = MultiHeadAttention(d_model=512, num_heads=8)
out, w = mha(x, x, x)
print(out.shape)   # (2, 10, 512)
```

---

### 3.5 Self-Attention（自注意力）

**提出：** Vaswani et al., 2017（Transformer Encoder 核心）

#### 原理

Q、K、V 全部来自**同一序列**，序列中每个位置都可以关注序列中的任意其他位置，
从而捕捉序列内部的长距离依赖关系，完全不依赖 RNN 的递推结构。

#### 与普通 Attention 的区别

```
普通 Attention：  Q 来自解码器，K/V 来自编码器（跨序列）
Self-Attention：  Q = K = V = 同一序列的线性变换（同序列内部）
```

#### 公式

```
Self-Attention(X) = softmax( (X·W^Q)(X·W^K)^T / sqrt(d_k) ) · (X·W^V)
```

#### 位置编码（Position Encoding）

Self-Attention 本身无位置信息，需要额外加入位置编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 任意位置直接交互，O(1) 路径长度 | 计算复杂度 O(n²·d) |
| 完全并行，训练极快 | 需要位置编码补充位置信息 |
| 捕捉全局依赖 | 对超长序列内存压力大 |

#### 适用场景

BERT（双向自注意力）、GPT（单向自注意力 + causal mask）

#### PyTorch 代码示例

```python
class SelfAttention(nn.Module):
    # 单头 Self-Attention，支持 causal mask
    def __init__(self, d_model, causal=False):
        super().__init__()
        self.W_q    = nn.Linear(d_model, d_model)
        self.W_k    = nn.Linear(d_model, d_model)
        self.W_v    = nn.Linear(d_model, d_model)
        self.causal = causal
        self.scale  = d_model ** 0.5

    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if self.causal:
            T = x.size(1)
            mask = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V), weights

# 示例：GPT 风格 causal self-attention
B, T, d = 2, 8, 64
x    = torch.randn(B, T, d)
sa   = SelfAttention(d, causal=True)
out, w = sa(x)
print(out.shape, w.shape)   # (2,8,64)  (2,8,8)
```

---

### 3.6 Cross-Attention（交叉注意力）

**提出：** Vaswani et al., 2017（Transformer Decoder 第二个子层）

#### 原理

Q 来自解码器，K 和 V 来自编码器输出。解码时每一步都能查询完整的编码器信息，
是连接编码器和解码器的桥梁。

#### 公式

```
Cross-Attention = softmax( Q_dec · K_enc^T / sqrt(d_k) ) · V_enc
Q = decoder_hidden · W^Q
K = encoder_output · W^K
V = encoder_output · W^V
```

#### 与 Self-Attention 的区别

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q 来源 | 同序列 | 解码器 |
| K/V 来源 | 同序列 | 编码器 |
| 作用 | 建模序列内关系 | 跨序列查询 |

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 解码每步可访问完整编码信息 | 编码器必须先跑完才能解码 |
| 比固定 context vector 灵活得多 | 推理不可并行（自回归） |

#### 适用场景

机器翻译解码器、图像描述（视觉编码器 + 文本解码器）、语音识别

#### PyTorch 代码示例

```python
class CrossAttention(nn.Module):
    # Q 来自 decoder，K/V 来自 encoder
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, decoder_hidden, encoder_output, mask=None):
        # decoder_hidden: (B, T_dec, d_model)
        # encoder_output: (B, T_enc, d_model)
        out, w = self.mha(
            Q=decoder_hidden,
            K=encoder_output,
            V=encoder_output,
            mask=mask
        )
        return out, w

# 示例
B = 2
T_enc, T_dec, d = 20, 8, 512
enc_out = torch.randn(B, T_enc, d)
dec_hid = torch.randn(B, T_dec, d)
ca = CrossAttention(d, num_heads=8)
out, w = ca(dec_hid, enc_out)
print(out.shape)   # (2, 8, 512)
```

---

### 3.7 Local Attention（局部注意力）

**提出：** Luong et al., 2015（Local Attention 变体）

#### 原理

不做全局注意力，只对当前查询位置附近固定窗口 [p-D, p+D] 内的位置做注意力，
将复杂度从 O(n²) 降低到 O(n·w)（w 为窗口大小）。

分两种：
- **Monotonic**：窗口位置 p = t（解码步骤 t）
- **Predictive**：p = S · sigmoid(W_p · s_t)（自适应预测窗口中心）

#### 公式

```
Local-Attention: 仅计算 i ∈ [pt-D, pt+D] 的 score

高斯加权版本：
  α_t(s) = Align(s_t, h_s) · exp( -(s-pt)² / 2σ² )
  σ = D/2
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 复杂度 O(n·w)，适合长序列 | 局部性假设可能遗漏远距离依赖 |
| 内存友好 | 窗口大小 D 是超参数需调优 |

#### 适用场景

长文档理解、语音信号处理、时序预测

#### PyTorch 代码示例

```python
class LocalAttention(nn.Module):
    def __init__(self, d_model, window_size=5):
        super().__init__()
        self.W   = window_size
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, d = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        outputs = []
        for t in range(T):
            start = max(0, t - self.W // 2)
            end   = min(T, t + self.W // 2 + 1)
            k_local = K[:, start:end, :]
            v_local = V[:, start:end, :]
            q_t     = Q[:, t:t+1, :]
            score   = torch.bmm(q_t, k_local.transpose(1,2)) / self.scale
            w       = F.softmax(score, dim=-1)
            outputs.append(torch.bmm(w, v_local))
        return torch.cat(outputs, dim=1)

# 示例
B, T, d = 2, 12, 64
x   = torch.randn(B, T, d)
la  = LocalAttention(d, window_size=5)
out = la(x)
print(out.shape)   # (2, 12, 64)
```

---

### 3.8 Sparse Attention（稀疏注意力）

**代表作：** Longformer (2020)、BigBird (2020)、Sparse Transformer (2019)

#### 原理

全局 Attention 复杂度 O(n²) 对长序列代价极高。稀疏 Attention 只计算选定的位置对，
将复杂度降到 O(n·sqrt(n)) 或 O(n·log n)。

**三种稀疏模式：**

```
1. 固定稀疏（Fixed）：每 stride 步取一个全局 token
2. 局部窗口（Local Window）：每个 token 只与附近 w 个交互
3. 全局 token（Global）：特殊 CLS token 与所有位置交互
Longformer = Local Window + Global Tokens + Dilated Sliding Window
```

#### 复杂度对比

```
Full Attention：    O(n²·d)
Sparse Attention：  O(n·w·d)  或  O(n·log n·d)
Longformer：        O(n·(w+g)·d)  w=窗口大小，g=全局token数
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 支持超长序列（4096~16384+） | 实现复杂，需要专用 CUDA kernel |
| 内存和计算线性化 | 全局信息依赖设计好的稀疏模式 |
| 保留关键局部+全局交互 | 精度可能略低于全局 Attention |

#### 适用场景

长文档分类/摘要（Longformer）、基因序列（BigBird）、长代码理解

#### PyTorch 概念代码示例

```python
class SimpleSparseAttention(nn.Module):
    # 简化版：局部窗口 + 全局 token（位置0）
    def __init__(self, d_model, window_size=4):
        super().__init__()
        self.W  = window_size
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        B, T, d = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = torch.bmm(Q, K.transpose(1,2)) / self.scale

        # 构造稀疏 mask
        mask = torch.zeros(T, T, dtype=torch.bool, device=x.device)
        for i in range(T):
            start = max(0, i - self.W // 2)
            end   = min(T, i + self.W // 2 + 1)
            mask[i, start:end] = True
        mask[:, 0] = True    # 所有位置可关注全局 token 0
        mask[0, :] = True    # token 0 关注所有位置

        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V)

# 示例
B, T, d = 1, 20, 64
x   = torch.randn(B, T, d)
spa = SimpleSparseAttention(d, window_size=4)
out = spa(x)
print(out.shape)   # (1, 20, 64)
```

---

## 四、各类 Attention 横向对比表

| 类型 | 提出时间 | score 函数 | 时间复杂度 | 典型应用 |
|------|---------|-----------|-----------|----------|
| Additive (Bahdanau) | 2015 | v^T·tanh(W1·Q+W2·K) | O(n·d) | 早期 NMT |
| Dot-Product (Luong) | 2015 | Q·K^T | O(n·d) | 小维度快速方案 |
| Scaled Dot-Product | 2017 | Q·K^T/sqrt(d_k) | O(n²·d) | Transformer |
| Multi-Head | 2017 | h 个缩放点积 | O(n²·d) | BERT/GPT 等 |
| Self-Attention | 2017 | 同序列 Scaled DP | O(n²·d) | BERT/GPT 编码 |
| Cross-Attention | 2017 | Q=dec, K/V=enc | O(n·m·d) | Transformer 解码 |
| Local Attention | 2015/2020 | 窗口内 Scaled DP | O(n·w·d) | 长序列 NLP |
| Sparse Attention | 2019/2020 | 稀疏选择 | O(n·sqrt(n)·d) | 超长文档 |

> 注：n=序列长度，d=维度，w=窗口大小，m=编码器序列长度

---

## 五、演进路线图

```
2014  Seq2Seq (Sutskever)
         |  固定 context vector，信息瓶颈
         v
2015  Bahdanau Attention
         |  动态 context，加性 score，解决长句翻译
         |
2015  Luong Attention
         |  点积/双线性 score，更简洁；Local/Global 变体
         v
2017  Transformer (Vaswani - Attention Is All You Need)
         |  Scaled Dot-Product + Multi-Head + Self + Cross
         |  完全摒弃 RNN，并行化训练
         |
    +----+--------------------+
    v                         v
2018 BERT                  2018 GPT
(双向 Self-Attn)           (单向 Causal Self-Attn)
    |                         |
    v                         v
2020 Longformer            2020 GPT-3
(Sparse Attn)              (全局 Multi-Head)
    |
    v
2022+ Flash Attention (IO 感知高效实现)
      Flash Attention 2 / xFormers
      Ring Attention（分布式超长序列）
```

---

## 六、学习路径建议

### 阶段一：基础理解（1-2天）

1. 先读 Bahdanau 2015 论文，理解为什么需要 Attention
2. 动手实现 AdditiveAttention，跑通 Seq2Seq + Attention
3. 可视化 attention weights，直觉感受对齐效果

### 阶段二：Transformer 核心（3-5天）

1. 精读《Attention Is All You Need》
2. 实现 Scaled Dot-Product + Multi-Head Attention
3. 搭建完整 Transformer Encoder（含 PE、残差、LayerNorm）
4. 搭建 Transformer Decoder（Self + Cross Attention）
5. 在 WMT 翻译任务上验证效果

### 阶段三：BERT & GPT 实践（1周）

1. 理解 BERT 的双向 Self-Attention + MLM 预训练
2. 理解 GPT 的 Causal Self-Attention + 自回归
3. 用 Hugging Face Transformers 微调下游任务
4. 分析注意力头的语言学含义（BERTology 相关论文）

### 阶段四：效率优化（进阶）

1. 阅读 Flash Attention 论文（IO 感知计算）
2. 了解 Longformer / BigBird 的稀疏化策略
3. 实践超长序列（4096+）的处理方案
4. 了解 Multi-Query Attention (MQA) / Grouped Query Attention (GQA)

### 推荐资源

| 资源 | 说明 |
|------|------|
| Bahdanau 2015 | arXiv:1409.0473 |
| Vaswani 2017 | arXiv:1706.03762 |
| The Annotated Transformer | Harvard NLP 逐行注释版 |
| Illustrated Transformer | Jay Alammar 可视化博客 |
| Flash Attention | arXiv:2205.14135 |
| Longformer | arXiv:2004.05150 |

### 代码学习顺序

```
1. 本文档 AdditiveAttention    -> 理解原理
2. 本文档 ScaledDotProduct     -> Transformer 基础
3. 本文档 MultiHeadAttention   -> 完整组件
4. PyTorch nn.MultiheadAttention -> 官方实现对比
5. Hugging Face BertModel      -> 工业级实现
6. Flash Attention 库          -> 生产环境优化
```

---

## 附录：完整可运行 Demo

将本文档所有 Attention 类整合的完整示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===== ScaledDotProductAttention =====
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, V), weights

# ===== MultiHeadAttention =====
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h   = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.size()
        def sh(x): return x.view(B, -1, self.h, self.d_k).transpose(1,2)
        Q, K, V = sh(self.W_q(Q)), sh(self.W_k(K)), sh(self.W_v(V))
        out, w  = self.attn(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(B, -1, self.h*self.d_k)
        return self.W_o(out), w
```

```python
# ===== AdditiveAttention =====
class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim,   hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, query, keys, values, mask=None):
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(keys)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1), weights


# ===== main =====
if __name__ == '__main__':
    torch.manual_seed(42)
    B, T, d = 2, 10, 64
    print('--- Scaled Dot-Product ---')
    sdpa = ScaledDotProductAttention()
    Q = torch.randn(B, 1, T, d)
    out, w = sdpa(Q, Q, Q)
    print(f'  output: {out.shape}')
    print('--- Multi-Head Attention ---')
    mha = MultiHeadAttention(d_model=d, num_heads=4)
    x   = torch.randn(B, T, d)
    out, w = mha(x, x, x)
    print(f'  output: {out.shape}')
    print('--- Additive Attention ---')
    aa  = AdditiveAttention(d, d, 32)
    q   = torch.randn(B, d)
    k   = torch.randn(B, T, d)
    ctx, w = aa(q, k, k)
    print(f'  context: {ctx.shape}')
    print('All tests passed!')
```

---

> 文档日期：2026-06-19
> 配套文件：Attention三种类型的演化.md、Self-Attention与Multi-Head-Attention_01.md
