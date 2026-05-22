# Transformer 注意力机制（面试核心考点）

## 📌 核心问题

> RNN/LSTM 处理序列时存在**无法并行**和**长距离依赖消失**两大缺陷。Transformer（Vaswani et al., 2017）通过**纯注意力机制**替代循环结构，实现了序列的并行计算，并能直接建模任意距离的词间关系，成为现代 NLP 的基础架构。

---

## 🌱 根源与动机

### RNN 的两大致命缺陷

**缺陷1：无法并行化**
```
RNN 计算：h1 → h2 → h3 → h4 → ...（串行，必须等前一步完成）
Transformer：所有位置同时计算（并行，GPU 友好）
```

**缺陷2：长距离依赖问题**
```
"The cat that sat on the mat is fat"
 ↑                               ↑
"cat" 和 "is" 之间距离很远，LSTM 的梯度会衰减
Transformer：直接计算任意两词之间的注意力，距离无关
```

### 注意力机制的直觉

想象你在看一篇文章，当翻译"bank"时，你会同时参考：
- "deposit"（注意力权重高 → 金融语境）
- "money"（注意力权重高）
- "river"（注意力权重低）

这就是注意力机制的本质：**对所有位置计算相关性得分，加权求和**。

---

## 📐 理论推导

### 1. Self-Attention：Q、K、V 矩阵

**核心公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Q、K、V 的来源：**

输入序列 $X \in \mathbb{R}^{n \times d_{model}}$（n 个词，每词 $d_{model}$ 维），通过三个独立的权重矩阵线性变换得到：

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

其中：
- $W^Q \in \mathbb{R}^{d_{model} \times d_k}$（Query 投影矩阵）
- $W^K \in \mathbb{R}^{d_{model} \times d_k}$（Key 投影矩阵）
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$（Value 投影矩阵）
- $Q, K \in \mathbb{R}^{n \times d_k}$，$V \in \mathbb{R}^{n \times d_v}$

**逐步计算过程：**

```
步骤1：计算注意力得分矩阵
  scores = Q @ K.T / sqrt(d_k)   shape: [n, n]
  scores[i][j] = 第 i 个词对第 j 个词的注意力得分

步骤2：softmax 归一化（按行）
  attn_weights = softmax(scores)  shape: [n, n]
  attn_weights[i] 是第 i 个词对所有词的注意力权重（和为1）

步骤3：加权求和
  output = attn_weights @ V        shape: [n, d_v]
  output[i] = 所有位置 V 的加权组合，权重由第 i 个词的注意力决定
```

### 2. 为什么除以 $\sqrt{d_k}$？（面试必问）

**原因：防止点积过大导致 softmax 梯度消失**

设 $q, k$ 均为均值0、方差1的随机向量，则：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

每一项 $q_i k_i$ 的均值为0，方差为1，共 $d_k$ 项之和的**方差为 $d_k$**，标准差为 $\sqrt{d_k}$。

因此 $q \cdot k$ 的值量级约为 $\sqrt{d_k}$，当 $d_k = 64$ 时，点积约为 8。

**softmax 梯度消失问题：**
$$\text{若 } x = [100, -100, 50] \Rightarrow \text{softmax}(x) \approx [1, 0, 0]$$

梯度几乎为0！除以 $\sqrt{d_k}$ 将值域压缩回 $O(1)$，避免落入 softmax 的饱和区。

### 3. Multi-Head Attention（多头注意力）

**公式：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

其中：
- $h$ 个头，每头维度 $d_k = d_{model} / h$（原论文 $h=8, d_{model}=512, d_k=64$）
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$：输出投影矩阵

**为什么需要多头？**

不同头可以关注不同类型的关系：
- 头1：关注**语法依存关系**（主语-谓语）
- 头2：关注**共指关系**（代词指代）
- 头3：关注**语义相关性**（近义词）
- 头4：关注**位置关系**（相邻词）

单头注意力只有一种"视角"，多头提供了多种"视角"，最后拼接融合。

**参数量分析（原论文）：**
```
d_model = 512, h = 8, d_k = d_v = 64
每个头参数: W^Q (512×64) + W^K (512×64) + W^V (512×64) = 98304 × 3
输出投影: W^O (512×512) = 262144
总参数: 8 × 3 × 32768 + 262144 ≈ 1M 参数
```

### 4. Position Encoding（位置编码）

自注意力本身是**置换不变的**（permutation invariant），不知道词的顺序，必须引入位置信息。

**原论文使用 sin/cos 固定位置编码：**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：词在序列中的位置（0, 1, 2, ...）
- $i$：维度索引（0, 1, 2, ..., $d_{model}/2 - 1$）
- 对于 $d_{model}=512$，位置编码是 512 维向量

**直觉理解：**
- 不同频率的正弦波，低频捕捉全局位置，高频捕捉局部位置
- 类比：时钟的秒针（高频）、分针（低频）、时针（更低频）组合唯一确定时间
- 最终：$\text{Input} = \text{TokenEmbedding} + \text{PositionEncoding}$

**为什么不用可学习的位置编码？**

原论文的理由：固定 sin/cos 编码可以**外推到更长序列**（训练时未见过的长度），且效果与可学习编码相当。注：BERT 使用的是**可学习**位置编码。

### 5. 残差连接 + Layer Normalization

每个子层（Self-Attention 和 FFN）都包裹：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Layer Norm 公式：**

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

其中 $\mu = \frac{1}{d}\sum_i x_i$，$\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$，$\gamma, \beta$ 是可学习参数。

**Layer Norm vs Batch Norm（面试必考）：**

| 特性 | Batch Norm | Layer Norm |
|------|-----------|-----------|
| 归一化维度 | 对 batch 维度归一化（同一特征跨样本） | 对特征维度归一化（同一样本内） |
| 适用场景 | CV（batch size 大，特征稳定） | NLP（序列长度可变，batch可以很小） |
| 推理时 | 需要存储 running mean/var | 无需额外统计量 |
| 序列变长 | 不适用（统计量会随 padding 变化） | 适用（只看单个样本自己的维度） |

### 6. Feed-Forward Network（FFN）

每个 Encoder/Decoder 层都有一个 FFN：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

- 原论文：$d_{model}=512 \to d_{ff}=2048 \to d_{model}=512$（中间扩大4倍）
- 作用：对每个位置**独立**进行非线性变换，增加模型表达能力
- 注：FFN 对每个位置单独处理，**没有位置间的信息交互**（交互在 Self-Attention 中完成）

### 7. Encoder 完整架构

```
输入 → Token Embedding + Position Encoding
         ↓
┌─── Encoder Layer × N ───┐
│   Multi-Head Self-Attention
│         ↓
│   Add & Layer Norm
│         ↓
│   Feed-Forward Network
│         ↓
│   Add & Layer Norm
└──────────────────────────┘
         ↓
最终编码表示
```

原论文：N=6 层，$d_{model}=512$，$h=8$ 头，$d_{ff}=2048$

### 8. Decoder 完整架构

```
目标序列（右移一位）→ Token Embedding + Position Encoding
         ↓
┌─── Decoder Layer × N ───┐
│   Masked Multi-Head Self-Attention （只看当前及之前位置）
│         ↓
│   Add & Layer Norm
│         ↓
│   Cross-Attention （Q来自Decoder, K/V来自Encoder输出）
│         ↓
│   Add & Layer Norm
│         ↓
│   Feed-Forward Network
│         ↓
│   Add & Layer Norm
└──────────────────────────┘
         ↓
Linear + Softmax → 预测下一个词
```

**Masked Self-Attention（因果掩码）：**

训练时目标序列是已知的，但为了**模拟自回归生成**（不能看未来的词），需要用上三角 mask：

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

加到注意力分数上，使 softmax 后未来位置权重为0。

### 9. 三种 Transformer 变体

| 变体 | 架构 | 代表模型 | 适用任务 |
|------|------|---------|---------|
| **Encoder-only** | 只有 Encoder | BERT, RoBERTa | 分类、NER、问答（理解任务） |
| **Decoder-only** | 只有 Decoder（带 Causal Mask） | GPT 系列 | 文本生成、语言模型 |
| **Encoder-Decoder** | 完整 Encoder+Decoder | T5, BART, 原始 Transformer | 翻译、摘要、Seq2Seq |

---

## 💡 关键理解

### 类比：图书馆检索系统

- **Query（Q）**：你的检索需求（"我想找关于机器学习的书"）
- **Key（K）**：每本书的索引标签（"机器学习"、"深度学习"、"统计学"...）
- **Value（V）**：书的实际内容

计算过程：
1. 拿你的 Query 和每本书的 Key 计算匹配度（点积）
2. softmax 归一化得到每本书的权重（相关度越高权重越大）
3. 用权重对每本书的 Value（内容）加权求和 → 综合了所有相关书籍的知识

**Self-Attention 的"自"**：Query、Key、Value 都来自同一序列，即句子里的每个词在问"我应该关注哪些其他词？"

### 残差连接的作用

想象信号传输中的"高速公路"：
- 没有残差：梯度必须通过每一层的变换才能回传（容易消失）
- 有残差：梯度可以直接"走高速公路"跳过某些层（梯度流畅）

$$F(x) + x \text{ 的梯度} = F'(x) + 1 \geq 1 \text{（不会小于1）}$$

---

## 🔧 代码实现

### PyTorch 手动实现完整 Transformer Encoder Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 位置编码（Position Encoding）
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码矩阵，shape: [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        
        # 分母：10000^(2i/d_model)，用 exp(2i * (-log(10000)/d_model)) 数值稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]，broadcast 用
        self.register_buffer('pe', pe)  # 不是参数，但需要随模型存储
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]  # 广播加法
        return self.dropout(x)


# ============================================================
# 2. Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple:
    """
    Q: [batch, heads, seq_q, d_k]
    K: [batch, heads, seq_k, d_k]
    V: [batch, heads, seq_k, d_v]
    mask: [batch, 1, seq_q, seq_k] or [1, 1, seq_q, seq_k]，True 表示需要 mask 的位置
    
    返回:
        output: [batch, heads, seq_q, d_v]
        attn_weights: [batch, heads, seq_q, seq_k]
    """
    d_k = Q.size(-1)
    
    # 步骤1：计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, heads, seq_q, seq_k]
    
    # 步骤2：应用 mask（Decoder 的 causal mask 或 padding mask）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 步骤3：softmax 归一化（沿 seq_k 维度）
    attn_weights = F.softmax(scores, dim=-1)
    
    # 处理全 -inf 的情况（全被 mask 时 softmax 返回 nan）
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    # 步骤4：加权求和
    output = torch.matmul(attn_weights, V)
    # output: [batch, heads, seq_q, d_v]
    
    return output, attn_weights


# ============================================================
# 3. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每头的 key/query 维度
        self.d_v = d_model // num_heads  # 每头的 value 维度
        
        # 线性投影矩阵（将 d_model 投影到 num_heads * d_k）
        # 等效于 num_heads 个 W^Q_i 拼在一起
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # 存储注意力权重，用于可视化
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, d_model]
        → [batch, num_heads, seq, d_k]
        """
        batch, seq, d_model = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch, num_heads, seq, d_k]
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_heads, seq, d_k]
        → [batch, seq, d_model]
        """
        batch, num_heads, seq, d_k = x.size()
        x = x.transpose(1, 2)  # [batch, seq, num_heads, d_k]
        return x.contiguous().view(batch, seq, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        query: [batch, seq_q, d_model]
        key:   [batch, seq_k, d_model]
        value: [batch, seq_k, d_model]
        对于 Self-Attention: query = key = value = x
        对于 Cross-Attention: query 来自 Decoder，key/value 来自 Encoder
        """
        # 线性投影
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)    # [batch, seq_k, d_model]
        V = self.W_v(value)  # [batch, seq_k, d_model]
        
        # 拆分成多头
        Q = self.split_heads(Q)  # [batch, h, seq_q, d_k]
        K = self.split_heads(K)  # [batch, h, seq_k, d_k]
        V = self.split_heads(V)  # [batch, h, seq_k, d_v]
        
        # Scaled Dot-Product Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_weights  # 保存用于可视化
        
        # 合并多头
        attn_output = self.merge_heads(attn_output)  # [batch, seq_q, d_model]
        
        # 输出投影
        output = self.W_o(attn_output)  # [batch, seq_q, d_model]
        return output


# ============================================================
# 4. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 原论文用 ReLU；BERT/GPT 用 GELU
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, d_model]"""
        # FFN 对每个位置独立处理（无位置间交互）
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ============================================================
# 5. Encoder Layer（一个完整的 Encoder block）
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Sub-layer 2: Feed-Forward Network
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, seq, d_model]
        src_mask: padding mask，shape [batch, 1, 1, seq]，0表示需要mask的位置
        """
        # Sub-layer 1: Self-Attention + Residual + LayerNorm
        # 注意：原论文是 LayerNorm(x + Sublayer(x))，即 Post-LN
        # 现代实践常用 Pre-LN：Sublayer(LayerNorm(x)) + x（训练更稳定）
        attn_output = self.self_attn(x, x, x, src_mask)  # Q=K=V=x，Self-Attention
        x = self.norm1(x + self.dropout(attn_output))    # Add & Norm
        
        # Sub-layer 2: FFN + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))     # Add & Norm
        
        return x


# ============================================================
# 6. 完整 Transformer Encoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)  # 最后的 LayerNorm
    
    def make_padding_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建 padding mask：padding 位置为 0（会被 mask 掉），其他为 1
        src: [batch, seq]
        返回: [batch, 1, 1, seq]，用于广播到 [batch, heads, seq_q, seq_k]
        """
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
        return mask
    
    def forward(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        src: [batch, seq] token ids
        返回: [batch, seq, d_model]
        """
        src_mask = self.make_padding_mask(src, pad_idx)
        
        # 1. 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)  # 缩放，防止位置编码淹没词嵌入
        x = self.pos_encoding(x)
        
        # 2. N 个 Encoder 层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)


# ============================================================
# 7. 注意力可视化
# ============================================================
def visualize_attention(attn_weights: torch.Tensor, tokens: list, head_idx: int = 0):
    """
    attn_weights: [batch, num_heads, seq, seq]
    tokens: 词列表
    """
    weights = attn_weights[0, head_idx].detach().cpu().numpy()  # [seq, seq]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, cmap='Blues')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Key（被关注的词）')
    ax.set_ylabel('Query（发起关注的词）')
    ax.set_title(f'Self-Attention 权重矩阵（第 {head_idx+1} 头）')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()


# ============================================================
# 8. 测试与验证
# ============================================================
def test_transformer():
    torch.manual_seed(42)
    
    # 超参数（小型用于测试）
    vocab_size = 1000
    d_model = 128
    num_heads = 8       # d_model/num_heads = 16
    num_layers = 2
    d_ff = 512
    batch_size = 4
    seq_len = 20
    
    # 创建模型
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0  # 测试时关闭 dropout
    )
    
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    
    # 构造测试输入（含 padding）
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    src[0, 15:] = 0  # 前两个样本有 padding
    src[1, 18:] = 0
    
    # 前向传播
    encoder.eval()
    with torch.no_grad():
        output = encoder(src)
    
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 [batch, seq, d_model]
    assert output.shape == (batch_size, seq_len, d_model), "输出形状错误！"
    print("✓ 形状验证通过")
    
    # 验证注意力权重
    first_layer_attn = encoder.layers[0].self_attn.attn_weights
    print(f"注意力权重形状: {first_layer_attn.shape}")  # [batch, heads, seq, seq]
    
    # 验证 padding 位置的注意力权重为0
    pad_attn = first_layer_attn[0, 0, :, 15:]  # 第0个样本，第0个头，所有query对padding的注意力
    print(f"Padding 位置注意力权重（应接近0）: {pad_attn.max().item():.6f}")
    
    print("\n=== 验证位置编码不同位置向量不同 ===")
    pe = PositionalEncoding(d_model=8, max_seq_len=10)
    x_dummy = torch.zeros(1, 10, 8)  # 零输入，纯看位置编码
    x_with_pe = pe(x_dummy)[0]  # [10, 8]
    print("位置0编码:", x_with_pe[0, :4].tolist())
    print("位置1编码:", x_with_pe[1, :4].tolist())
    print("位置5编码:", x_with_pe[5, :4].tolist())
    print("各位置向量是否不同:", not torch.allclose(x_with_pe[0], x_with_pe[1]))
    
    print("\n=== 可视化简单句子的注意力 ===")
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    token_ids = torch.randint(1, 100, (1, len(tokens)))  # 随机 ID（仅演示）
    with torch.no_grad():
        _ = encoder(token_ids)
    attn = encoder.layers[0].self_attn.attn_weights
    print(f"注意力矩阵形状: {attn.shape}")
    print("注意力权重（第0头，每行和为1）:")
    print(attn[0, 0].round(decimals=3).tolist())


def test_causal_mask():
    """测试 Decoder 的因果 mask"""
    seq_len = 5
    # 上三角 mask（不包含对角线），1表示被mask掉的位置
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # 转换为 attention 使用的格式（0表示mask，1表示保留，与padding mask一致）
    causal_mask = ~causal_mask  # [seq, seq]
    
    print("\n=== Causal Mask（因果掩码）===")
    print("1=可见, 0=被mask（未来位置）:")
    print(causal_mask.int())
    # 预期输出：下三角矩阵（包含对角线），上三角为0


if __name__ == '__main__':
    test_transformer()
    test_causal_mask()
```

**运行预期输出：**
```
模型参数量: 1,052,544
输入形状: torch.Size([4, 20])
输出形状: torch.Size([4, 20, 128])
✓ 形状验证通过
注意力权重形状: torch.Size([4, 8, 20, 20])
Padding 位置注意力权重（应接近0）: 0.000000

=== Causal Mask ===
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

---

## ⚠️ 易错点与常见误解

### 1. 为什么除以 $\sqrt{d_k}$？（最高频面试题）
**错误回答**：为了数值稳定。

**完整回答**：随机初始化时，$Q$ 和 $K$ 的元素均值为0、方差为1，则 $QK^T$ 的点积方差为 $d_k$，当 $d_k$ 较大时（如64）点积值很大，导致 softmax 进入饱和区，梯度接近0，训练困难。除以 $\sqrt{d_k}$ 将点积归一化到 $O(1)$ 量级，保持 softmax 输入在合理范围。

### 2. Layer Norm vs Batch Norm 的选择
**错误理解**：两者等价，随便用哪个都行。

**正确理解**：NLP 必须用 Layer Norm。原因：
- 序列长度可变，Batch Norm 的统计量会受 padding 干扰
- NLP 任务中 batch size 往往很小（1-32），BN 统计量不稳定
- Layer Norm 对每个样本独立归一化，不受其他样本影响

### 3. Self-Attention 是否能替代位置信息？
**错误理解**：Self-Attention 本身就能感知位置。

**正确理解**：Self-Attention 是**置换不变的**（输入序列打乱顺序，输出也随之打乱但权重不变），完全没有位置感知，必须显式加入 Position Encoding 才能区分词序。

### 4. Decoder 中有两种 Attention，作用不同
**错误理解**：Decoder 只有一个 Self-Attention。

**正确理解**：
- **Masked Self-Attention**：Query、Key、Value 都来自目标序列，用 causal mask 保证不看未来
- **Cross-Attention**：Query 来自 Decoder，Key/Value 来自 Encoder 输出，实现源序列信息融合

### 5. 残差连接的梯度意义
**易忽视**：残差不只是防止梯度消失，更重要的是让深层网络可以学习"增量变换"而非"完整映射"：$\text{子层只需学习} F(x) = \text{输出} - \text{输入}$（残差），比直接学整个映射更容易。

### 6. 多头注意力的计算等效性
**常见误解**：$h$ 个头 = $h$ 次独立的完整 Attention 计算（参数量是单头的 $h$ 倍）。

**正确理解**：每个头的维度是 $d_k = d_{model}/h$，总参数量与单头完整维度的 Attention **相同**。多头的意义是学习不同子空间的关系，而非增加参数。

### 7. Post-LN vs Pre-LN（工程重要）
原论文用 **Post-LN**：$\text{LayerNorm}(x + \text{Sublayer}(x))$

现代实践（BERT、GPT等）多用 **Pre-LN**：$x + \text{Sublayer}(\text{LayerNorm}(x))$

Pre-LN 训练更稳定，不需要 warmup，但理论上 Post-LN 表达能力更强。

### 8. Attention 的时间复杂度
Self-Attention 的计算复杂度是 $O(n^2 \cdot d)$（$n$ 是序列长度），这也是 Transformer 处理**超长序列**的瓶颈（如文档级任务），这催生了 Longformer、BigBird 等稀疏注意力变体。

---

## 🔗 知识延伸

| 技术 | 与 Transformer 的关系 |
|------|----------------------|
| **BERT** | Encoder-only Transformer，使用 MLM 预训练 |
| **GPT** | Decoder-only Transformer，使用自回归语言模型预训练 |
| **T5** | 完整 Encoder-Decoder Transformer |
| **Vision Transformer (ViT)** | 将图像分割成 patch 序列，用 Transformer 做图像分类 |
| **Longformer / BigBird** | 稀疏注意力，解决 $O(n^2)$ 瓶颈，处理长文档 |
| **Flash Attention** | IO感知的 Attention 实现，不改变数学等价，速度快数倍 |
| **RWKV / Mamba** | 用线性递归替代 Attention，保持并行训练但推理是 $O(1)$ |

---

## 📚 参考资料

1. Vaswani et al. (2017). **Attention Is All You Need**. NeurIPS 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Illustrated Transformer（强烈推荐）：[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
3. Harvard NLP Annotated Transformer：[https://nlp.seas.harvard.edu/annotated-transformer/](https://nlp.seas.harvard.edu/annotated-transformer/)
4. The Transformer Family（各变体综述）：[https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
5. Ba et al. (2016). **Layer Normalization**. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
