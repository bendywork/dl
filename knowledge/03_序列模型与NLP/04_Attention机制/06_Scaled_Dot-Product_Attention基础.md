# 01 · Scaled Dot-Product Attention（放缩点积注意力）

是所有 Transformer Attention 的数学基础。后续所有变体（MHA/MQA/GQA/FlashAttention）都是它的工程优化。

## 公式

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

## 三步走

| 步骤 | 操作 | 含义 |
|------|------|------|
| 1 | `score = Q · Kᵀ` | 每个 query 和每个 key 做点积，得到一个 "相关性" 分值 |
| 2 | `÷ √d_k` → softmax | 缩放防梯度饱和 + 转成概率权重（每行和为 1） |
| 3 | `× V` | 按权重从 Value 中取信息，加权求和 |

## 为什么必须 ÷ √d_k

当 d_k（向量维度）很大时，点积值的方差 ≈ d_k。极端情况 d_k=512 → std ≈ 22：

```
不加缩放: softmax([30, 10, 5]) → [~1.0, ~0, ~0]  ← 几乎 one-hot
          ∂softmax/∂z ≈ 0  → 梯度消失 → 模型学不动

加缩放:   softmax([1.3, 0.4, 0.2]) → [0.55, 0.25, 0.20] ← 分布平滑
          ∂softmax/∂z 正常 → 梯度健康 → 能学
```

**本质**：方差压缩回 1 → softmax 落在梯度健康区 → 模型训练成功。

## 为什么不用加法 Attention（Bahdanau 2014 版）

| | Bahdanau（加法） | Scaled Dot-Product |
|------|-----------|------|
| 相似度计算公式 | `vᵀ · tanh(W_q·q + W_k·k)` | `q · kᵀ / √d_k` |
| 有可训练参数 | 有（W_q, W_k, v） | 无（纯数学运算） |
| 计算效率 | O(d²)，每个 qk 对过小网络 | 矩阵乘法一次性算完，GPU 友好 |
| 可堆叠性 | 慢，多层撑不住 | 快，可以堆 100 层 |

**Scaled Dot-Product Attention = 用数学技巧换速度，让 Transformer 可以堆成摩天大楼。**

## Mask 机制

Attention 有两种 Mask，作用位置都在 softmax 之前：

```
score = Q · Kᵀ / √d_k
score = score + mask        ← 加 mask（-∞ 位置 softmax 后 → 0）
weight = softmax(score)
output = weight · V
```

| Mask 类型 | 用在哪 | 作用 |
|-----------|--------|------|
| Padding Mask | Encoder + Decoder | 把 `<PAD>` 位置置 -∞，防止注意到填充符 |
| Causal Mask（上三角） | Decoder Self-Attention | 位置 i 只能看 ≤ i 的位置，保证自回归生成不泄露未来信息 |

Causal Mask 示意（序列长度=4）：

```
     pos0  pos1  pos2  pos3
pos0 [  0   -∞    -∞    -∞  ]
pos1 [  0    0    -∞    -∞  ]
pos2 [  0    0     0    -∞  ]
pos3 [  0    0     0     0  ]
```

加到 score 上后，-∞ 经过 softmax → 0，等价于"看不到"。

## 复杂度分析

设序列长度 = n，向量维度 = d_k：

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Q · Kᵀ | O(n² · d_k) | O(n²) |
| softmax | O(n²) | O(n²) |
| weight · V | O(n² · d_k) | O(n · d_k) |
| **总计** | **O(n² · d_k)** | **O(n²)** |

**瓶颈是 n²**：序列长 1k → attention 矩阵 1M；序列长 100k → 10B，直接 OOM。这就是 FlashAttention / 稀疏 Attention 要解决的核心问题。

## 代码实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_k, d_v)
    mask: (batch, 1, seq_q, seq_k) 或 None
    """
    d_k = Q.size(-1)

    # 1. 计算相似度
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, heads, seq_q, seq_k)

    # 2. 应用 mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. softmax 归一化
    weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(weights, V)
    # output: (batch, heads, seq_q, d_v)

    return output, weights


# 使用示例
batch, heads, seq_len, d_k = 2, 8, 10, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

# Causal mask（下三角为1，上三角为0）
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(output.shape)   # (2, 8, 10, 64)
print(weights.shape)  # (2, 8, 10, 10)
```

PyTorch 2.0+ 内置了优化版本：

```python
# F.scaled_dot_product_attention 自动选用 FlashAttention（如果硬件支持）
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
```

## 与后续变体的关系

Scaled Dot-Product Attention 是基础模块，所有变体都在改造它的某个环节：

```
Scaled Dot-Product Attention
        ↓ 并行多组
Multi-Head Attention          ← Transformer 原文
        ↓ 解决 n² 显存瓶颈
FlashAttention                ← IO 感知，分块计算，显存 O(n)
        ↓ 减少 KV 头数
Multi-Query / Grouped-Query   ← LLaMA2 / Mistral，推理提速
        ↓ 改相对位置编码
RoPE Attention                ← LLaMA / Qwen 系列
        ↓ 稀疏化
Sparse / Sliding Window       ← Longformer，处理超长序列
```

| 变体 | 解决的问题 | 代表模型 |
|------|-----------|---------|
| Multi-Head Attention | 捕获多种关系 | Transformer |
| FlashAttention | 显存 OOM | GPT-4, LLaMA |
| GQA（分组查询） | 推理 KV Cache 过大 | LLaMA2-70B, Mistral |
| RoPE | 外推到更长序列 | LLaMA, Qwen |
| Sparse Attention | n² 计算瓶颈 | Longformer, BigBird |

## 总结

Scaled Dot-Product Attention 核心就三件事：

1. **Q·Kᵀ** → 算每对 token 的相关性
2. **÷√d_k** → 防止梯度消失
3. **softmax → ·V** → 用相关性加权聚合信息

这个结构简单、可并行、效果强，是现代大模型的基石。
