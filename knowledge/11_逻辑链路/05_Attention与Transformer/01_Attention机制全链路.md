# Attention 机制全链路

## 逻辑链

```
Score 函数(度量相似度) → Softmax(归一化) → 加权求和(聚合)
    → Self-Attention(Q/K/V 同源) + Cross-Attention(Q≠K/V)
    → Mask(反作弊+忽略Pad) → Multi-Head(多子空间) → Scaled(√dk)
```

## 核心公式（必经之路）

```
Attention(Q,K,V) = softmax(QK^T / √d_k) V

QK^T：计算每个 query 和所有 key 的相似度 → 得到注意力分数矩阵
√d_k：缩放，防止点积太大导致 softmax 梯度消失
Softmax：每个位置的注意力权重，和为 1
× V：按权重加权求和，得到上下文表示
```

## Attention 分类树

```
Attention
├── 按 Q/K/V 来源
│   ├── Self-Attention：Q=K=V 来自同一序列
│   ├── Cross-Attention：Q 来自 Decoder，K/V 来自 Encoder
│   └── Causal Self-Attention：Self + 反作弊 mask
├── 按 Score 函数
│   ├── 加性 (Bahdanau)：v^T tanh(W[Q;K])
│   ├── 点积 (Luong)：Q^T K
│   └── Scaled Dot-Product：Q^T K / √d_k
└── 按关注范围
    ├── 全局 Attention：每个 Q 看所有 K
    ├── 局部 Attention：窗口内看
    └── 稀疏 Attention：Longformer/BigBird
```

## Mask 机制

| Mask 类型 | 作用 | 设置位置 |
|-----------|------|---------|
| Padding Mask | 忽略 <pad> token | Softmax 前设 -inf |
| Causal Mask | 防止看到未来 token | 上三角设 -inf，反作弊 |

**原理**：Softmax(e^{-∞}) = 0 → 对应位置权重为 0，不参与加权。

## 知识文件索引

### Attention 基础
| 文件 | 核心内容 |
|------|---------|
| `02_Seq2Seq/03_Attention解决了Seq2Seq什么瓶颈.md` | Attention 动机 |
| `02_Seq2Seq/04_Bahdanau_Attention详解.md` | 加性 Attention |
| `02_Seq2Seq/08_QKV理解.md` | Q/K/V 本质+动画 |
| `02_Seq2Seq/09_Seq2Seq_Attention学习记录.md` | 学习笔记 |

### Attention 进阶
| 文件 | 核心内容 |
|------|---------|
| `05_Transformer与BERT/06_Score函数计算与Attention维度.md` | Score 函数+V矩阵维度 |
| `05_Transformer与BERT/03_注意力机制详解.md` | Self/Cross/Multi-Head+动画 |
| `05_Transformer与BERT/08_LSTM_ht_vs_ct_设计哲学.md` | ht vs ct → Q/K/V 类比 |

### Scaled Dot-Product Attention
| 文件 | 核心内容 |
|------|---------|
| `04_序列模型/01_Scaled_Dot-Product_Attention原理详解.md` | 公式+缩放原因 |
| `04_序列模型/02_Scaled_Dot-Product_Attention_PyTorch实现.md` | 代码实现 |
| `04_序列模型/03_Attention分类详解.md` | 分类全景 |

## 理解检查点

1. Q/K/V 是什么关系？（Q 问谁和我相关，K 是我能提供什么，V 是实际信息）
2. 为什么要除以 √d_k？（d_k 大→点积大→softmax 进入饱和区→梯度小）
3. Causal Mask 为什么是"反作弊"？（不让当前位置偷看未来 token，保证自回归）
4. Multi-Head 解决什么？（不同 head 关注不同子空间：位置、语法、语义等）
