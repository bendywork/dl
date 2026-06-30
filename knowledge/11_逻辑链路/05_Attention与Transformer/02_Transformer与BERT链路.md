# Transformer 与 BERT 链路

## 逻辑链

```
RNN 串行瓶颈 → Transformer(纯Attention,并行计算)
    → Encoder(双向Self-Attn) + Decoder(Causal+Cross-Attn)
    → BERT(只用Encoder,双向预训练) → Fine-tuning(下游任务)
```

## Transformer 架构核心

```
Encoder 层 × N:
    Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm

Decoder 层 × N:
    Masked Multi-Head Self-Attention(反作弊) → Add & Norm
    → Cross-Attention(Q=Decoder, K/V=Encoder) → Add & Norm
    → FFN → Add & Norm
```

## 位置编码

Attention 是置换等变的 → 没有位置信息 → 加 Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

## BERT = Transformer 的 Encoder 部分

```
BERT = 堆叠 N 层 Transformer Encoder
     + 无监督预训练(MLM + NSP)
     + 下游 Fine-tuning

核心创新：
  MLM(Masked Language Model)：随机遮住 15% 的词让它猜 → 双向理解
  NSP(Next Sentence Prediction)：判断两句是否相邻 → 句子关系
```

## 知识文件索引

### Transformer 原理
| 文件 | 核心内容 |
|------|---------|
| `05_Transformer与BERT/01_RNN到Transformer演进.md` | 为什么从 RNN 到 Transformer |
| `05_Transformer与BERT/02_Transformer原理与结构.md` | 完整结构 |
| `05_Transformer与BERT/03_注意力机制详解.md` | Self/Cross/Multi-Head+动画 |
| `05_Transformer与BERT/06_Score函数计算与Attention维度.md` | 维度细节 |
| `04_序列模型/01_Scaled_Dot-Product_Attention原理详解.md` | Scaled 原理 |
| `04_序列模型/02_Scaled_Dot-Product_Attention_PyTorch实现.md` | 代码 |

### BERT
| 文件 | 核心内容 |
|------|---------|
| `05_Transformer与BERT/04_BERT模型预训练详解.md` | MLM+NSP |
| `05_Transformer与BERT/05_BERT微调与下游任务.md` | Fine-tuning+动画 |
| `05_Transformer与BERT/07_Transformer与BERT完整对比.md` | 架构对比 |

### 对比总结
| 文件 | 核心内容 |
|------|---------|
| `05_Transformer与BERT/08_LSTM_ht_vs_ct_设计哲学.md` | LSTM→Q/K/V 哲学 |
| `05_Transformer与BERT/09_FastText与预训练词向量.md` | 静态词向量→预训练 |

## Add & Norm：残差连接 + LayerNorm

与 ResNet 同样的思想：
```
输出 = LayerNorm(x + Sublayer(x))
```
x 是残差"高速通道"，Sublayer 是 Attention 或 FFN。

## FFN 层

```
FFN(x) = W2 · GELU(W1 · x + b1) + b2
d_model → d_ff(4×) → d_model

作用：Attention 负责"查关系"，FFN 负责"自己琢磨"
     ——每个 token 独立做非线性变换，学习知识存储
```

## 理解检查点

1. Transformer 核心优势？（并行计算，直接建模任意距离依赖）
2. Encoder vs Decoder 关键区别？（Decoder 有 Causal Mask + Cross-Attention）
3. BERT 预训练的两个任务分别训练什么能力？（MLM=词级理解，NSP=句间关系）
4. Transformer 的 FFN 有什么用？（token 独立非线性变换，存储知识）
