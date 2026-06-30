# Seq2Seq → Attention 演进链

## 逻辑链

```
Encoder-Decoder(Seq2Seq) → 固定长度上下文瓶颈 → Attention(动态加权)
    → Bahdanau(加性Attention) → Luong(乘性Attention) → Scaled Dot-Product
```

## 核心演进：为什么需要 Attention

```
Seq2Seq：Encoder 把所有输入压成最后一个 ht → Decoder 只用这一个向量解码
         ↓ 瓶颈：长句子信息丢失
Attention：Decoder 每步直接访问 Encoder 所有时刻的 hidden state
          → 加权求和 → 动态上下文，不再依赖固定向量
```

**一句话：Encoder 不用把所有信息压给最后一个状态了，Decoder 自己选要看什么。**

## 知识文件索引

### Seq2Seq 基础
| 文件 | 核心内容 |
|------|---------|
| `04_序列模型/01_Seq2Seq为什么叫编码器解码器.md` | Seq2Seq 命名来源 |
| `04_序列模型/02_Seq2Seq与Attention的关联.md` | 两类 Attention 梳理 |
| `04_序列模型/05_自编码器翻译任务.md` | 翻译实验 |
| `04_序列模型/06_翻译任务实验分析.md` | 实验分析 |
| `04_序列模型/07_自编码NN的理解.md` | AutoEncoder→Seq2Seq |

### Attention 登场
| 文件 | 核心内容 |
|------|---------|
| `04_序列模型/03_Attention解决了Seq2Seq什么瓶颈.md` | 固定向量瓶颈 |
| `04_序列模型/04_Bahdanau_Attention详解.md` | 加性 Attention 公式 |
| `04_序列模型/08_QKV理解.md` | Q/K/V 来源本质+动画 |
| `04_序列模型/09_Seq2Seq_Attention学习记录.md` | 学习记录 |
| `04_序列模型/10_Seq2Seq+Attention链路总结.md` | 完整链路 |

## 理解检查点

1. Seq2Seq 的 bottleneck 是什么？（所有 Encoder 信息压入一个固定向量）
2. Attention 的核心公式？（softmax(score(h_dec, h_enc)) · h_enc 加权求和）
3. Q/K/V 分别是什么？（Q=Decoder状态，K=Encoder全部状态，V=Encoder全部状态 → 加权后的上下文）
