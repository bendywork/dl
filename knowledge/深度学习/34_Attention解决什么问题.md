# Attention 解决什么问题

## 📌 核心问题
> Seq2Seq 的上下文向量 c 是信息瓶颈，Attention 如何解决？

## 🌱 根源与动机

Seq2Seq 把整个输入序列压成一个固定长度的向量 c，Decoder 每步只能看这个 c。长句子时信息混在一起，Decoder 无法定位到具体位置的细节。

## 📐 Attention 机制

Decoder 每生成一个词，都能回头看 Encoder 每一步的输出，动态计算关注度，得到当前步专属的上下文向量。

```
Encoder 输出:  h1  h2  h3  h4  h5
                ↓   ↓   ↓   ↓   ↓
Decoder 查询 q_t
                ↓   ↓   ↓   ↓   ↓
            算 q_t 和每个 h_i 的相似度 → 权重 α
                ↓   ↓   ↓   ↓   ↓
            加权求和: c_t = α1·h1 + α2·h2 + ... + α5·h5
```

- **q_t**：Decoder 当前状态，"我现在想知道什么"
- **α_i**：q_t 和 h_i 的相关程度
- **c_t**：当前步专属上下文向量，每一步都不同

## 💡 关键理解

| | Seq2Seq | Attention |
|---|---|---|
| 信息获取 | 所有步共享一个 c | 每步有自己的 c_t |
| 类比 | 只能看一个压缩包 | 每步翻原文，挑相关的看 |

Attention 没有改 RNN 内部结构，改的是**信息获取方式**：从被动接收压缩包，变成主动按需查询。

## ⚠️ 易错点与常见误解

1. **Attention 是网络结构改进吗？** — 不是结构改进，是信息获取方式的改进。Encoder/Decoder 内部仍然是 RNN/LSTM/GRU。

2. **α 权重怎么算？** — 常见方式：q_t 和 h_i 做点积或加性网络，过 softmax 归一化。核心就是"相似度高的权重大"。

3. **Transformer 和这里的 Attention 什么关系？** — Transformer 把 Attention 推到极致：连 RNN 都不要了，全靠 Attention 在任意两个位置之间直接传信息。

## 🔗 知识延伸

- [[26_Seq2Seq为什么出现]] — Attention 出现的前置背景
- [[24_LSTM如何解决长时依赖]] — Encoder/Decoder 内部机制
- Transformer — 用 Attention 完全替代 RNN

## 📚 参考资料

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", 2015
