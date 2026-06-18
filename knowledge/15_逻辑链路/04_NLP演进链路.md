# NLP 演进链路自查

---

## 一、整体演进时间线

```
词袋模型 (BoW)
    ↓ 问题：无语义、无顺序
TF-IDF
    ↓ 问题：仍是统计，无语义
Word2Vec / GloVe（静态词向量）
    ↓ 问题：一词一义，无上下文
RNN / LSTM（序列建模）
    ↓ 问题：长程依赖弱，串行慢
Seq2Seq（编码器-解码器）
    ↓ 问题：信息瓶颈（context vector）
Attention 机制
    ↓ 问题：RNN 仍然串行
Self-Attention / Transformer
    ↓
BERT（双向预训练）/ GPT（单向生成）
    ↓
GPT-2/3/4、LLaMA 等大模型
```

---

## 二、各阶段核心问题与解法

### 阶段1：词袋 → 词向量
**问题：** "苹果手机" 和 "苹果水果" 的"苹果"是同一个向量，语义混淆。

**Word2Vec 解法：** 通过上下文预测词，让语义相近的词在向量空间里距离近。

- [ ] 能解释 Word2Vec 的训练思路（CBOW / Skip-gram）
- [ ] 知道词向量是静态的，同一个词永远是同一个向量
- [ ] 知道词向量的类比能力：`king - man + woman ≈ queen`

---

### 阶段2：静态词向量 → 上下文表示
**问题：** "我去银行取钱" vs "河岸边的银行"，"银行"向量相同但语义不同。

**ELMo/LSTM 解法：** 根据上下文动态生成词向量。

- [ ] 知道 ELMo 用双向 LSTM 生成上下文相关的词向量
- [ ] 理解为什么静态词向量无法解决多义词问题

---

### 阶段3：RNN → Seq2Seq
**问题：** 需要处理变长输入输出（如翻译），RNN 单输出无法处理。

**Seq2Seq 解法：** 编码器把输入压成 context vector，解码器逐步生成输出。

- [ ] 能说出 Seq2Seq 的编码器和解码器各自做什么
- [ ] 知道 Teacher Forcing：训练时喂真实标签，推理时喂自己预测的结果
- [ ] 知道 Exposure Bias：训练和推理输入分布不一致导致的误差累积
- [ ] 能说出 Seq2Seq 的核心缺陷：信息瓶颈（整个输入压成一个向量）

---

### 阶段4：Seq2Seq → Attention
**问题：** context vector 信息瓶颈，长句子前面的词信息丢失。

**Attention 解法：** 解码每个词时，动态关注编码器所有位置，而不是只用最后一个状态。

- [ ] 能说出 Attention 的 Q/K/V 分别是什么
- [ ] 知道 Cross-Attention 和 Self-Attention 的本质区别（Q来自哪里）
- [ ] 能说出 Attention 权重的物理含义（当前词对每个输入词的关注程度）

---

### 阶段5：Attention → Transformer → BERT/GPT
**问题：** RNN 串行，无法并行，训练慢。

**Transformer 解法：** 完全用 Self-Attention 替代 RNN，全部并行计算。

- [ ] 知道为什么 Transformer 比 RNN 快（并行 vs 串行）
- [ ] 知道 BERT 和 GPT 的核心区别（双向理解 vs 单向生成）
- [ ] 知道预训练 + 微调的范式是什么意思
