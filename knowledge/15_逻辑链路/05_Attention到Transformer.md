# Attention 到 Transformer 推导链路

---

## 一、为什么需要 Attention（从问题出发）

Seq2Seq 把整个输入序列压成一个固定长度向量：
```
"我今天去北京出差了" → [0.3, 0.7, ...] → "I went to Beijing on a business trip"
```
序列越长，信息损失越严重，早期词的信息几乎消失。

**Attention 的核心思想：**
解码每个词时，不只看最后一个状态，而是看编码器所有位置，动态决定关注哪里。

---

## 二、Attention 公式拆解

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

| 符号 | 含义 | 来源 |
|------|------|------|
| Q (Query) | 我想查什么 | 解码器当前状态 |
| K (Key) | 数据库的索引 | 编码器所有位置 |
| V (Value) | 查到后取出的内容 | 编码器所有位置 |

计算步骤：
```
1. Q @ K^T      → 每个Q和每个K的相似度得分 [seq_q, seq_k]
2. / √d_k       → 缩放，防止点积过大导致 softmax 梯度消失
3. softmax      → 转为注意力权重（概率分布，和为1）
4. @ V          → 加权求和，得到融合了上下文的表示
```

- [ ] 能解释 Q/K/V 各自的含义
- [ ] 能解释为什么除以 √d_k
- [ ] 能解释 softmax 之后的权重是什么含义

---

## 三、Self-Attention vs Cross-Attention

| | Q 来自 | K/V 来自 | 作用 |
|---|---|---|---|
| Self-Attention | 自身序列 | 自身序列 | 理解序列内部关系 |
| Cross-Attention | 解码器 | 编码器 | 对齐两个序列 |

Self-Attention 让每个词直接和序列里所有词计算相关性，距离不再是障碍。

- [ ] 能说出 Self-Attention 解决了 RNN 的哪个问题（长程依赖 + 并行）
- [ ] 能说出 Cross-Attention 中 Q 和 K/V 分别来自哪里
- [ ] 知道 Decoder-Only 模型（GPT）没有 Cross-Attention

---

## 四、Multi-Head Attention

单头 Attention 只能从一个角度看序列关系，Multi-Head 让模型从多个角度同时关注：

```
输入 X
  ↓ 分成 h 个头，每个头有独立的 W_Q, W_K, W_V
Head1 = Attention(XW_Q1, XW_K1, XW_V1)
Head2 = Attention(XW_Q2, XW_K2, XW_V2)
...
Headh = Attention(XW_Qh, XW_Kh, XW_Vh)
  ↓ 拼接所有头的输出
Concat(Head1,...,Headh) @ W_O → 最终输出
```

- [ ] 能解释多头的好处（不同头关注不同类型的关系）
- [ ] 知道每个头的维度是 `d_model / h`，总计算量不变

---

## 五、位置编码

Self-Attention 本身没有顺序感知，"我爱你"和"你爱我"算出来结果一样。

解决方法：把位置信息加到词向量里。

```
最终输入 = 词向量 + 位置编码向量
```

两种方式：
- **正弦位置编码**（原始 Transformer）：用不同频率 sin/cos 函数生成，可外推
- **可学习位置编码**（GPT/BERT）：每个位置一个向量，训练时学习

- [ ] 能解释为什么 Self-Attention 需要位置编码
- [ ] 知道位置编码是加法，不是拼接

---

## 六、Transformer 完整结构

```
输入 tokens
    ↓
Embedding + 位置编码
    ↓
┌──────────────────────────────┐
│        Encoder（N层）         │
│  Multi-Head Self-Attention   │
│           ↓                  │
│      Add & LayerNorm         │
│           ↓                  │
│      Feed Forward (MLP)      │
│           ↓                  │
│      Add & LayerNorm         │
└──────────────────────────────┘
    ↓ 编码器输出（K/V）
┌──────────────────────────────┐
│        Decoder（N层）         │
│  Masked Self-Attention       │← 只看已生成的词
│           ↓                  │
│      Add & LayerNorm         │
│           ↓                  │
│  Cross-Attention             │← Q来自Decoder，K/V来自Encoder
│           ↓                  │
│      Add & LayerNorm         │
│           ↓                  │
│      Feed Forward            │
│           ↓                  │
│      Add & LayerNorm         │
└──────────────────────────────┘
    ↓
Linear + Softmax → 预测下一个词
```

- [ ] 能说出 Encoder 和 Decoder 各包含哪些子层
- [ ] 知道 Masked Self-Attention 为什么要 Mask（推理时不能看未来）
- [ ] 知道 Add & LayerNorm 的作用（残差连接防梯度消失 + 归一化稳定训练）
- [ ] 能区分 Encoder-Decoder（翻译）/ Encoder-Only（BERT）/ Decoder-Only（GPT）三种架构
