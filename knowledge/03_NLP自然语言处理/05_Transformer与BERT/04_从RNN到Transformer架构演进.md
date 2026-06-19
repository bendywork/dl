# 从 RNN/LSTM 到 Transformer：NLP 架构演进脉络

## 📌 核心问题
> NLP 的神经网络架构是怎么一步步演进到 Transformer 的？为什么 RNN 被淘汰了？

## 🌱 核心脉络（一句话）

```
Seq2Seq + RNN  →  Seq2Seq + RNN + Attention  →  Transformer（纯 Attention）
   信息丢失          解决丢失但串行慢            Attention 取代 RNN 本身
```

## 📐 三个演进阶段

### 第一阶段：Seq2Seq + RNN（2014 前）

最早处理"输入序列 → 输出序列"任务（如机器翻译）的结构：

```
编码器(RNN): "我 爱 学习" → 一步步压缩 → 上下文向量 C
                                        ↓
解码器(RNN): C → 一步步生成 → "I love learning"
```

**核心问题：** 所有输入信息被压缩成**一个固定长度的向量 C**。

```
"我 爱 学习" → C（比如 256 维）
"我 爱 学习 深度 学习 并 研究 Transformer ..." → C（还是 256 维）
```

句子一长，信息塞不进一个向量，**信息丢失严重**。

---

### 第二阶段：Seq2Seq + RNN + Attention（2014，Bahdanau）

**改进：** 编码器不再只输出一个向量，而是**每一步的输出都保留**，解码时按需挑选。

```
编码器(RNN): "我 爱 学习" → [h1, h2, h3]（每一步都留着）
                                ↓
解码器(RNN): 生成 "I" 时
             → Attention：h1, h2, h3 哪个最相关？→ 主要看 h1
             → 生成 "love" 时 → 主要看 h2
             → 生成 "learning" 时 → 主要看 h3
```

**解决了信息丢失**，但**没解决 RNN 的串行计算问题**——RNN 还是一步步算，无法并行，训练慢。

---

### 第三阶段：Transformer = 纯 Attention（2017）

**关键洞察：** 既然 Attention 这么有效，**为什么还要 RNN？直接用 Attention 不就行了？**

```
编码器(Self-Attention): "我 爱 学习" → 每个词直接看所有词 → 新表示
                                            ↓
解码器(Self-Attention + Cross-Attention) → 生成译文
```

**论文标题就叫《Attention Is All You Need》**——意思就是"你只需要 Attention，不需要 RNN"。

---

## 💡 关键理解：是替代，不是叠加

### 常见误解

> ❌ "Transformer 是 Seq2Seq 结构 + 加上 Attention 机制"

这说法把 Attention 当成附属品，好像 RNN 还在。**其实 Attention 把 RNN 整个替代了。**

### 正确理解

| 阶段 | RNN 的角色 | Attention 的角色 |
|------|-----------|-----------------|
| 阶段1 | **主体**（编码/解码都靠它） | 无 |
| 阶段2 | **主体**（编码/解码还靠它） | **辅助**（帮解码器挑信息） |
| 阶段3 | **被淘汰**（没有了） | **主体**（自己就是编码/解码结构） |

**Transformer 不是"Seq2Seq + Attention"，它就是"Attention 重新编排的结构本身"。**

---

## 🔑 Transformer 比 RNN 强在哪

| 维度 | RNN/LSTM | Transformer |
|------|---------|-------------|
| **看多远** | 只能看前面，远了就忘 | **一眼看所有位置**（Self-Attention） |
| **并行计算** | 必须串行（等前一步算完） | **全并行**（矩阵乘法一次算完） |
| **信息传递** | h_t 逐步传递，衰减 | **直达**，任意两位置直接关联 |
| **长序列** | 梯度消失/爆炸 | 残差连接 + 直达路径，更稳定 |
| **训练速度** | 慢（GPU 并行用不上） | 快（GPU 并行吃满） |

---

## 🔧 不同任务用 Transformer 的不同部分

Transformer 完整结构是**编码器 + 解码器**，但不是所有任务都需要全部：

| 任务 | 用哪部分 | 例子 |
|------|---------|------|
| **分类**（情感、意图） | 只要编码器 | 股吧情感分析（本项目 Route A） |
| **生成**（续写、摘要） | 只要解码器 | GPT 系列 |
| **翻译/Seq2Seq** | 编码器 + 解码器 | 机器翻译、T5 |

你做的股吧情感分析，**只用 Transformer 的编码器部分**——Self-Attention 理解句子 → 分类头输出三分类。不需要解码器（没有生成任务）。

---

## 📊 架构演进时间线

```
1997  LSTM 提出（解决 RNN 长依赖问题）
  ↓
2014  Seq2Seq + RNN（机器翻译）
  ↓
2014  Seq2Seq + RNN + Attention（Bahdanau）
  ↓
2017  Transformer = 纯 Attention（Vaswani et al.）
  ↓
2018  BERT = Transformer 编码器 + MLM 预训练
  ↓
2018  GPT = Transformer 解码器 + 自回归预训练
  ↓
2019+ 整个 NLP 被 Transformer 统一，RNN 退居特定场景
```

## ⚠️ 易错点与常见误解

1. **以为 Transformer 是"RNN + Attention"** → 是"用 Attention 替代 RNN"，RNN 被淘汰了
2. **以为 Attention 必须依附 RNN** → Attention 可以独立工作，这就是 Transformer 的核心
3. **以为分类任务需要完整 Transformer** → 分类只用编码器，不需要解码器
4. **以为 Self-Attention 和 Cross-Attention 是不同东西** → Self-Attention 是 Q/K/V 同源的特例
5. **以为 RNN 完全没用了** → 长序列（超长上下文）Transformer 计算量爆炸，RNN 在某些场景仍有用（如 Mamba 等新架构回归线性复杂度）

## 🔗 知识延伸

- [[Self-Attention与Multi-Head-Attention]] — Transformer 的核心组件细节
- [[RNN四种输入输出结构]] — RNN 时代的输入输出模式
- [[LSTM为什么拆分输出与状态]] — LSTM 如何缓解 RNN 的信息衰减
- BERT vs GPT — 同为 Transformer，编码器路线 vs 解码器路线
- Mamba/SSM — 2023+ 回归 RNN 思想的新架构，挑战 Transformer 长序列霸权

## 📚 参考资料

- Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)
- Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014) — Attention 引入
- Attention Is All You Need (Vaswani et al., 2017) — Transformer 原论文
- The Illustrated Transformer (Jay Alammar) — 可视化图解
