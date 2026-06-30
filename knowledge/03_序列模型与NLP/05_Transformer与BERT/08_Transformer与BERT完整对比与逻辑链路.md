# Transformer 与 BERT：完整对比与逻辑链路

## 阅读导引

本文档回答一个核心问题：**BERT 相比原始 Transformer 改了哪些东西？每一条改动背后的原因和逻辑链路是什么？**

读完本文你将理解：
- 为什么 BERT 只用 Encoder，不用 Decoder
- 为什么 BERT 把 sin/cos 位置编码换成了可学习位置编码（以及"双重位置编码"的本质）
- 为什么 BERT 多了 Segment Embedding，Transformer 没有
- 为什么 BERT 的预训练 + 微调范式是革命性的
- 每一条设计决策背后的工程与理论约束

---

## 第一章 · 架构总览：Encoder-Decoder vs Encoder-Only

### 1.1 结构对比

```
原始 Transformer（Vaswani 2017）          BERT（Devlin 2018）
─────────────────────────────────         ─────────────────────
Encoder × N                               Encoder × N
  ├─ Self-Attention                         ├─ Self-Attention
  ├─ Add & Norm                             ├─ Add & Norm
  ├─ FFN                                    ├─ FFN
  └─ Add & Norm                             └─ Add & Norm
         ↓                                        ↓
Decoder × N                                [CLS] 输出 → 分类头
  ├─ Masked Self-Attn                      或 每个 token 输出 → 序列标注头
  ├─ Add & Norm
  ├─ Cross-Attention ← Encoder 输出
  ├─ Add & Norm
  ├─ FFN
  └─ Add & Norm
         ↓
Linear + Softmax
```

### 1.2 为什么 BERT 丢弃了 Decoder？

这是最根本的架构决策，背后的逻辑链如下：

**原始 Transformer 的 Decoder 是用来做自回归生成的**——逐词预测下一个词（`y_t = f(y_{<t}, encoder_output)`）。Decoder 的两个关键组件都是为生成服务的：

1. **Masked Self-Attention**（因果掩码）：确保当前词只能看到过去，不能看到未来。这是自回归生成的前提——如果看到了未来，生成就没意义了。
2. **Cross-Attention**：把 Encoder 的源语言信息融合进目标语言生成过程。这是 Seq2Seq 翻译任务的刚需。

**BERT 的设计目标完全不同**：它要做一个**通用的语言理解模型**，而不是翻译或生成模型。理解任务需要双向上下文——要知道一个词的意思，必须同时看它的左边和右边。

```
句子："我把苹果吃了"
问题："苹果"在这里是什么意思？

GPT（单向）：只能看"我把" → "苹果"可能是手机
BERT（双向）：看"我把"和"吃了" → "苹果"显然是水果
```

**逻辑链：**
```
目标：通用语言理解
  → 需要深度双向上下文
  → 不能用 Masked Self-Attention（它阻止看右边）
  → 不能用 Cross-Attention（没有源语言输入）
  → Decoder 的两个核心组件都没用
  → 整个 Decoder 不需要了
  → BERT = Encoder-only
```

**反面案例验证**：GPT 系列是 Decoder-only，因为它的目标是**生成**——逐词续写文本，天然需要因果约束。BERT 和 GPT 的架构选择完全由任务目标决定：**理解用 Encoder，生成用 Decoder**。

---

## 第二章 · 位置编码：最容易被误读的差异

这是 BERT 和 Transformer 差异中**最需要讲清楚**的一点，也是你最初问题的核心。

### 2.1 原始 Transformer：固定正弦位置编码（Sinusoidal PE）

Transformer 使用数学公式计算位置编码，**不是可学习参数**：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

核心特征：
- 不同频率的正弦波叠加：低频捕获全局位置，高频捕获局部位置
- 类比：时钟的时针（低频）、分针（中频）、秒针（高频）三者组合唯一确定时间
- **可以外推**：训练时最长 512，推理时可以到 1024（因为公式是确定的）

**为什么 Transformer 选 sin/cos？**
1. **可外推性**：机器翻译的句子长度变化大，可能碰到比训练时更长的句子
2. **相对位置的自然表达**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换，让模型隐式学习相对位置关系
3. **论文实验**：sin/cos 编码效果与可学习编码相当，且具有上述优势

### 2.2 BERT：可学习位置嵌入（Learnable Position Embedding）

BERT 直接用一个 `nn.Embedding(max_position_embeddings, hidden_size)` 来存储每个位置的向量，在预训练过程中通过反向传播学习。

```python
# BERT 的位置嵌入本质
self.position_embeddings = nn.Embedding(512, 768)  # max_len=512, hidden=768
# position_ids = [0, 1, 2, 3, ..., 511]
# position_embeddings(position_ids) → [seq_len, 768]
```

**为什么 BERT 改用可学习编码？这里有四个层次的逻辑：**

**第一层（表面原因）：固定长度不需要外推**
- BERT 的所有输入都 pad/trunc 到 512，不需要像翻译那样处理变长序列
- 既然不需要外推，可学习编码就没了主要劣势

**第二层（深层原因）：让预训练自动发现最优位置表示**
- BERT 的哲学是"让模型自己学"：词的意思让 MLM 自学，句子关系让 NSP 自学，位置信息也应该让梯度自己优化
- 可学习编码让模型根据预训练数据的实际分布，学出最适合的位置表示模式

**第三层（工程原因）：与 [CLS]/[SEP] 等特殊 token 的位置语义统一**
- BERT 有特殊 token（`[CLS]`, `[SEP]`），它们的"位置"有特殊语义（首位置=聚合，分隔位置=边界）
- 固定 sin/cos 无法区分这些特殊位置，而可学习编码可以给位置 0（`[CLS]`）和分隔位学出独特的表示

**第四层（终极原因）：BERT 不需要用位置编码来表征相对位置**
- Transformer 的 sin/cos 有一个重要优势是相对位置的可线性表达性
- 但 BERT 有**双向 Self-Attention**（无 mask），每个词可以**直接**看到所有其他词，不需要通过相对位置编码来间接计算距离
- 生成模型（GPT）需要可学习的位置编码来辅助因果推理，BERT 则依赖更直接的全局注意力

### 2.3 "双重位置编码"的本质

这是一个经常被误解的概念。BERT 所谓的"双重位置编码"指的是：

**BERT 同时使用两种位置相关的嵌入**：

| 嵌入类型 | 编码什么 | 维度 | 学习方式 |
|---------|---------|------|---------|
| **Position Embedding** | token 在序列中的绝对位置（第0个、第1个…） | `[512, 768]` | 预训练可学习 |
| **Segment Embedding** | token 属于句子 A 还是句子 B | `[2, 768]` | 预训练可学习 |

Segment Embedding 本质是一种**粗粒度的句子级位置信息**——它不告诉你词在第几个位置，但告诉你词属于第一句话还是第二句话。

```
输入: [CLS] I love NLP [SEP] It is amazing [SEP]
Pos:   E_0   E_1 E_2 E_3  E_4   E_5 E_6 E_7    E_8
Seg:   E_A   E_A E_A E_A  E_A   E_B E_B E_B    E_B
                                       ↑
                        句子 B 的所有 token 共享同一个 Segment Embedding E_B
```

**为什么需要 Segment Embedding？**
1. **NSP 任务需要**：判断两句话是否连续，模型必须知道哪些词属于句子 A，哪些属于句子 B
2. **词序消歧**：相同的 Position Embedding 在两个句子中可能出现两次（位置 1 在句子 A 和 B 各有一个），Segment Embedding 帮助区分
3. **句子边界感知**：配合 `[SEP]` token，让模型明确知道句子的分界点

### 2.4 位置编码差异的完整对比

| 维度 | Transformer | BERT |
|------|------------|------|
| 编码类型 | sin/cos 固定函数 | 可学习 Embedding |
| 是否可训练 | ❌ 不可学习 | ✅ 预训练中学习 |
| 最大序列长度 | 理论无限（公式外推） | 固定 512（只训练了 0-511） |
| Segment 信息 | ❌ 无 | ✅ Segment Embedding（句子 A/B） |
| 相对位置编码 | 可通过线性变换表达 | 依赖 Self-Attention 全局感知 |
| 特殊位置处理 | 无特殊位置 | [CLS] 位置0, [SEP] 分隔位 → 可学习 |
| 设计哲学 | 数学严谨、可外推 | 让模型自己学、任务适配 |


---

## 第三章 · 注意力机制的差异：双向 vs 单向 vs 交叉

### 3.1 注意力类型的根本不同

Transformer 里存在三种注意力模式，BERT 只保留了其中一种：

```
原始 Transformer 的注意力体系：

Encoder Self-Attention          BERT 保留了这种（核心）
  ├─ Q、K、V 全部来自同一序列
  ├─ 无 mask（每个词可以看所有词）
  └─ 双向 → 适用于"理解"

Decoder Masked Self-Attention   BERT 丢弃了
  ├─ Q、K、V 来自目标序列
  ├─ 有 causal mask（只看当前及之前）
  └─ 单向 → 适用于"生成"

Decoder Cross-Attention          BERT 丢弃了
  ├─ Q 来自 Decoder
  ├─ K、V 来自 Encoder 输出
  └─ 源-目标信息融合 → 适用于"翻译"
```

### 3.2 为什么 BERT 的双向 Self-Attention 是"真正的深度双向"？

**ELMo 的"伪双向"**：
```
前向 LSTM: the → cat → sat → on → the → mat
后向 LSTM: the ← cat ← sat ← on ← the ← mat
"双向"表示 = 拼接([前向输出, 后向输出])

问题：前向 LSTM 和后向 LSTM 各自独立训练，没有交互
     每个方向的表示只知道"半边天"
```

**BERT 的"真双向"**：
```
Self-Attention: 每个词在每一层都同时看所有词
Layer 1: "cat" 看到 [the, cat, sat, on, the, mat]
Layer 2: "cat" 看到 [更新后的 the, 更新后的 cat, ...]
...
Layer 12: "cat" 的表示融合了 12 层深度双向交互

关键：每一层都是双向的，不是最后拼起来的
```

**逻辑链：为什么深度双向对理解任务至关重要？**
```
理解"bank"的语义：
  → 单向（GPT）：只能看到左边 → "he deposited money in the" → 缺右边的 "account" 消歧信号
  → 浅层双向（ELMo）：前向看"deposited"，后向看"account"，但两个信息没有融合
  → 深度双向（BERT）：每层的"bank"同时融合"deposited"和"account" → 最精准的消歧
```

### 3.3 Padding Mask：BERT 也有的 mask

虽然 BERT 没有 causal mask（未来掩码），但它有 **padding mask**——这是工程必需的：

```python
# Padding Mask 逻辑
输入: [CLS] I love NLP [SEP] [PAD] [PAD]
       ↓
attention_mask: [1,    1, 1,   1,  1,    0,    0]
                 ↑                               ↑
          真实 token 正常参与 attention     padding 位置的 attention 权重被设为 -inf
```

**为什么 padding mask 必须存在？**
- batch 内的句子长度不同，必须 padding 到相同长度
- `[PAD]` token 没有语义信息，不应该被关注
- 否则模型会对 padding 位置"无中生有"地提取信息


---

## 第四章 · 输入表示的差异：从 2 层到 3 层嵌入

### 4.1 逐元素对比

```
原始 Transformer 的输入：               BERT 的输入：
─────────────────────                  ─────────────
Token Embedding                        Token Embedding (WordPiece)
    +                                       +
Position Encoding (sin/cos)             Position Embedding (可学习)
                                            +
                                        Segment Embedding (句子 A/B)
```

两处核心差异：
1. **Position 的来源和形式不同**（第二章已详述）
2. **BERT 多了 Segment Embedding**

### 4.2 Segment Embedding 的深层逻辑

**为什么 Transformer 不需要 Segment Embedding？**
- 翻译任务天然有"源语言序列"和"目标语言序列"，它们在不同的输入通道（Encoder 输入 vs Decoder 输入）
- 不需要额外标记来区分两个句子

**为什么 BERT 需要 Segment Embedding？**
- BERT 的输入是**单序列**：`[CLS] 句子A [SEP] 句子B [SEP]`
- 两个句子被拼接成了一个序列送入**同一个** Encoder
- 如果不加 Segment Embedding，两个句子中的相同位置可能有相同的位置编码，模型无法区分

```
没有 Segment Embedding：
  [CLS]  I   love  NLP   [SEP]  I   love  it   [SEP]
  E_0    E_1 E_2   E_3   E_4    E_5 E_6   E_7  E_8

  问题：I 出现两次（E_1 和 E_5），但如果两个 I 都在各自句子的位置 1 呢？
       位置编码相同 → 模型无法分辨

有 Segment Embedding：
  [CLS]  I   love  NLP   [SEP]  I   love  it   [SEP]
  E_A   E_A  E_A   E_A   E_A    E_B  E_B  E_B   E_B
   ↑                              ↑
  句子A的"I"                     句子B的"I" → 可区分！
```

### 4.3 [CLS] 和 [SEP] 特殊 Token 的设计逻辑

这是 BERT 独有的设计，Transformer 没有这两个 token：

**`[CLS]`（Classification Token）**
- 加在序列最开头
- 经过所有 Transformer 层后，`[CLS]` 的输出聚合了整个序列的信息
- 用于句子级分类任务（情感分析、NSP 等）

**为什么 `[CLS]` 能聚合全序列信息？**
```
逻辑链：
  Self-Attention 让每个 token 可以关注所有其他 token
  → [CLS] 在每一层都关注全序列
  → 经过 12 层，[CLS] 的表示 = 12 次加权聚合后的全局信息
  → NSP 预训练任务强制 [CLS] 学会"读懂整个句子对"的能力
  → 微调时分类损失进一步强化这个能力
```

**注意事项**：`[CLS]` 的表示能力并非天生，而是预训练任务"训练"出来的。对于语义相似度任务，Sentence-BERT 证明均值池化通常优于 `[CLS]`。

**`[SEP]`（Separator Token）**
- 标记句子边界（在 NSP 任务中分隔句子 A 和 B）
- 在单句任务中标记句子结束
- 配合 Segment Embedding 提供句子边界信息


---

## 第五章 · 训练范式的根本不同：任务专训 vs 预训练+微调

这是 Transformer 和 BERT 之间**最革命性的差异**，它改变了整个 NLP 的工作方式。

### 5.1 两种范式的对比

```
原始 Transformer 的训练范式：          BERT 的训练范式：
─────────────────────────────         ───────────────────
每个任务从零开始训练                    第一阶段：预训练（Pre-training）
  ↓                                      ↓
翻译任务 → 准备平行语料 → 训练        用海量无标注文本训练 MLM + NSP
  ↓                                      ↓
翻译模型（只能翻译）                    通用语言模型（不知道具体做什么任务）
                                         ↓
                                      第二阶段：微调（Fine-tuning）
                                         ↓
                                      加一个分类/NER/QA 头 → 少量标注数据训练
                                         ↓
                                      任务专用模型
```

### 5.2 为什么这改变了 NLP？

**之前的世界（Transformer 时代）：**
```
翻译任务 → 收集 1000 万平行句对 → 训练 Transformer → 翻译模型
分类任务 → 收集 100 万标注数据 → 训练 TextCNN     → 分类模型
NER 任务 → 收集 50 万标注数据  → 训练 BiLSTM-CRF → NER 模型

每个任务 = 大量的标注数据 + 从零训练一个模型
```

**之后的世界（BERT 时代）：**
```
预训练（只需一次，用无标注文本）：
  Wikipedia + BookCorpus → BERT → 通用语言理解模型

微调（每个任务，只需少量标注数据）：
  分类任务 → 1 万标注数据 → 微调 3 epochs → 分类模型（90%+ 准确率）
  NER 任务 → 5 万标注数据 → 微调 3 epochs → NER 模型
  QA 任务  → 10 万标注数据 → 微调 2 epochs → QA 模型
```

**逻辑链：为什么预训练+微调如此有效？**
```
无标注文本是无限的（Wikipedia、网页、书籍...）
  → BERT 在预训练阶段学会了：
      1. 词义消歧（同一个词在不同上下文中的含义）
      2. 语法结构（主谓宾、从句边界）
      3. 常识知识（巴黎是法国的首都）
      4. 语义关系（同义词、反义词、上下位词）
  → 微调阶段只需学习"怎么把这通用知识应用到特定任务"
  → 就像先学会"读懂文章"（预训练），再学"做阅读理解题"（微调）
  → 少量标注数据就够用，因为大部分语言能力已经学会了
```

### 5.3 原始 Transformer 为什么不能预训练+微调？

这是一个很好的问题。理论上可以，但实际上存在障碍：

1. **架构不是为"通用理解"设计的**：Encoder-Decoder 架构天然耦合了"理解→生成"的流程，不生成时 Decoder 是浪费
2. **没有无监督预训练目标**：翻译需要平行语料（有监督），不能利用无标注文本
3. **缺乏统一的输入输出格式**：不同任务（分类 vs 翻译 vs 摘要）需要完全不同的输入输出处理，难以统一微调


---

## 第六章 · 预训练任务的深层逻辑：MLM 与 NSP

### 6.1 MLM（Masked Language Model）：为什么这样设计？

**任务定义**：随机遮蔽 15% 的输入 token，让模型根据上下文预测被遮蔽的词。

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \hat{x}; \theta)$$

**80/10/10 规则（面试核心考点）**：
- **80%** 被选中的 token → 替换为 `[MASK]`
- **10%** → 替换为随机词
- **10%** → 保持原词不变

**为什么不是全部 mask？逻辑链：**
```
如果 100% mask：
  → 预训练时模型只见过 [MASK] token 的上下文
  → 微调/推理时输入中没有 [MASK] token
  → 预训练和微调的输入分布不匹配（distribution mismatch）
  → 模型性能大幅下降

80/10/10 的策略让模型对每个位置的 token 都不确定：
  → 是 [MASK] 吗？→ 需要预测
  → 是随机词吗？→ 需要纠错
  → 是原词吗？→ 需要验证
  → 结果：模型被迫对每个位置都认真建模上下文
  → 学到的是真正的"语言理解"，而非"填空技巧"
```

**为什么是 15% 而不是更多或更少？**
```
太少（如 5%）：任务太简单，模型学不到足够的上下文理解
太多（如 50%）：上下文信息太少，模型无法推断被遮蔽词
15%：经验平衡点，既提供足够的训练信号，又保留足够的上下文
```

### 6.2 NSP（Next Sentence Prediction）：设计意图与后续证伪

**任务定义**：给定句子对 (A, B)，判断 B 是否是 A 的真实下一句。
- 50% 正样本：B 确实紧跟在 A 之后
- 50% 负样本：B 从其他文档随机抽取

**设计意图的逻辑链：**
```
QA 和 NLI 等下游任务需要理解句子之间的关系
  → 只靠 MLM（词级理解）不够
  → 需要句子级理解能力
  → NSP 任务强制模型判断两句话的连贯性
  → 通过 [CLS] token 的输出做二分类
```

**为什么后来被证明作用有限甚至有害？（RoBERTa 的发现）**
```
逻辑链：
1. NSP 的负样本来自"不同文档"
   → 模型只需判断"话题是否相同"就能做对
   → 没有真正学到句子级别的逻辑连贯性
   → 这是一道太简单的"送分题"

2. NSP 和 MLM 在训练中共享 Encoder 参数
   → 两个任务的目标可能冲突（MLM 要求关注词级细节，NSP 要求关注句子级主题）
   → 模型在两个任务间折中，每个都做不到最优

3. RoBERTa 去掉 NSP → 效果反而更好
   → 证明 MLM 单独训练 + 更充分的数据和训练 → 已经足够学到句子级理解
   → NSP 的贡献被高估了

4. ALBERT 把 NSP 换成 SOP（Sentence Order Prediction）
   → 正负样本都来自同一文档，只是顺序不同
   → 难度更高、信号更纯 → 效果更好
   → 证明改进的句子对任务仍有价值，只是 NSP 太简单了
```


---

## 第七章 · 子词分词：BPE vs WordPiece

### 7.1 分词方式的对比

| 维度 | 原始 Transformer | BERT |
|------|-----------------|------|
| 分词方式 | BPE（Byte Pair Encoding）或其他 | WordPiece |
| 词表大小 | 论文中约 37K（英德共享） | 30,522 |
| 核心算法 | 统计频率、贪心合并 | 似然最大化、贪心合并 |
| 未登录词标记 | 子词拆分 | `##` 前缀标记非词首子词 |

### 7.2 BPE 与 WordPiece 的差异

**BPE（原始 Transformer 及 GPT 使用）：**
```
算法：统计所有相邻字符对的频率，合并频率最高的对，重复直到词表达标

"unaffordable" 的拆分过程：
  → low → low er → lower → ...
  → un + afford + able → unaffordable

核心原则："最常一起出现的字符就应该合并"
```

**WordPiece（BERT 使用）：**
```
算法：用似然（likelihood）而非频率来决定合并
  选择合并后能使训练语料似然增加最多的字符对

"unaffordable" → "un", "##afford", "##able"
  ## 前缀 = "这是词的延续部分，不是词首"

核心原则："合并后让语言模型概率最大的对"
```

### 7.3 为什么 BERT 需要 `##` 前缀？

这是 WordPiece 和 BPE 的一个关键区别：

```
BPE: "unaffordable" → "un", "afford", "able"
     问题："afford" 作为独立词和作为子词在向量表中是同一个 → 含义混淆

WordPiece: "unaffordable" → "un", "##afford", "##able"
           "afford" 作为独立词 → "afford"（不同的 token ID）
           含义不同 → 向量不同 → 更精准
```

**逻辑链：`##` 前缀让模型区分"词首"和"词中"的子词**
```
"##ing" ≠ "ing"
  → "##ing"（在 "playing" 中）= 进行时态后缀，语义是"进行中"
  → "ing"（如果作为一个词出现）= 可能是一个缩写或专有名词
  → 前端标记区分了语法角色 → 更丰富的表示能力
```


---

## 第八章 · 下游任务适配：微调策略的本质差异

### 8.1 原始 Transformer：任务和架构紧耦合

```
Transformer 设计 = 为翻译而生
  Encoder 处理源语言 → Decoder 自回归生成目标语言
  → 换一个任务（如分类）→ 架构需要大改
  → 没有统一的"预训练→微调"接口
```

### 8.2 BERT：统一的微调接口

BERT 的核心创新之一是**用一个统一的输入格式 + 不同的输出头覆盖各种下游任务**：

```
所有任务的输入格式统一：
  [CLS] + 文本（+ [SEP] + 文本） + [SEP]

不同任务只换输出头：
  分类（情感分析、主题分类）：
    → 只用 [CLS] 输出 → Linear(hidden, num_classes)
  
  NER（命名实体识别）：
    → 用每个 token 的输出 → Linear(hidden, num_labels)
  
  QA（抽取式问答）：
    → 用每个 token 的输出 → Linear(hidden, 2) → start_logits + end_logits
  
  句对分类（NLI、语义相似度）：
    → [CLS] 句子A [SEP] 句子B [SEP] → [CLS] 输出 → 分类
```

### 8.3 微调的关键超参数（面试必问）

| 参数 | 推荐值 | 为什么 |
|------|-------|--------|
| 学习率 | 1e-5 ~ 5e-5 | 预训练权重已经很好，大 lr 会灾难性遗忘 |
| Epochs | 2 ~ 4 | 标注数据少，多 epoch 会过拟合 |
| Batch size | 16 ~ 32 | BERT 参数量大，显存限制 |
| Warmup | 前 10% 步 | 避免初始阶段梯度不稳定破坏预训练权重 |
| 学习率衰减 | 线性衰减到 0 | 训练末期精细化收敛 |

**为什么学习率必须这么小？**
```
逻辑链：
  预训练时，BERT 已经学会了"语言"（词义、语法、常识）
  → 预训练权重是一个近乎最优的语言理解空间
  → 微调只需在这个空间里找到"任务判别边界"
  → 大学习率相当于"推倒重来" → 预训练知识被洗掉
  → 这被称为"灾难性遗忘"（Catastrophic Forgetting）
```


---

## 第九章 · 模型规模与工程细节对比

### 9.1 参数量对比

| 模型 | 层数 | 隐藏维度 | 注意力头数 | FFN 维度 | 参数量 |
|------|------|---------|-----------|---------|--------|
| Transformer (base) | 6E + 6D | 512 | 8 | 2048 | ~65M |
| Transformer (big) | 6E + 6D | 1024 | 16 | 4096 | ~213M |
| BERT-Base | 12E | 768 | 12 | 3072 | 110M |
| BERT-Large | 24E | 1024 | 16 | 4096 | 340M |

**逻辑链：为什么 BERT 可以堆更深？**
```
Transformer 需要训练 Encoder + Decoder，两者都深 → 参数量大、训练难
BERT 只有 Encoder → 同样的参数量预算可以堆更多层
  → 更深的网络 = 更丰富的层次化表示
  → BERT 的下层学语法，中层学语义，上层学任务相关特征
```

### 9.2 Post-LN vs Pre-LN（工程重要差异）

```
原始 Transformer 使用 Post-LN：
  x → Sublayer(x) → Add(x, Sublayer(x)) → LayerNorm
  问题：梯度路径经过 LayerNorm → 深层训练不稳定 → 需要 warmup

BERT/GPT 普遍使用 Pre-LN：
  x → LayerNorm(x) → Sublayer(LayerNorm(x)) → Add(x, output)
  优势：梯度路径"绕开" LayerNorm → 训练更稳定 → 不需要精心调 warmup
```

### 9.3 激活函数：ReLU vs GELU

```
原始 Transformer FFN：ReLU(x) = max(0, x)
BERT FFN：            GELU(x) = x · Φ(x)（高斯误差线性单元）

GELU 相比 ReLU 的优势：
  → ReLU 在 0 处不可导，负半轴梯度为 0（"死神经元"）
  → GELU 处处可导，负半轴有小幅负值输出
  → 对 NLP 的精细语义建模更友好
  → 后来成为 Transformer 类模型的事实标准
```


---

## 第十章 · 各自的局限与演进方向

### 10.1 原始 Transformer 的局限

| 局限 | 原因 | 解决方案 |
|------|------|---------|
| O(n²) 复杂度 | Self-Attention 计算 n×n 矩阵 | Longformer, BigBird（稀疏注意力） |
| 需要大量平行语料 | Seq2Seq 监督训练 | BERT 式预训练（无监督） |
| 位置编码外推有限 | sin/cos 在高频维度上外推不准 | RoPE, ALiBi |
| 训练不稳定 | Post-LN + 深层网络 | Pre-LN, 更好的初始化 |

### 10.2 BERT 的局限

| 局限 | 原因 | 解决方案 |
|------|------|---------|
| 序列长度限制 512 | 可学习位置编码只训练了 0-511 | Longformer, BigBird (稀疏注意力) |
| 无法生成 | Encoder-only，双向注意力违反因果性 | UniLM, BART (统一框架) |
| [MASK] 在微调时不可见 | 预训练-微调分布不匹配 | 80/10/10 策略缓解，XLNet (排列语言模型) |
| 每个 token 计算量相同 | 没有 focus 机制 | Funnel-Transformer (逐步压缩序列) |
| 只支持单语 | 英文预训练 | mBERT, XLM-R (多语言预训练) |
| NSP 有害 | 任务太简单, 与 MLM 冲突 | RoBERTa (去掉), ALBERT (换成 SOP) |

### 10.3 演进图谱

```
                    原始 Transformer (2017)
                    Encoder-Decoder, sin/cos PE
                         /          \
                        /            \
              BERT (2018)           GPT (2018)
           Encoder-only            Decoder-only
           可学习 PE                可学习 PE
           MLM + NSP               自回归 LM
                |                      |
          RoBERTa (2019)          GPT-2 (2019)
          去 NSP 提纯 MLM           更大规模
                |                      |
          ALBERT (2019)            GPT-3 (2020)
          参数共享压缩               175B, 少样本学习
                |                      |
          DeBERTa (2020)           ChatGPT (2022)
          解耦注意力                 RLHF 对齐
                |                      |
          -------------------------------------------------
                            |
                    Modern LLMs (2023+)
                    RoPE, GQA, Flash Attention, MoE...
```


---

## 第十一章 · 总结：每一条差异背后的完整逻辑链

### 11.1 差异全景图

| # | 差异维度 | Transformer | BERT | 逻辑根因 |
|---|---------|------------|------|---------|
| 1 | **架构** | Encoder-Decoder | Encoder-only | BERT 不做生成，只需理解 → Decoder 是负担 |
| 2 | **位置编码** | sin/cos 固定函数 | 可学习 Embedding | BERT 序列长度固定(512)，不需要外推 → 可学习更适配 |
| 3 | **Segment 编码** | ❌ 无 | ✅ Segment Embedding | BERT 把两个句子拼成一个序列 → 需要区分句子归属 |
| 4 | **特殊 Token** | ❌ 无 | [CLS], [SEP], [MASK] | 预训练任务(NSP/MLM)和微调接口需要的"协议 token" |
| 5 | **注意力方向** | 双向 + 单向 + 交叉 | 仅双向 Self-Attention | 理解任务 = 深度双向上下文；生成和翻译不需要 |
| 6 | **预训练任务** | 无（任务专训） | MLM + NSP | 核心创新：利用无标注文本学习通用语言表示 |
| 7 | **训练范式** | 每个任务从零训 | 预训练 + 微调 | 预训练学会"语言"，微调学会"任务" → 少量标注即可 |
| 8 | **分词方式** | BPE | WordPiece (##前缀) | WordPiece 的 ## 前缀区分词首/词中子词的语法角色 |
| 9 | **激活函数** | ReLU | GELU | GELU 处处可导，对 NLP 精细语义建模更友好 |
| 10 | **层归一化** | Post-LN | Pre-LN（实践中） | Pre-LN 训练更稳定，深层网络不依赖精心 warmup |
| 11 | **序列长度** | 理论不限(sin/cos外推) | 固定 512 | 可学习位置编码的代价：训练过的位置才有效 |
| 12 | **输出方式** | Decoder 逐词生成 | [CLS] / token 输出 → 任务头 | 理解 vs 生成 → 不同的输出消费方式 |

### 11.2 根本分歧点：一切差异的源头

所有差异都可以追溯到**两条根本不同的设计路线**：

```
路线A：原始 Transformer                        路线B：BERT
─────────────────────                        ─────────────
"如何完成序列到序列的转换？"                  "如何学到通用的语言理解表示？"
          ↓                                            ↓
    需要 Encoder（理解源语言）                   只需要 Encoder（理解文本）
    需要 Decoder（生成目标语言）                 不需要 Decoder（不生成）
    需要 Cross-Attention（融合源-目标）          不需要 Cross-Attention（没有目标序列）
    需要 Masked-Attention（因果生成约束）       不需要 Mask（需要双向上下文）
          ↓                                            ↓
    每个任务从零开始训练                          预训练 MLM + NSP → 微调
    有监督（需要平行/标注数据）                    自监督（用无标注文本）
    架构决定任务                                   任务适配架构
          ↓                                            ↓
    翻译/Seq2Seq 专用                              通用 NLP 任务
```

### 11.3 为什么 BERT 的设计"赢了"但最终被 GPT "超越"？

这是一个更深层的问题，值得展开：

**BERT "赢"在 2018-2020 的理解任务：**
- NLU benchmark（GLUE, SuperGLUE）全线超越
- 微调范式极大降低了 NLP 任务的标注成本
- Encoder-only 架构对理解任务是最优解

**GPT 路线"后来居上"的原因（2020+）：**
- **生成是更通用的能力**：理解了文本不一定能生成，但能生成一定意味着理解
- **Scaling Law**：Decoder-only 架构在规模化扩展时表现更稳定（GPT-3 175B）
- **In-Context Learning**：超大 Decoder-only 模型涌现了零样本/少样本学习能力，不需要微调
- **统一任务格式**：所有任务都可以转化为"续写"——分类、翻译、问答都一样

```
BERT 路线：理解专用，需要为每个任务微调 → 上限受微调数据限制
GPT 路线：生成通用，大规模后涌现零样本能力 → 上限受模型规模限制
         → Scale 是更"容易"的路线（更多算力 vs 更多标注数据）
```

### 11.4 核心认知

学习 Transformer 到 BERT 的差异，最重要的认知收获不是记住了哪些参数不同，而是理解：

> **每个设计决策都源于"你要解决什么问题"——任务决定了架构，架构决定了每个组件的形式。**

- Transformer 要解决**翻译** → 需要 Encoder-Decoder
- BERT 要解决**理解** → 只需要 Encoder
- GPT 要解决**生成** → 只需要 Decoder

理解了这条"任务 → 架构 → 组件"的因果链，就不会再混淆三者之间的差异。


---

## 附录 A：关键公式速查

| 组件 | 公式 |
|------|------|
| Scaled Dot-Product Attention | $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| Multi-Head Attention | $\text{MultiHead} = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$ |
| 正弦位置编码 | $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$ |
| BERT 输入 | $\text{Input} = E_{token} + E_{position} + E_{segment}$ |
| MLM 损失 | $\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \hat{x})$ |
| FFN（Transformer） | $\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$ |
| FFN（BERT） | $\text{FFN}(x) = \text{GELU}(xW_1+b_1)W_2+b_2$ |
| Layer Normalization | $\text{LayerNorm}(x) = \gamma \cdot \frac{x-\mu}{\sigma} + \beta$ |


## 附录 B：面试高频问题速答

**Q1: BERT 为什么不能做文本生成？**
> BERT 是 Encoder-only，用双向 Self-Attention（无 causal mask）。生成需要自回归（第 t 步只看 1~t-1），双向注意力天然违反因果性。硬要用 BERT 生成只能用 [MASK] 逐步替换，速度慢、质量差。

**Q2: 为什么要除以 √d_k？**
> 假设 Q、K 各分量独立、均值 0、方差 1，则 Q·K 的方差 = d_k，标准差 = √d_k。当 d_k=64 时点积约 8，输入 softmax 会饱和 → 梯度消失。除以 √d_k 把方差拉回 1。

**Q3: 为什么 BERT 用可学习位置编码而不是 sin/cos？**
> BERT 序列长度固定 512，不需要 sin/cos 的外推优势。可学习编码让梯度自己优化出最适合预训练数据的位置表示。同时 [CLS]/[SEP] 的特殊位置语义需要在训练中学习。

**Q4: MLM 为什么是 80/10/10 而不是全 mask？**
> 全 mask → 预训练有 [MASK]，微调没有 → 分布不匹配。80/10/10 让模型对每个位置都不确定（是 mask/随机词/原词？），被迫认真建模上下文。

**Q5: NSP 为什么后来被去掉了？**
> 负样本来自不同文档，模型只需判断"话题是否相同"→ 太简单，学不到真正的句子连贯性。RoBERTa 去掉 NSP 后效果反而更好。ALBERT 换成 SOP（同文档、顺序不同，难度更高）效果更好。

**Q6: [CLS] token 的表示能力是天生的吗？**
> 不是。是 NSP 预训练任务"训练"出来的——NSP 的梯度让 [CLS] 学会聚合全序列信息。对于语义相似度任务，均值池化通常优于 [CLS]（Sentence-BERT 的发现）。

**Q7: 为什么 BERT 微调学习率要这么小（1e-5~5e-5）？**
> 预训练权重已经学好了语言理解。大学习率 → 灾难性遗忘（预训练知识被洗掉）。小学习率 = 在现有语言理解空间里找到"任务判别边界"。


## 附录 C：相关文档索引

本文档目录下的相关文档：
- [[01_Transformer注意力机制]] — Transformer 注意力机制的完整推导与代码实现
- [[02_BERT预训练与微调]] — BERT 预训练任务与微调的详细代码示例
- [[03_Attention的Score函数F对比]] — 加性 vs 点积注意力的深度对比
- [[04_从RNN到Transformer架构演进]] — NLP 架构从 RNN 到 Transformer 的演进脉络
- [[05_FastText子词机制]] — 子词思想的起源与 BPE/WordPiece 的前身
- [[35_Transformer原理与实现]] — Transformer 精简版总结
- [[36_BERT预训练模型]] — BERT 精简版总结
- [[LSTM_ht_vs_ct_Attention选K_V的逻辑]] — LSTM 输出与 Attention K/V 选择的深层逻辑


## 附录 D：参考资料

1. Vaswani et al. (2017). **Attention Is All You Need**. NeurIPS 2017.
2. Devlin et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL 2019.
3. Liu et al. (2019). **RoBERTa: A Robustly Optimized BERT Pretraining Approach**.
4. Lan et al. (2020). **ALBERT: A Lite BERT for Self-supervised Learning**.
5. Reimers & Gurevych (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**.
6. Jay Alammar. **The Illustrated Transformer / Illustrated BERT**. (强烈推荐的可视化教程)
7. Harvard NLP. **The Annotated Transformer**. (代码级详细实现)
8. Lilian Weng. **The Transformer Family**. (各变体综述)

