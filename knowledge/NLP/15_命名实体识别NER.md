# 命名实体识别（NER）

## 📌 核心问题
> 如何让模型自动识别文本中的"人名、地名、机构名"等有意义的实体？NER 为什么是典型的序列标注任务？

---

## 🌱 根源与动机

### 什么是命名实体识别

NER（Named Entity Recognition）是 NLP 中的基础任务：给定一段文本，找出其中所有**命名实体**，并标注其类型。

常见实体类型：

| 类型 | 缩写 | 例子 |
|------|------|------|
| 人名 | PER | 张三、李四、马斯克 |
| 地名 | LOC | 北京、上海、纽约 |
| 机构名 | ORG | 北京大学、苹果公司 |
| 时间 | TIME | 2026年5月、明天下午 |
| 数量 | NUM | 三百万、50% |

### 为什么是序列标注任务

关键点：**实体的识别依赖上下文，且每个 token 都需要单独判断**。

- "张三" → 可能是人名，但也可能在其他语境下是其他意思
- 模型需要看每个 token 的周围上下文才能确定它是否是实体的一部分
- 最终输出是和输入**等长的标签序列**：每个 token 对应一个标签

这与**序列分类**（整句一个标签）本质不同：NER 是 **token 级别** 的多分类。

---

## 📐 理论推导

### BIO 标注体系

NER 最常用的标注格式是 **BIO（也叫 IOB2）**：

| 标签 | 含义 |
|------|------|
| **B-TYPE** | 实体的**开始**（Begin） |
| **I-TYPE** | 实体的**内部**（Inside） |
| **O** | **非实体**（Outside） |

### 标注示例

句子：`张三 在 北京大学 读书`

```
张  →  B-PER   （人名开始）
三  →  I-PER   （人名内部）
在  →  O       （非实体）
北  →  B-ORG   （机构名开始）
京  →  I-ORG   （机构名内部）
大  →  I-ORG   （机构名内部）
学  →  I-ORG   （机构名内部）
读  →  O       （非实体）
书  →  O       （非实体）
```

### 标签集构造

如果有 PER、LOC、ORG 三种实体类型，标签集为：

```
O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
→ 共 7 个标签（1 + 3×2）
```

每个位置做 7 分类。

### RNN 做 NER 的完整流程

```
输入: ["张", "三", "在", "北", "京", "大", "学", "读", "书"]
         ↓  token → id
token_id: [5, 23, 7, 88, 91, 34, 12, 67, 45]
         ↓  Embedding
词向量:   [e_1, e_2, e_3, ..., e_9]   每个 shape: (embed_dim,)
         ↓  RNN（双向更好）
hidden:   [h_1, h_2, h_3, ..., h_9]   每个 shape: (hidden_dim,)
         ↓  FC（每步独立）
logits:   [l_1, l_2, l_3, ..., l_9]   每个 shape: (num_tags,)
         ↓  argmax
标签:     [B-PER, I-PER, O, B-ORG, I-ORG, I-ORG, I-ORG, O, O]
```

---

## 💡 关键理解

### NER vs 序列分类的本质区别

| 对比维度 | 序列分类 | 序列标注（NER） |
|---------|----------|----------------|
| 输出粒度 | 整句一个标签 | 每个 token 一个标签 |
| 用到的 hidden | 只用最后一步 $h_T$ | 每步 $h_t$ 都用 |
| 典型任务 | 情感分析、主题分类 | NER、词性标注、分词 |
| 输出 shape | `(batch, num_classes)` | `(batch, seq_len, num_tags)` |

### BIO 为什么不直接用 PER/LOC/ORG/O？

不用 B/I 区分时，连续两个同类型实体会粘连：

```
"张三 李四" 没有 B/I 区分：PER PER → 无法判断是一个实体还是两个
"张三 李四" 有 B/I 区分：B-PER I-PER B-PER I-PER → 明确是两个独立实体
```

BIO 体系让模型能处理**相邻同类型实体**的边界问题。

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 数据准备
# ============================================================

# 词汇表（简化版）
word2idx = {"<PAD>": 0, "<UNK>": 1, "张": 2, "三": 3, "在": 4,
            "北": 5, "京": 6, "大": 7, "学": 8, "读": 9, "书": 10}

# 标签集（BIO 体系）
tag2idx = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
           "B-LOC": 5, "I-LOC": 6}
idx2tag = {v: k for k, v in tag2idx.items()}

# 示例句子："张三在北京大学读书"
sentence = ["张", "三", "在", "北", "京", "大", "学", "读", "书"]
labels   = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O"]

# 转成 id
x = torch.tensor([[word2idx.get(w, 1) for w in sentence]])   # (1, 9)
y = torch.tensor([[tag2idx[t] for t in labels]])              # (1, 9)

print(f"输入 shape: {x.shape}")   # torch.Size([1, 9])
print(f"标签 shape: {y.shape}")   # torch.Size([1, 9])


# ============================================================
# 模型定义：RNN + FC 序列标注（TTwithFC 结构）
# ============================================================

class RNN_NER(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 用双向 RNN，每步同时看到左右上下文（NER 推荐做法）
        self.rnn = nn.RNN(embed_dim, hidden_dim,
                          batch_first=True, bidirectional=True)
        # 双向 RNN 输出维度是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))    # (batch, seq_len, embed_dim)
        out, _ = self.rnn(emb)                   # out: (batch, seq_len, hidden_dim*2)
        logits = self.fc(out)                    # (batch, seq_len, num_tags)
        return logits


# ============================================================
# 训练一步（演示前向传播和 loss 计算）
# ============================================================

VOCAB_SIZE = len(word2idx)
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_TAGS = len(tag2idx)

model = RNN_NER(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_TAGS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 前向传播
logits = model(x)                    # (1, 9, 7)
print(f"logits shape: {logits.shape}")

# CrossEntropyLoss 需要将 seq_len 维度展开
# logits: (1, 9, 7) → (9, 7)
# y:      (1, 9)    → (9,)
loss = criterion(logits.view(-1, NUM_TAGS), y.view(-1))
print(f"Loss: {loss.item():.4f}")

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()


# ============================================================
# 预测与解码
# ============================================================

model.eval()
with torch.no_grad():
    logits = model(x)                           # (1, 9, 7)
    pred_ids = logits.argmax(dim=-1)            # (1, 9)
    pred_tags = [idx2tag[i.item()] for i in pred_ids[0]]

print("\n预测结果：")
for word, true_tag, pred_tag in zip(sentence, labels, pred_tags):
    status = "✅" if true_tag == pred_tag else "❌"
    print(f"  {word}  真实: {true_tag:<8} 预测: {pred_tag:<8} {status}")
```

**运行输出示例：**
```
输入 shape: torch.Size([1, 9])
标签 shape: torch.Size([1, 9])
logits shape: torch.Size([1, 9, 7])
Loss: 1.9812

预测结果：
  张  真实: B-PER    预测: O        ❌
  三  真实: I-PER    预测: O        ❌
  ...（未训练，随机预测，正常）
```

> 注意：这里只演示了结构，未经过多轮训练，预测结果不准确是正常的。

---

## ⚠️ 易错点与常见误解

**1. loss 计算时维度错误（最常见错误）**

`nn.CrossEntropyLoss` 期望输入是 `(N, C)`，标签是 `(N,)`。
序列标注时 logits 是 `(batch, seq_len, num_tags)`，必须先 `.view(-1, num_tags)` 展开：

```python
# 错误写法（直接传 3D tensor）
loss = criterion(logits, y)  # ❌ 报错

# 正确写法（展开后计算）
loss = criterion(logits.view(-1, num_tags), y.view(-1))  # ✅
```

**2. BIO 标签体系中 I 标签不能单独出现**

预测结果里如果出现 "O → I-PER" 这种 I 开头没有 B 的情况，属于非法序列。
实际工程中会加 **CRF（条件随机场）层**约束标签转移，强制 B/I 合法。

**3. 双向 RNN 输出维度是 2 倍**

`bidirectional=True` 时，`out` 的最后一维是 `hidden_dim * 2`（正向 + 反向拼接）。
FC 层输入维度必须对应调整，否则报 shape 不匹配错误。

**4. padding 位置参与了 loss 计算**

批量处理变长序列时，padding 位置也会产生 loss。
应使用 `nn.CrossEntropyLoss(ignore_index=0)` 忽略 padding 位置的 loss。

**5. NER 评估指标不用 accuracy，用 F1**

NER 实体稀疏（O 标签占多数），accuracy 虚高没意义。
标准评估用**实体级别的 Precision / Recall / F1**（seqeval 库）：
```python
from seqeval.metrics import f1_score
f1_score(true_tags, pred_tags)
```

---

## 🔗 知识延伸

- **BiLSTM-CRF**：NER 的经典架构，LSTM 提取特征，CRF 约束标签转移合法性
- **BERT + NER**：用预训练模型做 token embedding，大幅提升 NER 精度
- **词性标注（POS Tagging）**：同为序列标注任务，模型结构与 NER 完全一样，只是标签集不同
- **分词（Chinese Word Segmentation）**：中文 NLP 基础任务，也可用 BIO 体系（B=词开始，I=词内部）
- **BIOES 标注体系**：BIO 的扩展，增加 E（实体结束）和 S（单字实体），更精细

---

## 📚 参考资料

- Lafferty et al., *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data* (2001)
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019)
- seqeval 库：https://github.com/chakki-works/seqeval
- Stanford NER：https://nlp.stanford.edu/software/CRF-NER.html
