# 命名实体识别（NER）

## 📌 核心问题
> 给定一段文本，自动找出其中"有名字的东西"——人名、地名、机构名、时间、金额等，并标出它们的边界和类型。这是 NLP 中最基础也最实用的序列标注任务之一。

---

## 🌱 根源与动机

### 什么是命名实体识别？

**定义：** 从非结构化文本中识别出具有特定意义的实体（Named Entity），并将其分类到预定义的类别中。

**输入：** 一段自然语言文本（token 序列）
**输出：** 每个 token 对应的实体标签

**例子：**

```
输入: "马云 创立了 阿里巴巴 ，总部位于 杭州"
输出:  B-PER O   B-ORG I-ORG  O    O   B-LOC
```

### 常见实体类型

| 类型 | 英文缩写 | 示例 |
|------|---------|------|
| 人名 | PER | 张三、马云、乔布斯 |
| 地名 | LOC | 北京、杭州、硅谷 |
| 组织机构 | ORG | 阿里巴巴、清华大学、联合国 |
| 时间 | TIME | 2024年、下午三点、昨天 |
| 金额 | MONEY | 100万元、$500、三十亿 |
| 产品/品牌 | PROD | iPhone、微信、ChatGPT |
| 非实体 | O | 的、了、在、创立 |

### 为什么 NER 是序列标注任务？

核心原因：**实体边界信息存在于每一个 token 上**，不是整句话的一个标签。

- 情感分析："这部电影很好看" → 一个标签（正面）
- NER："马云创立阿里巴巴" → 每个 token 都需要一个标签

如果只输出"发现了人名：马云，组织名：阿里巴巴"，我们需要知道每个词属于哪个实体的哪个部分。因此必须对**每个 token 独立打标**。

---

## 📐 理论推导

### BIO 标注体系

BIO（也叫 IOB）是 NER 中最常用的标注方案：

| 前缀 | 含义 | 用途 |
|------|------|------|
| B- | Begin（开始） | 实体的第一个 token |
| I- | Inside（内部） | 实体的非第一个 token |
| O | Outside（外部） | 不属于任何实体 |

**为什么需要区分 B 和 I？**

考虑两个相邻的同类实体：
```
"北京 上海 都是 大城市"
 B-LOC B-LOC  O    O
```
如果不区分 B/I，两个地名连在一起会被误判为一个实体。

**完整标签集合示例（3类实体）：**
```
O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
共 7 个标签
```

**实际标注例子：**

```
句子:  "李 明 在 北 京 大 学 学 习"
BIO:   B  I  O  B  I  I  I  O  O
类型: PER  -  -  ORG  -  -  -  -  -

完整标签:
李: B-PER
明: I-PER
在: O
北: B-ORG
京: I-ORG
大: I-ORG
学: I-ORG
学: O
习: O
```

### 用 RNN 做 NER 的网络结构

```
输入 token 序列
      ↓
  Embedding 层     (token id → 稠密词向量)
      ↓
   RNN 层          (每步更新 hidden state，融合上下文)
      ↓
  FC 层（每步）    (hidden_dim → num_tags)
      ↓
 标签预测序列      (每个 token 对应一个标签)
```

**数学表达：**

$$e_t = \text{Embedding}(x_t)$$
$$h_t = \text{RNN}(e_t, h_{t-1})$$
$$\hat{y}_t = \text{softmax}(W_{fc} \cdot h_t + b)$$

**为什么用 RNN 而不是逐词独立分类？**

RNN 的 $h_t$ 融合了"北"之前的所有上下文（"在"、主语等），所以模型在判断"北"是 B-ORG 还是 B-LOC 时，能参考前面已经看过的词。

---

## 💡 关键理解

### BIO 解码规则

预测完每个 token 的标签后，按以下规则还原实体：

```
规则1: B-X 开启一个类型为 X 的实体
规则2: I-X 紧跟 B-X 或 I-X 时，属于同一个实体（类型必须一致）
规则3: O 或类型不匹配的 I-X 结束当前实体
```

```python
def decode_bio(tokens, labels):
    """将 BIO 标签序列还原为实体列表"""
    entities = []
    current_entity = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # 新实体开始
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {'type': entity_type, 'tokens': [token], 'start': i}

        elif label.startswith('I-') and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['tokens'].append(token)
            else:
                # 类型不匹配，结束前一个实体
                entities.append(current_entity)
                current_entity = None
        else:
            # O 标签，结束当前实体
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities
```

### 类别不平衡问题

在真实数据中，O 标签占绝大多数（通常 70%~90%），实体标签稀少。
- 模型容易"偷懒"：全预测 O 也能获得高准确率
- 解决方案：加权交叉熵损失、使用 F1 而非 accuracy 评估

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. 构建词表和标签表
# ============================================================

# 模拟小型词表
vocab = {'<PAD>': 0, '<UNK>': 1, '马': 2, '云': 3, '创': 4, '立': 5,
         '了': 6, '阿': 7, '里': 8, '巴': 9, '总': 10, '部': 11,
         '位': 12, '于': 13, '杭': 14, '州': 15}

# BIO 标签表（3类实体：人名PER，组织ORG，地名LOC）
tag2idx = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6
}
idx2tag = {v: k for k, v in tag2idx.items()}
num_tags = len(tag2idx)  # 7


# ============================================================
# 2. RNN-NER 模型定义
# ============================================================

class RNN_NER(nn.Module):
    """
    基于 RNN 的命名实体识别模型
    结构：Embedding → RNN → FC → 每步标签预测
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=0    # PAD token 对应的 embedding 全0
        )
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False  # 可改为 True 使用双向RNN
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        """
        x: (batch, seq_len)  — token id 序列
        返回: (batch, seq_len, num_tags)  — 每步各标签的 logit
        """
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        output, _ = self.rnn(embedded)        # output: (batch, seq_len, hidden_dim)
        output = self.dropout(output)

        logits = self.fc(output)              # (batch, seq_len, num_tags)
        return logits


# ============================================================
# 3. 构造训练数据（示例）
# ============================================================

# 句子："马云创立了阿里巴巴"
# 标签：B-PER I-PER O O O B-ORG I-ORG I-ORG I-ORG
sentence_ids = [2, 3, 4, 5, 6, 7, 8, 9, 9]   # token ids
label_ids    = [1, 2, 0, 0, 0, 3, 4, 4, 4]   # tag ids

# 转为 tensor，增加 batch 维度
x_train = torch.tensor([sentence_ids], dtype=torch.long)   # (1, 9)
y_train = torch.tensor([label_ids],    dtype=torch.long)   # (1, 9)


# ============================================================
# 4. 训练示例
# ============================================================

vocab_size = len(vocab)   # 16
embed_dim  = 32
hidden_dim = 64

model = RNN_NER(vocab_size, embed_dim, hidden_dim, num_tags)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 注意：CrossEntropyLoss 接受 (N, C, ...) 或先 reshape
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(30):
    optimizer.zero_grad()
    logits = model(x_train)                      # (1, 9, 7)

    # reshape: (batch*seq_len, num_tags) vs (batch*seq_len,)
    loss = criterion(
        logits.view(-1, num_tags),               # (9, 7)
        y_train.view(-1)                         # (9,)
    )
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ============================================================
# 5. 推理与 BIO 解码
# ============================================================

def decode_bio(tokens, labels):
    """将 BIO 标签还原为实体列表"""
    entities, current = [], None
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            if current:
                entities.append(current)
            current = {'type': label[2:], 'text': token, 'start': i}
        elif label.startswith('I-') and current and label[2:] == current['type']:
            current['text'] += token
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities


model.eval()
with torch.no_grad():
    logits = model(x_train)                        # (1, 9, 7)
    pred_ids = logits.argmax(dim=-1)[0].tolist()   # (9,)

tokens = ['马', '云', '创', '立', '了', '阿', '里', '巴', '巴']
pred_labels = [idx2tag[i] for i in pred_ids]

print("\n=== 推理结果 ===")
for token, label in zip(tokens, pred_labels):
    print(f"  {token} → {label}")

entities = decode_bio(tokens, pred_labels)
print("\n识别到的实体：")
for e in entities:
    print(f"  [{e['type']}] {e['text']} (位置:{e['start']})")


# ============================================================
# 6. 评估：NER 常用 F1 指标
# ============================================================

def compute_f1(pred_entities, gold_entities):
    """
    计算实体级别的 F1（完全匹配：类型+边界都正确才算 TP）
    """
    pred_set = set([(e['type'], e['text']) for e in pred_entities])
    gold_set = set([(e['type'], e['text']) for e in gold_entities])

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall    = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


gold_labels = [idx2tag[i] for i in label_ids]
gold_entities = decode_bio(tokens, gold_labels)
metrics = compute_f1(entities, gold_entities)
print(f"\nF1 评估: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
```

---

## ⚠️ 易错点与常见误解

**1. I- 标签必须紧跟 B- 或同类型 I-，不能独立出现**
```
错误的标签序列：O I-PER B-PER   ← I-PER 出现在开头，违反规范
正确的标签序列：B-PER I-PER O
```
解码时遇到孤立的 I- 标签，通常视为预测错误，直接忽略或转为 B-。

**2. NER 不用 accuracy，要用实体级 F1**
- accuracy = (预测正确的 token 数) / 总 token 数，因 O 类别占比大，accuracy 虚高
- 标准评估：精确率、召回率、F1（seqeval 库是标准工具）
```python
from seqeval.metrics import f1_score, classification_report
```

**3. 两个相邻同类实体的边界**
```
"北京 上海"（两个地名）
正确: B-LOC B-LOC    ← 第二个必须是 B，不能是 I
错误: B-LOC I-LOC    ← 会被解码为一个实体"北京上海"
```

**4. 损失函数 reshape 顺序**
```python
# 必须先 reshape 再计算 loss
# 正确
loss = criterion(logits.view(-1, num_tags), targets.view(-1))

# 错误（不要在 seq_len 维度上操作）
loss = criterion(logits, targets)  # 维度不匹配会报错
```

**5. 双向 RNN 后 FC 的输入维度要乘2**
```python
# 单向 RNN
self.fc = nn.Linear(hidden_dim, num_tags)

# 双向 RNN（bidirectional=True）
self.fc = nn.Linear(hidden_dim * 2, num_tags)  # 前向+后向拼接
```

---

## 🔗 知识延伸

- **LSTM-NER**：用 LSTM 替换 RNN，缓解梯度消失，处理更长实体依赖
- **BiLSTM-CRF**：工业界经典方案，CRF 层在 FC 层之后建模标签之间的转移约束（如 O 后不能直接跟 I-PER）
- **BERT-NER**：预训练 Transformer 作为 Encoder，每个 token 的 [CLS] representation 接 FC，目前性能最强
- **BIOES 标注体系**：比 BIO 更细，多了 E-（End）和 S-（Single），边界更明确
- **数据集**：CoNLL-2003（英文）、OntoNotes 5.0、MSRA NER（中文）

---

## 📚 参考资料

- Lample et al. (2016). *Neural architectures for named entity recognition.* NAACL.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
- seqeval 库文档：https://github.com/chakki-works/seqeval
- 《自然语言处理入门》第 9 章：序列标注
