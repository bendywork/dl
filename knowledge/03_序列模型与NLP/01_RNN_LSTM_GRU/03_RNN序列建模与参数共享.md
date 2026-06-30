# RNN序列建模与参数共享

## 📌 核心问题
> FC（全连接）网络处理序列数据时有哪些本质缺陷？RNN 如何解决这些问题？

---

## 🌱 根源与动机

### FC 网络处理序列的局限

用 FC 网络处理文本/时序数据，面临两个根本问题：

**1. 固定长度输入**
FC 的输入维度在网络定义时就固定了。句子长度是变化的，要么截断要么 padding，都是信息损失或噪声引入。

**2. 忽略顺序信息**
FC 把输入向量铺平（flatten）后处理，"今天 天气 好" 和 "天气 好 今天" 对它是一样的。但序列的语义往往强依赖顺序。

**3. 参数量随长度爆炸**
如果强行把长度为 T 的序列铺平输入 FC，参数量是 `T × hidden_dim`，T 一大就不可控。

---

## 📐 理论推导

### RNN 的核心公式

RNN 的核心思想：**用一个隐藏状态（hidden state）来携带"记忆"，每步更新**。

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

符号说明：

| 符号 | 含义 |
|------|------|
| $x_t$ | 第 t 步的输入（如第 t 个 token 的词向量） |
| $h_t$ | 第 t 步的隐藏状态（携带历史信息） |
| $h_{t-1}$ | 上一步的隐藏状态 |
| $W_{hh}$ | 隐藏到隐藏的权重矩阵 |
| $W_{xh}$ | 输入到隐藏的权重矩阵 |
| $y_t$ | 第 t 步的输出（序列标注时每步都有） |

### 参数共享的关键

**所有时间步 t=1,2,...,T 共用同一套参数** $W_{hh}, W_{xh}, W_{hy}$。

这带来两个好处：
- 参数量与序列长度无关（固定大小）
- 天然支持变长序列（循环处理，直到序列结束）

---

## 💡 关键理解

### Embedding 层的作用

在 RNN 之前，输入通常是 token id（整数），需要先经过 Embedding 层转成稠密向量：

```
token_id → Embedding → dense vector (词向量)
```

为什么不用 one-hot？
- one-hot 维度 = 词汇表大小（可能几万维），非常稀疏
- 词向量（dense representation）低维（如 128 维），且能捕获语义相似性
- Embedding 层本质是一个可学习的查找表（lookup table）

### RNN 的两种使用场景

| 场景 | 操作 | 典型任务 |
|------|------|----------|
| **序列分类** | 只取最后一步 $h_T$，接 FC 输出类别 | 情感分析、文本分类 |
| **序列标注** | 每一步 $h_t$ 都接 FC 输出标签 | NER、词性标注、机器翻译 |

### TTwithFC 结构（序列标注）

```
输入序列:   x_1   x_2   x_3   ...   x_T
              ↓     ↓     ↓           ↓
Embedding: emb_1 emb_2 emb_3  ...  emb_T
              ↓     ↓     ↓           ↓
RNN:        h_1 → h_2 → h_3 → ... → h_T
              ↓     ↓     ↓           ↓
FC层:       y_1   y_2   y_3   ...   y_T
```

每个位置都有独立输出，适合需要对每个 token 打标签的任务。

### FC 网络 vs RNN 核心区别

| 对比维度 | FC 网络 | RNN |
|---------|---------|-----|
| 输入长度 | 固定（网络定义时确定） | 可变（循环处理） |
| 顺序信息 | 忽略（flatten 后无序） | 保留（h 传递历史） |
| 参数共享 | 每层独立参数 | 所有时间步共享参数 |
| 适合任务 | 图像分类、表格数据 | 序列分类、序列标注、时序预测 |
| 记忆机制 | 无 | hidden state 传递上下文 |

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn

# ============================================================
# 场景一：序列分类（取最后一步 hidden state）
# ============================================================

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) — token id 序列
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        out, h_n = self.rnn(emb)         # out: (batch, seq_len, hidden_dim)
                                         # h_n: (1, batch, hidden_dim) — 最后一步
        last_h = h_n.squeeze(0)          # (batch, hidden_dim)
        logits = self.fc(last_h)         # (batch, num_classes)
        return logits


# ============================================================
# 场景二：序列标注（每个时间步都输出标签）TTwithFC 结构
# ============================================================

class RNNTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)  # 每步都过这个 FC

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        out, _ = self.rnn(emb)           # out: (batch, seq_len, hidden_dim)
        logits = self.fc(out)            # (batch, seq_len, num_tags) ← 每步都打标签
        return logits


# ============================================================
# 验证两种结构的输出 shape
# ============================================================

VOCAB_SIZE = 1000
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 3   # 情感分类：正/中/负
NUM_TAGS = 5      # NER 标签：O, B-PER, I-PER, B-LOC, I-LOC

batch_size = 4
seq_len = 10

x = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len))  # 模拟 token id

# 序列分类
classifier = RNNClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES)
out_cls = classifier(x)
print(f"序列分类输出 shape: {out_cls.shape}")   # (4, 3)

# 序列标注
tagger = RNNTagger(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_TAGS)
out_tag = tagger(x)
print(f"序列标注输出 shape: {out_tag.shape}")   # (4, 10, 5)
```

**运行输出：**
```
序列分类输出 shape: torch.Size([4, 3])
序列标注输出 shape: torch.Size([4, 10, 5])
```

---

## ⚠️ 易错点与常见误解

**1. `batch_first` 参数容易忽略**
PyTorch 的 `nn.RNN` 默认输入格式是 `(seq_len, batch, input_size)`，而不是 `(batch, seq_len, input_size)`。
务必加 `batch_first=True`，否则维度顺序错误，结果完全乱掉。

**2. `out` 和 `h_n` 的区别**
`rnn(emb)` 返回两个值：
- `out`：**所有**时间步的 hidden state，shape `(batch, seq_len, hidden_dim)`
- `h_n`：**最后一步**的 hidden state，shape `(1, batch, hidden_dim)`（1 层时）

序列分类用 `h_n`（或 `out[:, -1, :]`，两者等价）；序列标注用 `out`。

**3. Embedding `padding_idx` 不要忘**
pad token 对应的 id 应设 `padding_idx=0`，使其词向量始终为全零，不参与梯度更新，避免 padding 位置污染模型。

**4. 基础 RNN 有梯度消失问题**
长序列时 $h_T$ 对 $h_1$ 的梯度几乎为 0，"记不住"很远的信息。实际工程中优先用 LSTM 或 GRU，而非 `nn.RNN`。

**5. 序列分类不要用 `out` 的所有步求平均（初学常见错误）**
用平均池化（`out.mean(dim=1)`）虽然也能跑，但有 padding 位置时会引入噪声。建议取最后有效位置的 hidden state，或做 masked 均值。

---

## 🔗 知识延伸

- **LSTM / GRU**：解决 RNN 梯度消失问题，引入门控机制，是 RNN 的升级版
- **双向 RNN（BiRNN）**：正向 + 反向两个 RNN，每个位置同时看到左右上下文，NER 常用
- **Attention 机制**：让模型动态关注序列的不同位置，而不依赖单一的 $h_T$
- **Transformer**：彻底抛弃循环，用纯注意力建模序列，是现代 NLP 的主流架构
- **梯度消失详解**：见 `10_梯度消失与ReLU详解.md`

---

## 📚 参考资料

- Stanford CS224n: Natural Language Processing with Deep Learning
- PyTorch 官方文档：`torch.nn.RNN`
- Goodfellow et al., *Deep Learning*, Chapter 10: Sequence Modeling
