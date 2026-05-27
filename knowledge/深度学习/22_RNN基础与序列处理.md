# RNN基础与序列处理

## 📌 核心问题
> FC（全连接）网络如何处理"一句话"这样的序列数据？做不到——它不记得前面说了什么。RNN 通过引入 hidden state（隐状态），让网络在处理每个 token 时都能"记住"之前的上下文。

---

## 🌱 根源与动机

### FC 网络处理序列的两大致命局限

**局限一：固定输入长度**

FC 网络的第一层 `Linear(in_features, out_features)` 要求输入维度固定。
- 句子 A："我爱北京" → 4 个 token
- 句子 B："今天天气真不错" → 7 个 token

不同长度的句子无法喂给同一个 FC 网络（除非强行 padding 到最大长度，但这浪费计算且引入噪声）。

**局限二：忽略顺序（无记忆）**

即使强行 flatten 成固定长度向量，FC 也会丢失顺序信息：
- "猫吃鱼" 和 "鱼吃猫" 经 flatten 后对网络来说差别极小
- 每次处理 token 时，网络对"已经处理过什么"毫无记忆

**直觉类比：**
FC 处理句子就像让你蒙着眼睛一次性摸一堆积木，然后猜整体形状——你摸完就忘。
RNN 则像让你睁眼从左到右一块一块看，每看完一块都记在脑子里，最后再判断。

---

## 📐 理论推导

### RNN 核心公式

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

**符号含义：**

| 符号 | 含义 | 典型维度 |
|------|------|---------|
| $x_t$ | 第 $t$ 步的输入（词向量） | `(batch, embed_dim)` |
| $h_{t-1}$ | 上一步的隐状态（记忆） | `(batch, hidden_dim)` |
| $h_t$ | 当前步的隐状态（更新后的记忆） | `(batch, hidden_dim)` |
| $y_t$ | 当前步的输出（可选） | `(batch, output_dim)` |
| $W_{xh}$ | 输入到隐层的权重矩阵 | `(hidden_dim, embed_dim)` |
| $W_{hh}$ | 隐层到隐层的权重矩阵 | `(hidden_dim, hidden_dim)` |

**信息流图：**

```
x_1   x_2   x_3   x_4
 │     │     │     │
 ▼     ▼     ▼     ▼
[RNN]→[RNN]→[RNN]→[RNN]
 h_0   h_1   h_2   h_3   h_4
```

每个 RNN 单元接收：当前输入 $x_t$ + 上一步隐状态 $h_{t-1}$，输出新的隐状态 $h_t$。

### RNN vs FC 网络对比

| 维度 | FC 网络 | RNN |
|------|--------|-----|
| 输入形状 | 固定 `(batch, features)` | 可变长序列 `(seq_len, batch, embed_dim)` |
| 信息流 | 单次前向，无时序 | 时间步展开，有先后顺序 |
| 记忆 | 无（每次独立处理） | 有（$h_t$ 携带历史信息） |
| 参数共享 | 每层独立参数 | 所有时间步共享同一组参数 |
| 处理变长序列 | 困难（需 padding/截断） | 天然支持 |

---

## 💡 关键理解

### 1. Embedding 层：为什么不用 one-hot？

one-hot 向量的问题：
- **维度爆炸**：词表大小 10000，每个词就是 10000 维向量，99.99% 是 0
- **无语义关系**："猫" 和 "狗" 的 one-hot 向量正交（距离完全一样），毫无相关性
- **计算浪费**：稀疏矩阵乘法效率极低

Embedding 层的本质：
- 一张可学习的"查表"：`nn.Embedding(vocab_size, embed_dim)`
- 每个 token ID → 一个稠密的低维向量（比如 128 维）
- **语义相近的词，向量距离更近**（通过反向传播自动学习）
- 本质上就是一个 `(vocab_size, embed_dim)` 的权重矩阵，根据 token id 取对应行

```
token_id = 42
embedding_weight[42]  →  [0.3, -0.1, 0.8, ..., 0.2]  (128维向量)
```

### 2. TTwithFC 结构

"Token-by-Token with FC"：每个时间步的 hidden state 都接一个 FC 层输出预测。

```
x_1  x_2  x_3  x_4
 ↓    ↓    ↓    ↓
RNN→ RNN→ RNN→ RNN
 ↓    ↓    ↓    ↓
FC   FC   FC   FC
 ↓    ↓    ↓    ↓
y_1  y_2  y_3  y_4
```

每个位置都有输出，适合**序列标注**任务（NER、词性标注等）。

### 3. 两种核心使用场景

**场景一：序列分类（取最后 h）**

```
"这部电影真好看" → RNN → ... → h_7 → FC → 正面/负面
```
- 情感分析、文本分类
- 只用最后一个时间步的 $h_T$，它理论上包含了整句话的信息
- 代码：`output, h_n = rnn(x)` → 用 `h_n` 做分类

**场景二：序列标注（每个 h 都输出）**

```
"张三 在 北京 上班" → RNN → h_1, h_2, h_3, h_4 → FC → B-PER O B-LOC O
```
- 命名实体识别、词性标注
- 每个时间步都输出预测标签
- 代码：用 `output`（shape: `seq_len, batch, hidden_dim`）每步接 FC

### 4. 参数共享机制

RNN 在所有时间步使用**同一组参数** $(W_{xh}, W_{hh}, b_h)$：

- 优点：参数量固定，与序列长度无关；天然支持变长序列
- 直觉：就像人类阅读——你处理第1个词和第100个词用的是同一套语言理解能力
- 对比 Transformer：每层的参数虽然也是共享的，但注意力机制可以直接建立任意两个位置的关系

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn

# ============================================================
# 1. 基础组件演示
# ============================================================

# 超参数
vocab_size = 1000    # 词表大小
embed_dim = 64       # 词向量维度
hidden_dim = 128     # RNN 隐状态维度
num_classes = 2      # 分类数（正面/负面）
seq_len = 10         # 序列长度
batch_size = 4       # 批大小

# Embedding 层：vocab_size 个词，每个映射到 embed_dim 维向量
embedding = nn.Embedding(vocab_size, embed_dim)

# RNN 层
rnn = nn.RNN(
    input_size=embed_dim,    # 每步输入维度（词向量维度）
    hidden_size=hidden_dim,  # 隐状态维度
    batch_first=False        # 默认 (seq_len, batch, features)
)

# 输出层
fc = nn.Linear(hidden_dim, num_classes)

# ============================================================
# 2. 场景一：序列分类（取最后 hidden state）
# ============================================================

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)  — token id 序列
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(embedded)
        # output: (batch, seq_len, hidden_dim) — 每步的隐状态
        # h_n:    (1, batch, hidden_dim)       — 最后一步的隐状态

        last_hidden = h_n.squeeze(0)          # (batch, hidden_dim)
        logits = self.fc(last_hidden)         # (batch, num_classes)
        return logits


# 测试分类模型
classifier = RNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
x_cls = torch.randint(0, vocab_size, (batch_size, seq_len))   # (4, 10)
logits = classifier(x_cls)
print(f"分类输出 shape: {logits.shape}")   # (4, 2)
print(f"预测类别: {logits.argmax(dim=1)}")


# ============================================================
# 3. 场景二：序列标注（每步都输出，TTwithFC）
# ============================================================

class RNNTagger(nn.Module):
    """
    序列标注模型：每个 token 都预测一个标签
    适用于 NER、词性标注等任务
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)  # 每步都接 FC

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)              # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(embedded)
        # output: (batch, seq_len, hidden_dim) — 每步的隐状态

        tag_logits = self.fc(output)              # (batch, seq_len, num_tags)
        return tag_logits


# 测试标注模型（5个标签类别）
num_tags = 5
tagger = RNNTagger(vocab_size, embed_dim, hidden_dim, num_tags)
x_tag = torch.randint(0, vocab_size, (batch_size, seq_len))    # (4, 10)
tag_logits = tagger(x_tag)
print(f"\n标注输出 shape: {tag_logits.shape}")    # (4, 10, 5)
print(f"每步预测标签: {tag_logits.argmax(dim=-1)}")  # (4, 10)


# ============================================================
# 4. 参数共享验证
# ============================================================

print("\n=== RNN 参数验证 ===")
rnn_test = nn.RNN(embed_dim, hidden_dim, batch_first=True)
for name, param in rnn_test.named_parameters():
    print(f"{name}: {param.shape}")
# weight_ih_l0: (hidden_dim, embed_dim)   — W_xh
# weight_hh_l0: (hidden_dim, hidden_dim)  — W_hh
# bias_ih_l0, bias_hh_l0
# 无论序列多长，只有这几组参数（共享！）


# ============================================================
# 5. 手动展开 RNN 验证公式（教学用）
# ============================================================

print("\n=== 手动展开 RNN ===")
W_xh = torch.randn(hidden_dim, embed_dim)
W_hh = torch.randn(hidden_dim, hidden_dim)
b_h  = torch.zeros(hidden_dim)

# 模拟一个句子的 embedding 序列
x_seq = torch.randn(seq_len, embed_dim)    # (seq_len, embed_dim)
h = torch.zeros(hidden_dim)               # 初始 hidden state

for t in range(seq_len):
    x_t = x_seq[t]                        # (embed_dim,)
    h = torch.tanh(W_xh @ x_t + W_hh @ h + b_h)  # 核心公式
    print(f"t={t}, h norm={h.norm():.4f}")
```

---

## ⚠️ 易错点与常见误解

**1. `batch_first` 参数忘记设置**
- 默认 `batch_first=False`：输入形状是 `(seq_len, batch, features)`
- 设置 `batch_first=True`：输入形状是 `(batch, seq_len, features)`
- 混淆会导致维度错误或结果错误，但 PyTorch 不会报错！

**2. `h_n` 和 `output` 的区别**
- `output`: 每个时间步的隐状态，shape `(batch, seq_len, hidden_dim)`
- `h_n`: 最后一步的隐状态，shape `(num_layers, batch, hidden_dim)`
- `output[:, -1, :]` == `h_n[-1]`（单层 RNN 时等价）
- 序列分类用 `h_n`，序列标注用 `output`

**3. Embedding 的输入必须是 LongTensor（整数）**
```python
# 错误
x = torch.tensor([1.0, 2.0, 3.0])   # float tensor
embedding(x)  # 报错！

# 正确
x = torch.tensor([1, 2, 3])         # long tensor (int64)
embedding(x)  # OK
```

**4. RNN 不是万能的记忆**
- 理论上 $h_t$ 包含所有历史信息，但实际上**梯度消失**导致长距离依赖学不好
- 句子很长时，$h_T$ 往往"忘记"了句子开头的信息
- 解决方案：LSTM（门控机制）、GRU、或直接用 Transformer

**5. 序列分类时不应该取平均 hidden state**
- 初学者常用 `output.mean(dim=1)` 代替 `h_n`
- 理论上两者都可以，但语义不同：`h_n` 是"读完整句后的状态"，mean 是"所有时刻状态的平均"
- 标准做法是取 `h_n`

---

## 🔗 知识延伸

- **LSTM**：通过遗忘门、输入门、输出门解决 RNN 的梯度消失问题
- **GRU**：LSTM 的简化版，两个门（重置门、更新门），效果相近但参数更少
- **双向 RNN（BiRNN）**：同时从左到右和从右到左处理序列，$h_t$ 同时包含左右上下文
- **Transformer**：抛弃了循环结构，用自注意力机制直接建立全局依赖，彻底解决长距离依赖问题
- **梯度消失详解**：见 `01_RNN原理与梯度消失.md`

---

## 📚 参考资料

- Elman, J.L. (1990). *Finding structure in time.* Cognitive Science.
- PyTorch 官方文档：`torch.nn.RNN`
- Stanford CS224N: Natural Language Processing with Deep Learning
- 《动手学深度学习》第 9 章：循环神经网络
