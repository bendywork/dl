# 01 Word2Vec 基本理解

## 📌 核心问题
> Word2Vec 解决了什么问题？

在 NLP 领域，计算机要理解语言，必须先把"词"转成数字。
最简单的做法是 **One-Hot 编码**：词表有 10000 个词，就用 10000 维的向量，对应词的位置是 1，其余全是 0。

**One-Hot 的致命缺陷：**
- 向量维度爆炸（词表多大，向量就多长）
- 任意两个词的余弦相似度都是 0，无法表达词义关系
  - `猫` 和 `狗` 的相似度 = 0
  - `猫` 和 `飞机` 的相似度 = 0
  - 完全没有语义信息

Word2Vec 的目标是：**把每个词映射成一个低维稠密的实数向量，且语义相近的词在向量空间中距离更近。**

---

## 🌱 根源与动机

### 语言学直觉：你可以通过它的邻居来认识一个词

> "You shall know a word by the company it keeps." —— John Firth, 1957

这句话是 Word2Vec 最本质的思想：
**一个词的意义，由它经常出现在哪些词旁边来决定。**

举例：
- "我今天吃了一个苹果" → `苹果` 出现在 `吃`、`食物` 相关词附近
- "苹果发布了新手机" → `苹果` 出现在 `发布`、`手机`、`科技` 相关词附近

训练数据够多的话，模型会自动发现：
- `苹果(水果)` 经常和 `吃`、`甜`、`果汁` 共现
- `苹果(公司)` 经常和 `iPhone`、`发布`、`股价` 共现

**核心思想**：用一个词的**上下文**来定义这个词的含义。

---

## 📐 理论基础

### 词向量是什么？

Word2Vec 训练后，每个词对应一个 **d 维实数向量**（通常 d = 100~300）：

```
苹果  → [0.32, -0.14, 0.87, 0.23, ..., 0.55]  # d 维
香蕉  → [0.30, -0.12, 0.85, 0.21, ..., 0.51]  # 与苹果很像
手机  → [-0.71, 0.93, -0.22, 0.44, ..., -0.33]  # 与苹果(水果)差异大
```

这些向量捕获了词义关系，最著名的例子：

```
向量(国王) - 向量(男) + 向量(女) ≈ 向量(女王)
```

### 余弦相似度衡量词义距离

两个词向量 **v** 和 **u** 之间的相似度：

```
cos(θ) = (v · u) / (|v| × |u|)
```

- 结果在 [-1, 1] 之间
- 越接近 1 → 词义越相似
- 越接近 0 → 词义无关
- 越接近 -1 → 词义相反（如：`好` vs `坏`）

---

## 💡 Word2Vec 的两种训练方式

Word2Vec 是一个框架，包含两种具体模型：

| 模型 | 输入 | 输出 | 特点 |
|------|------|------|------|
| **CBOW** | 上下文词 → 中心词 | 给定周围词，预测中心词 | 快速，适合大语料 |
| **Skip-gram** | 中心词 → 上下文词 | 给定中心词，预测周围词 | 对稀有词效果好 |

---

## 🏗️ 网络结构概览

两种模型都使用同一种简单的 3 层神经网络：

```
输入层 (One-Hot)
    ↓  × W_in (嵌入矩阵)
隐藏层 (词向量 = Embedding)
    ↓  × W_out (输出权重)
输出层 (Softmax → 概率分布)
```

**关键点**：
- 隐藏层没有激活函数（线性变换）
- 训练完成后，**W_in 就是我们想要的词向量矩阵**
- W_in 的第 i 行 = 第 i 个词的词向量

---

## 🔢 训练流程（总体）

1. **准备训练数据**：大量文本语料（维基百科、新闻、书籍等）
2. **滑动窗口扫描**：设定窗口大小（如 window=2），生成 (输入, 标签) 训练对
3. **前向传播**：计算预测概率
4. **损失计算**：交叉熵损失（预测分布 vs 真实分布）
5. **反向传播**：更新 W_in 和 W_out
6. **迭代训练**：重复直到收敛
7. **提取词向量**：取 W_in 的对应行

---

## 🔧 代码实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# ============================================================
# 1. 基础概念演示：One-Hot vs 词向量
# ============================================================

# 假设词表
vocab = ["我", "喜欢", "苹果", "香蕉", "吃", "水果"]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)  # 词表大小 = 6

# One-Hot 编码
def one_hot(word, word2idx):
    vec = np.zeros(len(word2idx))
    vec[word2idx[word]] = 1
    return vec

print("One-Hot '苹果':", one_hot("苹果", word2idx))
# 输出: [0. 0. 1. 0. 0. 0.]

# 余弦相似度
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

v_apple = one_hot("苹果", word2idx)
v_banana = one_hot("香蕉", word2idx)
print("One-Hot 余弦相似度(苹果, 香蕉):", cosine_similarity(v_apple, v_banana))
# 输出: 0.0  → 无法表达相似性！

# ============================================================
# 2. 简单词向量演示（随机初始化，未训练）
# ============================================================

D = 3  # 词向量维度（演示用，实际用 100-300）
embedding_matrix = np.random.randn(V, D)  # shape: (6, 3)

print("\n随机初始化的词向量矩阵 (6个词 × 3维):")
for word, idx in word2idx.items():
    print(f"  {word}: {embedding_matrix[idx].round(3)}")

# 查找词向量：直接通过索引
apple_vec = embedding_matrix[word2idx["苹果"]]
banana_vec = embedding_matrix[word2idx["香蕉"]]
print(f"\n未训练时 余弦相似度(苹果, 香蕉): {cosine_similarity(apple_vec, banana_vec):.4f}")
# 随机值，没有语义意义

# ============================================================
# 3. 滑动窗口生成训练数据
# ============================================================

def build_training_data(corpus, window_size=2):
    """生成 (中心词, 上下文词) 数据对"""
    pairs = []
    for sentence in corpus:
        tokens = sentence.split()
        for i, center_word in enumerate(tokens):
            # 左边 window_size 个词
            start = max(0, i - window_size)
            # 右边 window_size 个词
            end = min(len(tokens), i + window_size + 1)
            
            context_words = tokens[start:i] + tokens[i+1:end]
            for context_word in context_words:
                if center_word in word2idx and context_word in word2idx:
                    pairs.append((center_word, context_word))
    return pairs

corpus = ["我 喜欢 吃 苹果", "我 喜欢 香蕉 水果"]
pairs = build_training_data(corpus, window_size=2)

print("\n滑动窗口生成的训练对 (中心词, 上下文词):")
for p in pairs:
    print(f"  {p}")

# ============================================================
# 4. PyTorch 实现简单词向量模型（Skip-gram 版本）
# ============================================================

class SimpleWord2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 嵌入层：本质是一个查找表 (lookup table)
        # 形状: (vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 输出层
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, center_word_idx):
        # center_word_idx: (batch,)
        embed = self.embedding(center_word_idx)  # (batch, embed_dim)
        logits = self.output(embed)               # (batch, vocab_size)
        return logits

# 训练
model = SimpleWord2Vec(V, D)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 转为索引
train_data = [(word2idx[c], word2idx[ctx]) for c, ctx in pairs 
              if c in word2idx and ctx in word2idx]

print("\n开始训练 (100 epochs)...")
for epoch in range(100):
    total_loss = 0
    for center_idx, context_idx in train_data:
        center_tensor = torch.tensor([center_idx])
        context_tensor = torch.tensor([context_idx])
        
        logits = model(center_tensor)
        loss = loss_fn(logits, context_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Loss = {total_loss:.4f}")

# 提取训练好的词向量
trained_vectors = model.embedding.weight.detach().numpy()
print("\n训练后词向量:")
for word, idx in word2idx.items():
    print(f"  {word}: {trained_vectors[idx].round(3)}")

# 语义相似度
apple_vec = trained_vectors[word2idx["苹果"]]
banana_vec = trained_vectors[word2idx["香蕉"]]
eat_vec = trained_vectors[word2idx["吃"]]
print(f"\n余弦相似度(苹果, 香蕉): {cosine_similarity(apple_vec, banana_vec):.4f}")
print(f"余弦相似度(苹果, 吃): {cosine_similarity(apple_vec, eat_vec):.4f}")
print("苹果和香蕉都是水果，相似度应该 > 苹果和吃")
```

---

## ⚠️ 易错点与常见误解

1. **误解：Word2Vec 是直接学习词义的**
   - 正确：Word2Vec 学的是**共现统计规律**，词义相似只是副产品
   - 如果语料有偏见，词向量也会有偏见

2. **误解：隐藏层是词向量**
   - 正确：**权重矩阵 W_in 才是词向量**，隐藏层的输出是词向量的查找结果

3. **误解：余弦相似度 = 欧氏距离**
   - 余弦相似度看的是**方向**，不管长度
   - 欧氏距离看的是**绝对距离**
   - Word2Vec 用余弦相似度，因为词向量的方向比长度更重要

4. **误解：One-Hot 向量太长，只是浪费空间**
   - 更本质的问题：One-Hot 无法表达语义关系，任意两词相似度为 0

5. **误解：CBOW 和 Skip-gram 效果一样**
   - CBOW 对高频词更准确，训练更快
   - Skip-gram 对低频词更好，适合小语料

---

## 🔗 知识延伸

- **CBOW（连续词袋模型）**：→ 见 [02_CBOW详解.md]
- **Skip-gram**：→ 见 [03_Skip-gram详解.md]
- **负采样 (Negative Sampling)**：解决 Softmax 计算慢的问题
- **GloVe**：基于全局共现矩阵的词向量，与 Word2Vec 互补
- **FastText**：加入了子词信息，处理 OOV（词表外词汇）更好
- **BERT/GPT**：上下文相关的动态词向量，取代了静态词向量

---

## 📚 参考资料
- Mikolov et al., 2013: "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al., 2013: "Distributed Representations of Words and Phrases and their Compositionality"
