# 02 CBOW（连续词袋模型）详解

## 📌 核心问题
> CBOW 解决了什么问题？为什么说 CBOW 本质上通过余弦相似度预测中心词？

CBOW（Continuous Bag of Words）的任务非常直观：
**给定一个词的周围词（上下文），预测这个词本身是什么。**

例如："我 喜欢 吃 ___ 和 香蕉"，根据上下文预测中间那个词是"苹果"。

---

## 🌱 根源与动机

### 从语言统计规律出发

回到 Firth 定律："一个词由它的邻居来定义。"

CBOW 把这个思想直接做成了监督学习任务：
- **输入**（上下文词）→ **输出**（中心词）
- 模型被迫学习：什么样的上下文对应什么样的词？

通过大量训练，模型学到的**隐藏层权重**就成了词向量。

### 为什么是"连续词袋"？

- **Bag（词袋）**：上下文词的顺序无关，只关心有哪些词（丢弃了位置信息）
- **Continuous（连续）**：向量是连续的实数，不是离散的 One-Hot

---

## 📐 CBOW 完整流程（逐步拆解）

### 设定参数

```
词表大小 V = 10（只有10个词，便于演示）
词向量维度 D = 3（实际用 100-300）
窗口大小 window = 2（左右各2个词）
```

以句子 "我 爱 北京 天安门 广场" 为例，以"北京"为中心词：
- 上下文词：["我", "爱", "天安门", "广场"]
- 目标：预测出"北京"

---

### Step 1：One-Hot 编码输入

为每个上下文词创建 One-Hot 向量（维度 = V）：

```
词表：[我=0, 爱=1, 北京=2, 天安门=3, 广场=4, ...]

"我"      → [1,0,0,0,0,0,0,0,0,0]
"爱"      → [0,1,0,0,0,0,0,0,0,0]
"天安门"  → [0,0,0,1,0,0,0,0,0,0]
"广场"    → [0,0,0,0,1,0,0,0,0,0]
```

---

### Step 2：通过嵌入矩阵查找词向量

嵌入矩阵 **W_in** 形状为 (V × D) = (10 × 3)

每个 One-Hot 向量乘以 W_in，本质是**按行索引查找**：

```
One-Hot("我") × W_in = W_in 的第 0 行 = v_我
One-Hot("爱") × W_in = W_in 的第 1 行 = v_爱
...
```

得到 4 个词向量，各自维度为 D=3：

```
v_我      = [0.2, 0.4, -0.1]
v_爱      = [0.3, 0.5, -0.2]
v_天安门  = [0.8, 0.1,  0.7]
v_广场    = [0.9, 0.2,  0.6]
```

---

### Step 3：求上下文词向量的平均值（CBOW 核心）

**CBOW 把所有上下文词向量取平均，作为隐藏层的输出：**

```
h = (v_我 + v_爱 + v_天安门 + v_广场) / 4

h = ([0.2,0.4,-0.1] + [0.3,0.5,-0.2] + [0.8,0.1,0.7] + [0.9,0.2,0.6]) / 4
h = [2.2, 1.2, 1.0] / 4
h = [0.55, 0.30, 0.25]
```

**这个 h 向量代表"上下文的综合语义"。**

---

### Step 4：通过输出权重矩阵计算得分

输出权重矩阵 **W_out** 形状为 (D × V) = (3 × 10)

计算隐藏层向量 h 与 W_out 的乘积，得到每个词的得分（logit）：

```
z = h × W_out    →  形状 (1 × V)，即每个词一个得分
```

W_out 的**每一列**就是某个词的"输出向量"（记为 u_j）

```
z_j = h · u_j（点积）= |h| × |u_j| × cos(θ)
```

**这就是为什么说 CBOW 本质上是余弦相似度！**

完整展开：

```
z_j = h · u_j
    = |h| × |u_j| × cos(h, u_j)
```

点积越大 → cos(θ) 越大 → h（上下文向量）和 u_j（目标词向量）方向越相似 → 这个词被选中的概率越高。

---

### Step 5：Softmax 归一化为概率

```
P(w_j | context) = exp(z_j) / Σ_k exp(z_k)
```

所有词的概率之和为 1，形成一个概率分布。

"北京"的概率应该是最大的。

---

### Step 6：计算交叉熵损失

```
Loss = -log P(北京 | 上下文)
```

越接近 1 → 损失越小 → 模型越好。

---

### Step 7：反向传播更新权重

通过梯度下降，同时更新：
- **W_in**（嵌入矩阵）：调整输入词的词向量表示
- **W_out**（输出权重）：调整输出词的表示

训练完成后，**W_in 的每一行就是对应词的词向量**。

---

## 💡 为什么说 CBOW 本质上是余弦相似度？

这是整个 Word2Vec 最精妙的设计，详细拆解如下：

### 数学层面

隐藏层向量 h 代表**上下文的综合语义向量**。

对于词表中的每个词 j，其输出向量为 u_j（W_out 的第 j 列）。

模型计算的 logit（得分）是：

```
z_j = h · u_j = Σ_k h_k × u_jk
```

这就是**点积**，也叫**内积**。

点积与余弦相似度的关系：

```
h · u_j = |h| × |u_j| × cos(θ)
```

其中：
- |h| = h 的 L2 范数（长度）
- |u_j| = u_j 的 L2 范数（长度）
- cos(θ) = h 和 u_j 之间夹角的余弦值

**如果向量已经归一化（|h| = |u_j| = 1），则点积就完全等于余弦相似度！**

### 直觉层面

想象词向量空间是一个高维空间：

- h 是"上下文语义方向"（指向一个方向）
- u_j 是"词 j 的语义方向"
- cos(θ) 衡量两个方向有多接近

**当 h 和 u_北京 方向最接近时，cos(θ) 最大，z_北京 最大，P(北京|上下文) 最大。**

所以，CBOW 在问的问题是：**词表里哪个词的语义方向，和当前上下文的语义方向最接近？**

这本质上就是余弦相似度搜索！

### 训练目标的几何含义

训练 CBOW 的过程，就是在调整词向量，使得：
- 经常出现在相同上下文中的词 → 向量方向趋于一致（余弦相似度高）
- 很少共现的词 → 向量方向趋于垂直（余弦相似度≈0）
- 语义相反的词 → 向量方向相反（余弦相似度为负）

---

## 🏗️ CBOW 网络结构图

```
输入层（上下文词的 One-Hot 向量）
  x1=[0,1,0,...,0]  ← "我"
  x2=[1,0,0,...,0]  ← "爱"
  x3=[0,0,0,1,...0] ← "天安门"
  x4=[0,0,0,0,1,...] ← "广场"
        ↓  × W_in（嵌入矩阵，V×D）
        
嵌入层（查找词向量）
  v1 = [0.2, 0.4, -0.1]  ← 我
  v2 = [0.3, 0.5, -0.2]  ← 爱
  v3 = [0.8, 0.1,  0.7]  ← 天安门
  v4 = [0.9, 0.2,  0.6]  ← 广场
        ↓  求平均
        
隐藏层（上下文向量）
  h = [0.55, 0.30, 0.25]
        ↓  × W_out（D×V）
        
输出层（每个词的得分）
  z = [z_我, z_爱, z_北京, z_天安门, z_广场, ...]
        ↓  Softmax
        
概率分布
  P = [0.02, 0.01, 0.85, 0.08, 0.02, ...]
                   ↑ 北京的概率最高
```

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# 1. 构建语料和词表
# ============================================================

corpus = [
    "我 爱 北京 天安门",
    "我 喜欢 吃 苹果 和 香蕉",
    "苹果 和 香蕉 都是 水果",
    "北京 是 中国 首都",
]

# 构建词表
all_words = []
for sent in corpus:
    all_words.extend(sent.split())

word2idx = {}
idx2word = {}
for i, word in enumerate(sorted(set(all_words))):
    word2idx[word] = i
    idx2word[i] = word

V = len(word2idx)
print(f"词表大小: {V}")
print(f"词表: {word2idx}")

# ============================================================
# 2. 生成 CBOW 训练数据（上下文词 → 中心词）
# ============================================================

def generate_cbow_data(corpus, word2idx, window_size=2):
    """
    返回：(context_indices_list, center_idx) 的列表
    context_indices_list: 上下文词的索引列表
    center_idx: 中心词的索引
    """
    data = []
    for sentence in corpus:
        words = sentence.split()
        for i in range(len(words)):
            center_word = words[i]
            if center_word not in word2idx:
                continue
            
            # 收集上下文词（左右各 window_size 个）
            context_words = []
            for j in range(i - window_size, i + window_size + 1):
                if j == i:
                    continue  # 跳过中心词自己
                if 0 <= j < len(words) and words[j] in word2idx:
                    context_words.append(word2idx[words[j]])
            
            if context_words:
                data.append((context_words, word2idx[center_word]))
    return data

train_data = generate_cbow_data(corpus, word2idx, window_size=2)
print(f"\n训练样本数量: {len(train_data)}")
print(f"示例 (上下文索引列表, 中心词索引):")
for ctx, ctr in train_data[:3]:
    ctx_words = [idx2word[i] for i in ctx]
    print(f"  上下文: {ctx_words} → 中心词: {idx2word[ctr]}")

# ============================================================
# 3. CBOW 模型定义
# ============================================================

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # W_in：嵌入矩阵，形状 (V, D)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # W_out：输出矩阵，形状 (D, V) → 用 Linear 表示
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, context_idxs):
        """
        context_idxs: (batch, context_size) 上下文词的索引
        
        CBOW 核心步骤：
        1. 查找每个上下文词的嵌入向量
        2. 对所有上下文词向量取平均
        3. 线性变换得到得分（点积 = 余弦相似度的核心）
        """
        # Step 1: 查找嵌入向量，形状 (batch, context_size, embed_dim)
        embeds = self.embedding(context_idxs)
        
        # Step 2: 对上下文词向量取平均，形状 (batch, embed_dim)
        # 这是 CBOW 的关键：上下文向量的平均
        h = embeds.mean(dim=1)
        
        # Step 3: 计算每个词的得分（点积 = 与每个输出词向量的内积）
        # 内积本质上是余弦相似度（×长度）
        logits = self.linear(h)  # 形状 (batch, V)
        
        return logits

# ============================================================
# 4. 训练
# ============================================================

D = 10  # 词向量维度
model = CBOW(V, D)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 将可变长度的上下文列表处理为固定长度（不足则填充）
def collate_cbow_batch(data_list, max_context=4):
    """将 (context_idxs, center_idx) 列表整理为 tensor"""
    contexts = []
    centers = []
    for ctx, ctr in data_list:
        # 如果上下文不够，重复最后一个词填充
        padded_ctx = ctx[:max_context]
        while len(padded_ctx) < max_context:
            padded_ctx.append(padded_ctx[-1])
        contexts.append(padded_ctx)
        centers.append(ctr)
    return torch.tensor(contexts), torch.tensor(centers)

print("\n开始训练 CBOW...")
for epoch in range(200):
    total_loss = 0
    for ctx_idxs, center_idx in train_data:
        # 单样本处理
        ctx_tensor = torch.tensor([ctx_idxs])   # (1, context_size)
        center_tensor = torch.tensor([center_idx])  # (1,)
        
        logits = model(ctx_tensor)
        loss = loss_fn(logits, center_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ============================================================
# 5. 提取词向量并验证语义相似度
# ============================================================

def cosine_similarity(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.dot(v1, v2)

# 提取训练好的词向量
word_vectors = model.embedding.weight.detach().numpy()  # (V, D)

print("\n===== 词向量语义相似度 =====")
test_pairs = [
    ("苹果", "香蕉"),   # 都是水果，应该相似
    ("苹果", "北京"),   # 语义不相关，应该低
    ("苹果", "水果"),   # 应该有一定相关性
]

for w1, w2 in test_pairs:
    if w1 in word2idx and w2 in word2idx:
        v1 = word_vectors[word2idx[w1]]
        v2 = word_vectors[word2idx[w2]]
        sim = cosine_similarity(v1, v2)
        print(f"  cos({w1}, {w2}) = {sim:.4f}")

# ============================================================
# 6. 验证"余弦相似度"是 CBOW 预测的核心
# ============================================================

print("\n===== 验证余弦相似度 = 模型预测核心 =====")
# 取训练样本：上下文["我","爱","天安门"] → 中心词"北京"
ctx_words = ["我", "爱", "天安门"]
ctx_idxs = [word2idx[w] for w in ctx_words if w in word2idx]

# 计算隐藏层向量 h（上下文词向量平均）
ctx_vecs = [word_vectors[idx] for idx in ctx_idxs]
h = np.mean(ctx_vecs, axis=0)
print(f"上下文向量 h = {h.round(3)}")

# 计算 h 与每个词的输出向量之间的余弦相似度
output_weights = model.linear.weight.detach().numpy()  # (V, D)
print("\nh 与各词的余弦相似度（越高 = 越可能是预测结果）:")
similarities = []
for word, idx in word2idx.items():
    u_j = output_weights[idx]
    sim = cosine_similarity(h, u_j)
    similarities.append((word, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
for word, sim in similarities[:5]:
    print(f"  {word}: {sim:.4f}")
```

---

## ⚠️ 易错点与常见误解

1. **误解：CBOW 中的"词袋"说明顺序不重要**
   - 正确：CBOW 确实不考虑词序（这是局限性），但实际语言中词序很重要
   - 这也是 Transformer 后来引入位置编码的原因之一

2. **误解：CBOW 和 Skip-gram 的隐藏层都是词向量**
   - 正确：两者的 **W_in 矩阵的每一行**才是输入词向量
   - CBOW 还有 W_out 可以作为另一组词向量（有时两者平均）

3. **误解：取平均会丢失信息，CBOW 效果很差**
   - 正确：对于高频词、大语料，CBOW 通常与 Skip-gram 效果相当
   - CBOW 的优势：训练速度更快，对噪声更鲁棒

4. **误解：点积等于余弦相似度**
   - 正确：**点积 = 余弦相似度 × 两向量长度之积**
   - 只有当两向量都是单位向量时，点积才等于余弦相似度
   - 但 CBOW 用点积（而非归一化的余弦）进行排序，效果是一样的

5. **CBOW 为什么用 Softmax 而不是直接取最大余弦相似度？**
   - Softmax 把分数转成概率，便于用交叉熵计算损失
   - 训练时需要可微的损失函数，Softmax+CrossEntropy 是标准配合

---

## 🔗 知识延伸

- **负采样（Negative Sampling）**：Softmax 对整个词表计算代价太高，负采样只随机选几个"错误答案"作对比
- **层次 Softmax（Hierarchical Softmax）**：用 Huffman 树加速 Softmax 计算
- **Skip-gram vs CBOW**：→ 见 [03_Skip-gram详解.md]
- **GloVe**：基于全局共现矩阵，弥补了 CBOW 只看局部上下文的不足

---

## 📚 参考资料
- Mikolov et al., 2013: "Efficient Estimation of Word Representations in Vector Space"
- CS224N Stanford NLP Lecture Notes
