# 03 Skip-gram 详解

## 📌 核心问题
> Skip-gram 解决了什么问题？它和 CBOW 有什么根本区别？

Skip-gram 的任务与 CBOW 完全相反：
**给定中心词，预测它周围可能出现哪些词。**

- CBOW：`["我", "爱", "天安门", "广场"]` → 预测 `"北京"`
- Skip-gram：`"北京"` → 预测 `"我"`, `"爱"`, `"天安门"`, `"广场"`

**"Skip-gram" 名字的来源**：训练时从句子中跳跃（skip）地取词对，用中心词预测"跳跃式"的周围词。

---

## 🌱 根源与动机

### 为什么要"反过来"预测？

CBOW 用多个词预测一个词，天然把稀有词"稀释"了。当某个词只出现过几次，它提供的信号很弱。

Skip-gram 反向训练：**对每个中心词，生成多个 (中心词, 上下文词) 训练对**。

如果中心词出现了 5 次，且窗口大小为 2，就能产生 5×4=20 个训练样本。
低频词也能获得足够多的训练信号。

**结论：Skip-gram 对低频词、稀有词效果更好。**

---

## 📐 Skip-gram 完整流程（逐步拆解）

### 设定参数

```
词表大小 V = 10
词向量维度 D = 3
窗口大小 window = 2
```

以句子 "我 爱 北京 天安门 广场" 为例，以"北京"为中心词：

生成训练对（中心词, 上下文词）：
```
("北京", "我")
("北京", "爱")
("北京", "天安门")
("北京", "广场")
```

每个训练对独立训练，这就是为什么 Skip-gram 训练数据量更大。

---

### Step 1：中心词 One-Hot 编码

只对中心词"北京"做 One-Hot 编码：

```
"北京" → [0,0,1,0,0,0,0,0,0,0]   （第2个位置为1）
```

注意：CBOW 输入是多个词，Skip-gram 输入是一个词。

---

### Step 2：通过 W_in 查找中心词向量

```
v_北京 = One-Hot("北京") × W_in
       = W_in 的第 2 行
       = [0.8, 0.1, 0.6]   (示例值)
```

这个向量 **v_北京** 就是中心词的嵌入表示。

不需要求平均（只有一个词）！这是 Skip-gram 与 CBOW 的关键区别。

---

### Step 3：通过 W_out 计算各词得分

```
z = v_北京 × W_out   → 形状 (1 × V)
```

z_j = v_北京 · u_j（u_j 是 W_out 的第 j 列）

同样是**点积 = 余弦相似度的核心**：

```
z_j = v_北京 · u_j = |v_北京| × |u_j| × cos(θ)
```

词表中哪个词的输出向量与 v_北京 方向最接近，得分就最高。

---

### Step 4：Softmax 转概率

对每个 (中心词, 目标上下文词) 对：

```
P("天安门" | "北京") = exp(z_天安门) / Σ_k exp(z_k)
```

模型对每个上下文词单独计算一次损失。

---

### Step 5：损失计算与反向传播

对每个 (中心词, 上下文词) 训练对：

```
Loss = -log P(context_word | center_word)
```

一个中心词对应多个上下文词，所有损失加总：

```
Total Loss = -[log P("我"|"北京") + log P("爱"|"北京") + 
               log P("天安门"|"北京") + log P("广场"|"北京")]
```

通过反向传播更新 W_in 和 W_out。

---

## 💡 Skip-gram 的余弦相似度本质

Skip-gram 和 CBOW 一样，预测的核心机制都是**向量方向匹配（余弦相似度）**。

不同之处：
- CBOW：h（上下文平均向量）与 u_j（候选词输出向量）的余弦相似度
- Skip-gram：v_i（中心词输入向量）与 u_j（候选词输出向量）的余弦相似度

**Skip-gram 直觉**：
- "北京"的向量指向某个方向
- "天安门"的输出向量方向与"北京"很接近（因为经常共现）
- 余弦相似度高 → 预测概率高

训练的过程就是：调整所有词向量，使得**经常共现的词对向量方向趋近，不共现的词对向量方向趋远**。

---

## 🆚 CBOW vs Skip-gram 详细对比

| 维度 | CBOW | Skip-gram |
|------|------|-----------|
| **任务方向** | 上下文 → 中心词 | 中心词 → 上下文 |
| **输入** | 多个上下文词向量（取平均） | 单个中心词向量 |
| **训练样本数** | 每个位置1个样本 | 每个位置 2×window 个样本 |
| **对低频词** | 效果较弱（被"稀释"） | 效果更好（更多训练信号） |
| **对高频词** | 效果稳定 | 效果稳定 |
| **训练速度** | 更快 | 更慢（样本多） |
| **适用场景** | 大语料、快速训练 | 需要覆盖低频词 |

**经验结论（Mikolov 团队建议）：**
- 小数据集：Skip-gram 更好
- 大数据集：CBOW 足够，且更快
- 实践中 Skip-gram + 负采样 最常用

---

## 🚀 负采样（Negative Sampling）：解决 Softmax 太慢的问题

Softmax 需要对**整个词表**（几万~几十万词）都算一遍，太慢！

**负采样的思想**：不算全部词，只比较"正样本"（真实上下文词）和几个随机采样的"负样本"（随机词）。

原始 Softmax 目标：
```
maximize: P(context_word | center_word)
         = exp(z_context) / Σ_all_words exp(z_k)  # O(V) 计算
```

负采样目标（近似）：
```
maximize: σ(v_center · u_context)          # 正样本得分高
minimize: σ(v_center · u_neg_k) for k=1..K  # 负样本得分低
```

其中 σ 是 sigmoid 函数，K 通常取 5~20。

**效果**：把 O(V) 的计算降到 O(K)，提速几百倍。

负样本采样概率（不是均匀采样，而是按词频的 3/4 次方）：

```
P(w) ∝ count(w)^(3/4) / Σ_j count(w_j)^(3/4)
```

高频词被选为负样本的概率更大，但 3/4 次方"压平"了频率差距，给低频词更多机会。

---

## 🏗️ Skip-gram 网络结构图

```
输入层（中心词 One-Hot）
  x = [0,0,1,0,0,...]   ← "北京"（只有一个词）
        ↓  × W_in（V×D）
        
嵌入层（查找中心词向量）
  v_北京 = [0.8, 0.1, 0.6]   （W_in 的第 2 行）
        ↓  × W_out（D×V）
        
输出层（每个词的得分）
  z = [z_我, z_爱, z_北京, z_天安门, z_广场, ...]
        ↓  Softmax（每个上下文词独立）
        
多个概率分布（每个对应一个目标上下文词）
  P(我|北京) = 0.12
  P(爱|北京) = 0.08
  P(天安门|北京) = 0.75  ← 应该很高
  P(广场|北京) = 0.61    ← 应该很高
```

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random

# ============================================================
# 1. 构建语料和词表
# ============================================================

corpus = [
    "我 爱 北京 天安门",
    "我 喜欢 吃 苹果 和 香蕉",
    "苹果 和 香蕉 都是 水果",
    "北京 是 中国 首都",
    "中国 首都 是 北京",
    "我 喜欢 水果 苹果 香蕉",
]

all_words = []
for sent in corpus:
    all_words.extend(sent.split())

word2idx = {}
idx2word = {}
for i, word in enumerate(sorted(set(all_words))):
    word2idx[word] = i
    idx2word[i] = word

V = len(word2idx)
word_counts = Counter(all_words)
print(f"词表大小: {V}, 总词数: {len(all_words)}")

# ============================================================
# 2. 生成 Skip-gram 训练数据
# ============================================================

def generate_skipgram_data(corpus, word2idx, window_size=2):
    """
    返回：(center_idx, context_idx) 的列表
    每个中心词对应多个训练对
    """
    data = []
    for sentence in corpus:
        words = sentence.split()
        for i, center_word in enumerate(words):
            if center_word not in word2idx:
                continue
            center_idx = word2idx[center_word]
            
            # 生成所有上下文词对
            for j in range(i - window_size, i + window_size + 1):
                if j == i:
                    continue
                if 0 <= j < len(words) and words[j] in word2idx:
                    context_idx = word2idx[words[j]]
                    data.append((center_idx, context_idx))
    return data

train_data = generate_skipgram_data(corpus, word2idx, window_size=2)
print(f"\nSkip-gram 训练对数量: {len(train_data)}")
print("示例 (中心词, 上下文词):")
for c, ctx in train_data[:5]:
    print(f"  ({idx2word[c]}, {idx2word[ctx]})")

# ============================================================
# 3. 标准 Skip-gram 模型（Softmax 版本）
# ============================================================

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # W_in: 输入嵌入矩阵（中心词向量）
        self.embedding_in = nn.Embedding(vocab_size, embed_dim)
        # W_out: 输出权重（上下文词向量）
        self.embedding_out = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, center_idx):
        """
        center_idx: (batch,) 中心词索引
        """
        # 查找中心词向量，形状 (batch, embed_dim)
        v_center = self.embedding_in(center_idx)
        
        # 计算与所有词的点积得分（余弦相似度核心！）
        # 形状 (batch, V)
        logits = self.embedding_out(v_center)
        
        return logits, v_center

# ============================================================
# 4. 负采样版 Skip-gram（实践中更常用）
# ============================================================

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding_in = nn.Embedding(vocab_size, embed_dim)
        self.embedding_out = nn.Embedding(vocab_size, embed_dim)
        
        # 权重初始化：让点积初始值接近 0
        nn.init.uniform_(self.embedding_in.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.embedding_out.weight)
    
    def forward(self, center_idxs, context_idxs, neg_idxs):
        """
        center_idxs: (batch,)   中心词
        context_idxs: (batch,)  正样本（真实上下文词）
        neg_idxs: (batch, K)    负样本
        """
        # 中心词向量 (batch, D)
        v_center = self.embedding_in(center_idxs)
        
        # 正样本：中心词向量 · 上下文词向量
        u_pos = self.embedding_out(context_idxs)     # (batch, D)
        pos_score = torch.sum(v_center * u_pos, dim=1)  # (batch,)
        
        # 负样本：中心词向量 · 负样本词向量
        u_neg = self.embedding_out(neg_idxs)         # (batch, K, D)
        # v_center 扩展为 (batch, 1, D)
        neg_score = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)  # (batch, K)
        
        # 损失：正样本得分高、负样本得分低
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-8), dim=1)
        
        loss = (pos_loss + neg_loss).mean()
        return loss

# ============================================================
# 5. 负采样采样器（按词频的 3/4 次方）
# ============================================================

def build_neg_sampler(word_counts, word2idx, power=0.75):
    """构建负采样概率表"""
    total = sum(count ** power for count in word_counts.values())
    sampling_probs = {}
    for word, count in word_counts.items():
        if word in word2idx:
            sampling_probs[word2idx[word]] = (count ** power) / total
    return sampling_probs

def sample_negatives(center_idx, context_idx, sampling_probs, K=5):
    """采样 K 个负样本（避开正样本）"""
    neg_indices = []
    idx_list = list(sampling_probs.keys())
    prob_list = [sampling_probs[i] for i in idx_list]
    
    while len(neg_indices) < K:
        neg_idx = random.choices(idx_list, weights=prob_list, k=1)[0]
        if neg_idx != center_idx and neg_idx != context_idx:
            neg_indices.append(neg_idx)
    return neg_indices

sampling_probs = build_neg_sampler(word_counts, word2idx)

# ============================================================
# 6. 训练负采样 Skip-gram
# ============================================================

D = 10
K = 5  # 负样本数量

model_ns = SkipGramNegSampling(V, D)
optimizer = optim.Adam(model_ns.parameters(), lr=0.025)

print("\n开始训练 Skip-gram + 负采样...")
for epoch in range(300):
    total_loss = 0
    random.shuffle(train_data)
    
    for center_idx, context_idx in train_data:
        neg_idxs = sample_negatives(center_idx, context_idx, sampling_probs, K)
        
        center_t = torch.tensor([center_idx])
        context_t = torch.tensor([context_idx])
        neg_t = torch.tensor([neg_idxs])  # (1, K)
        
        loss = model_ns(center_t, context_t, neg_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ============================================================
# 7. 验证词向量质量
# ============================================================

def cosine_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.dot(v1_norm, v2_norm)

def find_most_similar(word, word_vectors, word2idx, idx2word, top_k=3):
    """找到最相似的词"""
    if word not in word2idx:
        return []
    
    query_vec = word_vectors[word2idx[word]]
    sims = []
    for w, idx in word2idx.items():
        if w != word:
            sim = cosine_similarity(query_vec, word_vectors[idx])
            sims.append((w, sim))
    
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

word_vectors = model_ns.embedding_in.weight.detach().numpy()

print("\n===== 语义相似词查找 =====")
test_words = ["苹果", "北京", "我"]
for word in test_words:
    if word in word2idx:
        similar = find_most_similar(word, word_vectors, word2idx, idx2word)
        print(f"\n  '{word}' 的最相似词:")
        for w, sim in similar:
            print(f"    {w}: {sim:.4f}")

print("\n===== 词对相似度 =====")
pairs = [("苹果", "香蕉"), ("苹果", "北京"), ("北京", "首都"), ("北京", "中国")]
for w1, w2 in pairs:
    if w1 in word2idx and w2 in word2idx:
        v1 = word_vectors[word2idx[w1]]
        v2 = word_vectors[word2idx[w2]]
        print(f"  cos({w1}, {w2}) = {cosine_similarity(v1, v2):.4f}")
```

---

## ⚠️ 易错点与常见误解

1. **误解：Skip-gram 只能预测一个上下文词**
   - 正确：每个中心词生成多个 (中心词, 上下文词) 训练对，每对单独训练
   - 一个中心词对应 2×window 个训练对

2. **误解：负采样和随机采样一样**
   - 正确：负采样按词频的 3/4 次方采样，而非均匀随机
   - 3/4 次方是 Mikolov 实验发现的最优值，压平了高频词的优势

3. **误解：Skip-gram 输出的是"词义"**
   - 正确：Skip-gram 学到的是**共现统计关系**，不是词义本身
   - "苹果(公司)" 和 "苹果(水果)" 如果在相同语境中出现，它们会有相似的向量！

4. **误解：window 越大越好**
   - 大 window：捕获主题相关性（"苹果"和"维生素"）
   - 小 window：捕获语法相关性（"苹果"和"吃"）
   - 通常 window = 2~10，根据任务选择

5. **误解：Word2Vec 的词向量可以用于所有 NLP 任务**
   - 局限：同一个词只有一个向量，无法区分多义词（"苹果"水果 vs 公司）
   - 解决：BERT 等上下文词向量模型，同一个词根据上下文有不同的向量

---

## 🔗 知识延伸

- **CBOW 对比**：→ 见 [02_CBOW详解.md]
- **Word2Vec 基础**：→ 见 [01_Word2Vec基本理解.md]
- **GloVe**：全局向量，结合了全局共现矩阵
- **FastText**：在 Skip-gram 基础上加入 n-gram 子词，处理形态变化
- **ELMo**：上下文相关词向量的雏形
- **BERT**：完全基于 Transformer，完全替代了静态词向量

---

## 📚 参考资料
- Mikolov et al., 2013: "Distributed Representations of Words and Phrases and their Compositionality"
- Goldberg & Levy, 2014: "word2vec Explained"
