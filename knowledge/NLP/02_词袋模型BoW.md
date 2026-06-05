# 词袋模型 BoW（Bag of Words）

## 📌 核心问题
> 如何将一篇文档（变长的词序列）转换为固定长度的数值向量，使得机器学习模型可以处理文本？

---

## 🌱 根源与动机

### 历史背景
1954 年 Zellig Harris 提出"分布假说"（Distributional Hypothesis）：**词的语义由其出现的上下文决定**。在此基础上，最早的文本数值化方案就是词袋模型——把文档当作一"袋子"，里面装着若干词，不关心顺序，只统计每个词出现了多少次。

词袋模型的核心假设：**文档的类别/主题主要由它包含哪些词决定，词的顺序是次要的**。

这个假设在文本分类、垃圾邮件过滤等任务中出奇地有效——即使忽略了语法结构，高频的关键词依然能传达文档的主题信息。

### 直觉类比
想象你在超市结账时，收银员不关心你放商品的顺序，只关心购物袋里有哪些商品、各买了几件。BoW 就是用这种"购物袋清单"的方式来表示文档。

---

## 📐 理论推导

### 词频矩阵构建

设语料库有 $N$ 篇文档，词汇表大小为 $V$。

**词频（Term Frequency，TF）矩阵** $\mathbf{M} \in \mathbb{R}^{N \times V}$：

$$
\mathbf{M}_{ij} = \text{count}(w_j, d_i) = \text{词 } w_j \text{ 在文档 } d_i \text{ 中出现的次数}
$$

**示例：**

| 文档 | I | love | NLP | hate | boring |
|------|---|------|-----|------|--------|
| d1: "I love NLP I love" | 2 | 2 | 1 | 0 | 0 |
| d2: "I hate boring NLP" | 1 | 0 | 1 | 1 | 1 |
| d3: "NLP is not boring" | 0 | 0 | 1 | 0 | 1 |

### 矩阵性质分析

**稀疏性**：词汇表 $V$ 通常数万到数十万，但每篇文档只包含其中极少数词。典型情况下，矩阵非零元素比例 < 0.1%。

**存储复杂度**：
- 密集存储：$O(N \times V)$，$10^4$ 篇文档 × $10^5$ 词汇 = $10^9$ 元素（约 4GB float32）
- 稀疏存储（CSR）：$O(\text{非零元素数})$，通常只需几 MB

**文档长度归一化**（避免长文档天然得分更高）：

$$
\text{TF\_norm}(w_j, d_i) = \frac{\text{count}(w_j, d_i)}{|d_i|}
$$

其中 $|d_i|$ 是文档 $d_i$ 的总词数。

### 文档相似度计算

两篇文档的 BoW 向量之间的**余弦相似度**：

$$
\text{sim}(d_i, d_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

这是信息检索中最常用的相似度度量方式。

---

## 💡 关键理解

### BoW 的三大局限

**1. 忽略语序（最致命）**
- "狗咬了人" 和 "人咬了狗" 得到完全相同的 BoW 向量
- "I love not hate this movie" 与 "I hate not love this movie" 完全相同

**2. 无语义**
- "car" 和 "automobile" 是不同的词，在 BoW 中距离为 0（完全不相似）
- 同义词、反义词、近义词全部等距

**3. 高频词主导（引出 TF-IDF 的动机）**
- "the"、"a"、"is" 等停用词频率极高，会主导相似度计算，但毫无语义价值
- 解决方案：去除停用词 + TF-IDF 加权

### N-gram BoW：对语序的有限补偿
在词袋模型基础上，把连续 N 个词作为一个 token，可以保留局部语序信息：
- Unigram（默认 BoW）: ["I", "love", "NLP"]
- Bigram: ["I_love", "love_NLP"]
- Trigram: ["I_love_NLP"]

---

## 🔧 代码实现

```python
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

# ============================================================
# 1. 手动实现词袋模型
# ============================================================

def build_vocab(corpus: list[str], min_freq: int = 1) -> dict:
    """
    构建词汇表
    
    Args:
        corpus: 文档列表，每个文档是字符串
        min_freq: 词频阈值，低于此频率的词被过滤
    
    Returns:
        word2idx: 词 -> 索引的映射
    """
    # 统计全局词频
    global_freq = Counter()
    for doc in corpus:
        global_freq.update(doc.lower().split())
    
    # 过滤低频词，构建词汇表（排序保证一致性）
    vocab_words = sorted([w for w, c in global_freq.items() if c >= min_freq])
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    return word2idx


def docs_to_bow(corpus: list[str], word2idx: dict, normalize: bool = False) -> np.ndarray:
    """
    将文档列表转换为词频矩阵
    
    Args:
        corpus: 文档列表
        word2idx: 词汇表
        normalize: 是否按文档长度归一化
    
    Returns:
        matrix: shape=(N, V)，N=文档数，V=词汇表大小
    """
    N = len(corpus)
    V = len(word2idx)
    matrix = np.zeros((N, V), dtype=np.float32)
    
    for i, doc in enumerate(corpus):
        tokens = doc.lower().split()
        for token in tokens:
            if token in word2idx:
                matrix[i, word2idx[token]] += 1
        
        if normalize and len(tokens) > 0:
            matrix[i] /= len(tokens)
    
    return matrix


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """余弦相似度"""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# 测试数据
corpus = [
    "I love NLP I love deep learning",
    "I hate boring meetings NLP is great",
    "deep learning is amazing and powerful",
    "NLP and deep learning go hand in hand",
    "boring meetings are not productive",
]

word2idx = build_vocab(corpus, min_freq=1)
print(f"词汇表大小: {len(word2idx)}")
print(f"词汇表: {list(word2idx.keys())}")

# 构建 BoW 矩阵
bow_matrix = docs_to_bow(corpus, word2idx, normalize=False)
print(f"\nBoW 矩阵形状: {bow_matrix.shape}")
print("\nBoW 矩阵 (前3篇文档):")

idx2word = {v: k for k, v in word2idx.items()}
print(f"{'词':>15}", end="")
for doc_idx in range(3):
    print(f"  doc{doc_idx+1}", end="")
print()
for word_idx, word in sorted(idx2word.items()):
    row = [bow_matrix[doc_idx, word_idx] for doc_idx in range(3)]
    if sum(row) > 0:  # 只显示有非零值的词
        print(f"{word:>15}", end="")
        for v in row:
            print(f"  {v:5.0f}", end="")
        print()

# 计算文档间相似度
print("\n文档相似度矩阵:")
N = len(corpus)
sim_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        sim_matrix[i, j] = cosine_similarity(bow_matrix[i], bow_matrix[j])

print(np.round(sim_matrix, 3))


# ============================================================
# 2. 演示语序丢失问题
# ============================================================
print("\n" + "="*50)
print("语序丢失问题演示")
print("="*50)

problematic_docs = [
    "dog bit man",
    "man bit dog",
    "I love not hate this",
    "I hate not love this",
]

vocab2 = build_vocab(problematic_docs)
bow2 = docs_to_bow(problematic_docs, vocab2)

print("\n'dog bit man' vs 'man bit dog':")
print(f"  向量相同: {np.array_equal(bow2[0], bow2[1])}")
print(f"  余弦相似度: {cosine_similarity(bow2[0], bow2[1]):.4f}")

print("\n'I love not hate' vs 'I hate not love':")
print(f"  向量相同: {np.array_equal(bow2[2], bow2[3])}")
print(f"  余弦相似度: {cosine_similarity(bow2[2], bow2[3]):.4f}")


# ============================================================
# 3. N-gram BoW（补偿语序信息）
# ============================================================
print("\n" + "="*50)
print("N-gram BoW 演示")
print("="*50)

def ngram_tokenize(text: str, n: int) -> list[str]:
    """将文本转换为 N-gram token 列表"""
    tokens = text.lower().split()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append("_".join(tokens[i:i+n]))
    return ngrams


test_doc = "I love deep learning"
print(f"\n原始文档: '{test_doc}'")
print(f"Unigram: {ngram_tokenize(test_doc, 1)}")
print(f"Bigram:  {ngram_tokenize(test_doc, 2)}")
print(f"Trigram: {ngram_tokenize(test_doc, 3)}")


# ============================================================
# 4. sklearn CountVectorizer（工业级实现）
# ============================================================
print("\n" + "="*50)
print("sklearn CountVectorizer")
print("="*50)

# 基础用法
vectorizer = CountVectorizer(
    min_df=1,           # 词最少出现文档数（过滤极低频词）
    max_df=0.8,         # 词最多出现文档比例（过滤停用词）
    stop_words='english',  # 自动去除英语停用词
    ngram_range=(1, 2), # 同时使用 unigram 和 bigram
    max_features=100,   # 最多保留100个特征
)

X = vectorizer.fit_transform(corpus)  # 返回稀疏矩阵
print(f"\n特征矩阵形状: {X.shape}")
print(f"矩阵类型: {type(X)}")
print(f"稀疏度: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
print(f"\n部分特征名: {vectorizer.get_feature_names_out()[:20]}")

# 转为密集矩阵查看
print(f"\n密集矩阵 (前5列):\n{X.toarray()[:, :5]}")

# 对新文档进行变换（只用已有词汇，OOV 词被忽略）
new_docs = ["NLP is fascinating", "I love robots and AI"]
X_new = vectorizer.transform(new_docs)
print(f"\n新文档变换结果:\n{X_new.toarray()[:, :5]}")


# ============================================================
# 5. 高频词问题演示 + 停用词过滤效果
# ============================================================
print("\n" + "="*50)
print("高频词问题演示")
print("="*50)

# 不过滤停用词
vec_raw = CountVectorizer()
X_raw = vec_raw.fit_transform(corpus)
word_freq = np.asarray(X_raw.sum(axis=0)).flatten()
top_indices = word_freq.argsort()[-10:][::-1]
feature_names = vec_raw.get_feature_names_out()

print("\n出现频率最高的10个词（不过滤停用词）:")
for idx in top_indices:
    print(f"  '{feature_names[idx]}': {word_freq[idx]:.0f} 次")

print("\n结论: 'is', 'and', 'I' 等停用词频率高但无语义价值，")
print("这正是 TF-IDF 被发明的动机。")
```

---

## ⚠️ 易错点与常见误解

### 1. 忽略语序导致的语义错误
**误解**：词袋模型已经足够捕捉文档含义。

**正确理解**：BoW 完全丢失了词序信息。"not good"和"good not"的 BoW 向量完全相同，但语义相反。对于情感分析、问答等需要理解语义的任务，BoW 表现明显不足。

### 2. 高频词污染相似度计算
**误解**：直接用原始词频计算相似度就够了。

**正确理解**："the"、"is"、"a" 等停用词在几乎所有文档中都高频出现，会人为拉高不相关文档之间的相似度。必须要么去除停用词，要么使用 TF-IDF 加权来压制这些词的影响。

### 3. 词汇表大小与稀疏性
**误解**：词汇表越大越好，应该包含所有出现过的词。

**正确理解**：
- 低频词（如只出现1次）通常是噪声（拼写错误、专有名词），收录它们只会增加维度而不增加信息
- 通常设置 `min_df=2` 或 `min_df=5` 过滤低频词
- 词汇表过大会导致严重的维度灾难和稀疏性问题

### 4. OOV 的处理
**误解**：测试集新词可以直接加入词汇表。

**正确理解**：一旦模型训练完成，词汇表就固定了。测试时遇到 OOV 词，CountVectorizer 会**静默忽略**，这可能导致测试文档向量信息严重缺失。应在训练时合理设置词汇表大小，或使用 `<UNK>` token 处理未知词。

### 5. sklearn 稀疏矩阵默认行为
**误解**：`CountVectorizer.fit_transform()` 返回 numpy 数组。

**正确理解**：返回的是 `scipy.sparse.csr_matrix` 稀疏矩阵。直接调用 `.toarray()` 转为密集矩阵可能造成内存爆炸（对大语料库而言）。大多数 sklearn 模型可以直接接受稀疏矩阵输入，无需转换。

### 6. max_df 参数的含义
**误解**：`max_df=0.8` 表示词频超过80次就过滤。

**正确理解**：`max_df=0.8` 表示**出现在超过80%的文档中**的词会被过滤（这些词太常见，可能是停用词）。整数值如 `max_df=10` 才表示出现在超过10篇文档中的词被过滤。

---

## 🔗 知识延伸

| 概念 | 与 BoW 的关系 |
|------|-------------|
| **TF-IDF** | BoW 的加权改进版，用 IDF 压制高频无意义词 |
| **N-gram** | 用连续 N 个词作为 token，对 BoW 语序问题的有限修补 |
| **SVD/LSA** | 对 BoW 矩阵做 SVD 降维，得到语义更丰富的稠密向量 |
| **Word2Vec** | 彻底放弃 BoW 框架，用神经网络学习稠密词向量 |
| **朴素贝叶斯文本分类** | 在 BoW 假设下工作最好的经典分类器 |
| **One-Hot** | 单个词的 One-Hot 向量求和 = 文档的 BoW 向量 |

---

## 📚 参考资料

- Harris, Z. (1954). *Distributional Structure*. Word.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Chapter 6.
- scikit-learn 文档：`sklearn.feature_extraction.text.CountVectorizer`
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6: Vector Semantics and Embeddings
