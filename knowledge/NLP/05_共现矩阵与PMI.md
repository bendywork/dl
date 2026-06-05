# 共现矩阵与 PMI（点互信息）

## 📌 核心问题
> One-Hot 编码无法表示词与词之间的语义关系。如何从大量语料中自动挖掘"哪些词经常一起出现"这一统计信息，进而建立词的语义相似度？

---

## 🌱 根源与动机

### 分布假说（Distributional Hypothesis）
1954 年 Zellig Harris 提出，1957 年 J.R. Firth 经典表述：

> **"You shall know a word by the company it keeps."**
> — J.R. Firth

核心思想：**意思相近的词倾向于出现在相似的上下文中**。

- "猫"和"狗"都常与"喂食"、"宠物"、"可爱"共现 → 语义相似
- "国王"和"王后"都常与"王宫"、"统治"、"王位"共现 → 语义相关

这个假说是所有词向量方法（共现矩阵、Word2Vec、GloVe）的理论基础。

### 共现矩阵的直觉
就像社交网络中"经常在一起出现的人更可能是朋友"，词汇表中的词也是：在一个窗口（5-10个词）内经常同时出现的词，语义上往往相关。把这种共现关系用矩阵记录下来，就得到了共现矩阵。

---

## 📐 理论推导

### 共现矩阵定义

设词汇表 $\mathcal{V} = \{w_1, \ldots, w_V\}$，窗口大小为 $L$。

**共现矩阵** $\mathbf{X} \in \mathbb{R}^{V \times V}$，其中：

$$
X_{ij} = \text{在语料库中，} w_j \text{ 出现在 } w_i \text{ 的窗口内的次数}
$$

对称性：通常构建对称矩阵，即 $X_{ij} = X_{ji}$（左窗口和右窗口都统计）。

**示例**（窗口大小=1）：

语料："I love cats I love dogs cats are great"

| | I | love | cats | dogs | are | great |
|---|---|------|------|------|-----|-------|
| I | 0 | 2 | 0 | 0 | 0 | 0 |
| love | 2 | 0 | 1 | 1 | 0 | 0 |
| cats | 0 | 1 | 0 | 0 | 1 | 0 |
| dogs | 0 | 1 | 0 | 0 | 0 | 0 |
| are | 0 | 0 | 1 | 0 | 0 | 1 |
| great | 0 | 0 | 0 | 0 | 1 | 0 |

### PMI（Pointwise Mutual Information，点互信息）

原始共现次数有一个问题：高频词（如 "the"）与任何词共现次数都高，但并不意味着语义相关。

PMI 通过比较**实际共现概率**与**独立假设下的期望共现概率**来衡量真实的语义关联：

$$
\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) P(y)}
$$

用共现矩阵估计概率：

$$
P(x, y) = \frac{X_{xy}}{\sum_{i,j} X_{ij}}, \quad P(x) = \frac{\sum_j X_{xj}}{\sum_{i,j} X_{ij}}, \quad P(y) = \frac{\sum_i X_{iy}}{\sum_{i,j} X_{ij}}
$$

**PMI 的解读**：
- $\text{PMI}(x, y) > 0$：$x$ 和 $y$ 比独立时更经常共现（正相关）
- $\text{PMI}(x, y) = 0$：$x$ 和 $y$ 统计独立
- $\text{PMI}(x, y) < 0$：$x$ 和 $y$ 比独立时更少共现（负相关）

**示例**（语料 "I love cats I love dogs"）：

- $P(\text{love}, \text{cats}) = 1/11$（假设总共现对数=11）
- $P(\text{love}) = 4/11$（love 出现在4次共现对中）
- $P(\text{cats}) = 2/11$
- $\text{PMI}(\text{love}, \text{cats}) = \log_2 \frac{1/11}{(4/11)(2/11)} = \log_2 \frac{11}{8} \approx 0.46 > 0$

### PPMI（Positive PMI，正点互信息）

PMI 对低频词偏高（稀疏性导致，见易错点），负值语义不明确（不共现和反义词无法区分），实践中普遍使用 PPMI：

$$
\text{PPMI}(x, y) = \max(\text{PMI}(x, y), 0)
$$

**为什么截断负值**：
1. 负 PMI 表示两词的共现频率低于随机期望，但这可能只是因为数据稀疏，并非真正的语义排斥
2. 截断为 0 后，矩阵非负，数学性质更好（可以配合 SVD）
3. 实验表明 PPMI 的词相似度计算效果远优于原始 PMI

### PMI 与 IDF 的关系

PMI 和 IDF 本质都在惩罚高频词，只是视角不同：

$$
\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)P(c)} = \log \frac{N \cdot \text{count}(w, c)}{\text{count}(w) \cdot \text{count}(c)}
$$

类比 IDF：IDF 惩罚在所有文档中都出现的词，PMI 惩罚与所有词都高频共现的词。

---

## 💡 关键理解

### 共现矩阵与词-文档矩阵的区别

| | 词-文档矩阵 | 词-词共现矩阵 |
|---|------------|-------------|
| 行 | 词 | 词 |
| 列 | 文档 | 词（上下文词） |
| 元素 | TF / TF-IDF | 共现次数 |
| 应用 | 文档检索、分类 | 词义相似度 |

### PMI vs TF-IDF
- TF-IDF 衡量词对某篇文档的重要性（文档级别）
- PMI 衡量两个词在同一上下文中的语义关联强度（词对级别）

### PPMI 矩阵是 GloVe 的前身
GloVe 的核心目标函数就是让词向量的内积拟合共现矩阵的对数值（$\log X_{ij}$），与 PMI 有直接数学联系：

$$
\text{GloVe 目标} \approx \text{拟合 PPMI 矩阵（加权）}
$$

---

## 🔧 代码实现

```python
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from itertools import combinations

# ============================================================
# 1. 构建共现矩阵
# ============================================================

def build_cooccurrence_matrix(
    corpus: list[list[str]],
    window_size: int = 2,
    symmetric: bool = True
) -> tuple[np.ndarray, dict, dict]:
    """
    构建词-词共现矩阵
    
    Args:
        corpus: 句子列表，每个句子是词的列表
        window_size: 上下文窗口半径（左右各 window_size 个词）
        symmetric: 是否对称（左窗口和右窗口都统计）
    
    Returns:
        matrix: 共现矩阵，shape=(V, V)
        word2idx: 词 -> 索引映射
        idx2word: 索引 -> 词映射
    """
    # 1. 构建词汇表（按词频排序）
    all_words = [w for sent in corpus for w in sent]
    word_freq = Counter(all_words)
    vocab = sorted(word_freq.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    V = len(vocab)
    
    # 2. 统计共现
    matrix = np.zeros((V, V), dtype=np.float32)
    
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word not in word2idx:
                continue
            word_idx = word2idx[word]
            
            # 统计窗口内的共现词
            left = max(0, i - window_size)
            right = min(len(sentence), i + window_size + 1)
            
            for j in range(left, right):
                if j == i:
                    continue
                context_word = sentence[j]
                if context_word not in word2idx:
                    continue
                context_idx = word2idx[context_word]
                matrix[word_idx, context_idx] += 1
    
    # 确保对称性（通常已经满足，但明确处理）
    if symmetric:
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 0)
        # 去掉重复计数（上面操作会导致非对角线加倍）
        matrix /= 2
    
    return matrix, word2idx, idx2word


# ============================================================
# 2. 计算 PMI 和 PPMI 矩阵
# ============================================================

def compute_pmi_matrix(
    cooc_matrix: np.ndarray,
    positive: bool = True,
    laplace_smoothing: float = 0.0
) -> np.ndarray:
    """
    从共现矩阵计算 PMI / PPMI 矩阵
    
    Args:
        cooc_matrix: 共现矩阵
        positive: True=PPMI, False=PMI
        laplace_smoothing: 拉普拉斯平滑（防止对数中的0）
    
    Returns:
        pmi_matrix: PMI 或 PPMI 矩阵
    """
    # 加平滑
    M = cooc_matrix + laplace_smoothing
    
    # 总共现次数
    total = M.sum()
    if total == 0:
        return np.zeros_like(M)
    
    # P(x, y) = M_xy / total
    P_xy = M / total
    
    # P(x) = sum_y M_xy / total（行边际概率）
    P_x = M.sum(axis=1, keepdims=True) / total
    
    # P(y) = sum_x M_xy / total（列边际概率）
    P_y = M.sum(axis=0, keepdims=True) / total
    
    # PMI = log2( P(x,y) / (P(x)*P(y)) )
    # 避免 log(0)：在 P_xy=0 的位置，PMI 定义为 -inf 或 0
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(P_xy / (P_x * P_y))
    
    # 处理 0/0 和 log(0) 情况
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # PPMI：截断负值
    if positive:
        pmi = np.maximum(pmi, 0)
    
    return pmi


# ============================================================
# 3. 测试与验证
# ============================================================

# 构建测试语料
corpus = [
    ["I", "love", "cats", "and", "dogs"],
    ["I", "love", "dogs", "too"],
    ["cats", "are", "cute", "animals"],
    ["dogs", "are", "loyal", "animals"],
    ["I", "have", "a", "cat", "and", "a", "dog"],
    ["cats", "and", "dogs", "are", "great", "pets"],
    ["deep", "learning", "is", "powerful"],
    ["machine", "learning", "and", "deep", "learning"],
    ["neural", "networks", "learn", "representations"],
    ["natural", "language", "processing", "uses", "deep", "learning"],
]

# 构建共现矩阵
cooc_matrix, word2idx, idx2word = build_cooccurrence_matrix(
    corpus, window_size=2, symmetric=True
)

print(f"词汇表大小: {len(word2idx)}")
print(f"共现矩阵形状: {cooc_matrix.shape}")
print(f"矩阵总共现次数: {cooc_matrix.sum():.0f}")

# 展示部分共现矩阵
focus_words = ["cats", "dogs", "love", "learning", "deep"]
focus_indices = [word2idx[w] for w in focus_words if w in word2idx]

print("\n共现矩阵（focus词汇）:")
print(f"{'':>10}", end="")
for w in focus_words:
    if w in word2idx:
        print(f"  {w:>8}", end="")
print()

for w in focus_words:
    if w not in word2idx:
        continue
    print(f"{w:>10}", end="")
    for c in focus_words:
        if c not in word2idx:
            continue
        print(f"  {cooc_matrix[word2idx[w], word2idx[c]]:>8.1f}", end="")
    print()

# 计算 PPMI
ppmi_matrix = compute_pmi_matrix(cooc_matrix, positive=True, laplace_smoothing=1.0)

print("\nPPMI 矩阵（focus词汇）:")
print(f"{'':>10}", end="")
for w in focus_words:
    if w in word2idx:
        print(f"  {w:>8}", end="")
print()

for w in focus_words:
    if w not in word2idx:
        continue
    print(f"{w:>10}", end="")
    for c in focus_words:
        if c not in word2idx:
            continue
        print(f"  {ppmi_matrix[word2idx[w], word2idx[c]]:>8.3f}", end="")
    print()


# ============================================================
# 4. 基于 PPMI 向量计算词相似度
# ============================================================
print("\n" + "="*60)
print("基于 PPMI 向量的词相似度")
print("="*60)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def most_similar(word: str, matrix: np.ndarray, word2idx: dict,
                 idx2word: dict, top_k: int = 5) -> list:
    """找最相似的词"""
    if word not in word2idx:
        return []
    
    word_vec = matrix[word2idx[word]]
    sims = []
    
    for idx in range(matrix.shape[0]):
        if idx == word2idx[word]:
            continue
        sim = cosine_similarity(word_vec, matrix[idx])
        sims.append((idx2word[idx], sim))
    
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]


for query_word in ["cats", "dogs", "learning", "deep"]:
    if query_word in word2idx:
        similar = most_similar(query_word, ppmi_matrix, word2idx, idx2word)
        print(f"\n与 '{query_word}' 最相似的词:")
        for w, sim in similar:
            print(f"  {w:>15}: {sim:.4f}")


# ============================================================
# 5. PMI 对低频词偏高的演示（易错点）
# ============================================================
print("\n" + "="*60)
print("PMI 对低频词偏高的问题演示")
print("="*60)

demo_corpus = [
    # 'cat' 和 'whiskers' 只共现一次（低频稀有搭配）
    ["cat", "has", "whiskers"],
    # 'cat' 和 'animal' 共现多次（高频常见搭配）
    ["cat", "is", "an", "animal"],
    ["cat", "is", "an", "animal"],
    ["cat", "is", "an", "animal"],
    ["cat", "is", "an", "animal"],
    ["dog", "is", "an", "animal"],
    ["bird", "is", "an", "animal"],
]

demo_cooc, demo_w2i, demo_i2w = build_cooccurrence_matrix(demo_corpus, window_size=1)
demo_pmi = compute_pmi_matrix(demo_cooc, positive=False, laplace_smoothing=0)

if "cat" in demo_w2i and "whiskers" in demo_w2i and "animal" in demo_w2i:
    pmi_cat_whiskers = demo_pmi[demo_w2i["cat"], demo_w2i["whiskers"]]
    pmi_cat_animal = demo_pmi[demo_w2i["cat"], demo_w2i["animal"]]
    print(f"\nPMI(cat, whiskers) = {pmi_cat_whiskers:.3f}  ← 只共现1次但PMI可能偏高")
    print(f"PMI(cat, animal)   = {pmi_cat_animal:.3f}   ← 共现4次")
    print("\n原因：'whiskers' 极低频，P(whiskers) 极小，PMI 分母极小，导致 PMI 偏高")
    print("解决方案：使用 PPMI 并配合 SVD 降维来缓解低频词偏高问题")
```

---

## ⚠️ 易错点与常见误解

### 1. PMI 对低频词偏高（Low-frequency Bias）
**误解**：PMI 高一定代表两词语义强相关。

**正确理解**：若词 $y$ 非常罕见（$P(y)$ 极小），则 $P(x, y) / (P(x) \cdot P(y))$ 的分母极小，即使 $x$ 和 $y$ 只共现一次，PMI 也会很高。这是 PMI 对低频词的系统性偏高问题。

**解决方案**：
1. **PPMI**：截断负值，一定程度上缓解
2. **PMI²（Squared PMI）**：$\text{PMI}^2(x,y) = \log P(x,y)^2/(P(x)P(y))$
3. **SVD 降维**：低秩近似会抑制噪声（低频词的伪高 PMI 被"平均掉"）

### 2. PPMI 截断负值丢失信息？
**误解**：负 PMI 包含有价值的信息（反义词），不该截断。

**正确理解**：负 PMI 在实践中噪声极大，因为：
- 两词不共现可能只是因为数据稀疏，而非真正的语义排斥
- 即使是反义词（"热"和"冷"）也可能在语料中一起出现（"不热不冷"）
- 截断为 0 后矩阵非负，便于后续的 SVD 等操作

### 3. 窗口大小的影响
**误解**：窗口越大越好，能捕获更多上下文信息。

**正确理解**：
- **小窗口（1-5词）**：捕获句法关系（动词-宾语、修饰-名词）
- **大窗口（5-20词）**：捕获语义/主题关系（同一话题的词）

Levy & Goldberg (2014) 的实验表明，小窗口的 PPMI 在词汇关系（analogies）上表现更好。

### 4. 对称 vs 非对称共现矩阵
**误解**：共现矩阵一定是对称的。

**正确理解**：
- **对称共现**：$X_{ij}$ = 词 $j$ 在词 $i$ 的左右窗口内出现的次数（两个方向都算）
- **非对称共现**：可以分别统计 $j$ 出现在 $i$ 左边 vs 右边的次数

大多数实现使用对称版本，GloVe 也使用对称共现矩阵。

### 5. 共现矩阵 vs 词-文档矩阵
**误解**：两者本质相同，只是行列角色互换。

**正确理解**：
- **词-文档矩阵**：行是词，列是文档，元素是 TF/TF-IDF，用于**文档相似度/检索**
- **词-词共现矩阵**：行列都是词，元素是共现频率/PMI，用于**词义相似度/词向量**

两者的 SVD 降维产生的向量语义也不同（LSA vs 共现词向量）。

---

## 🔗 知识延伸

| 概念 | 与共现矩阵/PMI 的关系 |
|------|---------------------|
| **TF-IDF** | 词级别的 TF-IDF 与 IDF 本质上都是 PMI 的变体 |
| **SVD/LSA** | 对 PPMI 矩阵做 SVD 得到密集词向量，比原始 PPMI 效果更好 |
| **GloVe** | 显式利用全局共现矩阵，目标是拟合 $\log X_{ij}$（类 PMI） |
| **Word2Vec** | 隐式地拟合 PMI 矩阵（Levy & Goldberg 2014 的理论分析） |
| **Neural Word Embeddings** | 可以视为对 PPMI 矩阵进行低秩近似的神经网络实现 |

---

## 📚 参考资料

- Harris, Z. (1954). *Distributional Structure*. Word.
- Church, K. W., & Hanks, P. (1990). *Word Association Norms, Mutual Information, and Lexicography*. Computational Linguistics.
- Turney, P. D., & Pantel, P. (2010). *From Frequency to Meaning: Vector Space Models of Semantics*. JAIR.
- Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NIPS. - 证明 Word2Vec 隐式分解 PMI 矩阵
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6.3-6.5
