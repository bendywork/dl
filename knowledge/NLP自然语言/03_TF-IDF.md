# TF-IDF（词频-逆文档频率）

## 📌 核心问题
> 词袋模型中，"the"、"is"、"a" 等高频词会主导文档向量，掩盖真正有区分意义的词。如何自动识别并降低这些无意义词的权重，同时提升"深度学习"、"梯度下降"等有信息量词汇的权重？

---

## 🌱 根源与动机

### 历史背景
1972 年，Karen Spärck Jones 提出了 IDF（逆文档频率）的概念，发表在信息检索领域的经典论文中。她的直觉是：

**一个词越稀有，它对文档区分的贡献越大。**

- "深度学习" 只在 AI 相关文档中出现 → 高区分性 → 高 IDF
- "the" 几乎在所有英文文档中出现 → 无区分性 → 低 IDF（趋近于0）

TF-IDF 结合了两个直觉：
1. **TF（词频）**：词在当前文档中出现越多，对该文档越重要
2. **IDF（逆文档频率）**：词在整个语料库中越罕见，越能区分文档

这两个因素相乘，就能自动过滤掉停用词，保留关键词。

---

## 📐 理论推导

### TF（Term Frequency，词频）

$$
\text{TF}(t, d) = \frac{\text{count}(t, d)}{\sum_{t' \in d} \text{count}(t', d)}
$$

其中 $\text{count}(t, d)$ 是词 $t$ 在文档 $d$ 中的出现次数，分母是文档总词数。

**原始定义变体**（sklearn 默认使用原始词频，不归一化）：

$$
\text{TF}_{\text{raw}}(t, d) = \text{count}(t, d)
$$

### IDF（Inverse Document Frequency，逆文档频率）

$$
\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}
$$

其中 $N$ 是语料库总文档数，$|\{d : t \in d\}|$ 是包含词 $t$ 的文档数（document frequency，DF）。

**加1平滑（避免除零）：**

$$
\text{IDF}_{\text{smooth}}(t) = \log\frac{N + 1}{|\{d : t \in d\}| + 1} + 1
$$

sklearn 使用此变体（`smooth_idf=True`）。

### TF-IDF 最终权重

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

### 数值示例

设语料库有 100 篇文档（$N = 100$）：

| 词 | TF (在d中) | 包含该词的文档数 | IDF | TF-IDF |
|----|-----------|----------------|-----|--------|
| "the" | 10 | 99 | $\log(100/99) \approx 0.004$ | $\approx 0.04$ |
| "learning" | 3 | 20 | $\log(100/20) = \log(5) \approx 1.61$ | $\approx 4.83$ |
| "backpropagation" | 2 | 2 | $\log(100/2) = \log(50) \approx 3.91$ | $\approx 7.82$ |

"backpropagation" 的 TF-IDF 远高于 "the"，符合直觉。

### 为什么用 log？

若不用 log，IDF 的量纲变化过大：
- DF=1（极罕见词）：IDF = N/1 = 100（量纲爆炸）
- DF=N（无处不在）：IDF = N/N = 1

取 log 后：
- DF=1：IDF = log(100) ≈ 4.6
- DF=N：IDF = log(1) = 0（停用词权重恰好为0）
- IDF 的增长被压缩到对数尺度，数值稳定

**log 的第二个作用**：与信息论对应——IDF 近似等价于词的**自信息**（surprisal）：
$$
\text{IDF}(t) \approx -\log P(t \text{ appears in document})
$$

---

## 💡 关键理解

### TF-IDF 的核心权衡
| 情形 | TF | IDF | TF-IDF | 含义 |
|------|----|----|---------|------|
| 停用词（"the"） | 高 | 极低 | 低 | 普遍出现，无区分性 |
| 关键词（"backprop"） | 中 | 高 | 高 | 在本文档出现且稀有 |
| 非常罕见的词（"xyzzy"） | 低 | 极高 | 低 | 稀有但本文未重点讨论 |
| 主题词（"深度学习"） | 高 | 中 | 高 | 本文频繁提及且有区分性 |

### TF-IDF 的局限
1. **不能捕捉语义相似性**："car" 和 "automobile" 是完全不同的词，TF-IDF 无法知道它们相似
2. **忽略词序**：继承了 BoW 的缺陷
3. **不能捕捉上下文**：同一词在不同语境的含义不同（"bank" 银行 vs 河岸）

这些局限直接推动了 Word2Vec、BERT 等语义模型的发展。

### L2 归一化
sklearn 的 TfidfVectorizer 默认对每个文档向量做 L2 归一化：

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}
$$

归一化后，两个文档向量的**点积**直接等于**余弦相似度**，方便后续计算。

---

## 🔧 代码实现

```python
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ============================================================
# 1. 手动实现 TF-IDF
# ============================================================

class ManualTFIDF:
    """
    手动实现 TF-IDF，完整复现 sklearn 的计算逻辑
    """
    
    def __init__(self, smooth_idf: bool = True, norm: str = 'l2'):
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.vocab_ = {}
        self.idf_ = {}
    
    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()
    
    def fit(self, corpus: list[str]):
        """学习词汇表和 IDF 值"""
        N = len(corpus)
        
        # 统计每个词的文档频率 (DF)
        df = Counter()
        all_tokens = set()
        for doc in corpus:
            tokens = set(self._tokenize(doc))  # 每篇文档中每个词只统计一次
            df.update(tokens)
            all_tokens.update(tokens)
        
        # 构建词汇表
        self.vocab_ = {word: idx for idx, word in enumerate(sorted(all_tokens))}
        
        # 计算 IDF
        self.idf_ = {}
        for word, idx in self.vocab_.items():
            df_t = df[word]
            if self.smooth_idf:
                # sklearn 的 smooth_idf 公式
                idf_val = math.log((N + 1) / (df_t + 1)) + 1
            else:
                idf_val = math.log(N / df_t) + 1
            self.idf_[word] = idf_val
        
        return self
    
    def transform(self, corpus: list[str]) -> np.ndarray:
        """将文档转换为 TF-IDF 矩阵"""
        N = len(corpus)
        V = len(self.vocab_)
        matrix = np.zeros((N, V), dtype=np.float64)
        
        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            tf_raw = Counter(tokens)
            doc_len = len(tokens)
            
            for word, count in tf_raw.items():
                if word in self.vocab_:
                    j = self.vocab_[word]
                    tf = count  # sklearn 默认使用原始词频（不归一化）
                    matrix[i, j] = tf * self.idf_[word]
        
        # L2 归一化（sklearn 默认 norm='l2'）
        if self.norm == 'l2':
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免零向量除零
            matrix /= norms
        
        return matrix
    
    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)
    
    def get_feature_names(self) -> list[str]:
        return sorted(self.vocab_.keys())


# 测试语料
corpus = [
    "deep learning is a subset of machine learning",
    "machine learning uses statistical methods",
    "deep learning uses neural networks",
    "natural language processing is part of deep learning",
    "the the the the is is is",  # 故意加入停用词测试
]

# 手动实现
tfidf_manual = ManualTFIDF(smooth_idf=True, norm='l2')
X_manual = tfidf_manual.fit_transform(corpus)
print("手动实现 TF-IDF 矩阵 (前3篇文档, 前8列):")
print(np.round(X_manual[:3, :8], 4))

# sklearn 实现对比
tfidf_sklearn = TfidfVectorizer(smooth_idf=True, norm='l2')
X_sklearn = tfidf_sklearn.fit_transform(corpus).toarray()
print("\nsklearn TF-IDF 矩阵 (前3篇文档, 前8列):")
print(np.round(X_sklearn[:3, :8], 4))

# 验证两者是否接近（注意：词汇表排序可能不同）
print(f"\n手动实现与 sklearn 结果是否接近: 需要对齐词汇表后验证")


# ============================================================
# 2. IDF 值分析：验证停用词被压制
# ============================================================
print("\n" + "="*50)
print("IDF 值分析")
print("="*50)

tfidf = TfidfVectorizer(smooth_idf=True)
tfidf.fit(corpus)

# 获取词汇表和对应的 IDF 值
feature_names = tfidf.get_feature_names_out()
idf_values = tfidf.idf_

# 按 IDF 排序
idf_sorted = sorted(zip(feature_names, idf_values), key=lambda x: x[1])

print("\nIDF 值最低（最常见，信息量最低）:")
for word, idf in idf_sorted[:5]:
    print(f"  '{word}': IDF = {idf:.4f}")

print("\nIDF 值最高（最罕见，信息量最高）:")
for word, idf in idf_sorted[-5:]:
    print(f"  '{word}': IDF = {idf:.4f}")


# ============================================================
# 3. 逐步展示 TF-IDF 计算过程
# ============================================================
print("\n" + "="*50)
print("逐步计算示例")
print("="*50)

doc_example = "deep learning uses neural networks"
print(f"\n目标文档: '{doc_example}'")
print(f"语料库共 {len(corpus)} 篇文档\n")

tokens = doc_example.split()
tf_raw = Counter(tokens)
N = len(corpus)

print(f"{'词':<20} {'TF':>6} {'DF':>6} {'IDF':>8} {'TF-IDF':>10}")
print("-" * 55)

for word in sorted(tf_raw.keys()):
    tf = tf_raw[word]
    df = sum(1 for doc in corpus if word in doc.lower().split())
    idf = math.log((N + 1) / (df + 1)) + 1  # smooth_idf
    tfidf_val = tf * idf
    print(f"{word:<20} {tf:>6} {df:>6} {idf:>8.4f} {tfidf_val:>10.4f}")


# ============================================================
# 4. 文档相似度搜索（信息检索应用）
# ============================================================
print("\n" + "="*50)
print("TF-IDF 文档检索应用")
print("="*50)

# 构建文档索引
search_corpus = [
    "Python is great for data science and machine learning",
    "Java is used for enterprise software development",
    "Deep learning with PyTorch and TensorFlow",
    "Natural language processing with transformers and BERT",
    "Computer vision for image recognition and detection",
    "Reinforcement learning for game playing and robotics",
]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
doc_vectors = vectorizer.fit_transform(search_corpus)

def search(query: str, vectorizer, doc_vectors, corpus, top_k=3):
    """基于 TF-IDF 的文档检索"""
    query_vector = vectorizer.transform([query])
    
    # 余弦相似度（因为 TF-IDF 向量已经 L2 归一化，点积就是余弦相似度）
    similarities = (doc_vectors @ query_vector.T).toarray().flatten()
    
    # 排序取 top_k
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    print(f"\n查询: '{query}'")
    print("最相关文档:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. [相似度: {similarities[idx]:.4f}] {corpus[idx]}")

search("deep learning frameworks", vectorizer, doc_vectors, search_corpus)
search("language model NLP", vectorizer, doc_vectors, search_corpus)


# ============================================================
# 5. 可视化：TF-IDF 热力图
# ============================================================
print("\n" + "="*50)
print("TF-IDF 矩阵可视化（词云替代方案）")
print("="*50)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    
    mini_corpus = [
        "cat dog cat",
        "dog fish dog",
        "cat fish cat fish",
    ]
    
    tv = TfidfVectorizer()
    X_vis = tv.fit_transform(mini_corpus).toarray()
    words = tv.get_feature_names_out()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(X_vis, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=12)
    ax.set_yticks(range(len(mini_corpus)))
    ax.set_yticklabels([f"doc{i+1}" for i in range(len(mini_corpus))], fontsize=12)
    
    # 在格子中显示数值
    for i in range(len(mini_corpus)):
        for j in range(len(words)):
            ax.text(j, i, f"{X_vis[i, j]:.2f}", ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='TF-IDF value')
    plt.title("TF-IDF Matrix Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("/tmp/tfidf_heatmap.png", dpi=100, bbox_inches='tight')
    print("热力图已保存至 /tmp/tfidf_heatmap.png")
    plt.close()
except ImportError:
    print("matplotlib 未安装，跳过可视化")
```

---

## ⚠️ 易错点与常见误解

### 1. IDF 分母加1平滑的必要性
**误解**：$\text{IDF}(t) = \log(N / \text{DF}(t))$ 就够了。

**正确理解**：当某词出现在所有文档中时（DF = N），$\log(N/N) = 0$，该词权重为 0，合理。但若某词出现在测试集但不在训练集的任何文档中（DF = 0），则分母为 0，计算崩溃。加1平滑（`smooth_idf=True`）是防御性编程的好习惯：

$$
\text{IDF} = \log\frac{N+1}{\text{DF}+1} + 1
$$

加1后 IDF 最小值为 $\log(1) + 1 = 1$（不会为0），避免权重为0的极端情况。

### 2. 为什么 IDF 公式要加 +1（尾部的 +1）
**误解**：IDF 公式中的 +1 只是数值稳定性考虑。

**正确理解**：sklearn 在 IDF 公式尾部加 1（`+1`）是为了确保即使所有文档都包含某个词时，IDF 值也 >= 1（而非 0），防止 TF-IDF 向量退化为全零向量。这不是平滑，而是一个下界保证。

### 3. TF 定义的差异
**误解**：TF 就是词在文档中的出现次数。

**正确理解**：TF 有多种定义：
- 原始频率：$\text{TF} = \text{count}(t, d)$（sklearn 默认）
- 归一化频率：$\text{TF} = \text{count}(t, d) / |d|$
- 对数频率：$\text{TF} = 1 + \log(\text{count}(t, d))$（避免词频过大的词权重爆炸）

不同定义适合不同场景，使用时需确认工具使用的是哪种。

### 4. TF-IDF 不能捕捉语义相似性
**误解**：TF-IDF 向量的余弦相似度可以衡量词义相似度。

**正确理解**：TF-IDF 是词袋模型的扩展，"car" 和 "automobile" 对 TF-IDF 来说完全不相关（不同维度）。TF-IDF 只能衡量**词汇层面**的重叠，无法理解语义相似性。这是 Word2Vec、BERT 等模型解决的核心问题。

### 5. L2 归一化后的点积
**误解**：计算 TF-IDF 相似度必须显式计算余弦相似度公式。

**正确理解**：sklearn 的 `TfidfVectorizer` 默认 `norm='l2'`，输出向量已经 L2 归一化。此时两个文档向量的**点积**直接等于它们的余弦相似度：

$$
\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} \quad \text{（当 } \|\mathbf{u}\|=\|\mathbf{v}\|=1 \text{）}
$$

这在信息检索中极为高效，可以用矩阵乘法一次性计算所有文档对的相似度。

### 6. 子词/短语 TF-IDF
**误解**：TF-IDF 只能对单个词使用。

**正确理解**：通过 `ngram_range=(1, 2)` 等参数，TF-IDF 可以对词组（bigram）、短语使用，能捕获一定的局部语序信息（如 "deep learning" 作为一个 token）。

---

## 🔗 知识延伸

| 概念 | 与 TF-IDF 的关系 |
|------|----------------|
| **BoW** | TF-IDF 是 BoW 的加权改进，IDF 权重自动压制停用词 |
| **BM25** | TF-IDF 的改进版，引入文档长度归一化和饱和函数，广泛用于现代搜索引擎 |
| **PMI** | IDF 与 PMI（点互信息）有深刻联系，都是衡量词的"区分性" |
| **SVD/LSA** | 对 TF-IDF 矩阵做 SVD 降维，捕捉潜在语义结构 |
| **Word2Vec** | 从根本上解决 TF-IDF 无法捕捉语义相似性的问题 |
| **BERT** | 上下文感知的词表示，TF-IDF 的上下文无关性被彻底克服 |

---

## 📚 参考资料

- Spärck Jones, K. (1972). *A statistical interpretation of term specificity and its application in retrieval*. Journal of Documentation.
- Robertson, S. (2004). *Understanding Inverse Document Frequency*. Journal of Documentation.
- Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Chapter 6.
- scikit-learn 文档：`sklearn.feature_extraction.text.TfidfVectorizer`
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6
