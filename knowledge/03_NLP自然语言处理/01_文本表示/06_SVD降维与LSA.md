# SVD 降维与 LSA（潜在语义分析）

## 📌 核心问题
> PPMI 共现矩阵是 $V \times V$ 的高维稀疏矩阵（V 可达数万），直接用于计算词相似度既低效又噪声大。如何将其压缩成低维稠密的词向量，同时保留语义信息，甚至泛化出"潜在语义"（同义词、主题关联）？

---

## 🌱 根源与动机

### 历史背景
1990 年，Deerwester 等人在信息检索领域提出了**潜在语义分析（Latent Semantic Analysis，LSA）**，也称为**潜在语义索引（LSI，Latent Semantic Indexing）**。

核心发现：TF-IDF 矩阵/共现矩阵中存在大量"噪声"（拼写变体、同义词导致的矩阵稀疏），用 SVD 截断分解取低秩近似，能够"平滑掉"这些噪声，让语义上相关的词/文档在低维空间中聚集。

### 直觉：摄影的压缩类比
想象一张 $4000 \times 3000$ 的照片（1200万像素），若用 JPEG 压缩到 10KB，虽然丢失了细节，但人眼仍能识别主要内容——因为照片的主要信息集中在少数"主方向"上。

SVD 对矩阵做了类似的事：找出数据的主方向（奇异向量），按重要性（奇异值大小）排列，取前 $k$ 个保留主要信息，丢弃噪声。

---

## 📐 理论推导

### SVD（奇异值分解）

任意矩阵 $\mathbf{M} \in \mathbb{R}^{m \times n}$ 都可以精确分解为：

$$
\mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
$$

其中：
- $\mathbf{U} \in \mathbb{R}^{m \times m}$：左奇异向量矩阵（列正交：$\mathbf{U}^\top \mathbf{U} = \mathbf{I}$）
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$：奇异值对角矩阵，对角线 $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$：右奇异向量矩阵（列正交：$\mathbf{V}^\top \mathbf{V} = \mathbf{I}$）

### 截断 SVD（Truncated SVD）

取前 $k$ 个奇异值和对应的奇异向量：

$$
\mathbf{M} \approx \mathbf{M}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top
$$

其中 $\mathbf{U}_k \in \mathbb{R}^{m \times k}$，$\boldsymbol{\Sigma}_k \in \mathbb{R}^{k \times k}$，$\mathbf{V}_k \in \mathbb{R}^{n \times k}$。

**Eckart-Young 定理**：在所有秩为 $k$ 的矩阵中，$\mathbf{M}_k$ 是 $\mathbf{M}$ 的**最优低秩近似**（Frobenius 范数最小）：

$$
\mathbf{M}_k = \arg\min_{\text{rank}(\mathbf{B}) \leq k} \|\mathbf{M} - \mathbf{B}\|_F
$$

### LSA 的词向量提取

设 $\mathbf{M}$ 是词-文档 TF-IDF 矩阵（$V \times D$）或词-词 PPMI 矩阵（$V \times V$）。

SVD 分解后，词 $w_i$ 的低维向量为 $\mathbf{U}_k$ 的第 $i$ 行（乘以 $\boldsymbol{\Sigma}_k$ 后效果更好）：

$$
\text{词向量}(w_i) = \mathbf{U}_k[i, :] \cdot \boldsymbol{\Sigma}_k \in \mathbb{R}^k
$$

### 复杂度分析

**完整 SVD**：$O(\min(m^2n, mn^2))$
- 对 $V = 50000$ 的词-词矩阵：$O(50000^3)$ ≈ 无法完成

**截断 SVD（Lanczos/Randomized）**：$O(\text{nnz} \cdot k)$
- nnz 是非零元素数，$k$ 是目标维度（通常 100-300）
- 利用矩阵稀疏性，实际可行

---

## 💡 关键理解

### SVD 的"语义平滑"效果

**原始 PPMI 矩阵问题**：
- "汽车" 和 "轿车" 是不同列，直接相似度为 0
- 低频词 PMI 偏高，引入噪声

**SVD 降维后**：
- "汽车" 和 "轿车" 都与"驾驶"、"道路"、"发动机"高度共现，在低维空间中被映射到相近位置
- 低秩近似过滤掉低频噪声

这就是"潜在语义"（Latent Semantic）的含义——矩阵的低秩结构中包含了超越表面词汇的语义模式。

### SVD vs Word2Vec

| 特性 | SVD/LSA | Word2Vec |
|------|---------|----------|
| 训练方式 | 批量（全局矩阵） | 增量（随机梯度） |
| 内存需求 | $O(V^2)$（矩阵） | $O(V \times d)$（向量） |
| 增量更新 | 不支持（需重新计算） | 支持（继续训练） |
| 语义质量 | 中等 | 更好（神经网络泛化） |
| 速度 | 慢（大矩阵SVD） | 快（负采样） |

### 奇异值的信息量比例

$$
\text{方差解释比} = \frac{\sigma_i^2}{\sum_j \sigma_j^2}
$$

奇异值的平方正比于该方向上的信息量。通常前 100-300 个奇异值就能解释 80%+ 的方差。

---

## 🔧 代码实现

```python
import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# ============================================================
# 1. 手动 SVD 分解与词向量提取
# ============================================================

# 构建简单的词-词共现矩阵（PPMI）
# 词汇表：[cat, dog, animal, pet, car, vehicle, drive, road]
words = ["cat", "dog", "animal", "pet", "car", "vehicle", "drive", "road"]
V = len(words)
word2idx = {w: i for i, w in enumerate(words)}

# 手动构造近似 PPMI 矩阵（实际应从语料统计，这里为演示简化）
np.random.seed(42)
ppmi_manual = np.array([
    # cat   dog   animal  pet   car   vehicle  drive  road
    [0.0,  2.5,   3.0,   3.5,  0.0,   0.0,    0.0,  0.0],  # cat
    [2.5,  0.0,   2.8,   3.0,  0.0,   0.0,    0.0,  0.0],  # dog
    [3.0,  2.8,   0.0,   2.0,  0.5,   0.5,    0.0,  0.0],  # animal
    [3.5,  3.0,   2.0,   0.0,  0.0,   0.0,    0.0,  0.0],  # pet
    [0.0,  0.0,   0.5,   0.0,  0.0,   3.5,    3.0,  2.5],  # car
    [0.0,  0.0,   0.5,   0.0,  3.5,   0.0,    2.8,  2.0],  # vehicle
    [0.0,  0.0,   0.0,   0.0,  3.0,   2.8,    0.0,  3.5],  # drive
    [0.0,  0.0,   0.0,   0.0,  2.5,   2.0,    3.5,  0.0],  # road
], dtype=np.float32)

print("原始 PPMI 矩阵:")
print(ppmi_manual)

# ============================================================
# 2. 完整 SVD 分解
# ============================================================
U, sigma, Vt = svd(ppmi_manual, full_matrices=False)

print(f"\nSVD 分解结果:")
print(f"U 形状: {U.shape}  （词的左奇异向量）")
print(f"sigma 形状: {sigma.shape}  （奇异值）")
print(f"Vt 形状: {Vt.shape}  （上下文的右奇异向量）")

print(f"\n奇异值: {np.round(sigma, 3)}")

# 方差解释率
variance_explained = sigma**2 / (sigma**2).sum()
cumulative = np.cumsum(variance_explained)
print("\n奇异值方差解释率:")
for i, (sv, ve, cv) in enumerate(zip(sigma, variance_explained, cumulative)):
    print(f"  σ_{i+1} = {sv:.3f}  方差: {ve:.1%}  累积: {cv:.1%}")

# 验证重构
M_reconstructed = U @ np.diag(sigma) @ Vt
print(f"\n重构误差 (Frobenius): {np.linalg.norm(ppmi_manual - M_reconstructed):.6f}")


# ============================================================
# 3. 截断 SVD 提取词向量
# ============================================================
def extract_word_vectors_svd(M: np.ndarray, k: int) -> np.ndarray:
    """
    截断 SVD 提取 k 维词向量
    
    Args:
        M: 原始矩阵（PPMI 或 TF-IDF 矩阵）
        k: 目标维度
    
    Returns:
        word_vectors: shape=(V, k)，已 L2 归一化
    """
    U, sigma, Vt = svd(M, full_matrices=False)
    
    # 取前 k 个奇异值和向量
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    
    # 词向量 = U_k * Sigma_k（加权的奇异向量）
    word_vectors = U_k * sigma_k[np.newaxis, :]  # broadcasting
    
    # L2 归一化
    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    word_vectors = word_vectors / norms
    
    return word_vectors


def cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    """计算所有词对之间的余弦相似度矩阵"""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    M_norm = M / norms
    return M_norm @ M_norm.T


# 提取 2 维词向量（便于可视化）
word_vecs_2d = extract_word_vectors_svd(ppmi_manual, k=2)
word_vecs_3d = extract_word_vectors_svd(ppmi_manual, k=3)

print("\n2D 词向量（截断 SVD k=2）:")
for i, w in enumerate(words):
    print(f"  {w:>10}: [{word_vecs_2d[i, 0]:+.4f}, {word_vecs_2d[i, 1]:+.4f}]")

# 相似度矩阵（低维）
sim_matrix_svd = cosine_similarity_matrix(word_vecs_2d)

print("\n余弦相似度矩阵（SVD k=2）:")
print(f"{'':>10}", end="")
for w in words:
    print(f"  {w:>7}", end="")
print()
for i, wi in enumerate(words):
    print(f"{wi:>10}", end="")
    for j, wj in enumerate(words):
        print(f"  {sim_matrix_svd[i, j]:>7.3f}", end="")
    print()

# 相似度矩阵（原始 PPMI）
sim_matrix_raw = cosine_similarity_matrix(ppmi_manual)
print(f"\n原始 PPMI 中 (cat, dog): {sim_matrix_raw[0, 1]:.3f}")
print(f"SVD k=2 中 (cat, dog):  {sim_matrix_svd[0, 1]:.3f}")


# ============================================================
# 4. 可视化：2D 空间中的词聚类
# ============================================================
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：2D SVD 词向量散点图
    ax = axes[0]
    colors = ['blue', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red']
    for i, (word, vec) in enumerate(zip(words, word_vecs_2d)):
        ax.scatter(vec[0], vec[1], color=colors[i], s=100)
        ax.annotate(word, (vec[0], vec[1]),
                   textcoords="offset points", xytext=(5, 5), fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title("SVD 词向量 (k=2)\n蓝=动物 红=交通", fontsize=13)
    ax.set_xlabel("SVD 维度 1")
    ax.set_ylabel("SVD 维度 2")
    ax.grid(True, alpha=0.3)
    
    # 右图：奇异值能量图
    ax = axes[1]
    k_range = range(1, len(sigma) + 1)
    bars = ax.bar(k_range, variance_explained * 100, color='steelblue', alpha=0.7)
    ax.plot(k_range, cumulative * 100, 'r-o', label='累积方差解释率')
    ax.axhline(y=80, color='green', linestyle='--', label='80% 阈值')
    ax.set_xlabel("奇异值排序")
    ax.set_ylabel("方差解释率 (%)")
    ax.set_title("奇异值方差解释率", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/tmp/svd_lsa_visualization.png", dpi=100, bbox_inches='tight')
    print("\n可视化已保存至 /tmp/svd_lsa_visualization.png")
    plt.close()
except Exception as e:
    print(f"\n可视化跳过: {e}")


# ============================================================
# 5. sklearn TruncatedSVD（工业级，适合大稀疏矩阵）
# ============================================================
print("\n" + "="*60)
print("sklearn TruncatedSVD（适合大型稀疏矩阵）")
print("="*60)

# 构建文本语料的 TF-IDF 矩阵
corpus = [
    "I love cats and dogs",
    "cats are great pets",
    "dogs are loyal animals",
    "I drive cars on roads",
    "cars and vehicles on the highway",
    "deep learning is powerful",
    "machine learning uses algorithms",
    "neural networks learn features",
    "natural language processing NLP",
    "NLP uses deep learning models",
]

# TF-IDF 矩阵（文档 × 词汇）
tfidf = TfidfVectorizer(min_df=1)
X_tfidf = tfidf.fit_transform(corpus)  # 稀疏矩阵
print(f"TF-IDF 矩阵形状: {X_tfidf.shape}")
print(f"稀疏度: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.1%}")

# 截断 SVD（LSA）
svd_lsa = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
X_lsa = svd_lsa.fit_transform(X_tfidf)  # 文档在低维语义空间的表示
print(f"\nLSA 后文档向量形状: {X_lsa.shape}")

# 方差解释率
explained = svd_lsa.explained_variance_ratio_
print(f"前5个成分方差解释率: {[f'{v:.1%}' for v in explained]}")
print(f"累积方差解释率: {explained.sum():.1%}")

# 文档相似度（LSA 空间）
X_lsa_norm = normalize(X_lsa, norm='l2')
sim = X_lsa_norm @ X_lsa_norm.T

print("\n文档相似度矩阵（LSA k=5）:")
print(np.round(sim, 2))

# 找最相似的文档对
print("\n最相似的文档对:")
for i in range(len(corpus)):
    for j in range(i+1, len(corpus)):
        if sim[i, j] > 0.5:
            print(f"  doc{i+1} <-> doc{j+1}: {sim[i,j]:.3f}")
            print(f"    '{corpus[i]}'")
            print(f"    '{corpus[j]}'")


# ============================================================
# 6. 重构误差 vs k 的关系
# ============================================================
print("\n" + "="*60)
print("不同 k 值的重构误差")
print("="*60)

U_full, sigma_full, Vt_full = svd(ppmi_manual, full_matrices=False)
frobenius_norm_M = np.linalg.norm(ppmi_manual, 'fro')

print(f"{'k':>4} | {'重构误差':>10} | {'保留方差':>10} | {'压缩比':>8}")
print("-" * 40)
for k in range(1, len(sigma_full) + 1):
    M_k = U_full[:, :k] @ np.diag(sigma_full[:k]) @ Vt_full[:k, :]
    error = np.linalg.norm(ppmi_manual - M_k, 'fro')
    variance_kept = (sigma_full[:k]**2).sum() / (sigma_full**2).sum()
    compression = k * (V + V) / (V * V)
    print(f"{k:>4} | {error:>10.4f} | {variance_kept:>10.1%} | {compression:>8.1%}")
```

---

## ⚠️ 易错点与常见误解

### 1. SVD 计算复杂度问题
**误解**：可以直接对大型稀疏矩阵做完整 SVD。

**正确理解**：完整 SVD 的复杂度是 $O(mn^2)$ 或 $O(m^2n)$（取小值）。对于 $V=50000$ 的词-词矩阵，计算量约 $O(50000^3)$，内存需求 $50000^2 \times 4 \text{bytes} \approx 10\text{GB}$，完全不可行。

实践中必须使用：
- `scipy.sparse.linalg.svds`（稀疏矩阵截断 SVD）
- `sklearn.decomposition.TruncatedSVD`（底层用随机化 SVD）
- 这些方法复杂度是 $O(\text{nnz} \times k)$，nnz 是非零元素数量

### 2. 词向量应该取 U 还是 U×Σ？
**误解**：词向量就是 SVD 的左奇异向量 $\mathbf{U}$。

**正确理解**：
- $\mathbf{U}$ 是正交矩阵，列向量长度均为 1，没有反映奇异值的大小差异
- $\mathbf{U}_k \boldsymbol{\Sigma}_k$ 才是更好的词向量，因为奇异值 $\sigma_i$ 反映了第 $i$ 个方向上的信息量

实践中，也有直接用 $\mathbf{U}_k$ 并进行 L2 归一化的方案，效果相近。

### 3. LSA 无法增量更新
**误解**：可以把新文档/新词加入后快速更新 SVD 结果。

**正确理解**：SVD 是全局分解，加入新数据需要重新计算。虽然有折叠（folding-in）技巧（将新向量投影到已有的奇异向量空间），但这会逐渐降低精度。Word2Vec 的优势之一就是可以增量更新。

### 4. k 的选择
**误解**：k 越大越好，越接近原始矩阵越好。

**正确理解**：过大的 k 会包含噪声（稀疏矩阵中的低频词偏高 PMI），过小的 k 会丢失有用信息。典型的选择是 k=100-300，通常通过验证集上的词相似度任务来选择。

### 5. 奇异值分解 vs 特征值分解
**误解**：对称矩阵的 SVD 和特征值分解是完全不同的东西。

**正确理解**：对于**对称半正定矩阵**（如词-词共现矩阵，通常是对称的），SVD 和特征值分解等价：
$$
\mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top
$$
此时 $\mathbf{U} = \mathbf{V} = \mathbf{Q}$（特征向量矩阵），$\boldsymbol{\Sigma} = \boldsymbol{\Lambda}$（特征值=奇异值）。

对于非对称的词-文档矩阵，SVD 和特征值分解才是完全不同的。

### 6. LSA 与 LDA 的区别
**误解**：LSA 和 LDA 都是"潜在语义"方法，本质相同。

**正确理解**：
- **LSA（潜在语义分析）**：线性代数方法（SVD），输出连续的低维向量，无概率解释
- **LDA（潜在狄利克雷分配）**：概率图模型，输出主题分布，有清晰的概率解释（文档是主题的混合，主题是词的分布）

两者的"潜在"含义不同，LSA 的"潜在"指矩阵分解后的低维方向，LDA 的"潜在"指不可观测的主题变量。

---

## 🔗 知识延伸

| 概念 | 与 SVD/LSA 的关系 |
|------|----------------|
| **PPMI 矩阵** | SVD 的输入矩阵；对 PPMI 做 SVD 比对原始词频做效果更好 |
| **PCA** | 对中心化数据的协方差矩阵做 SVD = PCA（SVD 是 PCA 的通用形式） |
| **Word2Vec** | 神经网络方法，可以视为对 PMI 矩阵做隐式的低秩近似（但更灵活） |
| **GloVe** | 显式优化共现矩阵的对数，与 LSA 的关系最直接 |
| **Matrix Factorization** | SVD 是矩阵分解的特例，协同过滤（推荐系统）也用类似思想 |
| **随机化 SVD** | Halko et al. 2011，$O(\text{nnz} \cdot k)$ 复杂度，sklearn 的默认实现 |

---

## 📚 参考资料

- Deerwester, S., et al. (1990). *Indexing by Latent Semantic Analysis*. JASIST. - LSA 原论文
- Turney, P. D., & Pantel, P. (2010). *From Frequency to Meaning: Vector Space Models of Semantics*. JAIR.
- Halko, N., et al. (2011). *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*. SIAM Review. - 随机化 SVD
- Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NIPS.
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6.4: Word Embeddings with Dense Vectors
- sklearn 文档：`sklearn.decomposition.TruncatedSVD`
