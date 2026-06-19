# GloVe 词向量（Global Vectors for Word Representation）

## 📌 核心问题
> Word2Vec 使用局部滑动窗口，每次只看少数几个词的上下文，无法有效利用全局统计信息。如何设计一个词向量模型，既能利用**全局共现统计**（像 LSA 那样），又能获得**高质量的低维稠密向量**（像 Word2Vec 那样）？

---

## 🌱 根源与动机

### 历史背景
2014 年，Stanford NLP 组的 Pennington、Socher、Manning 发表论文《GloVe: Global Vectors for Word Representation》，提出了 GloVe 模型。

**GloVe 的核心洞察**：词向量模型的训练目标应该与词的**共现概率的比值**相关，而非直接拟合共现计数或预测上下文词。

### Word2Vec 的局限
Word2Vec 用滑动窗口逐个处理词，本质上在学习"给定中心词，预测上下文词"（Skip-gram）或反过来（CBOW）。这种训练方式：
1. **每次只看局部窗口**，对全局统计的利用效率低（同一共现对可能被多次重复训练）
2. 对低频词效果差（低频词的窗口样本少）

### GloVe 的核心直觉

考虑三个词：`ice`（冰）、`steam`（蒸汽）、`water`（水）、`solid`（固体）

| | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | 比值 |
|---|---|---|---|
| $k = \text{solid}$ | 大 | 小 | **远大于1** |
| $k = \text{gas}$ | 小 | 大 | **远小于1** |
| $k = \text{water}$ | 大 | 大 | **约等于1** |
| $k = \text{fashion}$ | 小 | 小 | **约等于1** |

**关键发现**：相对比值 $P(k \mid w_i) / P(k \mid w_j)$ 能区分 ice 和 steam 的语义差异，而单个条件概率不够。因此，词向量的内积应该能够表达这种比值关系。

---

## 📐 理论推导

### 从比值出发推导目标函数

设 $F(w_i, w_j, \tilde{w}_k)$ 是关于词向量的某函数，我们希望：

$$
F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} = \frac{P(k \mid i)}{P(k \mid j)}
$$

**推导约束**：
1. 词向量空间是线性的，$F$ 应该只依赖于 $w_i - w_j$（差向量）
2. $F$ 的参数是词向量 $w_i, w_j$ 和上下文向量 $\tilde{w}_k$，用内积形式：

$$
F\left((w_i - w_j)^\top \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}}
$$

3. 因为 $\frac{P_{ik}}{P_{jk}} = \frac{F(w_i^\top \tilde{w}_k)}{F(w_j^\top \tilde{w}_k)}$，自然令 $F = \exp$：

$$
\exp(w_i^\top \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}
$$

这意味着 $w_i^\top \tilde{w}_k \approx \log P_{ik} = \log P(k \mid i)$

### GloVe 目标函数

将 $\log P(k \mid i) = \log X_{ik} - \log X_i$ 代入（$X_i = \sum_j X_{ij}$），用偏置项 $b_i, \tilde{b}_k$ 吸收常数项：

$$
w_i^\top \tilde{w}_k + b_i + \tilde{b}_k \approx \log X_{ik}
$$

**加权最小二乘目标函数**：

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

**权重函数 $f(x)$**（解决两个问题）：

$$
f(x) = \begin{cases}
\left(\dfrac{x}{x_{\max}}\right)^{\alpha} & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}
$$

- 当 $x = 0$ 时，$f(0) = 0$（不统计未共现的词对，避免 $\log(0)$）
- 低频共现（$x$ 小）权重低，高频共现权重有上界（防止 "the"、"a" 等高频词主导训练）
- 论文建议 $x_{\max} = 100$，$\alpha = 3/4$

### 为什么有两套向量 $w$ 和 $\tilde{w}$？

$w_i$ 是词 $i$ 作为"中心词"时的向量，$\tilde{w}_j$ 是词 $j$ 作为"上下文词"时的向量。

对称共现矩阵满足 $X_{ij} = X_{ji}$，因此可以交换 $i, j$ 的角色。两套向量都需要训练，最终词向量通常取二者之和 $w_i + \tilde{w}_i$，实验表明这比只用一套效果更好（相当于对两次角色的向量取平均，降低方差）。

### GloVe 与 SVD/PPMI 的关系

Levy & Goldberg (2014) 证明，Word2Vec 的 Skip-gram 负采样（SGNS）隐式地分解如下矩阵：

$$
M_{ij} = \text{PMI}(w_i, c_j) - \log k
$$

而 GloVe 显式地最小化：

$$
\|w_i^\top \tilde{w}_j - \log X_{ij}\|^2
$$

两者都是对共现矩阵（的对数）做加权低秩分解，只是权重函数和优化方法不同。SVD/LSA 也是对共现矩阵做低秩分解，但 GloVe 通过 $f(X_{ij})$ 的加权机制更合理地处理了高频和零计数问题。

---

## 💡 关键理解

### GloVe vs Word2Vec 核心对比

| 特性 | Word2Vec（SGNS） | GloVe |
|------|-----------------|-------|
| 数据来源 | 逐窗口扫描（局部） | 预计算全局共现矩阵 |
| 训练目标 | 预测上下文词（softmax/负采样） | 拟合 log 共现计数（最小二乘） |
| 对高频词的处理 | 下采样（subsampling） | 权重函数 $f(X_{ij})$ 截断 |
| 对零共现的处理 | 负采样（从全词汇采样负例） | $f(0)=0$（直接忽略） |
| 低频词效果 | 较差（样本少） | 相对更好（权重接近0，但仍考虑全局分布） |
| 增量更新 | 支持（继续训练） | 不便（需重建共现矩阵） |
| 典型维度 | 100-300 | 50-300 |

### 两套向量的本质
可以将 $w_i$ 理解为词 $i$ 作为"主题词"的语义，$\tilde{w}_j$ 理解为词 $j$ 作为"背景词"的语义。取均值 $w_i + \tilde{w}_i$ 相当于对"我作为主角"和"我作为配角"两种视角的平均，能更全面地表达词义。

---

## 🔧 代码实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter

# ============================================================
# 1. 使用 torchtext 加载预训练 GloVe（推荐方式）
# ============================================================
print("方式1：加载预训练 GloVe 向量")
print("="*60)

# 注意：实际运行需要安装 torchtext 并且第一次会自动下载
# pip install torchtext
# 下载约 862MB（glove.6B.zip），含 50/100/200/300 维向量

USE_TORCHTEXT = False  # 设为 True 时运行实际下载

if USE_TORCHTEXT:
    try:
        from torchtext.vocab import GloVe
        
        # 加载 6B tokens 训练的 100 维 GloVe 向量
        glove = GloVe(name='6B', dim=100)
        
        # 获取词向量
        king_vec = glove['king']
        queen_vec = glove['queen']
        man_vec = glove['man']
        woman_vec = glove['woman']
        
        def cosine_sim(v1, v2):
            return torch.dot(v1, v2) / (v1.norm() * v2.norm())
        
        print(f"king 向量维度: {king_vec.shape}")
        print(f"cos(king, queen) = {cosine_sim(king_vec, queen_vec):.4f}")
        
        # 经典类比测试：king - man + woman ≈ queen
        analogy_vec = king_vec - man_vec + woman_vec
        print(f"king - man + woman ≈ ?")
        
        # 找最近邻
        all_vecs = glove.vectors  # (vocab_size, 100)
        sims = torch.mv(all_vecs, analogy_vec) / (
            all_vecs.norm(dim=1) * analogy_vec.norm()
        )
        top_k = sims.topk(5)
        for score, idx in zip(top_k.values, top_k.indices):
            print(f"  {glove.itos[idx]}: {score:.4f}")
    
    except ImportError:
        print("torchtext 未安装，请运行: pip install torchtext")
else:
    print("（跳过 torchtext 下载，演示手动实现）")
    print("若要使用预训练 GloVe，设置 USE_TORCHTEXT = True")


# ============================================================
# 2. 手动实现简化版 GloVe（从头训练）
# ============================================================
print("\n方式2：从头实现简化版 GloVe 训练")
print("="*60)

# ---------- 2.1 构建共现矩阵 ----------

def build_cooccurrence(corpus: list[list[str]], 
                        window_size: int = 2) -> tuple[dict, dict, dict]:
    """
    构建词-词共现矩阵（稀疏表示）
    
    Returns:
        word2idx: 词 -> 索引
        idx2word: 索引 -> 词
        cooc: {(i, j): count}，稀疏共现计数
    """
    # 统计词频，过滤低频词
    all_words = [w for sent in corpus for w in sent]
    word_freq = Counter(all_words)
    vocab = sorted([w for w, c in word_freq.items() if c >= 1])
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    # 统计共现（对称）
    cooc = defaultdict(float)
    
    for sentence in corpus:
        tokens = [w for w in sentence if w in word2idx]
        
        for i, word in enumerate(tokens):
            wi = word2idx[word]
            
            # 左右窗口
            for j in range(max(0, i - window_size), 
                           min(len(tokens), i + window_size + 1)):
                if i == j:
                    continue
                wj = word2idx[tokens[j]]
                
                # 距离加权（越近的词权重越大）
                distance = abs(i - j)
                weight = 1.0 / distance
                
                cooc[(wi, wj)] += weight
    
    return word2idx, idx2word, cooc


# ---------- 2.2 GloVe 模型（PyTorch） ----------

class GloVeModel(nn.Module):
    """
    GloVe 词向量模型
    
    目标函数: J = Σ f(X_ij) * (w_i @ w̃_j + b_i + b̃_j - log X_ij)²
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        
        # 两套词向量：中心词向量 + 上下文词向量
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 两套偏置
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # 初始化：均匀分布在 [-0.5/d, 0.5/d]
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.word_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.context_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.word_biases.weight)
        nn.init.zeros_(self.context_biases.weight)
    
    def forward(self, word_ids: torch.Tensor, 
                context_ids: torch.Tensor) -> torch.Tensor:
        """
        计算目标函数中的预测值: w_i @ w̃_j + b_i + b̃_j
        
        Args:
            word_ids: 中心词 id，shape=(batch,)
            context_ids: 上下文词 id，shape=(batch,)
        
        Returns:
            prediction: shape=(batch,)
        """
        w = self.word_embeddings(word_ids)           # (batch, d)
        wt = self.context_embeddings(context_ids)    # (batch, d)
        b = self.word_biases(word_ids).squeeze()     # (batch,)
        bt = self.context_biases(context_ids).squeeze()  # (batch,)
        
        # 内积 + 偏置
        dot = (w * wt).sum(dim=1)  # (batch,)
        return dot + b + bt
    
    def get_word_vectors(self) -> torch.Tensor:
        """
        获取最终词向量 = 两套向量之和（论文推荐）
        """
        return (self.word_embeddings.weight + 
                self.context_embeddings.weight).detach()


def weighting_function(x: torch.Tensor, 
                        x_max: float = 100.0, 
                        alpha: float = 0.75) -> torch.Tensor:
    """
    GloVe 权重函数 f(x)
    
    f(x) = (x/x_max)^alpha if x < x_max else 1
    """
    return torch.where(
        x < x_max,
        (x / x_max).pow(alpha),
        torch.ones_like(x)
    )


def train_glove(
    cooc: dict,
    word2idx: dict,
    embedding_dim: int = 50,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 0.05,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> GloVeModel:
    """训练 GloVe 模型"""
    
    vocab_size = len(word2idx)
    model = GloVeModel(vocab_size, embedding_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    
    # 将稀疏共现数据转为列表（便于批次训练）
    word_ids = []
    context_ids = []
    log_cooc = []
    weights = []
    
    for (wi, wj), count in cooc.items():
        word_ids.append(wi)
        context_ids.append(wj)
        log_cooc.append(np.log(count))
        weights.append(min(1.0, (count / x_max) ** alpha))
    
    word_ids = torch.LongTensor(word_ids)
    context_ids = torch.LongTensor(context_ids)
    log_cooc = torch.FloatTensor(log_cooc)
    weights_t = torch.FloatTensor(weights)
    
    N = len(word_ids)
    
    print(f"训练数据量: {N} 个共现对")
    print(f"词汇表大小: {vocab_size}，嵌入维度: {embedding_dim}")
    
    for epoch in range(epochs):
        # 随机打乱
        perm = torch.randperm(N)
        total_loss = 0.0
        
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            
            wi_batch = word_ids[idx]
            wj_batch = context_ids[idx]
            log_x_batch = log_cooc[idx]
            w_batch = weights_t[idx]
            
            # 前向传播
            pred = model(wi_batch, wj_batch)
            
            # GloVe 损失: Σ f(X_ij) * (pred - log X_ij)²
            loss = (w_batch * (pred - log_x_batch).pow(2)).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  Loss: {total_loss/N:.6f}")
    
    return model


# ---------- 2.3 训练与测试 ----------

# 构建训练语料
corpus = [
    ["I", "love", "cats", "and", "dogs"],
    ["I", "love", "dogs", "too"],
    ["cats", "are", "cute", "pets"],
    ["dogs", "are", "loyal", "pets"],
    ["cats", "and", "dogs", "are", "animals"],
    ["I", "have", "a", "cat"],
    ["I", "have", "a", "dog"],
    ["pet", "cats", "are", "great"],
    ["pet", "dogs", "are", "great"],
    ["deep", "learning", "is", "powerful"],
    ["machine", "learning", "uses", "deep", "networks"],
    ["neural", "networks", "learn", "representations"],
    ["language", "models", "use", "deep", "learning"],
    ["cats", "meow", "and", "dogs", "bark"],
    ["I", "love", "deep", "learning", "and", "NLP"],
]

word2idx, idx2word, cooc = build_cooccurrence(corpus, window_size=2)
print(f"\n词汇表大小: {len(word2idx)}")
print(f"共现对数量: {len(cooc)}")

# 训练 GloVe
print("\n开始训练 GloVe...")
model = train_glove(
    cooc=cooc,
    word2idx=word2idx,
    embedding_dim=10,
    epochs=100,
    lr=0.05,
)

# 获取词向量
word_vectors = model.get_word_vectors().numpy()  # (V, d)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def most_similar(word: str, word2idx: dict, idx2word: dict, 
                 vectors: np.ndarray, top_k: int = 5) -> list:
    """找最相似的词"""
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = vectors[idx]
    sims = []
    for i in range(len(word2idx)):
        if i == idx:
            continue
        sim = cosine_similarity(vec, vectors[i])
        sims.append((idx2word[i], sim))
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]


print("\n" + "="*60)
print("GloVe 词向量相似度测试")
print("="*60)

test_words = ["cats", "dogs", "learning", "deep"]
for w in test_words:
    if w in word2idx:
        similar = most_similar(w, word2idx, idx2word, word_vectors)
        print(f"\n'{w}' 最相似的词:")
        for sw, sim in similar:
            print(f"  {sw:>15}: {sim:.4f}")

# 验证 cats 和 dogs 的相似度 > cats 和 learning 的相似度
if all(w in word2idx for w in ["cats", "dogs", "learning"]):
    sim_cd = cosine_similarity(
        word_vectors[word2idx["cats"]], 
        word_vectors[word2idx["dogs"]]
    )
    sim_cl = cosine_similarity(
        word_vectors[word2idx["cats"]], 
        word_vectors[word2idx["learning"]]
    )
    print(f"\ncos(cats, dogs)    = {sim_cd:.4f}")
    print(f"cos(cats, learning) = {sim_cl:.4f}")
    print(f"cats 和 dogs 更相似: {sim_cd > sim_cl}")
```

---

## ⚠️ 易错点与常见误解

### 1. GloVe vs Word2Vec：全局 vs 局部
**误解**：GloVe 和 Word2Vec 只是实现细节不同，本质相同。

**正确理解**：
- **Word2Vec（SGNS/CBOW）**：局部窗口方法，通过预测任务间接学习共现统计，每对共现关系在训练中被多次处理
- **GloVe**：显式利用**全局**共现矩阵，先统计整个语料库的共现矩阵，再优化拟合对数共现值

**面试答法**：GloVe = 全局共现统计 + 加权最小二乘，Word2Vec = 局部窗口 + 预测任务 + 随机梯度下降

### 2. 为什么用 $\log X_{ij}$ 而非 $X_{ij}$
**误解**：直接拟合 $X_{ij}$ 更直接，为什么要取对数？

**正确理解**：
- $X_{ij}$ 的量纲变化极大（"the" 与 "the" 共现数万次，罕见词对只共现1次）
- 直接拟合 $X_{ij}$ 会被高频词对主导，低频词的信号被淹没
- $\log X_{ij}$ 压缩了量纲，各词对的贡献更均衡
- 同时，取对数后与 PMI 有自然联系：$\log P(i,j) = \log X_{ij} - \log X_{\text{total}}$

### 3. 权重函数 f(x) 的两个作用
**误解**：$f(x)$ 只是为了让 $f(0)=0$（处理零共现）。

**正确理解**：$f(x)$ 有两个作用：
1. **处理零计数**：$f(0)=0$，不惩罚未共现的词对（否则 $\log(0)$ 无定义）
2. **限制高频词权重**：当 $x \geq x_{\max}=100$ 时，$f(x)=1$，不允许高频词对（如"the the"）的权重超过1，防止高频词对主导训练

### 4. 两套向量的使用
**误解**：最终词向量只用 $w_i$（中心词向量），$\tilde{w}_i$ 是辅助变量。

**正确理解**：论文实验表明，取 $w_i + \tilde{w}_i$ 作为最终词向量，性能优于只用任意一套。原因是这相当于对同一词的两种视角（作为中心词 vs 作为上下文词）做了平均，降低了方差，增强了稳定性。

### 5. 预训练 GloVe 的维度选择
**误解**：维度越高越好，应该始终用 300 维。

**正确理解**：
- **50 维**：速度快，适合下游资源受限场景
- **100 维**：常用的平衡选择
- **200-300 维**：词语义推理任务上通常更好，但计算成本高
- 对于词相似度任务，100 维已经足够；对于复杂的类比任务，300 维有优势

### 6. GloVe 不能处理一词多义
**误解**：GloVe 训练后，每个词的向量能表达其所有含义。

**正确理解**：GloVe（和 Word2Vec）给每个词**唯一一个**向量，是对该词在所有上下文中含义的"平均"。"bank"（银行/河岸）只有一个向量，无法区分两种含义。

这是静态词向量的根本局限，也是 ELMo、BERT 等**上下文感知词嵌入**的核心贡献——同一个词在不同上下文中有不同的向量表示。

### 7. 预训练 GloVe 的 OOV 处理
**误解**：预训练 GloVe 涵盖了所有词，不存在 OOV 问题。

**正确理解**：预训练 GloVe（如 6B token 版本）覆盖约 40 万词，仍然存在 OOV。常见处理方式：
- 映射到随机向量
- 用字符级 n-gram 的平均向量（fastText 的方案）
- 用 `<UNK>` 向量（如果训练时包含 `<UNK>` 则直接用）

---

## 🔗 知识延伸

| 概念 | 与 GloVe 的关系 |
|------|--------------|
| **Word2Vec** | GloVe 的主要竞争方案，局部 vs 全局的对比 |
| **PMI/PPMI** | GloVe 目标 $\log X_{ij}$ ≈ PMI（加常数项）的加权最小二乘 |
| **SVD/LSA** | GloVe 是 LSA 的改进版：加权、非对称偏置、更好的频率处理 |
| **fastText** | Word2Vec/GloVe 的扩展，利用字符 n-gram 处理 OOV 和形态丰富语言 |
| **ELMo** | 用 BiLSTM 生成上下文感知词嵌入，彻底解决一词多义问题 |
| **BERT** | 用 Transformer + Masked LM 预训练，现代 NLP 的基础模型 |
| **词类比任务** | king-man+woman=queen，GloVe/Word2Vec 向量空间的线性结构体现 |

---

## 📚 参考资料

- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP. - GloVe 原论文
- Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NIPS. - 证明 Word2Vec 隐式分解 PMI 矩阵
- Mikolov, T., et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR. - Word2Vec 原论文
- GloVe 官方项目：https://nlp.stanford.edu/projects/glove/
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6.5: GloVe
- torchtext 文档：https://pytorch.org/text/stable/vocab.html
