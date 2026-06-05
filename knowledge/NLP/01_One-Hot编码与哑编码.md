# One-Hot 编码与哑编码

## 📌 核心问题
> 如何将离散的类别型变量（文字、词汇、标签）转换成神经网络/机器学习模型能够处理的数值向量？

---

## 🌱 根源与动机

### 历史背景
在机器学习早期，几乎所有算法的输入都要求是数值型。对于离散的类别变量（如词汇表中的单词、城市名称、颜色等），工程师们面临一个根本问题：

- 直接编码为整数（cat=0, dog=1, bird=2）会引入**虚假的大小关系**，让模型误以为 dog > cat，bird > dog
- 不能使用原始字符串

One-Hot 编码的直觉非常简单：**给每个类别一个专属的"开关"**，在该类别对应的位置置1，其余全部置0。

### 从投票类比理解
想象有 V 个候选人，每张选票只能投一个人。One-Hot 向量就是一张选票：只有被投的那个候选人对应的位置是1，其余全是0。

---

## 📐 理论推导

### One-Hot 编码

设词汇表大小为 $V$，词汇表 $\mathcal{V} = \{w_1, w_2, \ldots, w_V\}$

词 $w_i$ 的 One-Hot 向量定义为：

$$
\text{onehot}(w_i) = \mathbf{e}_i \in \mathbb{R}^V
$$

其中 $\mathbf{e}_i$ 是第 $i$ 个标准基向量：

$$
(\mathbf{e}_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}
$$

### 余弦相似度的致命缺陷

任意两个不同词的 One-Hot 向量余弦相似度恒为 0：

$$
\cos(\mathbf{e}_i, \mathbf{e}_j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} = \frac{0}{1 \cdot 1} = 0 \quad (i \neq j)
$$

**这意味着"猫"和"狗"的相似度 = "猫"和"飞机"的相似度 = 0**，完全无法捕捉语义关系。这是后来发展词嵌入（Word2Vec、GloVe）的根本动机。

### 哑编码（Dummy Encoding）

设某变量有 $k$ 个类别，One-Hot 编码产生 $k$ 列，哑编码只产生 $k-1$ 列（删去一个参考类别）。

| 颜色  | One-Hot: 红 | One-Hot: 绿 | One-Hot: 蓝 | 哑编码: 绿 | 哑编码: 蓝 |
|-------|------------|------------|------------|-----------|-----------|
| 红    | 1          | 0          | 0          | 0         | 0         |
| 绿    | 0          | 1          | 0          | 1         | 0         |
| 蓝    | 0          | 0          | 1          | 0         | 1         |

当哑编码两列都为 0 时，隐含表示"红色"（参考类别）。

### 多重共线性（Dummy Variable Trap）

在线性回归中，若使用 One-Hot 的全 $k$ 列，存在完美多重共线性：

$$
x_{\text{红}} + x_{\text{绿}} + x_{\text{蓝}} = 1 \quad \text{（对所有样本成立）}
$$

这导致设计矩阵 $\mathbf{X}$ 不满秩，$(\mathbf{X}^T\mathbf{X})$ 不可逆，最小二乘解无唯一解。

删去一列后，消除这种线性依赖，矩阵可逆，参数有唯一解。

### 稀疏性与维度

词汇量 $V=50000$ 时，每个词的向量有 $49999$ 个 0，只有 1 个 1。
- 存储效率极低（但可用稀疏矩阵格式 CSR/CSC 优化）
- 向量内积运算退化为查表操作（这也是 Embedding Layer 的基础）

---

## 💡 关键理解

### 类比：身份证号
One-Hot 就像给每个词颁发一张身份证，身份证号（位置）唯一标识这个词，但两张身份证之间没有任何"距离"或"相似"关系。

### One-Hot vs 哑编码使用场景
- **神经网络/NLP**：用 One-Hot（后接 Embedding Layer）
- **线性回归/统计模型**：用哑编码（避免多重共线性）
- **树模型（XGBoost/RandomForest）**：两者均可，树不受多重共线性影响

### One-Hot 是 Embedding 的前身
神经网络中，One-Hot 向量乘以权重矩阵 $\mathbf{W} \in \mathbb{R}^{V \times d}$，等价于查表取第 $i$ 行：

$$
\text{onehot}(w_i)^T \mathbf{W} = \mathbf{W}[i, :] \in \mathbb{R}^d
$$

这就是 `nn.Embedding` 层的本质——One-Hot 乘以嵌入矩阵的高效实现。

---

## 🔧 代码实现

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# ============================================================
# 1. 手动实现 One-Hot 编码
# ============================================================
def manual_onehot(tokens: list) -> tuple[dict, np.ndarray]:
    """
    手动实现 One-Hot 编码
    
    Args:
        tokens: 词汇列表，如 ['cat', 'dog', 'bird', 'cat']
    
    Returns:
        vocab: 词 -> 索引的映射字典
        matrix: One-Hot 矩阵，shape=(len(tokens), vocab_size)
    """
    # 1. 构建词汇表（去重 + 排序保证一致性）
    vocab = {word: idx for idx, word in enumerate(sorted(set(tokens)))}
    vocab_size = len(vocab)
    
    # 2. 构建 One-Hot 矩阵
    matrix = np.zeros((len(tokens), vocab_size), dtype=np.int32)
    for i, token in enumerate(tokens):
        matrix[i, vocab[token]] = 1
    
    return vocab, matrix


tokens = ['cat', 'dog', 'bird', 'cat', 'fish']
vocab, onehot_matrix = manual_onehot(tokens)

print("词汇表:", vocab)
print("One-Hot 矩阵:")
print(onehot_matrix)
print(f"\n矩阵形状: {onehot_matrix.shape}  (5个词, {len(vocab)}维)")

# 验证：任意两个不同词的余弦相似度 = 0
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

cat_vec = onehot_matrix[0]   # 'cat'
dog_vec = onehot_matrix[1]   # 'dog'
print(f"\ncos(cat, dog) = {cosine_sim(cat_vec, dog_vec):.4f}")  # 0.0
print(f"cos(cat, cat) = {cosine_sim(cat_vec, cat_vec):.4f}")  # 1.0


# ============================================================
# 2. 手动实现哑编码（删去第一列作为参考类别）
# ============================================================
def manual_dummy_encoding(tokens: list) -> tuple[dict, np.ndarray]:
    """
    哑编码 = One-Hot 去掉第一列（参考类别）
    """
    vocab, onehot = manual_onehot(tokens)
    dummy = onehot[:, 1:]  # 删去第0列（第一个类别作为参考）
    return vocab, dummy


_, dummy_matrix = manual_dummy_encoding(tokens)
print("\n哑编码矩阵（删去'bird'列作为参考）:")
print(dummy_matrix)
print(f"哑编码维度: {dummy_matrix.shape}  (少一列)")


# ============================================================
# 3. sklearn 实现
# ============================================================
print("\n" + "="*50)
print("sklearn 实现")
print("="*50)

# LabelEncoder: 将类别转为整数
le = LabelEncoder()
integer_encoded = le.fit_transform(tokens)
print(f"\nLabelEncoder 结果: {integer_encoded}")
print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

# OneHotEncoder: 整数 -> One-Hot
ohe = OneHotEncoder(sparse_output=False)  # sparse_output=False 返回密集矩阵
onehot_sklearn = ohe.fit_transform(integer_encoded.reshape(-1, 1))
print(f"\nOneHotEncoder 结果:\n{onehot_sklearn}")

# 哑编码: drop='first' 删去第一列
ohe_dummy = OneHotEncoder(sparse_output=False, drop='first')
dummy_sklearn = ohe_dummy.fit_transform(integer_encoded.reshape(-1, 1))
print(f"\n哑编码 (drop='first'):\n{dummy_sklearn}")


# ============================================================
# 4. OOV（Out-Of-Vocabulary）问题演示
# ============================================================
print("\n" + "="*50)
print("OOV 问题演示")
print("="*50)

train_tokens = ['cat', 'dog', 'bird']
test_tokens = ['cat', 'lion']  # 'lion' 在训练集中未见

vocab_train = {word: idx for idx, word in enumerate(sorted(set(train_tokens)))}

def encode_with_oov(tokens, vocab, oov_strategy='ignore'):
    """
    处理 OOV 的不同策略
    
    oov_strategy:
        'ignore'  : 跳过未知词
        'unk'     : 映射到特殊 <UNK> token
        'error'   : 抛出异常
    """
    vocab_size = len(vocab)
    if oov_strategy == 'unk':
        # 添加 <UNK> token
        vocab = {**vocab, '<UNK>': vocab_size}
        vocab_size += 1
    
    results = []
    for token in tokens:
        if token in vocab:
            vec = np.zeros(vocab_size, dtype=np.int32)
            vec[vocab[token]] = 1
            results.append((token, vec))
        elif oov_strategy == 'unk':
            vec = np.zeros(vocab_size, dtype=np.int32)
            vec[vocab['<UNK>']] = 1
            results.append((token, vec))
        elif oov_strategy == 'ignore':
            print(f"  警告: '{token}' 未在词汇表中，已跳过")
        elif oov_strategy == 'error':
            raise KeyError(f"'{token}' 不在词汇表中")
    
    return results

print("\n策略1: ignore")
encode_with_oov(test_tokens, vocab_train, 'ignore')

print("\n策略2: <UNK> token")
results = encode_with_oov(test_tokens, vocab_train, 'unk')
for token, vec in results:
    print(f"  '{token}': {vec}")


# ============================================================
# 5. 模拟 Embedding Layer 本质
# ============================================================
print("\n" + "="*50)
print("Embedding Layer 本质：One-Hot × 权重矩阵 = 查表")
print("="*50)

np.random.seed(42)
V, d = 4, 3  # 词汇量4，嵌入维度3
W = np.random.randn(V, d)  # 嵌入矩阵

# 方法1：One-Hot × W（矩阵乘法）
onehot_cat = np.array([1, 0, 0, 0])  # 'cat' 的 One-Hot（索引0）
embed_via_multiply = onehot_cat @ W
print(f"\n方法1 (One-Hot × W): {embed_via_multiply}")

# 方法2：直接查表（等价但高效）
embed_via_lookup = W[0]  # 取第0行
print(f"方法2 (W[0] 查表):   {embed_via_lookup}")
print(f"两种方法结果相同: {np.allclose(embed_via_multiply, embed_via_lookup)}")
```

---

## ⚠️ 易错点与常见误解

### 1. Dummy Variable Trap（哑变量陷阱）
**误解**：线性模型中可以直接用 One-Hot 的全 k 列。

**正确理解**：线性回归中，全 k 列的 One-Hot 存在完美多重共线性（所有列之和恒为1），导致设计矩阵奇异，参数无法唯一估计。必须删去一列（哑编码）。

**注意**：树模型、神经网络不受此限制，可以用完整的 One-Hot。

### 2. 哑编码"少一列"的语义
**误解**：哑编码丢失了信息（缺少参考类别的信息）。

**正确理解**：参考类别的信息被"编码"在截距项中。当哑编码所有列均为0时，模型截距就代表参考类别的基准效应，信息没有丢失，只是换了位置。

### 3. One-Hot 之间语义完全正交
**误解**：One-Hot 可以捕捉词汇之间的相似性。

**正确理解**：任意两个不同 One-Hot 向量的余弦相似度恒为0，点积恒为0。One-Hot 本质上是"独热"，词与词之间完全独立，没有任何语义关联。这是 One-Hot 的根本局限，也是发展词嵌入的核心动机。

### 4. OOV（Out-Of-Vocabulary）问题
**误解**：测试时遇到未见过的词可以随机初始化 One-Hot。

**正确理解**：One-Hot 的维度由训练时的词汇表决定。新词没有对应的维度，正确处理方式是：
- 映射到 `<UNK>` token（最常用）
- 使用字符级编码（不存在 OOV）
- 使用 BPE/SentencePiece 等子词分词（WordPiece、BPE）

### 5. 高维稀疏的存储问题
**误解**：One-Hot 向量必须用密集数组存储。

**正确理解**：One-Hot 可以用稀疏矩阵（`scipy.sparse.csr_matrix`）高效存储，只记录非零元素的位置。sklearn 的 `OneHotEncoder` 默认返回稀疏矩阵正是出于此原因。

---

## 🔗 知识延伸

| 概念 | 与 One-Hot 的关系 |
|------|-----------------|
| **词嵌入 (Word2Vec/GloVe)** | 解决 One-Hot 无法捕捉语义相似性的根本局限 |
| **Embedding Layer** | `nn.Embedding` 本质是 One-Hot × 权重矩阵的高效查表实现 |
| **词袋模型 (BoW)** | 对文档中所有词的 One-Hot 向量求和 |
| **TF-IDF** | 在 BoW 基础上加权，缓解高频无意义词问题 |
| **BPE/WordPiece** | 子词分词方案，从根本上解决 OOV 问题 |
| **Label Smoothing** | 训练时将 One-Hot 的"1"软化为 (1-ε)，提升泛化能力 |

---

## 📚 参考资料

- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 6
- scikit-learn 文档：`sklearn.preprocessing.OneHotEncoder`
- Bengio et al. (2003), *A Neural Probabilistic Language Model* - 最早用连续向量替代 One-Hot
- Harris (1954), *Distributional Structure* - 分布假说，词语义由其上下文决定
