# Word2Vec 两种优化方法详解

## 📌 核心问题

> 原始 Word2Vec 为什么慢？两种优化分别怎么解决？

原始 Word2Vec（CBOW / Skip-gram）在输出层需要对全词表做 Softmax：

$$P(w_o | w_I) = \frac{\exp(v'^{\top}_{w_o} v_{w_I})}{\sum_{w=1}^{V} \exp(v'^{\top}_w v_{w_I})}$$

分母要遍历全词表 V（通常 10万~100万），**每次反向传播要更新所有词的向量**，计算复杂度 O(V)，极慢。

---

## 🌱 优化一：负采样（Negative Sampling, NS）

### 核心思想

把"多分类问题"变成"多个二分类问题"。

- 正样本：真实出现在上下文的词对 `(中心词, 上下文词)` → 标签 1
- 负样本：随机采样 k 个不相关的词 → 标签 0

不再计算全词表 softmax，只需对 **1 + k 个词** 做二元逻辑回归。

### 目标函数

$$\mathcal{L} = \log \sigma(v'^{\top}_{w_o} v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[\log \sigma(-v'^{\top}_{w_i} v_{w_I})\right]$$

- 第一项：正样本得分尽量大（sigmoid → 1）
- 第二项：负样本得分尽量小（sigmoid → 0）
- k 通常取 5~20（小语料取大，大语料取小）

### 负样本采样概率

不是均匀采样，而是按词频的 3/4 次方：

$$P(w_i) = \frac{f(w_i)^{3/4}}{\sum_j f(w_j)^{3/4}}$$

**为什么是 3/4？**
- 高频词（如 "the"）如果按词频采样，会被过度采到，词向量训练不均匀
- 3/4 次方压缩了高频词的优势，提升低频词的采样概率
- 这是经验值（原论文实验得出）

### 复杂度对比

| | 原始 Softmax | 负采样 |
|--|--|--|
| 每步更新词数 | V（全词表）| k+1（通常 6~21）|
| 复杂度 | O(V) | O(k) |
| 加速比 | 1x | ~V/k 倍 |

### 直觉理解

> 你不需要证明"这个词是对的"方式是"看遍所有错误选项"，而是"只和几个随机抽到的坏答案比较"。

---

## 🌲 优化二：层次 Softmax（Hierarchical Softmax, HS）

### 核心思想

用一棵 **霍夫曼树（Huffman Tree）** 把 V 分类问题转成 log(V) 次二分类。

- 词表中每个词是叶节点
- 高频词 → 树的浅层（路径短，更新快）
- 低频词 → 树的深层（路径长）
- 从根节点到叶节点的每一步都是一个二分类（左走 or 右走）

### 目标函数

对于中心词 $w_I$ 预测输出词 $w_o$，设从根到 $w_o$ 的路径长度为 $L(w_o)$，路径上第 $j$ 个节点为 $n(w_o, j)$：

$$P(w_o | w_I) = \prod_{j=1}^{L(w_o)-1} \sigma\left(\mathbb{1}[n(w_o, j+1) = \text{左子节点}] \cdot v'^{\top}_{n(w_o,j)} v_{w_I}\right)$$

每个内部节点都有一个向量 $v'$，路径上的每步做一次 sigmoid 二分类。

### 霍夫曼树构建

```
词频：A=5, B=4, C=3, D=2, E=1

构建过程（每次合并两个最小频率节点）：
1. 合并 E(1)+D(2) → ED(3)
2. 合并 ED(3)+C(3) → EDC(6)
3. 合并 B(4)+EDC(6) → BEDC(10) ← 不对，应合并最小两个
   正确：合并 B(4)+C(3) → BC(7)，... 按最小堆迭代

最终高频词 A,B 在浅层，E 在深层
```

### 复杂度对比

| | 原始 Softmax | 层次 Softmax |
|--|--|--|
| 每步操作数 | V | log₂(V) |
| 词表 100万 | 100万次 | ~20次 |
| 参数数量 | V 个输出向量 | V-1 个内部节点向量 |

### 直觉理解

> 猜一个词，不是"和所有词比较"，而是"一路做是/否判断，逐步缩小范围"——像二分查找。

---

## ⚖️ 两种方法对比

| 维度 | 负采样（NS） | 层次Softmax（HS） |
|------|------------|-----------------|
| 原理 | 二分类替代多分类 | 哈夫曼树路径二分类 |
| 速度 | O(k)，k 很小 | O(log V) |
| 小语料 | 效果好 | 效果好 |
| 大语料 | **更常用** | 一般 |
| 低频词 | 依赖采样，可能不稳定 | 路径长但有保障 |
| 实现复杂度 | 简单 | 较复杂（需建树） |
| gensim 参数 | `negative=5` | `hs=1` |

**实际工程：负采样更常用**，gensim 默认也是负采样。

---

## 🔧 代码实现对比

```python
from gensim.models import Word2Vec

sentences = [["我", "爱", "北京", "天安门"], ["天安门", "上", "太阳", "升"]]

# 使用负采样（默认）
model_ns = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,          # Skip-gram
    negative=5,    # 负采样，5个负样本
    hs=0,          # 关闭层次softmax
    epochs=10
)

# 使用层次Softmax
model_hs = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=0,    # 关闭负采样
    hs=1,          # 开启层次softmax
    epochs=10
)

print(model_ns.wv['北京'])
print(model_hs.wv['北京'])
```

---

## ⚠️ 易错点与常见误解

1. **负采样不是随机均匀采样**：是按词频 3/4 次方采样，不是 uniform
2. **层次 Softmax 的内部节点向量不是词向量**：叶节点才对应词，内部节点向量只是分类器参数
3. **两者不能同时开启**：`hs=1` 且 `negative>0` 时 gensim 以 hs 为主
4. **负采样的 k 不是越大越好**：k 大 → 更稳定但更慢；小语料 k=15~20，大语料 k=5
5. **层次 Softmax 对高频词更有利**：因为高频词路径短，梯度传播路径也短

---

## 🔗 知识延伸

- [[Word2Vec基本理解]] → 为什么需要词向量
- [[CBOW详解]] / [[Skip-gram详解]] → 两种模型结构
- 下一步：FastText（在 Word2Vec 基础上加子词信息）
- 下一步：GloVe（全局共现矩阵 + 局部窗口的折中）

## 📚 参考资料

- Mikolov et al., 2013, "Distributed Representations of Words and Phrases and their Compositionality"
- 原论文提出负采样和层次Softmax，并给出3/4次方的实验依据
