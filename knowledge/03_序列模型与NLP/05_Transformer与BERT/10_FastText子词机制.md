# FastText 子词机制（Subword）

## 📌 核心问题

> Word2Vec 把每个词当"原子"处理，训练时没出现过的词（OOV，Out-Of-Vocabulary）完全没有向量。  
> FastText 用**字符级 n-gram 子词**来表示词，从根本上解决 OOV 问题，并让形态学相关词天然共享信息。

---

## 🌱 根源与动机

### Word2Vec 的致命缺陷

Word2Vec 的 lookup table 本质是一个词 → 向量的映射字典：

```
"eat"    → [0.2, 0.5, -0.1, 0.8]
"eating" → [0.3, 0.4, -0.2, 0.7]
"eats"   → 训练集没出现过 → ??? 没有向量
```

问题：
1. **OOV**：测试时遇到没训练过的词，直接报错或用零向量代替
2. **形态学盲**：`eat` / `eating` / `eaten` 在模型眼里是三个毫不相关的词，无法共享信息
3. **稀有词向量差**：出现次数少的词，向量训练不充分

### FastText 的解法：把词拆开

2017 年 Facebook 提出 FastText，核心思路：

> **词 = 字符 n-gram 的集合，词向量 = 所有子词向量的平均**

这样即使遇到没见过的词，只要它的子词在训练集中出现过，就能拼出一个合理的向量。

---

## 📐 子词拆分原理

### Step 1：加边界符

对每个词首尾加上 `<` 和 `>`，用于标记词的边界，区分前缀/后缀：

```
eating  →  <eating>
```

`<eat` 表示词首的 eat，`ing>` 表示词尾的 ing，与词中间的 `eat`/`ing` 是不同的子词。

### Step 2：滑窗提取 n-gram

对加了边界符的字符串，用长度 n=3 到 n=6 的滑窗提取所有子串：

```
<eating>  (长度8)

n=3: <ea  eat  ati  tin  ing  ng>
n=4: <eat eati atin ting ing>
n=5: <eati eatin ating ting>
n=6: <eatin eating ating>
     + 词本身: eating
```

完整子词集合（18个）：
```
<ea, <eat, <eati, <eatin,
eat, eati, eatin, eating,
ati, atin, ating, ating>,
tin, ting, ting>,
ing, ing>,
ng>
```

### Step 3：词向量 = 子词向量的平均

```
vec("eating") = mean(
    vec("<ea") + vec("<eat") + vec("eat") +
    vec("ati") + vec("ting") + vec("ing") + ...
)
```

每个子词有独立的向量（存在一个大的子词 embedding 表里），词向量动态合成，**不单独存储**。

---

## 💡 关键理解

### OOV 问题的解决

训练集从未出现过 `eats`，但：

```
eats  →  <eats>

子词: <ea, <eat, <eats, <eats>, eat, eats, ats, ats>, ts>

与 eating 的共同子词: <ea, <eat, eat
```

`<ea`、`<eat`、`eat` 这三个子词在训练 `eating`/`eaten`/`eater` 时已经充分训练过了，所以 `eats` 的向量能被合理估算出来。

**类比**：就像你没见过"不可思议"这个词，但你认识"不可"、"思议"这些字，能猜出大概意思。

### 形态学相关词天然相似

| 词组 | 共享子词 | 效果 |
|------|---------|------|
| eat / eating / eaten / eater | `eat`, `<eat` | 向量相近，语义相关 |
| run / running / runner | `run`, `<run` | 向量相近 |
| play / player / played / playing | `play`, `<play` | 向量相近 |
| un-happy / un-fair / un-known | `<un` | 前缀语义共享 |
| walk-ing / run-ning / talk-ing | `ing>` | 后缀语义共享 |

### 对拼写错误的鲁棒性

```
receive  →  子词: rec, ece, cei, eiv, ive, ive>  ...
recieve  →  子词: rec, eci, cie, iev, eve, ive>  ...

共同子词: rec, ive> 等
```

拼写错误的词与正确词仍有部分共同子词，向量不会差太远。




---

## 🔧 代码实现

```python
import math
from collections import defaultdict

# ============================================================
# Part 1: 子词提取
# ============================================================

def get_subwords(word, min_n=3, max_n=6):
    """提取一个词的所有字符 n-gram 子词"""
    bounded = '<' + word + '>'
    subwords = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(bounded) - n + 1):
            subwords.add(bounded[i:i+n])
    subwords.add(word)  # 词本身也加入
    return sorted(subwords)

# 演示
word = "eating"
subs = get_subwords(word)
print(f"词: {word}")
print(f"子词({len(subs)}个): {subs}")

# ============================================================
# Part 2: 模拟 FastText 词向量合成（简化版，随机初始化子词向量）
# ============================================================

import random
random.seed(42)

def random_vector(dim=4):
    return [random.uniform(-1, 1) for _ in range(dim)]

def vec_mean(vectors):
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

# 子词向量表（实际训练中通过 SGD 学习）
subword_table = defaultdict(lambda: random_vector(4))

def fasttext_word_vector(word, min_n=3, max_n=6):
    """FastText 词向量 = 所有子词向量的平均"""
    subwords = get_subwords(word, min_n, max_n)
    vectors = [subword_table[sw] for sw in subwords]
    return vec_mean(vectors)

# 测试：training 集里没有 eats，但能计算出向量
for w in ["eat", "eating", "eats", "eaten"]:
    vec = fasttext_word_vector(w)
    print(f"vec({w:8s}) = {[round(v,3) for v in vec]}")

# ============================================================
# Part 3: 验证 OOV 词与已知词的相似性
# ============================================================

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    return dot / (norm_a * norm_b + 1e-8)

vec_eating = fasttext_word_vector("eating")
vec_eats   = fasttext_word_vector("eats")    # OOV
vec_run    = fasttext_word_vector("run")     # 无关词

print(f"\ncos(eating, eats) = {cosine_similarity(vec_eating, vec_eats):.4f}  (应该较高)")
print(f"cos(eating, run)  = {cosine_similarity(vec_eating, vec_run):.4f}   (应该较低)")
```



---

## ⚠️ 易错点与常见误解

1. **误解：FastText 会改变词的 shape**
   - 错误：以为子词会拼接成更长的向量
   - 正确：子词向量全部取平均，输出维度与子词维度相同

2. **误解：FastText 存了每个词的向量**
   - 错误：以为 FastText 有词向量表
   - 正确：只存子词向量表，词向量每次动态合成，不单独存储

3. **误解：子词越多词向量越准**
   - 子词多说明词本身较长，平均后反而可能稀释关键特征；关键在于子词的训练质量

4. **误解：FastText = BPE（BERT 的分词方式）**
   - FastText：固定滑窗提取 n-gram，所有子词独立存在
   - BPE：基于频率统计合并字符对，词表是动态构建的
   - 两者都是子词思想，但机制不同

5. **n 的范围很重要**
   - 默认 min_n=3, max_n=6，太小（n=1,2）噪声多，太大子词数量爆炸
   - 中文通常用 n=1~2（因为汉字本身有语义）

---

## 🔗 知识延伸

| 技术 | 与 FastText 的关系 |
|------|-------------------|
| **Word2Vec** | FastText 的前身，FastText 在其基础上加了子词机制 |
| **BPE（GPT）** | 同为子词思想，但用统计频率动态合并，词表更紧凑 |
| **WordPiece（BERT）** | 类 BPE，用似然最大化决定合并，`##` 前缀标识非词首子词 |
| **SentencePiece** | 语言无关的子词分词器，支持 BPE 和 Unigram，T5/LLaMA 使用 |
| **RoPE（位置编码）** | 与子词无关，但同样是把信息融合进向量而不改变 shape 的思想 |

演进路线：
```
Word2Vec（词级，静态向量）
    -> FastText（n-gram 子词，静态向量）
    -> ELMo（动态向量，双向 LSTM）
    -> BERT（WordPiece 子词 + Transformer，动态上下文向量）
    -> GPT / LLaMA（BPE / SentencePiece + 自回归 Transformer）
```

---

## 📚 参考资料

- Bojanowski et al. (2017). *Enriching Word Vectors with Subword Information*. TACL.
- FastText 官方文档：https://fasttext.cc
- 相关文档：[[01_Transformer注意力机制]] [[02_BERT预训练与微调]]
