# N-gram 语言模型

## 📌 核心问题
> 如何让计算机判断一个句子是否"自然"？如何让模型预测下一个词？语言模型的核心就是对句子的概率进行建模：$P(\text{"深度学习很有趣"})$ 应该远大于 $P(\text{"学习深度有趣很"})$。

---

## 🌱 根源与动机

### 历史背景
语言模型（Language Model，LM）的概念可追溯至 Claude Shannon 1948 年的信息论论文。Shannon 把语言建模为一个随机过程，用概率来量化语言的统计规律。

N-gram 模型是最早被工业界大规模使用的语言模型，在 2003 年 Bengio 等人提出神经网络语言模型之前，N-gram 几乎垄断了整个 NLP 领域，支撑着语音识别、机器翻译、输入法等核心应用。

### 核心直觉
计算 $P(w_1 w_2 \ldots w_n)$ 很困难，因为语言空间是指数大的。但我们可以利用一个简化假设：

**马尔可夫假设（Markov Assumption）**：当前词只依赖于它之前的 $N-1$ 个词，与更早的词无关。

这将一个指数级的联合概率问题，简化为可以从语料库中直接统计的条件概率问题。

---

## 📐 理论推导

### 链式法则（Chain Rule）

对任意序列，联合概率可以精确分解：

$$
P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, w_2, \ldots, w_{i-1})
$$

**问题**：$P(w_i \mid w_1, \ldots, w_{i-1})$ 需要统计所有历史前缀的频率，数据极度稀疏。

### 马尔可夫假设

**Unigram（1-gram）**：每个词独立，不依赖历史：
$$
P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i)
$$

**Bigram（2-gram）**：每个词只依赖前一个词：
$$
P(w_1, \ldots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-1})
$$

**Trigram（3-gram）**：每个词依赖前两个词：
$$
P(w_1, \ldots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-2}, w_{i-1})
$$

**通用 N-gram**：
$$
P(w_1, \ldots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-N+1}, \ldots, w_{i-1})
$$

### MLE 估计（最大似然估计）

从语料库统计频率来估计条件概率：

$$
P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}
$$

对于 N-gram：
$$
P_{\text{MLE}}(w_i \mid w_{i-N+1}, \ldots, w_{i-1}) = \frac{\text{count}(w_{i-N+1}, \ldots, w_i)}{\text{count}(w_{i-N+1}, \ldots, w_{i-1})}
$$

### 数据稀疏问题

语料库中大量 N-gram 从未出现（zero count），导致 MLE 估计为 0。若某句子中有一个词组的 MLE = 0，整个句子的概率就为 0（连乘中的零）。

### 拉普拉斯平滑（Add-1 Smoothing）

$$
P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + 1}{\text{count}(w_{i-1}) + V}
$$

其中 $V$ 是词汇表大小。给所有 N-gram 加一个伪计数 1，使得所有 N-gram 的概率都大于 0。

**Add-k 平滑**（更灵活）：
$$
P_{\text{Add-k}}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i) + k}{\text{count}(w_{i-1}) + kV}
$$

### 困惑度（Perplexity）

评估语言模型质量的核心指标，本质是测试集上的平均每词逆概率：

$$
\text{PPL}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_{i-1})}}
$$

**对数形式（数值稳定）**：
$$
\log \text{PPL}(W) = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{i-1})
$$

**困惑度的直觉**：PPL 是模型在每个位置平均"困惑"于多少个候选词。
- 完美模型（每次都预测正确）：PPL = 1
- 随机猜测（词汇量 V）：PPL = V
- 越低越好（好的语言模型 PPL 通常在 50-200 之间）

**困惑度与信息熵的关系**：
$$
\text{PPL} = 2^H \quad \text{其中 } H = -\frac{1}{N}\sum_{i=1}^N \log_2 P(w_i \mid \text{context})
$$

---

## 💡 关键理解

### N 的选择权衡
| N | 优点 | 缺点 |
|---|------|------|
| 1 (Unigram) | 简单，无稀疏问题 | 完全忽略上下文，语言能力极弱 |
| 2 (Bigram) | 捕捉局部依赖，稀疏问题可控 | 只看一个前驱词，上下文信息有限 |
| 3 (Trigram) | 常用实用选择 | 稀疏性显著增加 |
| 5+ | 更强的上下文建模 | 极度稀疏，需要海量数据 |

### 为什么在对数空间计算
句子概率是多个小概率的连乘，当句子稍长（>20词），浮点数会下溢为 0。对数空间将乘法变为加法，完美解决数值稳定性问题：

$$
\log P(w_1, \ldots, w_n) = \sum_{i=1}^n \log P(w_i \mid w_{i-1})
$$

### 马尔可夫假设的局限
```
"The computer which I bought last week ___ broken."
```
这里填 "is" 还是 "are" 取决于主语 "computer"（7个词之前），而非紧邻的上下文。N-gram 无法处理这种长距离依赖，这是 RNN/LSTM/Transformer 的核心优势。

---

## 🔧 代码实现

```python
import numpy as np
import math
from collections import defaultdict, Counter
from typing import Optional

# ============================================================
# 1. Bigram 语言模型（完整实现）
# ============================================================

class BigramLanguageModel:
    """
    Bigram 语言模型，支持拉普拉斯平滑和困惑度计算
    """
    
    START_TOKEN = "<s>"   # 句子起始标记
    END_TOKEN = "</s>"    # 句子结束标记
    UNK_TOKEN = "<UNK>"  # 未知词标记
    
    def __init__(self, smoothing: str = 'laplace', k: float = 1.0):
        """
        Args:
            smoothing: 'none' | 'laplace' | 'add_k'
            k: add_k 平滑的 k 值
        """
        self.smoothing = smoothing
        self.k = k
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
    
    def _add_boundary_tokens(self, sentence: list[str]) -> list[str]:
        """添加句子边界标记"""
        return [self.START_TOKEN] + sentence + [self.END_TOKEN]
    
    def fit(self, corpus: list[list[str]]):
        """
        训练语言模型
        
        Args:
            corpus: 句子列表，每个句子是词的列表
        """
        for sentence in corpus:
            tokens = self._add_boundary_tokens(sentence)
            self.vocab.update(tokens)
            
            for i in range(len(tokens)):
                self.unigram_counts[tokens[i]] += 1
                if i > 0:
                    self.bigram_counts[tokens[i-1]][tokens[i]] += 1
        
        self.vocab.add(self.UNK_TOKEN)
        self.V = len(self.vocab)
        return self
    
    def _replace_unk(self, tokens: list[str]) -> list[str]:
        """将词汇表外的词替换为 <UNK>"""
        return [t if t in self.vocab else self.UNK_TOKEN for t in tokens]
    
    def log_prob_bigram(self, w_prev: str, w_curr: str) -> float:
        """
        计算条件概率 log P(w_curr | w_prev)
        """
        count_bigram = self.bigram_counts[w_prev][w_curr]
        count_unigram = self.unigram_counts[w_prev]
        
        if self.smoothing == 'none':
            if count_bigram == 0:
                return float('-inf')
            return math.log(count_bigram / count_unigram)
        
        elif self.smoothing in ('laplace', 'add_k'):
            k = 1.0 if self.smoothing == 'laplace' else self.k
            return math.log(
                (count_bigram + k) / (count_unigram + k * self.V)
            )
        
        raise ValueError(f"未知平滑方法: {self.smoothing}")
    
    def sentence_log_prob(self, sentence: list[str]) -> float:
        """计算句子的对数概率"""
        tokens = self._add_boundary_tokens(self._replace_unk(sentence))
        log_prob = 0.0
        for i in range(1, len(tokens)):
            lp = self.log_prob_bigram(tokens[i-1], tokens[i])
            if lp == float('-inf'):
                return float('-inf')
            log_prob += lp
        return log_prob
    
    def perplexity(self, test_corpus: list[list[str]]) -> float:
        """
        计算测试集上的困惑度（Perplexity）
        
        PPL = exp(-1/N * sum(log P(wi|wi-1)))
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in test_corpus:
            tokens = self._add_boundary_tokens(self._replace_unk(sentence))
            # 注意：统计词数时通常不算 <s>，但要算 </s>
            total_words += len(tokens) - 1
            
            for i in range(1, len(tokens)):
                lp = self.log_prob_bigram(tokens[i-1], tokens[i])
                if lp == float('-inf'):
                    return float('inf')
                total_log_prob += lp
        
        return math.exp(-total_log_prob / total_words)
    
    def generate(self, max_len: int = 20, seed: Optional[int] = None) -> list[str]:
        """
        基于语言模型生成文本（采样）
        """
        if seed is not None:
            np.random.seed(seed)
        
        tokens = [self.START_TOKEN]
        
        while len(tokens) < max_len + 1:
            prev = tokens[-1]
            
            # 获取所有可能的下一个词及其概率
            candidates = list(self.vocab - {self.START_TOKEN})
            log_probs = np.array([self.log_prob_bigram(prev, w) for w in candidates])
            
            # 转为概率并归一化
            log_probs = np.clip(log_probs, -100, 0)  # 避免 -inf
            probs = np.exp(log_probs - log_probs.max())
            probs /= probs.sum()
            
            next_word = np.random.choice(candidates, p=probs)
            tokens.append(next_word)
            
            if next_word == self.END_TOKEN:
                break
        
        return tokens[1:-1]  # 去掉 <s> 和 </s>


# ============================================================
# 2. 训练与测试
# ============================================================

# 训练语料
train_corpus = [
    ["deep", "learning", "is", "powerful"],
    ["machine", "learning", "is", "useful"],
    ["deep", "neural", "networks", "learn", "features"],
    ["language", "models", "predict", "next", "words"],
    ["natural", "language", "processing", "is", "fascinating"],
    ["deep", "learning", "models", "need", "large", "data"],
    ["machine", "learning", "algorithms", "are", "powerful"],
    ["neural", "networks", "deep", "learning", "models"],
]

# 初始化模型（用 Laplace 平滑）
lm = BigramLanguageModel(smoothing='laplace', k=1.0)
lm.fit(train_corpus)

print("=" * 60)
print("Bigram 语言模型训练完成")
print(f"词汇表大小: {lm.V}")
print("=" * 60)

# 测试句子概率
test_sentences = [
    ["deep", "learning", "is", "powerful"],    # 训练集句子
    ["machine", "learning", "is", "fascinating"],  # 部分出现
    ["deep", "learning", "models", "are", "useful"],  # 组合
    ["cats", "fly", "into", "space"],           # 完全陌生
]

print("\n句子概率测试:")
for sent in test_sentences:
    log_p = lm.sentence_log_prob(sent)
    print(f"  '{' '.join(sent)}'")
    print(f"    log P = {log_p:.4f}  |  P = {math.exp(log_p):.6f}")


# ============================================================
# 3. 困惑度对比：有无平滑
# ============================================================
print("\n" + "="*60)
print("平滑方法对比：困惑度")
print("="*60)

test_corpus = [
    ["deep", "learning", "is", "useful"],
    ["neural", "networks", "are", "powerful"],
    ["natural", "language", "models", "predict"],
]

for smoothing, k in [('none', 1), ('laplace', 1), ('add_k', 0.1), ('add_k', 0.01)]:
    lm_temp = BigramLanguageModel(smoothing=smoothing, k=k)
    lm_temp.fit(train_corpus)
    ppl = lm_temp.perplexity(test_corpus)
    name = smoothing if smoothing != 'add_k' else f'add_k(k={k})'
    print(f"  {name:20s}: PPL = {ppl:.2f}")


# ============================================================
# 4. 文本生成
# ============================================================
print("\n" + "="*60)
print("文本生成（采样）")
print("="*60)

lm_gen = BigramLanguageModel(smoothing='add_k', k=0.5)
lm_gen.fit(train_corpus)

print("\n生成的句子:")
for i in range(5):
    generated = lm_gen.generate(max_len=10, seed=i*42)
    print(f"  {i+1}: {' '.join(generated)}")


# ============================================================
# 5. N-gram 频率统计（通用 N-gram 实现）
# ============================================================
print("\n" + "="*60)
print("N-gram 频率统计")
print("="*60)

def count_ngrams(corpus: list[list[str]], n: int) -> Counter:
    """统计语料库中所有 N-gram 的频率"""
    ngram_counts = Counter()
    for sentence in corpus:
        tokens = ["<s>"] * (n-1) + sentence + ["</s>"]
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_counts[ngram] += 1
    return ngram_counts

# 统计 bigram 和 trigram
bigram_counts = count_ngrams(train_corpus, 2)
trigram_counts = count_ngrams(train_corpus, 3)

print("\nTop 10 Bigrams:")
for ngram, count in bigram_counts.most_common(10):
    print(f"  {ngram}: {count}")

print("\nTop 5 Trigrams:")
for ngram, count in trigram_counts.most_common(5):
    print(f"  {ngram}: {count}")


# ============================================================
# 6. 困惑度的直觉验证
# ============================================================
print("\n" + "="*60)
print("困惑度直觉验证")
print("="*60)

# 极端情况1：完全随机语言模型（均匀分布）
V = lm.V
ppl_random = V  # 均匀分布下 PPL = V
print(f"\n随机猜测（词汇量={V}）的 PPL = {V}")
print(f"训练集困惑度: {lm.perplexity(train_corpus):.2f}")
print(f"测试集困惑度: {lm.perplexity(test_corpus):.2f}")
print("（训练集 PPL 应 <= 测试集 PPL）")
```

---

## ⚠️ 易错点与常见误解

### 1. 数据稀疏问题的严重性
**误解**：只要语料库够大，稀疏问题就不是问题。

**正确理解**：即使拥有万亿词级别的语料库，Trigram/4-gram 的稀疏问题依然严重。英语词汇量 ~50万，4-gram 的可能空间是 $500000^4 = 6.25 \times 10^{22}$，任何语料库都无法覆盖。平滑是不可避免的。

### 2. 为什么需要平滑（Smoothing）
**误解**：遇到 0 概率就直接忽略这些词组。

**正确理解**：如果某个词组的概率为 0，整个句子的概率就是 0（连乘）。这导致语言模型对训练集外的任何句子都输出 0 概率，完全无法泛化。平滑的本质是从高概率的 N-gram 处"借走"一点概率质量，分给未见过的 N-gram。

### 3. 困惑度 = 每步预测的平均分支数
**误解**：困惑度越低代表模型越复杂。

**正确理解**：困惑度可以理解为模型在每个预测位置平均"不确定于几个选项"。PPL=100 意味着模型平均在 100 个候选词中无法判断哪个更可能。PPL=1 是理想状态（每次都预测正确）。越低越好，与模型复杂度无直接关系。

### 4. 边界标记 `<s>` 和 `</s>` 的重要性
**误解**：可以不添加句子边界标记。

**正确理解**：没有 `<s>` 标记，模型无法对句子开头词建模（第一个词没有前驱词）；没有 `</s>` 标记，模型不知道何时停止生成。边界标记是 N-gram 语言模型的必要组成部分。

### 5. Laplace 平滑的过度平滑问题
**误解**：Laplace 平滑是最佳选择，k=1 是最好的。

**正确理解**：Add-1 平滑往往过度平滑——它把太多概率质量分给了未见过的 N-gram，同时把常见 N-gram 的概率压缩得过多。实践中，更好的平滑方法有：
- **Add-k（k < 1）**：比 k=1 更保守
- **Good-Turing 平滑**：基于频率的频率估计
- **Kneser-Ney 平滑**：目前 N-gram 最优平滑方法，考虑词在不同上下文中出现的多样性

### 6. 困惑度在对数空间计算
**误解**：直接连乘概率计算困惑度。

**正确理解**：句子长度 > 20 时，连乘多个 < 1 的概率会导致浮点数下溢为 0。必须在对数空间计算：$\log \text{PPL} = -\frac{1}{N} \sum \log P(w_i \mid \text{context})$，最后取 $\exp$ 恢复。

---

## 🔗 知识延伸

| 概念 | 与 N-gram 的关系 |
|------|---------------|
| **Kneser-Ney 平滑** | 目前最优的 N-gram 平滑方法，Backoff 机制 |
| **NNLM（Bengio 2003）** | 用神经网络替代 N-gram，实现词嵌入 + 上下文建模 |
| **RNN 语言模型** | 理论上可以处理任意长度的历史，克服马尔可夫假设 |
| **LSTM** | 解决 RNN 的梯度消失，能捕捉更长范围依赖 |
| **GPT** | Transformer 架构的自回归语言模型，现代 N-gram 的终极替代 |
| **困惑度** | 所有语言模型的核心评估指标，从 N-gram 延续至 GPT |

---

## 📚 参考资料

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Chapter 3: N-gram Language Models
- Chen & Goodman (1999). *An empirical study of smoothing techniques for language modeling*. Computer Speech & Language.
- Bengio, Y., et al. (2003). *A Neural Probabilistic Language Model*. JMLR. - N-gram 的神经网络替代方案
- Kneser, R., & Ney, H. (1995). *Improved backing-off for M-gram language modeling*. ICASSP.
