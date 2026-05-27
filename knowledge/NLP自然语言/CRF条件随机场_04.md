# CRF 条件随机场

## 📌 核心问题
> 序列标注时，为什么不用 softmax 而要用 CRF？CRF 是怎么保证标签序列合法的？

## 🌱 根源与动机

softmax 对每个位置**独立**分类，会产生非法序列如 `I-PER` 接 `B-LOC`。

**CRF** 对整个序列的标签路径打分，选择**全局最优的合法路径**：
- 引入转移矩阵约束：某些标签跳转是非法的
- Viterbi 算法高效求全局最优路径

## 📐 理论推导

### 线性链 CRF 评分

对于输入序列 x 和标签序列 y：
```
score(x, y) = Σₜ [emission(yₜ, xₜ) + transition(yₜ₋₁, yₜ)]
```

- **emission score**：神经网络（BERT/BiLSTM）输出的 logits
- **transition score**：可学习的标签转移矩阵 T[i][j]

### 训练：最大化条件对数似然

```
P(y|x) = exp(score(x,y)) / Σ_{y'} exp(score(x,y'))

分母（配分函数）用前向算法高效计算
loss = -log P(y|x)
```

### 推理：Viterbi 算法

动态规划找全局最优路径：
```
dp[t][k] = max_{k'} (dp[t-1][k'] + T[k'][k]) + emission[t][k]
```
时间复杂度：O(T × K²)，T=序列长度，K=标签数

## 💡 关键理解

- **CRF 的核心优势**：全局归一化（vs softmax 的局部归一化），避免标签跳转错误
- 转移矩阵是**可学习的**，训练时自动学会哪些转移合理
- **BERT + CRF**：BERT 提供强语义特征，CRF 保证序列合法性，互补
- CRF 计算配分函数是 O(T×K²)，T 很长时有点慢，但 NER 序列通常 ≤ 512

## 🔧 代码实现

NER 项目中 CRF 使用：`knowledge/NLP自然语言/项目实践/命名实体识别/models/bert_ner.py`

```python
from torchcrf import CRF

crf = CRF(num_tags, batch_first=True)

# 训练：计算 loss
loss = -crf(emissions, tags, mask=mask)

# 推理：Viterbi 解码
predictions = crf.decode(emissions, mask=mask)
```

## ⚠️ 易错点与常见误解

1. **CRF 需要 mask**：padding 位置不参与计算，否则影响配分函数
2. **emission 是 logits，不是概率**：不要先 softmax 再传给 CRF
3. **`batch_first=True` 要显式设置**，默认是 `(seq_len, batch, tags)`
4. **Viterbi 比 argmax 慢**，但质量显著更好，生产中值得付出

## 🔗 知识延伸

- [[命名实体识别]] — CRF 的主要应用场景
- [[中文分词模型]] — 分词也用 BIO + CRF

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/扩展_03_CRF条件随机场.pdf`
