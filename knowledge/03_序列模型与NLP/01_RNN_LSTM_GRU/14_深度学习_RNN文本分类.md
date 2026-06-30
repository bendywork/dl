# 深度学习-RNN 文本分类总结

## 一、核心思想

用 RNN（循环神经网络）处理文本序列，每个词依次输入，RNN 逐步累积语义信息，最后用累积的状态做分类。

## 二、任务流程

```
文本数据 → 分词 → 构建词汇表(token2id) → 文本转id序列 → padding
    → Embedding → RNN/LSTM → 取最后时刻状态 → Linear → softmax → 类别
```

## 三、关键步骤

### 1. 文本预处理
- jieba 分词 → token 列表
- 构建词表：`{token: id}`，留 0 给 padding
- 序列截断/填充到统一长度

### 2. Embedding 层
```
[id_1, id_2, ..., id_n] → Embedding → [vec_1, vec_2, ..., vec_n]
```
查表操作，把离散 id 变成连续稠密向量。可随机初始化从头训，也可加载预训练词向量（word2vec）。

### 3. RNN 层
```
vec_1 → RNN → h_1
vec_2 + h_1 → RNN → h_2
...
vec_n + h_{n-1} → RNN → h_n  ← 取最后状态做分类
```
h_n 包含了整句话压缩后的语义信息。

### 4. 分类层
```
h_n → Linear(hidden_size, num_classes) → softmax → 预测类别
```

## 四、LSTM 相对 RNN 的改进

| RNN | LSTM |
|-----|------|
| 直接叠加，远距离梯度消失 | 三个门 + 细胞状态 C，长序列可训 |
| 普通 RNN → 序列 > 20 就忘 | LSTM → 序列可达 100+ |

## 五、训练注意点

- **变长序列**用 `pad_sequence` + `pack_padded_sequence`，避免 pad 位参与计算
- **Loss** 用 `CrossEntropyLoss`，ignore_index=0 忽略 pad
- **双向 LSTM** 提高效果：正向+反向各跑一遍，拼接两个方向的最后状态
