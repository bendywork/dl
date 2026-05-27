# 双向RNN与hidden状态理解

## 📌 核心问题
> 双向RNN的 hidden 第一维为什么会变成 `num_layers × 2`？hidden shape 和 output shape 分别是什么？

## 🌱 根源与动机

单向RNN只从左到右处理序列，每个时间步只能看到过去的上下文。双向RNN同时引入一个从右到左的方向，让每个时间步都能同时看到过去和未来的上下文，从而获得更丰富的序列表示。

这在NLP任务中尤为重要：理解一个词的含义往往需要同时参考它前后的词语。

## 📐 理论推导

### hidden shape 公式

```
num_directions = 2 if bidirectional else 1
hidden 第一维 = num_layers × num_directions
```

| 配置 | hidden shape |
|------|-------------|
| `num_layers=1, bidirectional=False` | `[1, bs, hidden_size]` |
| `num_layers=1, bidirectional=True`  | `[2, bs, hidden_size]` |
| `num_layers=3, bidirectional=False` | `[3, bs, hidden_size]` |
| `num_layers=3, bidirectional=True`  | `[6, bs, hidden_size]` |

### 正向 + 反向的处理过程

```
输入序列：[x0, x1, x2, x3, x4]

正向：x0 → x1 → x2 → x3 → x4 → 得到最终隐状态 h_forward
反向：x4 → x3 → x2 → x1 → x0 → 得到最终隐状态 h_backward

hidden = [h_forward, h_backward]  # shape: [2, bs, hidden_size]
```

- `hidden[0]`：正向跑完整个序列后的隐状态（看完整个序列从左到右的记忆）
- `hidden[1]`：反向跑完整个序列后的隐状态（看完整个序列从右到左的记忆）

### output shape

- 单向：`output → [bs, t, hidden_size]`
- 双向：`output → [bs, t, hidden_size × 2]`

双向时，每个时间步的输出是正向隐状态和反向隐状态在 `dim=-1` 上的拼接：

```
output[:, t, :] = concat(h_forward_t, h_backward_t)  # dim=-1 拼接
```

## 💡 关键理解

**类比理解：** 把双向RNN想象成两个人同时读一篇文章。
- 正向：从第一页读到最后一页，读完后的总结 = `h_forward`
- 反向：从最后一页读到第一页，读完后的总结 = `h_backward`
- 两人读的是同一批样本（bs不变），只是方向不同，各自独立产生一个隐状态

**核心认知：** 双向让 hidden 第一维乘以2，是因为多了一个方向的隐状态，而不是多了样本。样本数永远在第二维（`batch_first=False` 时），不随双向改变。

## 🔧 代码实现

```python
import torch
import torch.nn as nn

# 验证单向 RNN 的 hidden shape
rnn_uni = nn.RNN(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
x = torch.randn(3, 5, 4)  # [bs=3, seq_len=5, input_size=4]
output_uni, hidden_uni = rnn_uni(x)
print(f"单向 output shape: {output_uni.shape}")   # [3, 5, 8]
print(f"单向 hidden shape: {hidden_uni.shape}")   # [1, 3, 8]

# 验证双向 RNN 的 hidden shape
rnn_bi = nn.RNN(input_size=4, hidden_size=8, num_layers=1,
                batch_first=True, bidirectional=True)
output_bi, hidden_bi = rnn_bi(x)
print(f"双向 output shape: {output_bi.shape}")    # [3, 5, 16]  ← hidden_size × 2
print(f"双向 hidden shape: {hidden_bi.shape}")    # [2, 3, 8]   ← num_layers × 2

# 提取正向和反向的最终隐状态
h_forward  = hidden_bi[0]   # shape: [3, 8]
h_backward = hidden_bi[1]   # shape: [3, 8]
print(f"正向隐状态: {h_forward.shape}")
print(f"反向隐状态: {h_backward.shape}")

# 多层双向 RNN
rnn_multi = nn.RNN(input_size=4, hidden_size=8, num_layers=3,
                   batch_first=True, bidirectional=True)
output_multi, hidden_multi = rnn_multi(x)
print(f"3层双向 output shape: {output_multi.shape}")  # [3, 5, 16]
print(f"3层双向 hidden shape: {hidden_multi.shape}")  # [6, 3, 8]  ← 3×2=6
```

## ⚠️ 易错点与常见误解

1. **hidden 第一维的 `2` 不是 batch_size 的翻倍**
   - 错误理解：双向处理了两次样本，所以 bs 变成 2×bs
   - 正确理解：第一维是 `num_layers × num_directions`，bs 永远在第二维，双向不影响样本数

2. **output 双向时最后一维翻倍，不是序列长度翻倍**
   - 错误理解：双向 output 的 `seq_len` 会变成 `2×t`
   - 正确理解：每个时间步把正向和反向的隐状态在 `dim=-1` 拼接，所以最后一维变为 `hidden_size × 2`

3. **多层双向 hidden 的排列顺序**
   - PyTorch 的排列规则：`[layer0_forward, layer0_backward, layer1_forward, layer1_backward, ...]`
   - 取第 `l` 层正向：`hidden[l*2]`；取第 `l` 层反向：`hidden[l*2+1]`

4. **LSTM 的双向 hidden**
   - LSTM 返回 `(h_n, c_n)`，两者都遵循相同的 shape 规则：`[num_layers × num_directions, bs, hidden_size]`

## 🔗 知识延伸

- **Bidirectional LSTM（BiLSTM）**：最常用的双向变体，NLP 中用于序列标注（NER）、情感分析等任务
- **Transformer**：通过 Self-Attention 天然实现了"双向"感知，每个位置都能看到整个序列，是双向RNN的现代替代方案
- **ELMo**：使用双向 LSTM 为每个词生成上下文相关的词向量，是 BERT 之前的重要预训练模型
- **batch_first 参数**：`batch_first=True` 时 output shape 为 `[bs, t, hidden_size]`，但 hidden shape 不受影响，始终为 `[num_layers×num_directions, bs, hidden_size]`

## 📚 参考资料

- PyTorch 官方文档：[torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- PyTorch 官方文档：[torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- 论文：Schuster & Paliwal (1997) - Bidirectional Recurrent Neural Networks
