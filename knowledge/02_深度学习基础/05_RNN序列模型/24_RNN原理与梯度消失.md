# RNN原理与梯度消失

## 📌 核心问题
> 普通神经网络无法处理序列数据（文本、语音、时间序列），因为它对每个输入独立处理，没有"记忆"能力。

## 🌱 根源与动机

语言是序列，词与词之间有时间顺序依赖：
- "我今天很**开心**" → 最后这个词的意思依赖前面所有词
- 普通网络：每个输入完全独立，无法建模这种依赖

RNN 的解法：给网络加一条"记忆线"，让每一步的计算依赖上一步的状态。

类比（Java工程师视角）：
```java
// 普通网络：无状态，每次调用独立
Output process(Input x) { return transform(x); }

// RNN：有状态，每次调用依赖上一次的状态
Output process(Input x, State h_prev) {
    State h_new = transform(x, h_prev);
    return output_layer(h_new);
}
```

## 📐 核心公式

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)   # 新记忆 = f(当前输入, 历史记忆)
y_t = W_y · h_t + b_y                          # 当前输出
```

- `x_t`：当前时刻输入（词向量）
- `h_{t-1}`：上一时刻的隐状态（记忆）
- `h_t`：新的隐状态
- `y_t`：当前时刻输出

**"两条线"= 输入线(x_t) + 记忆线(h_{t-1})**

## 💡 关键理解

RNN 在时间维度上"展开"后，本质是一个权重共享的深层网络：
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T → 输出
              ↑共享同一套W↑
```
权重 W 在所有时间步共享，这既是优点（参数少）也是缺点（导致梯度消失）。

## ⚠️ 致命问题：梯度消失

长序列反向传播时，梯度需要连乘 T 次权重矩阵：

```
梯度 ∝ W^T
```

- W < 1 → 连乘T次 → 趋近0（**梯度消失**，早期词的信息被遗忘）
- W > 1 → 连乘T次 → 爆炸（**梯度爆炸**，训练崩溃）

**结论：RNN 无法学习"长距离依赖"**，这是 LSTM 出现的直接动机。

## 🔧 PyTorch 代码

```python
import torch
import torch.nn as nn

# 定义RNN层
# input_size: 每个时间步输入的特征维度（词向量维度）
# hidden_size: 隐状态维度（记忆的容量）
rnn = nn.RNN(input_size=100, hidden_size=128, batch_first=True)

# 模拟输入：batch_size=32, 序列长度=10, 词向量维度=100
x = torch.randn(32, 10, 100)

# 前向传播
output, h_n = rnn(x)
# output: (32, 10, 128) — 每个时间步的输出
# h_n:   (1, 32, 128)  — 最后一个时间步的隐状态（最终记忆）
print(output.shape)  # torch.Size([32, 10, 128])
print(h_n.shape)     # torch.Size([1, 32, 128])
```

## 🔗 知识延伸
- RNN 的梯度消失问题 → 直接导致 **LSTM** 的诞生
- LSTM 通过"门控机制"控制信息的保留与遗忘
- 更长距离依赖 → **Attention 机制** → **Transformer**
