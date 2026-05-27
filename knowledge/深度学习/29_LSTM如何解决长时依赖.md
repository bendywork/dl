# LSTM 如何解决 RNN 的长时依赖问题

## 📌 核心问题
> RNN 的梯度消失导致无法学习远距离依赖，LSTM 是如何从结构层面解决的？

## 🌱 根源与动机

RNN 递推 `h_t = tanh(W · h_{t-1} + ...)` 是**乘法传递**，反向传播梯度 = `(tanh' × W)^(T-1)`，连乘导致指数衰减。

LSTM 的核心思路：**把乘法传递改成加法传递**，让梯度有一条"高速公路"直达远处。

## 📐 LSTM 结构详解

### 细胞状态 C_t —— 加法通道

```
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ C̃_t
      ↑ 保留旧信息      ↑ 写入新信息
```

### 三个门控

**遗忘门 f_t**（旧信息留多少）：
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    → 输出 0~1
```

**输入门 i_t + 候选 C̃_t**（新信息写多少）：
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    → 输出 0~1
C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  → 候选新内容
```

**输出门 o_t**（暴露多少给外界）：
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

### 为什么解决梯度消失

C_t 递推对 C_{t-1} 的导数：`∂C_t / ∂C_{t-1} = f_t`

链式求导：
```
∂L/∂C_1 = ∂L/∂C_T × f_T × f_{T-1} × ... × f_2
```

当 f_t ≈ 1（遗忘门选择保留）时，梯度 ≈ 1，几乎无损传到第 1 步。

```
RNN:  梯度 = (0.4)^50  ≈ 0       ← 消失
LSTM: 梯度 = (1.0)^50  = 1       ← 畅通
```

本质：C_t = f_t ⊙ C_{t-1} + ... 中的**加法连接**，类比 ResNet 的 skip connection。

## 💡 关键理解

LSTM 信息的"高速公路"：C_{t-1} 通过加法直达 C_t，不经过 tanh 挤压，梯度可以无损流过。门控机制让网络自己决定什么时候保留（f_t≈1）、什么时候遗忘（f_t≈0），是主动选择而非被动衰减。

## 🔧 代码验证

```python
import torch

T = 50

# RNN: 乘法传递
grad_rnn = torch.tensor(1.0)
for t in range(T):
    grad_rnn = grad_rnn * 0.4

# LSTM: 加法传递
grad_lstm = torch.tensor(1.0)
for t in range(T):
    f_t = torch.tensor(1.0)
    grad_lstm = grad_lstm * f_t

print(f"RNN  传{T}步后梯度: {grad_rnn:.2e}")   # ≈ 0
print(f"LSTM 传{T}步后梯度: {grad_lstm:.2e}")   # ≈ 1
```

## ⚠️ 易错点与常见误解

1. **f_t 永远等于 1？** — 不是，f_t 是学出来的。需要遗忘时 f_t 会变小，需要长时记忆时 f_t 接近 1。门控 = 让网络自己决定保留多少。

2. **LSTM 完全不会梯度消失？** — 如果网络学到 f_t < 1，梯度还是会衰减，但这是有意义的衰减（主动遗忘），不是 RNN 那种被动的结构性衰减。

3. **C_t 和 h_t 的区别？** — C_t 是内部长期记忆，信息可以直通不经过激活函数挤压；h_t 是对外输出，必须经过 tanh(C_t) 再由 o_t 门控过滤。C_t 保留"原味"信息，h_t 是"过滤后的公开版本"。

## 🔗 知识延伸

- [[01_RNN原理与梯度消失]] — RNN 梯度消失的数学推导
- ResNet skip connection — 与 LSTM 加法连接思想同源
- GRU — LSTM 的简化版，合并遗忘门和输入门为更新门

## 📚 参考资料

- Hochreiter & Schmidhuber, "Long Short-Term Memory", 1997
- Christopher Olah, "Understanding LSTM Networks", 2015
