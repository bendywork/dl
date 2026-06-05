# LSTM 门计算：加法写法 vs Cat 写法的等价性

## 📌 核心问题

> LSTM 中对遗忘门（以及其他三个门）有两种常见写法，它们是否等价？哪种更优？

**写法 A（加法形式）：**
```python
f_t = sigmoid(x_t @ W_xf + h_prev @ W_hf + b_f)
```

**写法 B（Cat 形式）：**
```python
combined = torch.cat([x_t, h_prev], dim=1)   # shape: (bs, input_dim + hidden_dim)
f_t = sigmoid(combined @ W_f + b_f)
```

---

## 🌱 根源与动机

LSTM 的门控机制本质是：**同时接收当前输入 x_t 和上一时刻隐状态 h_{t-1}，计算一个门控信号。**

最自然的数学表达来自神经网络的线性变换：

```
f_t = σ(W_f · [x_t, h_{t-1}] + b_f)
```

这是原始论文（Hochreiter & Schmidhuber 1997）的写法——将两个向量拼接后乘以一个大权重矩阵。

但在工程实现中，许多代码把它拆成：

```
f_t = σ(U_f · x_t + W_f · h_{t-1} + b_f)
```

两者看似不同，实为同一运算的两种等价表示。

---

## 📐 理论推导

### 分块矩阵证明等价性

设：
- `x_t`：形状 `(bs, d_x)`
- `h_prev`：形状 `(bs, d_h)`
- `W_xf`：形状 `(d_x, d_h_out)`，作用于输入 x_t
- `W_hf`：形状 `(d_h, d_h_out)`，作用于隐状态 h_{t-1}

**加法形式展开：**

```
x_t @ W_xf + h_prev @ W_hf
= [row vectors of x_t] × W_xf  +  [row vectors of h_prev] × W_hf
```

**Cat 形式展开：**

将 W_xf 和 W_hf 竖向拼接，构造大矩阵 W_f：

```
W_f = [ W_xf ]    形状：(d_x + d_h, d_h_out)
      [ W_hf ]

combined = [x_t | h_prev]    形状：(bs, d_x + d_h)
```

则：

```
combined @ W_f
= [x_t | h_prev] @ [ W_xf ]
                   [ W_hf ]

= x_t @ W_xf + h_prev @ W_hf    ← 分块矩阵乘法展开，与加法形式完全相同
```

**结论：两种写法在数学上严格等价。**

加上偏置项 b_f 后：
```
combined @ W_f + b_f  ≡  x_t @ W_xf + h_prev @ W_hf + b_f
```

---

## 💡 关键理解

### 为什么论文和手写代码倾向于加法写法？

| 原因 | 说明 |
|------|------|
| **内存效率** | 不需要额外分配拼接后的 combined 张量，节省 `(bs × (d_x + d_h))` 的临时内存 |
| **灵活性** | 可以对 x_t 和 h_{t-1} 分别施加不同的正则化（如对 W_xf 和 W_hf 分别 dropout） |
| **可读性** | 明确区分了"输入贡献"与"记忆贡献"，物理意义清晰 |
| **对应论文符号** | 大多数教材和论文把两路贡献分开写，更贴近理论推导 |

### 为什么 PyTorch 官方 nn.LSTM 用 Cat 写法？

PyTorch 内部实现将四个门的权重矩阵合并为一个大矩阵：

```python
# PyTorch 内部等效逻辑（简化）
W_ih: (4 * hidden_size, input_size)   # 4个门对 x_t 的权重
W_hh: (4 * hidden_size, hidden_size)  # 4个门对 h_{t-1} 的权重

# 实际计算用一次大矩阵乘法完成所有门
gates = x_t @ W_ih.T + h_prev @ W_hh.T + b
# 然后 split 成 i, f, g, o 四个门
```

这样做的核心原因是 **GPU GEMM（通用矩阵乘法）优化**：

- GPU 上矩阵乘法的最优性能要求矩阵足够大（充分利用 SM 并行度）
- 四个小矩阵乘法分别调度 vs 一个大矩阵乘法：后者吞吐量更高
- cuBLAS 对大矩阵有更好的 tiling 策略，kernel launch 开销更低
- 一次 cat + 一次大 GEMM 比四次独立 GEMM 快约 2-4 倍（实测依硬件而定）

### 命名约定：W vs U 的来历

在学术文献中存在两套命名习惯：

**习惯一（经典 RNN 文献）：**
```
W  专指作用于 h_{t-1}（循环权重，Recurrent Weight）
U  专指作用于 x_t  （输入权重，Input Weight）
```
来源：早期 RNN 论文将循环连接权重命名为 W，强调其"循环"性质，而输入权重用 U 区分。

**习惯二（工程代码）：**
```
W_xf  作用于 x_t（下标 x 表示 input）
W_hf  作用于 h_{t-1}（下标 h 表示 hidden）
```

两套命名描述的是**同一件事**，只是符号选择不同。在自己的代码里用下标区分（`w_xf` / `w_hf`）完全合理，甚至比 W/U 更直观——因为下标直接说明了"作用对象"。

---

## 🔧 代码示例：两种写法的等价验证

```python
import torch
import torch.nn as nn

# 超参数
batch_size = 4
input_dim = 8
hidden_dim = 16

# 随机输入
x_t    = torch.randn(batch_size, input_dim)
h_prev = torch.randn(batch_size, hidden_dim)

# ---- 写法 A：加法形式 ----
W_xf = torch.randn(input_dim, hidden_dim)
W_hf = torch.randn(hidden_dim, hidden_dim)
b_f  = torch.zeros(hidden_dim)

out_A = torch.sigmoid(x_t @ W_xf + h_prev @ W_hf + b_f)

# ---- 写法 B：Cat 形式 ----
# W_f 由 W_xf 和 W_hf 竖向拼接而成
W_f = torch.cat([W_xf, W_hf], dim=0)  # shape: (input_dim + hidden_dim, hidden_dim)

combined = torch.cat([x_t, h_prev], dim=1)  # shape: (batch_size, input_dim + hidden_dim)
out_B = torch.sigmoid(combined @ W_f + b_f)

# ---- 验证等价性 ----
print("最大误差:", (out_A - out_B).abs().max().item())
# 输出应接近 0（浮点误差范围内，约 1e-7 量级）

assert torch.allclose(out_A, out_B, atol=1e-6), "两种写法不等价！"
print("验证通过：两种写法完全等价 ✅")
```

---

## ⚠️ 易错点与常见误解

**1. 误以为两种写法参数量不同**
- 错误理解：Cat 写法只有一个矩阵，加法写法有两个矩阵，参数量不一样
- 正确理解：Cat 写法的大矩阵 W_f 的参数量 = W_xf + W_hf 的参数量之和，完全相同

**2. 混淆 W/U 命名导致的理解偏差**
- 错误理解：看到 `W_f · h_{t-1}` 以为 W 是作用于输入的，看到 `U_f · x_t` 以为 U 是作用于隐状态的
- 正确理解：W 作用于 h（循环），U 作用于 x（输入）——这是 RNN 文献的经典约定，与字母本身无关

**3. Cat 维度拼接方向搞错**
- 错误写法：`torch.cat([x_t, h_prev], dim=0)`（沿 batch 维度拼，会混合不同样本）
- 正确写法：`torch.cat([x_t, h_prev], dim=1)`（沿特征维度拼，每个样本的特征拼在一起）

**4. 以为 PyTorch nn.LSTM 的 W_ih / W_hh 各自对应一个门**
- 错误理解：W_ih 是遗忘门的权重
- 正确理解：W_ih 形状为 `(4*hidden_size, input_size)`，**包含了 i、f、g、o 四个门的权重**，通过 chunk(4) 拆分

**5. 在加法写法中漏掉偏置**
- 加法写法中 `x_t @ W_xf + h_prev @ W_hf + b_f` 只有一个偏置 b_f
- Cat 写法等价的偏置也只有一个
- 不要写成 `x_t @ W_xf + b_x + h_prev @ W_hf + b_h`（多加了一个偏置，参数不等价）

---

## 🔗 知识延伸

- LSTM 四个门（i/f/g/o）均使用同样的等价变换，只是激活函数不同（sigmoid vs tanh）
- GRU 同样有这两种等价写法，结构更简单（只有两个门）
- Transformer 的 Attention 中 Q/K/V 的计算本质也是同样的线性变换分解思想
- PyTorch `nn.LSTMCell` 暴露了底层实现，可以用来验证单步计算行为

## 📚 参考资料

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*
- PyTorch 官方文档：[nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- Goodfellow et al., *Deep Learning*, Chapter 10: Sequence Modeling
