# PyTorch Linear 层为什么 W 是 (out, in) 而不是 (in, out)

## 📌 核心问题
> `nn.Linear(in_features=3, out_features=2)` 的 weight 形状是 `(2, 3)`，直觉上 3 输入×2 输出应该是 `(3, 2)`，为什么反了？

## 🌱 根源：数学惯例 vs PyTorch 批量计算

### 数学课本怎么写

```
y = W @ x + b

W: (m, n)   m=输出维度, n=输入维度
x: (n,)     列向量
y: (m,)     列向量

W 的每一行 = 一个输出神经元对全部输入的权重
```

W 天然就是 **(out, in)** 形状，数学上从来如此。

### PyTorch 怎么算

PyTorch 输入不是单个列向量，而是**批量行向量**：

```
Y = X @ Wᵀ + b

X:  (batch, n)     ← 一整批样本，每行一个样本
W:  (m, n)         ← 和数学一样 (out, in)
Wᵀ: (n, m)         ← 转置后放右边
b:  (m,)
Y:  (batch, m)     ← 一整批输出
```

**W 的形状没有反，是 (out, in)，和数学一致。** 反直觉的是 PyTorch 把 W 转置后放右边乘。

### 维度验证

```
x:   (1, 3)
W:   (2, 3)    →  Wᵀ: (3, 2)

y = (1,3) @ (3,2) + (2,) = (1, 2) ✓
```

## 📐 关键对比

| | 数学 | PyTorch |
|---|------|---------|
| 公式 | `y = W @ x + b` | `y = x @ Wᵀ + b` |
| W 形状 | (out, in) | (out, in) — **一样** |
| x 形状 | (n,) 列向量 | (batch, n) 行向量 |
| W 在哪一侧 | 左边 | **转置后**在右边 |
| 结果形状 | (m,) | (batch, m) |

## 💡 关键理解

1. **W 形状从来没变过**，数学和 PyTorch 都是 (out, in)，每一行对应一个输出
2. **转置是计算位置决定的**：x 放左边必须乘 Wᵀ，W 放左边直接乘 x——结果相同，只是组织方式不同
3. **这和反向传播的 Wᵀ 是同一个转置**：前向 `x @ Wᵀ`，反向 `Wᵀ @ ∂L/∂y`，转置贯穿始终（见 [[13_反向传播中W转置的由来]]）

### 一句话

> W 存成 (out, in) 是数学惯例；PyTorch 因为批量行向量把 x 放左边，所以内部用 x @ Wᵀ——这个转置和你反向传播看到的 Wᵀ 是同一回事。

## 🔧 代码验证

```python
import torch
from torch import nn

linear = nn.Linear(in_features=3, out_features=2)
x = torch.randn(1, 3)

# PyTorch 内部等价于
y = x @ linear.weight.T + linear.bias

# 不等价于（维度都不对）
# y = linear.weight @ x  # ❌ (2,3) @ (1,3) 不匹配
```

## ⚠️ 易错点

1. **"W 形状是 (in, out)"** → 错。print 出来是 (out, in)，和直觉相反但和数学一致
2. **"PyTorch 和数学公式不一样"** → 不精确。数学就是 y=Wx，W=(out,in)。PyTorch 只是把 x 放左边导致要转置 W，等价变换
3. **"反向传播的 Wᵀ 和这个 Wᵀ 是两回事"** → 错。是**同一个转置**。前向 x@Wᵀ，反向 Wᵀ@∂L/∂y，Wᵀ 贯穿前向和反向
4. **手写 W@x 忘记 x 要是列向量** → 在 PyTorch 里 x 是行向量 (batch,n)，直接 W@x 维度不对，必须用 x@W.T

## 🔗 知识延伸

- [[13_反向传播中W转置的由来]]：Wᵀ 在反向传播中的数学根源
- [[05_BP梯度推导详解]]：完整 BP 推导
- [[11_PyTorch梯度计算开关详解]]：requires_grad 和 grad 的机制

## 📚 参考资料

- PyTorch 源码：`torch/nn/modules/linear.py` → `F.linear(input, weight, bias)` 即 `input @ weight.T + bias`
- 《Deep Learning》Goodfellow et al., Chapter 6.5
