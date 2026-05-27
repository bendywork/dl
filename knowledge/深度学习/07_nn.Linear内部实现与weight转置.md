# nn.Linear 内部实现与 weight 转置

## 📌 核心问题
为什么手动实现 nn.Linear 时要写 `xi @ linear.weight.T + linear.bias`？weight 为什么要转置？

## 🌱 根源与动机
数学上线性变换写作 y = Wx + b，W 是 (out, in)，x 是列向量 (in, 1)。
PyTorch 沿用这个约定把 weight 存成 (out, in)，但实际计算时 x 是行向量，所以计算时要转置。

## 📐 理论推导

nn.Linear 内部存储的参数：
```
linear = nn.Linear(128, 256)
linear.weight   shape: (256, 128)   ← (输出维度, 输入维度)
linear.bias     shape: (256,)
```

手动实现等价于：
```
y = x @ W.T + b
```

数学写法 vs PyTorch 写法：
```
数学：y = Wx + b    W是(out,in)，x是列向量(in,1)
PyTorch：y = xW^T + b    x是行向量(batch, in)
```

维度推导（以不同输入形状为例）：
```
一维：(128,)    @ (128, 256) + (256,) = (256,)
二维：(1, 128)  @ (128, 256) + (256,) = (1, 256)    ← 广播加bias
三维：(1,5,128) @ (128, 256) + (256,) = (1, 5, 256) ← batch维度透明传递
```

## 💡 关键理解

weight 存 (out, in) 是 PyTorch 历史习惯，沿用数学约定。
实际 x 是行向量，所以计算时必须 .T 转置成 (in, out) 才能和输入维度对上。
bias 形状 (out,) 通过广播加到结果的最后一维。

## 🔧 代码实现

```python
import torch
import torch.nn as nn

linear = nn.Linear(128, 256)

# 实际输入是二维矩阵
xi = torch.randn(1, 128)  # (batch, in_features)

# 方式1：直接调用
out1 = linear(xi)

# 方式2：手动实现（等价）
out2 = xi @ linear.weight.T + linear.bias
# (1, 128) @ (128, 256) + (256,) = (1, 256)

print(torch.allclose(out1, out2))  # True
print(out1.shape)  # (1, 256)

# 三维输入（含token序列）
xi_3d = torch.randn(1, 5, 128)
out3 = xi_3d @ linear.weight.T + linear.bias
print(out3.shape)  # (1, 5, 256)  ← batch维度透明传递
```

## ⚠️ 易错点与常见误解

1. 以为 weight 是 (in, out) → 实际是 (out, in)，手动计算必须 .T
2. 以为 x 必须是一维 → 实际支持任意维度，batch维度透明传递，只操作最后一维
3. bias 的广播：(1, 256) + (256,) 自动广播，不需要手动 reshape

## 🔗 知识延伸
- 与 Batched 矩阵乘法的关系：三维输入时就是 batched matmul，前面的维度是 batch
- 与广播机制的关系：加 bias 时触发广播，(256,) 扩展到 (1, 256)
