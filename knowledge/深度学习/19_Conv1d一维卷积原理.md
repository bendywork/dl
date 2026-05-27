# Conv1d 一维卷积原理

## 📌 核心问题
nn.Conv1d 的 kernel_size=3 为什么不是 3×3？Conv1d 和 Conv2d 的卷积核有什么区别？

## 🌱 根源与动机
卷积核的维度由输入数据的维度决定：
- 图像是二维数据（高×宽） → Conv2d → 卷积核 3×3
- 文本是一维数据（token序列） → Conv1d → 卷积核只有长度方向，大小为3

## 📐 理论推导

Conv1d 参数含义：
```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
nn.Conv1d(128,         256,          3,           1,      1)
#         输入特征维度  输出特征维度   一次看3个token 步长1  两端补0
```

卷积核真实形状：(out_channels, in_channels, kernel_size) = (256, 128, 3)
意思是：同时看3个token位置，每个token的128维特征全部参与计算。

Conv1d vs Conv2d：
```
Conv2d(3×3)：在高和宽两个方向滑动
■ ■ ■
■ ■ ■   ← 覆盖 3行 × 3列像素
■ ■ ■

Conv1d(3)：只在token位置方向滑动
■  ■  ■   ← 覆盖 3个token位置，特征维度(128)完整参与，不需要单独指定
```

padding 对序列长度的影响：
```
输入 t=5，kernel=3，stride=1

不加padding：输出 t = (5-3)/1 + 1 = 3   ← 序列变短
加padding=1：输出 t = (5+2-3)/1 + 1 = 5 ← 序列长度不变
```

## 💡 关键理解

Conv1d 和 Linear 的核心区别：
```
Linear：每个token独立变换，只看自己         → (bs, t, 2e)
Conv1d：每个token融合左右邻居，一次看3个token → (bs, t, 2e)
```

输出 shape 相同，但 Conv1d 的每个输出向量包含了局部上下文信息。

特征维度（e=128）不是卷积核滑动的方向，是每个位置的"深度"，会被完整处理。

## 🔧 代码实现

```python
import torch
import torch.nn as nn

bs, t, e = 1, 5, 128
dtype = torch.float32

conv1d = nn.Conv1d(e, 2*e, 3, 1, padding=1, dtype=dtype)

# 注意：Conv1d 输入格式是 (bs, channels, length)，与 Linear 不同
token_embs = torch.randn(bs, t, e, dtype=dtype)

# 需要转置：(bs, t, e) → (bs, e, t)
x = token_embs.transpose(1, 2)         # (1, 128, 5)
out = conv1d(x)                         # (1, 256, 5)
out = out.transpose(1, 2)              # (1, 5, 256)  ← 转回来

print(out.shape)  # (1, 5, 256)
```

## ⚠️ 易错点与常见误解

1. 以为 kernel_size=3 是 3×3 → Conv1d 只在序列方向滑动，卷积核只有一个维度
2. Conv1d 输入格式是 (bs, channels, length)，不是 (bs, length, channels)，需要 transpose
3. 不加 padding 序列会缩短，padding=1 配合 kernel=3 可保持序列长度不变

## 🔗 知识延伸
- 与 Conv2d 的关系：Conv2d 在高和宽两个方向滑动，Conv1d 只在序列长度方向滑动
- 与 Linear 的关系：Linear 是 token 独立变换，Conv1d 融合局部上下文，表达能力更强
- 与 Transformer 的关系：Transformer 用 Attention 做全局上下文融合，Conv1d 只做局部
