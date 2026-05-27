# Conv1d 卷积核维度理解

## 📌 核心问题
Conv1d 的 kernel_size=3 是 3×3 吗？为什么不是？和 Conv2d 的区别是什么？

## 🌱 根源与动机
卷积核的维度由输入数据的维度决定：
- 图像是二维数据 → Conv2d → 卷积核 3×3
- 文本序列是一维数据 → Conv1d → 卷积核只有长度方向，kernel_size=3 表示一次看3个token

## 📐 理论推导

Conv2d vs Conv1d 对比：

```
图像 (H, W)          → Conv2d(3) → 卷积核 3×3，在高和宽两个方向滑动
文本序列 (t,)        → Conv1d(3) → 卷积核 1×3，只在token位置方向滑动
```

Conv1d 卷积核真实形状：
```
(out_channels, in_channels, kernel_size)
(256,          128,          3)
```

意思是：同时看3个token位置，每个token的128维特征全部参与计算，特征维度方向被完整处理。

序列长度计算公式：
```
输出长度 = (输入长度 + 2×padding - kernel_size) / stride + 1

不加padding：(5 + 0 - 3) / 1 + 1 = 3   ← 序列变短
加padding=1：(5 + 2 - 3) / 1 + 1 = 5   ← 序列长度不变
```

## 💡 关键理解

Conv1d 和 Linear 的核心区别：
```
Linear：每个token独立变换，只看自己         → (bs, t, 2e)
Conv1d：每个token融合左右邻居上下文信息     → (bs, t, 2e)
```

输出 shape 一样，但 Conv1d 每个输出向量包含了局部上下文信息。

padding=1 的作用：让边缘 token 也能参与完整的卷积计算，同时保持序列长度不变。

## 🔧 代码实现

```python
import torch
import torch.nn as nn

e = 128
# 参数含义：in=128, out=256, kernel=3, stride=1, padding=1
conv1d = nn.Conv1d(e, 2*e, 3, 1, padding=1, dtype=torch.float32)

# Conv1d 输入要求：(bs, channels, length)，注意和Linear不同
# token_embs: (bs, t, e) → 需要转置 → (bs, e, t)
token_embs = torch.randn(1, 5, 128)
x = token_embs.transpose(1, 2)          # (1, 128, 5)  ← (bs, e, t)
out = conv1d(x)                          # (1, 256, 5)
out = out.transpose(1, 2)               # (1, 5, 256)  ← 转回 (bs, t, 2e)

print(out.shape)  # (1, 5, 256)

# 卷积核真实形状
print(conv1d.weight.shape)  # (256, 128, 3) ← (out_ch, in_ch, kernel_size)
```

## ⚠️ 易错点与常见误解

1. 以为 Conv1d kernel_size=3 是 3×3 → 只有序列长度方向是3，特征维度被完整处理
2. Conv1d 输入格式是 (bs, channels, length)，与 Linear 的 (bs, length, channels) 相反，需要 transpose
3. 不加 padding 序列长度会缩短，加 padding=1 且 kernel=3 stride=1 时长度保持不变

## 🔗 知识延伸
- Conv2d：图像二维卷积，卷积核在高和宽两个方向滑动
- 与 Linear 对比：Linear 每个token独立，Conv1d 融合局部上下文，是局部感受野的体现
- 与 Transformer 对比：Conv1d 只能看固定窗口(kernel_size)范围内的上下文，Transformer 的 Attention 能看全局
