# Conv1d 与 Conv2d 卷积核维度区别

## 📌 核心问题
为什么 Conv1d 的 kernel_size=3 不是 3×3，而 Conv2d 的 kernel_size=3 才是 3×3？

## 🌱 根源与动机
卷积核的维度由输入数据的维度决定：
- 图像有高和宽两个空间维度 → Conv2d → 卷积核 3×3
- 文本只有 token 位置一个序列维度 → Conv1d → 卷积核只有长度方向，大小为 3

## 📐 理论推导

Conv2d（图像）：
```
图像形状：(bs, C, H, W)
卷积核在 H 和 W 两个方向滑动
kernel_size=3 意味着 3×3，覆盖 3行×3列的像素区域
```

Conv1d（文本序列）：
```
序列形状：(bs, e, t)   ← 注意 Conv1d 要求 channel 在前
卷积核只在 t（token位置）方向滑动
kernel_size=3 意味着只覆盖 3个token位置
特征维度 e 不是滑动方向，被完整处理
```

Conv1d 卷积核真实形状：
```
(out_channels, in_channels, kernel_size)
(256,          128,          3)

对每个输出通道：卷积核是 (128, 3)
意思是：同时看3个token，每个token的128维特征全部参与计算
```

输出长度公式：
```
不加padding：t_out = (t - k) / s + 1 = (5-3)/1 + 1 = 3  ← 序列变短
加padding=1：t_out = (t + 2*p - k) / s + 1 = (5+2-3)/1 + 1 = 5  ← 长度不变
```

## 💡 关键理解

Conv1d 和 Linear 的核心区别：
```
Linear：每个token独立变换，只看自己         → (bs, t, 2e)
Conv1d：每个token融合左右邻居，看3个token   → (bs, t, 2e)
```
输出 shape 一样，但 Conv1d 的每个输出向量包含了局部上下文信息。

类比：
```
Conv2d(3×3)：看一块 3×3 的像素区域
Conv1d(3)：  看一段 3个token 的序列片段，每个token的所有特征维度都参与
```

## 🔧 代码实现

```python
import torch
import torch.nn as nn

bs, t, e = 1, 5, 128
conv1d = nn.Conv1d(e, 2*e, kernel_size=3, stride=1, padding=1)

# 注意：Conv1d 输入要求 (bs, channel, length)
token_embs = torch.randn(bs, t, e)
x = token_embs.transpose(1, 2)   # (1, 5, 128) → (1, 128, 5)
out = conv1d(x)                   # (1, 256, 5)
out = out.transpose(1, 2)         # (1, 5, 256)  ← 还原为 (bs, t, 2e)

print(out.shape)  # torch.Size([1, 5, 256])

# 卷积核真实形状
print(conv1d.weight.shape)  # (256, 128, 3)  ← (out_ch, in_ch, kernel_size)
```

## ⚠️ 易错点与常见误解

1. 以为 Conv1d kernel_size=3 是 3×3 → 只有序列长度方向是3，特征维度被完整卷积
2. Conv1d 输入格式是 (bs, channel, length)，与 Linear 的 (bs, length, channel) 相反，需要 transpose
3. 不加 padding 序列长度会缩短，加 padding=1 才能保持长度不变（kernel=3时）

## 🔗 知识延伸
- 与 Linear 的关系：Conv1d(kernel=1) 等价于 Linear，因为只看当前位置，不融合上下文
- 与 RNN 的关系：Conv1d 捕获局部上下文（固定窗口），RNN 捕获全局上下文（历史信息）
- 与 Transformer 的关系：Self-Attention 也是捕获全局上下文，但用注意力权重而非固定窗口
