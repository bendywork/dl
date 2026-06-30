# CNN 卷积神经网络链路

## 逻辑链

```
全连接网络局限（参数爆炸）→ 卷积核滑动（参数共享）→ Conv1d(时序)
    → Conv2d(图像) → 堆叠加深 梯度消失 → ResNet(残差连接) 解决退化
```

## CNN 核心洞察

**全连接 vs 卷积的核心区别**：参数共享 + 局部感受野

## 知识点 → 文件

| 知识点 | 文件 |
|--------|------|
| CNN 完整复习 | `02_深度学习基础/04_CNN卷积/01_CNN卷积神经网络完整复习.md` |
| Conv1d 原理 | `02_深度学习基础/04_CNN卷积/02_Conv1d一维卷积原理.md` |
| Conv1d 核维度 | `02_深度学习基础/04_CNN卷积/03_Conv1d卷积核维度理解.md` |
| Conv1d vs Conv2d | `02_深度学习基础/04_CNN卷积/04_Conv1d与Conv2d卷积核维度区别.md` |
| ResNet 原理 | `02_深度学习基础/04_CNN卷积/05_残差神经网络ResNet原理.md` |
| ResNet 设计哲学 | `02_深度学习基础/04_CNN卷积/06_ResNet残差网络的设计哲学与思维逻辑.md` |

## ResNet 的通用意义

```
残差连接 H(x) = F(x) + x → 梯度高速通道 → 解决深层退化
这个思想后来被 Transformer 继承为残差连接（Add & Norm）
```

## 向下连接

↓ `03_序列建模NLP`：Conv1d 处理时序 → 但无法捕获长距离依赖 → RNN 出场
↓ `05_Attention与Transformer`：ResNet 的残差 → Transformer 的 Add & Norm
