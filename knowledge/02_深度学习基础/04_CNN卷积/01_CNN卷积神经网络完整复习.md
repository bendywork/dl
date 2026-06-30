# CNN 卷积神经网络完整复习

## 📌 核心问题
> 卷积神经网络是什么？为什么图像任务用 CNN 而不是全连接网络？CNN 的每个组件（卷积、池化、步长、填充）背后的逻辑是什么？经典架构如何演进？

---

## 一、CNN 诞生的根源：全连接网络处理图像的三大灾难

### 灾难1：参数爆炸

一张 224×224×3 的图片，展平后是 150,528 维输入。如果第一层全连接有 1024 个神经元：

```
参数量 = 150,528 × 1024 + 1024 ≈ 1.54 亿
```

仅一层就 1.5 亿参数，深层网络根本训练不动。

### 灾难2：丢失空间结构

展平操作把 224×224 的二维图像拉成一条 150,528 维向量：

```
原图：          展平后：
■ ■ ■          ■ ■ ■ ■ ■ ■ ■ ■ ■
■ ■ ■    →     （相邻像素的关系完全丢失）
■ ■ ■          （"上面"和"下面"的概念消失了）
```

图像的核心信息在于**像素之间的空间关系**（边缘、纹理、形状），展平后全连接层无法利用这种关系。

### 灾难3：平移不变性缺失

同一个猫，在图片左上角和右下角，全连接网络会当成完全不同的输入。但我们的直觉是：**物体出现在不同位置，本质不变。**

### CNN 的三大应对

| 全连接的灾难 | CNN 的解法 | 核心思想 |
|-------------|-----------|---------|
| 参数爆炸 | 权重共享 | 同一个卷积核扫遍全图，参数量与位置无关 |
| 丢失空间结构 | 局部连接 | 卷积核只看局部区域，保留空间关系 |
| 平移不变性缺失 | 平移等变性 | 同一特征出现在任意位置，同一卷积核都能检测到 |

---

## 二、卷积操作：从信号处理到图像特征提取

### 2.1 数学根源

卷积的数学定义（一维连续）：

```
(f * g)(t) = ∫ f(τ) · g(t - τ) dτ
```

**直觉**：用一个函数 g（滤波器/核）去"扫描"另一个函数 f（信号），得到每个位置的"匹配程度"。

在图像处理中，卷积核就是一个**小的权重矩阵**，在图像上滑动，计算局部加权和：

```
图像局部区域        卷积核           点积
[1  2  1]         [1  0  -1]
[0  1  2]    *    [1  0  -1]    =  某个标量值
[3  1  0]         [1  0  -1]        （该位置的特征强度）
```

### 2.2 Conv2d 的四个核心参数

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
```

#### kernel_size（卷积核大小）

- 最常用 3×3，也有 1×1、5×5、7×7
- 3×3 是"最小的能捕获上下左右+中心"的尺寸
- 两个 3×3 堆叠的感受野 = 一个 5×5，但参数更少（18 vs 25）

#### stride（步长）

- 步长=1：每次移动 1 像素，输出尺寸基本不变
- 步长=2：每次移动 2 像素，输出尺寸减半（下采样）

```
stride=1:  ███ →  ███ →  ███      密密扫描
            ███    ███    ███
            ███    ███    ███

stride=2:  ███ →      ███          跳着扫描，输出更小
            ███        ███
```

#### padding（填充）

- 在输入边缘补 0，控制输出尺寸
- padding=1 + kernel=3 + stride=1 → 输出尺寸不变（same padding）
- 不加 padding → 输出缩小（valid padding）

```
原始 4×4：          padding=1 后 6×6：
┌──────┐           ┌────────┐
│ ■ ■ ■ │           │ 0 0 0 0 0 0│
│ ■ ■ ■ │           │ 0 ■ ■ ■ 0│
│ ■ ■ ■ │           │ 0 ■ ■ ■ 0│
│ ■ ■ ■ │           │ 0 ■ ■ ■ 0│
└──────┘           │ 0 0 0 0 0 0│
                   └────────┘
```

#### in_channels / out_channels

- in_channels：输入的通道数（RGB=3，上一层输出=64...）
- out_channels：输出的通道数 = 卷积核的个数
- **每个输出通道 = 一个不同的卷积核在所有输入通道上卷积后求和**

```
卷积核形状：(out_channels, in_channels, kH, kW)
例如：(64, 3, 3, 3) → 64 个 3×3×3 的核，每个核覆盖所有输入通道
```

### 2.3 输出尺寸计算公式

```
H_out = (H_in + 2×padding - kernel_size) / stride + 1
W_out = (W_out 同理)
```

**速算示例**：
```
输入 32×32，kernel=3，stride=1，padding=1
→ (32 + 2×1 - 3)/1 + 1 = 32  ← 尺寸不变

输入 32×32，kernel=3，stride=2，padding=1
→ (32 + 2×1 - 3)/2 + 1 = 16  ← 尺寸减半
```

---

## 三、池化层：降维与不变性

### 3.1 为什么需要池化？

1. **降低特征图尺寸** → 减少计算量和参数
2. **增大感受野** → 后续层能看到更大范围
3. **提供微小平移不变性** → 物体稍微移动，最大池化结果不变

### 3.2 两种池化

```python
nn.MaxPool2d(kernel_size=2, stride=2)   # 取最大值，保留最显著特征
nn.AvgPool2d(kernel_size=2, stride=2)   # 取平均值，更平滑
```

```
最大池化（2×2）：         平均池化（2×2）：
[1  3]                   [1  3]
[2  4]    → max = 4      [2  4]    → avg = 2.5

直觉：最大池化≈"这里有没有这个特征"
     平均池化≈"这里这个特征有多强"
```

### 3.3 全局平均池化（GAP）

```python
nn.AdaptiveAvgPool2d(1)  # 任意尺寸 → 1×1
```

现代 CNN 用 GAP 替代全连接层：
- 全连接层：Flatten → Linear，参数多
- GAP：直接把 H×W 压成 1×1，无参数，防止过拟合

---

## 四、CNN 的核心概念

### 4.1 感受野（Receptive Field）

**定义**：输出特征图上一个像素，对应输入图像上的区域大小。

```
1层 3×3 卷积：感受野 = 3×3
2层 3×3 卷积：感受野 = 5×5
3层 3×3 卷积：感受野 = 7×7
```

**为什么堆叠小卷积核优于大卷积核？**

| 方案 | 参数量 | 感受野 | 非线性层数 |
|------|--------|--------|-----------|
| 一个 7×7 | 7×7×C×C = 49C² | 7×7 | 1 |
| 三个 3×3 | 3×(3×3×C×C) = 27C² | 7×7 | 3 |

三个 3×3：参数更少（27C² < 49C²）、感受野相同、非线性更强（3层ReLU vs 1层）。这就是 VGG 的设计思想。

### 4.2 特征图（Feature Map）

每个卷积核在输入上滑动，输出一张二维"热力图"——记录了该特征在图像各位置的响应强度。

```
浅层特征图：边缘、颜色、纹理（低级特征）
中层特征图：部件、形状（中级特征）
深层特征图：物体语义（高级特征）
```

### 4.3 权重共享

同一个卷积核在所有位置使用相同的权重，这意味着：

```
"竖直边缘检测器"在图像左上角和右下角用同一组权重
→ 无论竖直边缘出现在哪里，都能被检测到
→ 参数量与图像大小无关！
```

全连接 vs 卷积参数对比：
```
输入 32×32×3，输出 32×32×64

全连接：32×32×3 × 32×32×64 = 6,442,450,944 参数（64亿！）
卷积 3×3：64 × (3×3×3) = 1,728 参数
```

差距 370 万倍，这就是权重共享的威力。

### 4.4 1×1 卷积的特殊用途

```python
nn.Conv2d(64, 128, kernel_size=1)  # 1×1 卷积
```

看似没有"看邻居"，但它有三大用途：

1. **通道数变换**：64 → 128（升维）或 256 → 64（降维，瓶颈结构）
2. **跨通道信息融合**：每个像素位置的 64 个通道做加权求和 → 1 个新通道
3. **增加非线性**：1×1 Conv + ReLU = 对每个像素做非线性变换，但不改变空间尺寸

---

## 五、经典 CNN 架构演进

### 5.1 演进时间线

```
LeNet-5 (1998)          → 开山之作，手写数字识别
    ↓
AlexNet (2012)           → 深度学习复兴，ReLU + Dropout + GPU
    ↓
VGGNet (2014)            → "更深更好"，统一 3×3 卷积
    ↓
GoogLeNet/Inception (2014) → 多尺度并行，1×1 瓶颈降维
    ↓
ResNet (2015)            → 残差连接，突破深度极限（152层+）
    ↓
EfficientNet (2019)      → 复合缩放，精度与效率平衡
    ↓
ConvNeXt (2022)          → 用 Transformer 设计思想改造 CNN
```

### 5.2 每个架构的核心贡献

#### LeNet-5（1998，LeCun）

```
输入 → Conv(5×5) → Pool → Conv(5×5) → Pool → FC → FC → 输出
       6通道         ↓       16通道       ↓
                  2×2下采样           2×2下采样
```

- 首次定义 CNN 标准范式：**卷积-池化-卷积-池化-全连接**
- 用于手写邮政编码识别，是 CNN 的起点

#### AlexNet（2012，Krizhevsky）

```
5个卷积层 + 3个全连接层
关键创新：ReLU、Dropout(0.5)、数据增强、GPU训练、LRN(已过时)
```

| 创新 | 解决的问题 | 为什么有效 |
|------|-----------|-----------|
| ReLU | sigmoid 梯度消失 | 正区间梯度恒为1，不饱和 |
| Dropout | 全连接层过拟合 | 随机丢弃50%，相当于集成学习 |
| 数据增强 | 训练数据不够 | 翻转/裁剪扩充数据，增强泛化 |
| GPU 训练 | 训练速度太慢 | 并行计算，加速 10-50 倍 |
| LRN | — | 现已被 BatchNorm 取代，历史产物 |

#### VGGNet（2014，Simonyan）

**核心思想**：只用 3×3 卷积，靠堆叠深度换取性能。

```
VGG-16 结构（16层有参数的层）：
[3×3,64] ×2  → Pool
[3×3,128] ×2 → Pool
[3×3,256] ×3 → Pool
[3×3,512] ×3 → Pool
[3×3,512] ×3 → Pool
FC(4096) → FC(4096) → FC(1000)
```

**设计哲学**：
- 大卷积核（7×7）→ 多个小卷积核（3×3 堆叠），参数更少、非线性更强
- 通道数翻倍规律：64 → 128 → 256 → 512，每次池化后翻倍
- 简洁统一，但参数量大（1.38 亿），主要在全连接层

#### GoogLeNet / Inception（2014，Szegedy）

**核心思想**：不同大小的特征应该并行提取。

```
Inception 模块：
输入 ──┬→ 1×1 Conv ──────────┐
       ├→ 1×1 Conv → 3×3 Conv ─┤→ 通道拼接 (Concat)
       ├→ 1×1 Conv → 5×5 Conv ─┤
       └→ 3×3 Pool → 1×1 Conv ─┘
```

关键创新：**1×1 瓶颈卷积**（降维再升维），大幅减少计算量：
```
直接 3×3：256 → 256，参数 = 3×3×256×256 ≈ 59万
瓶颈结构：256 → 64 → 256，参数 = 1×1×256×64 + 3×3×64×64 + 1×1×64×256 ≈ 7万
```

#### ResNet（2015，何恺明）

详见 [[39_残差神经网络ResNet原理]] 和 [[40_ResNet残差网络的设计哲学与思维逻辑]]

核心：`y = F(x) + x`，残差连接让梯度直接回传，突破深度限制。

---

## 六、CNN 的完整工作流程

以 AlexNetSmall（适配 CIFAR-10）为例：

```
输入图片 (3, 32, 32)
    │
    ▼
┌─────────────────────────────┐
│  卷积层1：提取低级特征         │  边缘、颜色、纹理
│  Conv(3→48, 3×3) + ReLU     │
│  MaxPool(2×2)               │  → (48, 16, 16)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  卷积层2：提取中级特征         │  形状、部件
│  Conv(48→128, 3×3) + ReLU   │
│  MaxPool(2×2)               │  → (128, 8, 8)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  卷积层3-5：提取高级特征       │  语义、物体
│  Conv(128→192→192→128)      │
│  MaxPool(2×2)               │  → (128, 4, 4)
└─────────────────────────────┘
    │
    ▼  Flatten: (128, 4, 4) → (2048,)
┌─────────────────────────────┐
│  全连接层：分类决策            │
│  FC(2048→1024) + Dropout     │
│  FC(1024→512)  + Dropout     │
│  FC(512→10)                  │  → 10类概率
└─────────────────────────────┘
```

**信息流的本质**：
```
像素 → 边缘 → 纹理 → 部件 → 物体 → 类别
（低级）                        （高级）
每经过一层，感受野增大，语义层次升高
```

---

## 七、CNN 中每个组件的"为什么"

| 组件 | 为什么需要它 | 不用它会怎样 |
|------|------------|-------------|
| 卷积 | 局部连接+权重共享 | 全连接参数爆炸，丢失空间结构 |
| ReLU | 非线性+梯度不饱和 | sigmoid 梯度消失，深层训练不动 |
| 池化 | 降维+平移不变性 | 特征图太大，计算量爆炸 |
| BatchNorm | 稳定每层输入分布 | 训练不稳定，学习率要很小 |
| Dropout | 防止全连接层过拟合 | 全连接层参数多，容易记住训练集 |
| 1×1 Conv | 通道变换+跨通道融合 | 无法高效改变通道数 |
| 残差连接 | 梯度直通，深层可训练 | 深层网络退化，梯度消失 |

---

## 八、Conv1d 与 Conv2d 的核心区别（快速回顾）

详见 [[19_Conv1d一维卷积原理]]、[[20_Conv1d卷积核维度理解]]、[[21_Conv1d与Conv2d卷积核维度区别]]

| | Conv2d | Conv1d |
|---|---|---|
| 输入形状 | (bs, C, H, W) | (bs, C, L) |
| 滑动方向 | H 和 W 两个方向 | L 一个方向 |
| kernel_size=3 | 3×3 二维核 | 1×3 一维核 |
| 典型应用 | 图像、视频 | 文本序列、时间序列 |
| 卷积核形状 | (out, in, kH, kW) | (out, in, kL) |

---

## ⚠️ 易错点与常见误解

1. **"卷积就是滤波"** → 不完全对。卷积核的权重是**学习出来**的，不是手动设计的。网络自己学会要检测什么特征。

2. **"padding=1 就是不改变尺寸"** → 只有 kernel=3, stride=1 时才成立。kernel=5 时需要 padding=2。

3. **"池化层有参数"** → 池化层**没有可学习参数**，只是固定的 max 或 avg 操作。

4. **"通道数越多越好"** → 通道数翻倍通常伴随池化减半，是计算量和表达力的权衡。太大 → 过拟合+计算慢。

5. **"CNN 只能处理图像"** → Conv1d 处理序列，3D CNN 处理视频/医学影像，图卷积处理非欧几里得数据。CNN 的核心思想（局部连接+权重共享）是通用的。

6. **"感受野覆盖了就一定能利用"** → 感受野是理论范围，实际有效感受野远小于理论值（高斯分布，中心区域权重高）。

7. **"卷积核越小越好"** → 1×1 只能做通道变换，无法捕获空间关系。3×3 是空间特征提取的最小单位。

---

## 🔗 知识延伸

- **CNN → Transformer**：ViT (Vision Transformer) 用自注意力替代卷积，但 ConvNeXt (2022) 证明纯 CNN 重新设计后仍可匹敌 Transformer
- **CNN → 多模态**：CNN 提取图像特征 → 与文本特征对齐 → CLIP 等多模态模型
- **CNN → 扩散模型**：U-Net（编码器-解码器 + skip connection）是扩散模型的核心架构
- **CNN → 目标检测**：在分类 CNN 基础上加 RPN（区域提议网络）→ Faster R-CNN / YOLO
- **权重共享 →** 不是只有卷积才共享权重，Transformer 的 Attention 也是全局共享的 QKV 投影

---

## 📚 参考资料

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998 (LeNet)
- Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks", 2012 (AlexNet)
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", 2014 (VGG)
- He et al., "Deep Residual Learning for Image Recognition", 2015 (ResNet)
- Liu et al., "A ConvNet for the 2020s", 2022 (ConvNeXt)

---

## 二、底层数学完整推导

### 2.1 信号处理中的卷积定义

数学意义上的连续卷积定义为：

```
(f * g)(t) = ∫ f(τ) · g(t - τ) dτ
```

离散形式：

```
(f * g)[n] = Σ f[k] · g[n - k]
```

卷积有一个关键性质：**翻转后滑动**。g 关于原点翻转后，再与 f 逐点相乘求和。

### 2.2 深度学习中实际使用的是"互相关"

CNN 中的"卷积"在数学上其实是**互相关（Cross-Correlation）**，不对卷积核做翻转：

```
(f ⊛ g)[i, j] = Σ_m Σ_n f[i+m, j+n] · g[m, n]
```

之所以不翻转，是因为卷积核权重是通过训练学到的——翻不翻转都能学到正确权重，不翻转实现更简单，PyTorch 的 `F.conv2d` 实际执行的是互相关。

### 2.3 二维卷积的矩阵形式

设输入特征图 X ∈ R^(H×W)，卷积核 K ∈ R^(k×k)，输出 Y ∈ R^(H'×W')：

```
Y[i, j] = Σ_{m=0}^{k-1} Σ_{n=0}^{k-1} X[i·s+m, j·s+n] · K[m, n] + b
```

其中 s 是 stride（步长），b 是偏置。

### 2.4 输出尺寸公式推导

设输入尺寸 H，卷积核 k，步长 s，填充 p，膨胀 d：

```
有效卷积核尺寸 k' = d × (k - 1) + 1
输出尺寸 H_out = floor((H + 2p - k') / s) + 1
             = floor((H + 2p - d×(k-1) - 1) / s) + 1
```

**常见特例记忆：**

| 配置 | 效果 | 公式结果 |
|------|------|---------|
| k=3, s=1, p=1, d=1 | 尺寸不变（same padding） | H_out = H |
| k=3, s=2, p=1, d=1 | 尺寸减半 | H_out = ceil(H/2) |
| k=1, s=1, p=0, d=1 | 尺寸不变，通道变换 | H_out = H |
| k=2, s=2, p=0, d=1 | 尺寸减半（无重叠） | H_out = H/2 |

### 2.5 感受野（Receptive Field）推导

感受野指输出特征图上一个像素对应原始输入的区域大小。

**单层感受野：** r = k（卷积核大小）

**多层堆叠时的感受野递推公式：**

```
r_l = r_{l-1} + (k_l - 1) × ∏_{i=1}^{l-1} s_i
```

例：3 层 3×3 卷积（stride=1）：
- 第1层：r = 3
- 第2层：r = 3 + (3-1)×1 = 5
- 第3层：r = 5 + (3-1)×1 = 7

**关键结论：** 2 个 3×3 卷积叠加 = 1 个 5×5 的感受野，但参数量更少（2×9 vs 25），非线性更强。这是 VGG 用小卷积替代大卷积的理论依据。

### 2.6 参数量与计算量（FLOPs）

**参数量：**

```
Conv2d 参数量 = C_out × (C_in/groups × k_h × k_w + 1[bias])
```

例：Conv2d(3, 64, 3) → 64 × (3×3×3 + 1) = 64 × 28 = 1,792 个参数

**计算量（FLOPs，乘加算一次）：**

```
FLOPs = C_out × H_out × W_out × (C_in/groups × k_h × k_w)
```

例：输入 (3,224,224)，Conv2d(3,64,3,padding=1)：
- H_out=224, W_out=224
- FLOPs = 64 × 224 × 224 × (3×3×3) = 86,704,128 ≈ 87M

### 2.7 反向传播梯度推导

设损失为 L，输出 Y，输入 X，卷积核 K：

**对卷积核的梯度（用于更新权重）：**

```
∂L/∂K[m,n] = Σ_{i,j} (∂L/∂Y[i,j]) · X[i·s+m, j·s+n]
```

本质：梯度是输入 X 与上游梯度的互相关。

**对输入的梯度（用于继续反向传播）：**

```
∂L/∂X = K_flipped * (∂L/∂Y 上采样)
```

这里是真正的卷积（翻转核）。这就是为什么数学卷积要翻转——反向传播时自然要翻转回来。

---

## 三、卷积超参数完整详解

### 3.1 kernel_size（卷积核大小）

控制每次感知的局部区域大小，直接决定感受野和参数量。

| 大小 | 用途 | 典型场景 |
|------|------|---------|
| 1×1 | 通道维度的线性变换，不捕获空间特征 | Inception 中降维、NiN、逐点卷积 |
| 3×3 | 最常用，平衡感受野与参数量 | VGG、ResNet、几乎所有现代网络 |
| 5×5 | 更大感受野，参数量是3×3的2.7倍 | AlexNet第一层，现在少用 |
| 7×7 | 通常只用于网络第一层 | ResNet 第一层 stem |
| 11×11 | 历史用法 | AlexNet 原始论文第一层 |

**Python 验证参数量差异：**

```python
import torch.nn as nn

conv3 = nn.Conv2d(64, 64, 3, padding=1)
conv5 = nn.Conv2d(64, 64, 5, padding=2)

params3 = sum(p.numel() for p in conv3.parameters())  # 36,928
params5 = sum(p.numel() for p in conv5.parameters())  # 102,464
print(f"3x3: {params3}, 5x5: {params5}, 比值: {params5/params3:.1f}x")
# 输出：3x3: 36928, 5x5: 102464, 比值: 2.8x
```

### 3.2 stride（步长）

控制卷积核滑动的步幅，stride>1 会降低特征图分辨率。

```python
import torch
import torch.nn as nn

x = torch.randn(1, 3, 224, 224)
conv_s1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)  # 224→224
conv_s2 = nn.Conv2d(3, 64, 3, stride=2, padding=1)  # 224→112

print(conv_s1(x).shape)  # torch.Size([1, 64, 224, 224])
print(conv_s2(x).shape)  # torch.Size([1, 64, 112, 112])
```

**stride vs MaxPool 的区别：**
- MaxPool：取区域内最大值，有选择地保留信息
- stride=2：均匀跳步，无参数，计算更快
- 现代网络（ResNet变体）倾向用 stride=2 替代 MaxPool

### 3.3 padding（填充）

在输入边缘补0，控制输出尺寸，防止边缘信息丢失。

**三种填充模式（padding_mode 参数）：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| zeros（默认） | 补0 | 通用 |
| reflect | 镜像填充 | 图像修复、风格迁移 |
| replicate | 复制边缘像素 | 语义分割边缘保留 |
| circular | 循环填充 | 处理周期性信号 |

**padding 与 same/valid 对应：**

```python
# same padding（输出与输入同尺寸）：p = (k-1)/2（k为奇数时整除）
nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)   # 3×3 same
nn.Conv2d(C_in, C_out, kernel_size=5, padding=2)   # 5×5 same
nn.Conv2d(C_in, C_out, kernel_size=7, padding=3)   # 7×7 same

# PyTorch 1.9+ 可直接写 padding='same'（但不支持 stride>1）
nn.Conv2d(C_in, C_out, 3, padding='same')
```

### 3.4 dilation（空洞率/膨胀率）

在卷积核元素之间插入空隙，以相同参数量获得更大感受野。

```
dilation=1：标准卷积，3×3核感受野=3
dilation=2：空洞卷积，3×3核感受野=5（元素间距为2）
dilation=4：感受野=9
dilation=8：感受野=17
```

```python
import torch, torch.nn as nn, torch.nn.functional as F

# 空洞卷积可视化
x = torch.zeros(1, 1, 7, 7)
x[0, 0, 3, 3] = 1  # 中心点

conv_d1 = nn.Conv2d(1, 1, 3, dilation=1, padding=1)
conv_d2 = nn.Conv2d(1, 1, 3, dilation=2, padding=2)

# dilation=1 输出：3×3 区域非零
# dilation=2 输出：5×5 区域非零（等效感受野）
```

**多尺度空洞卷积（DeepLab ASPP 核心）：**

```python
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r)
            for r in [6, 12, 18]
        ])
    def forward(self, x):
        return sum(conv(x) for conv in self.convs)
```

### 3.5 groups（分组卷积）

将输入通道分成若干组，每组独立做卷积，大幅减少参数量。

```
标准卷积参数量：C_out × C_in × k × k
分组卷积参数量：C_out × (C_in/groups) × k × k  （减少 groups 倍）
```

**三种特殊情况：**

```python
# 1. 标准卷积（groups=1，默认）
nn.Conv2d(32, 64, 3, groups=1)   # 参数：64×32×9 = 18,432

# 2. 分组卷积（groups=N，AlexNet 最早使用）
nn.Conv2d(32, 64, 3, groups=2)   # 参数：64×16×9 = 9,216（减半）

# 3. 深度可分离卷积第一步：逐通道卷积（groups=in_channels）
nn.Conv2d(32, 32, 3, groups=32)  # 参数：32×1×9 = 288（减少32倍）
# 第二步：逐点卷积（1×1 conv），混合通道信息
nn.Conv2d(32, 64, 1)             # 参数：64×32×1 = 2,048
# 合计：288+2048=2,336 vs 标准卷积 18,432 → 减少 8 倍
```

**深度可分离卷积（MobileNet 核心）完整实现：**

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.pw(self.dw(x))
```

---

## 四、PyTorch API 完整讲解

### 4.1 nn.Conv2d 完整签名

```python
nn.Conv2d(
    in_channels,          # 输入通道数
    out_channels,         # 输出通道数（卷积核个数）
    kernel_size,          # int 或 (h, w) tuple
    stride=1,             # int 或 (h, w) tuple
    padding=0,            # int / (h,w) / 'same' / 'valid'
    dilation=1,           # int 或 (h, w) tuple
    groups=1,             # 分组数
    bias=True,            # 是否加偏置（接BN时一般False）
    padding_mode='zeros', # 'zeros'/'reflect'/'replicate'/'circular'
    device=None,
    dtype=None
)
```

**权重形状：** `weight.shape = (out_channels, in_channels/groups, kH, kW)`

**偏置形状：** `bias.shape = (out_channels,)`

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
print(conv.weight.shape)  # torch.Size([64, 3, 3, 3])
print(conv.bias.shape)    # torch.Size([64])

x = torch.randn(8, 3, 224, 224)     # batch=8
y = conv(x)
print(y.shape)  # torch.Size([8, 64, 224, 224])
```

### 4.2 nn.Conv1d — 用于序列数据

```python
nn.Conv1d(in_channels, out_channels, kernel_size, ...)
# 输入：(batch, channels, length)
# 输出：(batch, out_channels, length_out)

# 用于 NLP：文本卷积（TextCNN）
text_conv = nn.Conv1d(128, 256, kernel_size=3, padding=1)
x = torch.randn(32, 128, 50)  # 32个句子，128维embedding，50个词
y = text_conv(x)              # torch.Size([32, 256, 50])
```

### 4.3 nn.Conv3d — 用于视频/医学影像

```python
nn.Conv3d(in_channels, out_channels, kernel_size, ...)
# 输入：(batch, channels, depth, H, W)

video_conv = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
x = torch.randn(2, 3, 16, 112, 112)  # 2个视频，3通道，16帧，112×112
y = video_conv(x)  # torch.Size([2, 64, 16, 112, 112])
```

### 4.4 nn.ConvTranspose2d — 转置卷积（反卷积）

用于上采样，是编码器-解码器（U-Net、GAN）的核心组件。

```python
nn.ConvTranspose2d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, output_padding=0,  # output_padding 控制奇偶尺寸
    groups=1, bias=True, dilation=1
)

# 经典用法：stride=2 实现2倍上采样
up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
x = torch.randn(1, 64, 14, 14)
y = up(x)   # torch.Size([1, 32, 28, 28])

# 上采样公式：H_out = (H_in - 1) × stride - 2×padding + kernel_size + output_padding
```

**转置卷积 vs 双线性插值+卷积：**

| 方法 | 参数 | 可学习 | 常见问题 |
|------|------|--------|---------|
| ConvTranspose2d | 有 | 是 | 棋盘格伪影（checkerboard artifact） |
| Upsample + Conv2d | 有 | 是（Conv部分） | 无棋盘格，推荐 |

```python
# 推荐写法（避免棋盘格伪影）
up_block = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
)
```

### 4.5 F.conv2d — 函数式 API

更底层，可以手动传入权重，常用于自定义操作。

```python
import torch.nn.functional as F

weight = torch.randn(64, 3, 3, 3)  # (out_ch, in_ch, kH, kW)
bias   = torch.randn(64)
x      = torch.randn(1, 3, 224, 224)

y = F.conv2d(x, weight, bias, stride=1, padding=1)
# torch.Size([1, 64, 224, 224])

# 自定义高斯模糊核（不可学习）
def gaussian_blur(x, sigma=1.0):
    k = 5
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]          # (5, 5)
    kernel = kernel.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=k//2, groups=x.shape[1])
```

### 4.6 池化层 API

```python
# 最大池化
nn.MaxPool2d(kernel_size=2, stride=2)             # 标准2倍降采样
nn.MaxPool2d(3, stride=2, padding=1)              # stride不等于kernel_size

# 平均池化
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))                      # 全局平均池化（任意输入→1×1）
nn.AdaptiveAvgPool2d((7, 7))                      # 自适应到固定大小

# 全局平均池化是分类网络最后一层的标准写法
class Classifier(nn.Module):
    def forward(self, x):
        x = self.features(x)              # (B, C, H, W)
        x = nn.AdaptiveAvgPool2d(1)(x)   # (B, C, 1, 1)
        x = x.flatten(1)                  # (B, C)
        return self.fc(x)
```

### 4.7 BatchNorm2d — 必须掌握的搭档

```python
# 标准 Conv-BN-ReLU 模块（工业标准写法）
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            # bias=False：BN 有自己的 beta 参数，Conv 的 bias 冗余
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# BN 参数：gamma(weight) 和 beta(bias)，形状均为 (out_ch,)
bn = nn.BatchNorm2d(64)
print(bn.weight.shape)  # torch.Size([64])
print(bn.bias.shape)    # torch.Size([64])
```

**BN 在训练/推理时行为不同：**

```python
model.train()   # 用当前 batch 统计量，同时更新 running_mean/running_var
model.eval()    # 用 running_mean/running_var（全局统计），不更新
```

---

## 五、经典网络架构演进

### 5.1 LeNet-5（1998）— CNN 的起点

LeCun 提出，用于手写数字识别，奠定了 CNN 的基本框架。

```
输入 32×32 → Conv(6,5×5) → AvgPool(2×2) → Conv(16,5×5)
          → AvgPool(2×2) → Flatten → FC(120) → FC(84) → 输出10
```

**关键设计：** 局部连接、权重共享、池化降维，三大 CNN 基本组件首次完整出现。

### 5.2 AlexNet（2012）— 深度学习复兴里程碑

ImageNet LSVRC-2012 冠军，错误率从 26% 降至 15.3%。

**五大创新：**

| 创新 | 意义 |
|------|------|
| ReLU 代替 Sigmoid | 训练速度快 6 倍，缓解梯度消失 |
| Dropout(0.5) | 防止过拟合，相当于模型集成 |
| 数据增强 | 随机裁剪+翻转，扩充数据 |
| 双 GPU 并行 | 突破显存限制，开创多 GPU 训练 |
| 局部响应归一化(LRN) | 现已被 BN 取代 |

### 5.3 VGG（2014）— 深度的力量

Oxford 提出，提出了"小卷积核堆叠"的设计哲学。

**核心思想：** 用 2-3 个 3×3 卷积替代更大的卷积核：

```python
# 2个3×3 = 5×5的感受野，但参数量：2×(3×3)=18 vs 25，且多一次非线性
# 3个3×3 = 7×7的感受野，但参数量：3×(3×3)=27 vs 49

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(in_ch if i==0 else out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)
```

### 5.4 ResNet（2015）— 残差连接解决梯度消失

He 等人提出，ImageNet 错误率降至 3.57%，超越人类水平（5.1%）。

**核心问题：** 网络越深，梯度消失/爆炸越严重，深层网络反而比浅层网络差（退化问题）。

**解决方案：** 残差连接（shortcut connection）

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity      # 残差连接
        return self.relu(out)
```

**为什么有效：**

```
梯度 = ∂L/∂x = ∂L/∂(F(x)+x) = ∂L/∂F(x) + 1
```

加上恒等映射后，梯度中始终存在 "+1" 项，即使 F(x) 的梯度很小，整体梯度也不为零，从根本上解决梯度消失。

**下采样 Block（通道数变化时）：**

```python
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 1×1 conv 调整维度
        return self.relu(out)
```

### 5.5 轻量级网络对比

| 网络 | 关键技术 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| MobileNetV1 | 深度可分离卷积 | ~4M | 移动端 |
| MobileNetV2 | 倒置残差 + 线性瓶颈 | ~3.4M | 移动端 |
| ShuffleNet | Channel Shuffle + 分组卷积 | ~1.4M | 极低资源 |
| EfficientNet | 复合缩放（宽度/深度/分辨率） | 5-66M | 高精度通用 |

**MobileNetV2 倒置残差（核心思想）：**

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # 1×1 升维
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            # 3×3 深度卷积
            nn.Conv2d(hidden, hidden, 3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            # 1×1 降维（无激活函数！）
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)
```

---

## 六、现代应用与实战技巧

### 6.1 迁移学习（Transfer Learning）

```python
import torchvision.models as models

# 1. 加载预训练模型
model = models.resnet50(pretrained=True)

# 2. 冻结主干网络（只训练分类头）
for param in model.parameters():
    param.requires_grad = False

# 3. 替换分类头
model.fc = nn.Linear(2048, num_classes)  # 只有 fc 层有梯度

# 4. 微调模式（解冻部分层）
for param in model.layer4.parameters():
    param.requires_grad = True

# 5. 不同层用不同学习率（推荐做法）
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```

### 6.2 数据增强（Data Augmentation）

```python
from torchvision import transforms

# 训练集增强（随机性强）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),  # ImageNet 均值/标准差
])

# 测试集（只做确定性变换）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
```

### 6.3 特征图可视化

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 方法1：hook 提取中间层特征图
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.resnet18(pretrained=True)
model.layer1.register_forward_hook(get_activation('layer1'))

x = torch.randn(1, 3, 224, 224)
model(x)

feat = activation['layer1']  # (1, 64, 56, 56)
# 可视化前 16 个通道
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(feat[0, i].numpy(), cmap='viridis')
    ax.axis('off')
plt.tight_layout()

# 方法2：可视化卷积核本身
conv1_weights = model.conv1.weight.data  # (64, 3, 7, 7)
# 取第一个卷积核
kernel = conv1_weights[0].permute(1, 2, 0)  # (7, 7, 3)
plt.imshow((kernel - kernel.min()) / (kernel.max() - kernel.min()))
```

### 6.4 梯度加权类激活映射（Grad-CAM）

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam  # 热力图，叠加到原图即可

# 使用
grad_cam = GradCAM(model, model.layer4[-1])
heatmap = grad_cam(x, class_idx=243)
```

### 6.5 常见训练技巧

**学习率调度：**

```python
# 余弦退火（最常用）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 带热重启的余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# OneCycleLR（超收敛，训练快速）
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01,
    steps_per_epoch=len(train_loader), epochs=30
)
```

**混合精度训练（AMP）：**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()
    with autocast():                   # 自动选择 fp16/fp32
        outputs = model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()      # 缩放梯度防止下溢
    scaler.step(optimizer)
    scaler.update()
```

**Label Smoothing（防止过自信）：**

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# 硬标签 [0,0,1,...] → 软标签 [0.01,0.01,0.91,...,0.01]
```

---

## 七、易错点总结

| 错误 | 正确做法 |
|------|---------|
| Conv 后接 BN 时设置 bias=True | 应设 bias=False，BN 的 beta 替代 bias |
| 测试时忘记 model.eval() | 必须 model.eval() 关闭 BN/Dropout 的训练模式 |
| 特征图 shape 混乱 | 记住：(B, C, H, W)，通道在第2维 |
| 感受野不够大 | 增加深度或用 dilation，而不是单纯用大卷积核 |
| ConvTranspose2d 棋盘格 | 改用 Upsample(bilinear) + Conv2d |
| 对同一 batch 用不同 BN | 确认 train/eval 切换正确 |
| stride > 1 时用 padding='same' | PyTorch 不支持，需手动计算 padding |

---

*文档涵盖：历史背景 → 数学推导 → 超参数 → PyTorch API → 经典网络 → 现代应用*
