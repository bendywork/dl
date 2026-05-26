# -*- coding: utf-8 -*-
"""
卷积操作基础演示
- 手动实现2D卷积，理解滑动扫描 + 点乘求和的过程
- 对比不同卷积核提取的不同特征（边缘、锐化、模糊）
- 验证PyTorch nn.Conv2d与手动实现结果一致
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. 手动实现2D卷积
# ============================================================
def conv2d_manual(input_matrix, kernel, stride=1, padding=0):
    """
    手动实现2D卷积（实际为互相关运算，与PyTorch一致）
    :param input_matrix: 输入矩阵 [H, W]
    :param kernel: 卷积核 [kH, kW]
    :param stride: 步长
    :param padding: 填充圈数
    :return: 输出特征图 [out_H, out_W]
    """
    if padding > 0:
        input_matrix = np.pad(input_matrix, padding, mode='constant')

    H, W = input_matrix.shape
    kH, kW = kernel.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = input_matrix[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# 2. 经典卷积核定义
# ============================================================
FILTERS = {
    '横向边缘检测': np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32),

    '纵向边缘检测': np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32),

    '锐化': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32),

    '模糊(均值)': np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32),

    '拉普拉斯(二阶边缘)': np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32),
}


# ============================================================
# 3. 在简单矩阵上演示卷积过程
# ============================================================
def demo_simple_matrix():
    """用小矩阵直观展示卷积滑动过程"""
    print("=" * 60)
    print("简单矩阵卷积演示")
    print("=" * 60)

    input_matrix = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ], dtype=np.float32)

    kernel = FILTERS['纵向边缘检测']

    print(f"输入矩阵:\n{input_matrix}")
    print(f"\n卷积核(纵向边缘检测):\n{kernel}")

    # 逐步展示滑动过程
    kH, kW = kernel.shape
    for i in range(3):
        for j in range(3):
            region = input_matrix[i:i+kH, j:j+kW]
            result = np.sum(region * kernel)
            print(f"\n位置({i},{j}): 取区域\n{region}")
            print(f"点乘求和 = {result}")

    # 完整输出
    output = conv2d_manual(input_matrix, kernel)
    print(f"\n完整特征图:\n{output}")


# ============================================================
# 4. 在真实图片上演示不同卷积核效果
# ============================================================
def demo_image_convolution():
    """在图片上展示不同卷积核提取的特征"""
    print("\n" + "=" * 60)
    print("图片卷积演示——不同卷积核提取不同特征")
    print("=" * 60)

    # 生成一个简单的测试图案（渐变+边缘）
    img = np.zeros((64, 64), dtype=np.float32)
    # 左半边暗，右半边亮 → 有纵向边缘
    img[:, 32:] = 1.0
    # 上半边加一点渐变 → 有横向边缘
    for i in range(32):
        img[i, :] += i / 32.0 * 0.5

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # 原图
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('原图')
    axes[0, 0].axis('off')

    # 不同卷积核效果
    for idx, (name, kernel) in enumerate(FILTERS.items()):
        if idx >= 5:
            break
        output = conv2d_manual(img, kernel)
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(output, cmap='gray')
        axes[row, col].set_title(name)
        axes[row, col].axis('off')

    plt.suptitle('不同卷积核提取的特征对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('conv_filters_demo.png', dpi=150)
    plt.show()
    print("图片已保存: conv_filters_demo.png")


# ============================================================
# 5. 验证手动实现与PyTorch一致
# ============================================================
def demo_pytorch_match():
    """验证手动卷积与PyTorch Conv2d输出一致"""
    print("\n" + "=" * 60)
    print("验证：手动实现 vs PyTorch nn.Conv2d")
    print("=" * 60)

    input_np = np.random.randn(1, 1, 5, 5).astype(np.float32)
    kernel_np = FILTERS['纵向边缘检测']

    # 手动实现
    manual_out = conv2d_manual(input_np[0, 0], kernel_np)

    # PyTorch实现
    conv_layer = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    conv_layer.weight.data = torch.tensor(kernel_np.reshape(1, 1, 3, 3))
    input_tensor = torch.tensor(input_np)
    pytorch_out = conv_layer(input_tensor).detach().numpy()[0, 0]

    print(f"手动实现输出:\n{manual_out}")
    print(f"PyTorch输出:\n{pytorch_out}")
    print(f"差异: {np.max(np.abs(manual_out - pytorch_out)):.6f}")
    assert np.allclose(manual_out, pytorch_out, atol=1e-5), "结果不一致！"
    print("✅ 验证通过：手动实现与PyTorch输出一致")


# ============================================================
# 6. 多通道卷积演示（RGB图像）
# ============================================================
def demo_rgb_convolution():
    """RGB三通道卷积：卷积核深度必须匹配输入通道数"""
    print("\n" + "=" * 60)
    print("多通道(RGB)卷积演示")
    print("=" * 60)

    # 模拟一张 5×5 的 RGB 图片
    rgb_input = np.random.randn(1, 3, 5, 5).astype(np.float32)

    # 卷积核形状: [out_channels, in_channels, kH, kW]
    # 1个卷积核，3个输入通道（R/G/B各一个），3×3
    # 每个通道的核可以不同，最终3个通道的点乘结果求和
    kernel = np.array([
        # R通道核
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        # G通道核
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        # B通道核
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    ], dtype=np.float32).reshape(1, 3, 3, 3)

    conv_layer = nn.Conv2d(3, 1, kernel_size=3, bias=False)
    conv_layer.weight.data = torch.tensor(kernel)
    input_tensor = torch.tensor(rgb_input)
    output = conv_layer(input_tensor).detach().numpy()

    print(f"输入形状: {rgb_input.shape}  (1张图, 3通道, 5×5)")
    print(f"卷积核形状: {kernel.shape}  (1个核, 3通道深度, 3×3)")
    print(f"输出形状: {output.shape}  (1张图, 1个特征图, 3×3)")
    print(f"\n计算过程:")
    print(f"  对R通道做3×3点乘 → 得到一个值")
    print(f"  对G通道做3×3点乘 → 得到一个值")
    print(f"  对B通道做3×3点乘 → 得到一个值")
    print(f"  三个值求和 → 输出特征图的一个像素")
    print(f"\n  → 64个卷积核就输出64张特征图叠在一起")


if __name__ == '__main__':
    demo_simple_matrix()
    demo_image_convolution()
    demo_pytorch_match()
    demo_rgb_convolution()
