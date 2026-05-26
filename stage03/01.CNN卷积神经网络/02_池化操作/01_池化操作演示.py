# -*- coding: utf-8 -*-
"""
池化操作演示
- 最大池化 vs 平均池化的区别
- 池化的作用：降维 + 保留显著特征 + 增大感受野
- stride和kernel_size对输出尺寸的影响
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(__file__), '../../../../base/images')


# ============================================================
# 1. 手动实现池化
# ============================================================
def max_pool2d_manual(input_matrix, kernel_size=2, stride=2):
    """
    手动实现最大池化
    """
    H, W = input_matrix.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = input_matrix[i*stride:i*stride+kernel_size,
                                  j*stride:j*stride+kernel_size]
            output[i, j] = np.max(region)

    return output


def avg_pool2d_manual(input_matrix, kernel_size=2, stride=2):
    """
    手动实现平均池化
    """
    H, W = input_matrix.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = input_matrix[i*stride:i*stride+kernel_size,
                                  j*stride:j*stride+kernel_size]
            output[i, j] = np.mean(region)

    return output


# ============================================================
# 2. 池化过程可视化演示
# ============================================================
def demo_pooling_visual():
    """直观展示池化的降维过程"""
    print("=" * 60)
    print("池化操作演示")
    print("=" * 60)

    # 4×4 输入
    input_matrix = np.array([
        [1, 3, 2, 1],
        [4, 6, 5, 2],
        [7, 3, 9, 4],
        [2, 1, 6, 8],
    ], dtype=np.float32)

    print(f"输入 (4×4):\n{input_matrix}")

    # 最大池化 2×2, stride=2
    max_out = max_pool2d_manual(input_matrix, kernel_size=2, stride=2)
    print(f"\n最大池化 2×2 (输出 2×2):\n{max_out}")
    print("  每个区域取最大值 → 保留最显著的特征")

    # 平均池化 2×2, stride=2
    avg_out = avg_pool2d_manual(input_matrix, kernel_size=2, stride=2)
    print(f"\n平均池化 2×2 (输出 2×2):\n{avg_out}")
    print("  每个区域取平均值 → 保留整体趋势")

    print(f"\n对比:")
    print(f"  最大池化保留: {max_out.flatten()}  ← 突出强特征")
    print(f"  平均池化保留: {avg_out.flatten()}  ← 保留整体信息")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(input_matrix, cmap='YlOrRd', vmin=0, vmax=10)
    axes[0].set_title('输入 (4×4)')
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{input_matrix[i,j]:.0f}', ha='center', va='center')

    axes[1].imshow(max_out, cmap='YlOrRd', vmin=0, vmax=10)
    axes[1].set_title('Max Pooling (2×2)')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{max_out[i,j]:.0f}', ha='center', va='center')

    axes[2].imshow(avg_out, cmap='YlOrRd', vmin=0, vmax=10)
    axes[2].set_title('Avg Pooling (2×2)')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, f'{avg_out[i,j]:.1f}', ha='center', va='center')

    for ax in axes:
        ax.axis('off')
    plt.suptitle('池化操作对比', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '09_CNN_池化操作对比.png'), dpi=150)
    plt.show()


# ============================================================
# 3. 池化在卷积后的实际效果
# ============================================================
def demo_conv_then_pool():
    """卷积 → 池化的完整流程演示"""
    print("\n" + "=" * 60)
    print("卷积 + 池化完整流程")
    print("=" * 60)

    # 生成测试图：8×8，带边缘
    img = np.zeros((8, 8), dtype=np.float32)
    img[:, 4:] = 1.0  # 左暗右亮，中间有纵向边缘

    # 纵向边缘检测卷积核
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=np.float32)

    # 卷积
    conv_out = np.zeros((6, 6), dtype=np.float32)
    for i in range(6):
        for j in range(6):
            conv_out[i, j] = np.sum(img[i:i+3, j:j+3] * kernel)

    # 池化
    pool_out = max_pool2d_manual(conv_out, kernel_size=2, stride=2)

    print(f"原图 (8×8) → 卷积后 (6×6) → 池化后 (3×3)")
    print(f"  尺寸变化: 8→6→3, 逐步压缩")
    print(f"  池化保留卷积检测到的最强边缘信号")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title('原图 8×8')
    axes[1].imshow(conv_out, cmap='gray'); axes[1].set_title('卷积后 6×6')
    axes[2].imshow(pool_out, cmap='gray'); axes[2].set_title('池化后 3×3')
    for ax in axes:
        ax.axis('off')
    plt.suptitle('卷积 + 池化', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '10_CNN_卷积加池化流程.png'), dpi=150)
    plt.show()


# ============================================================
# 4. PyTorch池化操作
# ============================================================
def demo_pytorch_pooling():
    """PyTorch池化API使用"""
    print("\n" + "=" * 60)
    print("PyTorch池化API")
    print("=" * 60)

    x = torch.randn(1, 1, 4, 4)
    print(f"输入形状: {x.shape}")

    # 最大池化
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_out = max_pool(x)
    print(f"MaxPool2d(2,2) 输出形状: {max_out.shape}")

    # 平均池化
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_out = avg_pool(x)
    print(f"AvgPool2d(2,2) 输出形状: {avg_out.shape}")

    # 全局平均池化（常用于替代全连接层）
    global_pool = nn.AdaptiveAvgPool2d(1)  # 输出固定为 1×1
    global_out = global_pool(x)
    print(f"AdaptiveAvgPool2d(1) 输出形状: {global_out.shape}")
    print(f"  → 全局平均池化：不管输入多大，输出都是1×1")
    print(f"  → 常用于ResNet等网络最后一层，替代展平+全连接")


# ============================================================
# 5. 池化的三大作用总结
# ============================================================
def demo_pooling_effect():
    """展示池化的三大作用"""
    print("\n" + "=" * 60)
    print("池化的三大作用")
    print("=" * 60)

    print("""
1. 降维（减少计算量）
   224×224 → 池化 → 112×112，面积缩小4倍，后续计算量大幅减少

2. 保留显著特征（平移鲁棒性）
   同一个特征在2×2区域内稍微偏移，最大池化结果不变
   → 提供了微小的平移不变性

3. 增大感受野
   浅层卷积核只看3×3区域
   → 池化后尺寸减半，同样3×3卷积核在原图上看到的区域翻倍
   → 层层叠加，深层神经元"看"到的原图区域越来越大

   示例（3×3卷积 + 2×2池化 堆叠）：
   层1: 卷积3×3 → 感受野 3×3
   层2: 池化2×2 → 等效感受野 5×5  (3 + (3-1)×1)
   层3: 卷积3×3 → 等效感受野 7×7
   层4: 池化2×2 → 等效感受野 14×14
   层5: 卷积3×3 → 等效感受野 18×18
   → 越深看得越广！
""")


if __name__ == '__main__':
    demo_pooling_visual()
    demo_conv_then_pool()
    demo_pytorch_pooling()
    demo_pooling_effect()
