# ============================================================
# 文件: 03_张量创建API.py
# 主题: PyTorch 张量创建 API 精讲
# 涵盖: torch.tensor / zeros / ones / full / rand / randn /
#        randint / arange / linspace / eye / empty
# 说明: 每个 API 独立封装为函数，含参数说明和输出形状
# ============================================================

import torch
SEP = "=" * 50

def demo_tensor():
    """
    torch.tensor: 从 Python 列表/数组手动构造张量
    - data: 输入数据（列表、元组、NumPy数组）
    - dtype: 指定数据类型（默认从数据推断）
    - requires_grad: 是否追踪梯度
    """
    print(SEP)
    print("1. torch.tensor --- 从数据手动构造张量")
    print(SEP)

    t1 = torch.tensor([1, 2, 3])
    print(f"  从列表创建: {t1}, dtype={t1.dtype}, shape={t1.shape}")

    t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"  从嵌套列表创建 2D shape={t2.shape}:")
    print(f"  {t2}")

    t3 = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(f"  指定 dtype=float32: {t3}")

    t4 = torch.tensor([1.0, 2.0], requires_grad=True)
    print(f"  requires_grad=True: {t4}")

def demo_zeros_ones_full():
    """
    torch.zeros: 全零张量
    torch.ones:  全一张量
    torch.full:  指定填充值的张量
    参数: size/shape 元组, dtype, device
    """
    print(SEP)
    print("2. torch.zeros / ones / full --- 常数填充张量")
    print(SEP)

    z = torch.zeros(2, 3)
    print(f"  zeros(2,3) shape={z.shape}:")
    print(f"  {z}")

    o = torch.ones(3, 4, dtype=torch.int32)
    print(f"  ones(3,4) dtype=int32 shape={o.shape}:")
    print(f"  {o}")

    fl = torch.full((2, 2), fill_value=7.0)
    print(f"  full((2,2), fill_value=7.0) shape={fl.shape}:")
    print(f"  {fl}")

def demo_rand_randn_randint():
    """
    torch.rand:    均匀分布随机张量 [0, 1)
    torch.randn:   标准正态分布随机张量 N(0,1)
    torch.randint: 整数随机张量 [low, high)
    """
    print(SEP)
    print("3. torch.rand / randn / randint --- 随机张量")
    print(SEP)

    torch.manual_seed(42)
    r = torch.rand(2, 3)
    print(f"  rand(2,3) 均匀分布[0,1) shape={r.shape}:")
    print(f"  {r}")

    rn = torch.randn(2, 3)
    print(f"  randn(2,3) 标准正态N(0,1) shape={rn.shape}:")
    print(f"  {rn}")

    ri = torch.randint(low=0, high=10, size=(2, 4))
    print(f"  randint(0,10, size=(2,4)) shape={ri.shape}:")
    print(f"  {ri}")

def demo_arange_linspace():
    """
    torch.arange:   等差序列（类似 Python range，步长可浮点）
      arange(start, end, step) — 左闭右开
    torch.linspace: 线性等间隔序列
      linspace(start, end, steps) — 两端均闭，steps 为点数
    区别: arange 指定步长; linspace 指定点数
    """
    print(SEP)
    print("4. torch.arange / linspace --- 序列张量")
    print(SEP)

    a1 = torch.arange(0, 10, 2)
    print(f"  arange(0,10,2)={a1}")

    a2 = torch.arange(0.0, 1.0, 0.2)
    print(f"  arange(0.0,1.0,0.2)={a2}")

    ls = torch.linspace(0, 1, steps=5)
    print(f"  linspace(0,1,steps=5)={ls}")

    import math
    angles = torch.linspace(0, 2*math.pi, steps=7)
    print(f"  linspace(0,2pi,7)={angles.round(decimals=3)}")

def demo_eye():
    """
    torch.eye: 单位矩阵 / 对角矩阵
      n: 行数; m: 列数（默认=n）
    常见应用: one-hot 编码、初始化权重、残差跳连初始化
    """
    print(SEP)
    print("5. torch.eye --- 单位矩阵")
    print(SEP)

    e1 = torch.eye(3)
    print("  eye(3) 3x3 单位矩阵:")
    print(f"  {e1}")

    e2 = torch.eye(3, 5)
    print("  eye(3,5) 3x5 非方形:")
    print(f"  {e2}")

    labels = torch.tensor([0, 2, 1])
    onehot = torch.eye(3)[labels]
    print(f"  labels={labels.tolist()} one-hot:")
    print(f"  {onehot}")

def demo_empty():
    """
    torch.empty: 未初始化张量（内存残留值，不确定）
    速度最快，适合后续立即赋值的场景
    警告: 直接使用可能出现 NaN/Inf/随机噪声
    """
    print(SEP)
    print("6. torch.empty --- 未初始化张量")
    print(SEP)

    e = torch.empty(2, 3)
    print(f"  empty(2,3) 内存残留（值不确定）shape={e.shape}:")
    print(f"  {e}")

    buf = torch.empty(1000)
    buf.fill_(0.5)
    print(f"  empty(1000).fill_(0.5) 均值={buf.mean():.4f} (应为0.5)")


def demo_like_apis():
    """
    *_like 系列: 根据已有张量的 shape/dtype/device 创建新张量
      zeros_like / ones_like / rand_like / full_like
    工程常用: mask = torch.zeros_like(output)
    """
    print(SEP)
    print("7. *_like 系列 --- 依形创建")
    print(SEP)

    ref = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"  参考张量 ref shape={ref.shape}:")
    print(f"  {ref}")

    print("  zeros_like(ref):")
    print(f"  {torch.zeros_like(ref)}")
    print("  ones_like(ref):")
    print(f"  {torch.ones_like(ref)}")
    print("  rand_like(ref):")
    print(f"  {torch.rand_like(ref)}")
    print("  full_like(ref, 3.14):")
    print(f"  {torch.full_like(ref, 3.14)}")


if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    demo_tensor()
    demo_zeros_ones_full()
    demo_rand_randn_randint()
    demo_arange_linspace()
    demo_eye()
    demo_empty()
    demo_like_apis()
    print(SEP)
    print("所有张量创建 API 演示完毕！")
    print(SEP)