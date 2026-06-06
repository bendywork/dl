# ============================================================
# 文件: 04_张量属性与转换.py
# 主题: PyTorch 张量属性访问 + 类型转换 精讲
# 涵盖: .shape/.dtype/.device/.requires_grad/.ndim/.numel()
#        .item()/.tolist()/.numpy()/from_numpy()
#        .to(dtype=)/.to(device=)/.float()/.long()/.bool()
# 重点: tensor() vs from_numpy() 内存共享区别
# ============================================================

import torch
import numpy as np

SEP = "=" * 55
def demo_basic_properties():
    """
    张量基本属性（只读，不会触发计算）
    .shape       -> torch.Size 对象，可下标访问 t.shape[0]
    .dtype       -> 元素数据类型（torch.float32 等）
    .device      -> 所在设备（cpu / cuda:0 等）
    .requires_grad -> 是否被自动微分追踪
    .ndim        -> 张量维度数（等价于 len(t.shape)）
    .numel()     -> 张量元素总数（shape 所有维度之积）
    """
    print(SEP)
    print("1. 基本属性: shape / dtype / device / requires_grad / ndim / numel")
    print(SEP)
    t = torch.randn(3, 4, 5)
    print(f"  t = torch.randn(3, 4, 5)")
    print(f"  shape      = {t.shape}       <- torch.Size 对象")
    print(f"  dtype      = {t.dtype}    <- 默认 float32")
    print(f"  device     = {t.device}          <- 当前在 CPU")
    print(f"  requires_grad = {t.requires_grad}    <- 未开启梯度")
    print(f"  ndim       = {t.ndim}           <- 3维张量")
    print(f"  numel()    = {t.numel()}         <- 3*4*5=60 个元素")
    print(f"  shape[0]   = {t.shape[0]}           <- 可下标访问")

def demo_item_tolist():
    """
    .item()   -> 将标量张量转成 Python 原生数值（float/int）
                 只能用于单个元素（numel==1）的张量
    .tolist() -> 将张量转成 Python 嵌套列表（任意形状）
    常见用法: loss.item() 获取损失值用于打印/记录
    """
    print(SEP)
    print("2. .item() / .tolist() --- 转 Python 原生类型")
    print(SEP)
    scalar = torch.tensor(3.14)
    print(f"  scalar tensor = {scalar}")
    print(f"  .item() = {scalar.item()}  type={type(scalar.item())}")

    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"  2D tensor shape={t.shape}")
    print(f"  .tolist() = {t.tolist()}")
    print(f"  第一行    = {t[0].tolist()}")

    # 工程常用: 记录 loss 时用 .item()")
    fake_loss = torch.tensor(0.2345, requires_grad=True)
    log_val = fake_loss.item()
    print(f"  loss={fake_loss}, log_val={log_val:.4f} (type={type(log_val).__name__})")

def demo_numpy_conversion():
    """
    tensor.numpy()      -> 将 CPU 张量转成 NumPy 数组（共享内存！）
    torch.from_numpy()  -> 将 NumPy 数组转成 Tensor（共享内存！）
    torch.tensor(arr)   -> 从 NumPy 数组创建张量（复制数据，不共享）

    内存共享意味着:
      修改一方会影响另一方
      适合零拷贝数据交换，但要注意副作用
    """
    print(SEP)
    print("3. tensor.numpy() vs from_numpy() --- 内存共享")
    print(SEP)

    # --- from_numpy: 共享内存 ---
    arr = np.array([1.0, 2.0, 3.0])
    t_shared = torch.from_numpy(arr)
    print("  [from_numpy] 共享内存演示:")
    print(f"  arr={arr}, t_shared={t_shared}")
    arr[0] = 99.0   # 修改 numpy 数组
    print(f"  arr[0]=99 后: arr={arr}, t_shared={t_shared} (同步变化!)")

    # --- tensor(): 复制数据 ---
    arr2 = np.array([1.0, 2.0, 3.0])
    t_copy = torch.tensor(arr2)    # 会拷贝数据
    arr2[0] = 99.0
    print("  [torch.tensor()] 复制数据演示:")
    print(f"  arr2[0]=99 后: arr2={arr2}, t_copy={t_copy} (t_copy 不变!)")

    # --- .numpy() ---
    t2 = torch.ones(3)
    np_arr = t2.numpy()
    t2[1] = 88.0
    print("  [.numpy()] 共享内存演示:")
    print(f"  t2[1]=88 后: t2={t2}, np_arr={np_arr} (同步!)")

    # cuda tensor 不能直接 .numpy()")
    print("  注意: GPU 张量需先 .cpu() 再 .numpy()")

def demo_dtype_conversion():
    """
    类型转换三种方式（等价）:
    1. tensor.to(dtype=torch.float32)  最通用，推荐写法
    2. tensor.float()   -> float32 简写
    3. tensor.long()    -> int64 简写
    4. tensor.bool()    -> bool 简写
    5. tensor.half()    -> float16 简写（混合精度训练常用）
    """
    print(SEP)
    print("4. dtype 转换: .to(dtype) / .float() / .long() / .bool()")
    print(SEP)

    t = torch.tensor([1, 0, 2, 0, 3])
    print(f"  原始: {t}  dtype={t.dtype}")

    tf = t.float()
    print(f"  .float()  -> dtype={tf.dtype}: {tf}")

    tl = t.long()
    print(f"  .long()   -> dtype={tl.dtype}: {tl}")

    tb = t.bool()
    print(f"  .bool()   -> dtype={tb.dtype}: {tb}")

    tf2 = t.to(dtype=torch.float16)
    print(f"  .to(float16) -> dtype={tf2.dtype}: {tf2}")

    # 常见场景: 分类标签需要 long，损失函数才能接受
    labels = torch.tensor([0.0, 1.0, 2.0])  # float
    labels_long = labels.long()
    print(f"  labels.long() for CrossEntropy: {labels_long} dtype={labels_long.dtype}")

def demo_device_conversion():
    """
    tensor.to(device) / tensor.cuda() / tensor.cpu()
    - 模型和数据必须在同一设备才能运算
    - 推荐写法: device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    print(SEP)
    print("5. device 转换: .to(device) / .cpu() / .cuda()")
    print(SEP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  当前设备: {device}")

    t = torch.tensor([1.0, 2.0, 3.0])
    print(f"  原始: device={t.device}")

    t_dev = t.to(device)
    print(f"  .to(device) -> device={t_dev.device}")

    t_cpu = t_dev.cpu()
    print(f"  .cpu()      -> device={t_cpu.device}")

    # 多 GPU 时指定编号
    print("  多GPU写法: t.to(device=torch.device('cuda:0'))")
    print("  注意: GPU Tensor 不能直接 .numpy()，需先 .cpu()")


if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    demo_basic_properties()
    demo_item_tolist()
    demo_numpy_conversion()
    demo_dtype_conversion()
    demo_device_conversion()
    print(SEP)
    print("所有张量属性与转换演示完毕！")
    print(SEP)