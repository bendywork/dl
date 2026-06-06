# ============================================================
# 文件: 07_数学运算API.py
# 主题: PyTorch 数学运算 API 精讲
# 涵盖:
#   - 逐元素运算: add/sub/mul/div/pow/sqrt/abs/exp/log/log2/log10
#   - 归约运算:   sum/mean/max/min/std/var（含 dim 参数）
#   - 矩阵运算:   matmul/@/dot/mm/bmm
#   - 比较运算:   eq/ne/gt/lt/ge/le/torch.max(返回值与indices)
#   - 其他实用:   clamp/round/floor/ceil/sign
# ============================================================

import torch

SEP = "=" * 58
def demo_elementwise_ops():
    """
    逐元素运算: 对张量中每个元素独立执行操作，输出 shape 不变
    函数形式  t.add(x)  等价于  torch.add(t, x)  等价于  t + x
    带下划线的版本 add_() 是 原地操作，节省内存但会修改原张量
    """
    print(SEP)
    print("1. 逐元素运算: add/sub/mul/div/pow/sqrt/abs/exp/log")
    print(SEP)

    t = torch.tensor([1.0, 4.0, 9.0, 16.0])
    print(f"  t = {t}")

    print(f"  t + 1        = {t + 1}")
    print(f"  t - 2        = {t - 2}")
    print(f"  t * 3        = {t * 3}")
    print(f"  t / 2        = {t / 2}")
    print(f"  t ** 2       = {t ** 2}   (pow)")
    print(f"  torch.sqrt(t)= {torch.sqrt(t)}")
    print(f"  torch.abs(-t)= {torch.abs(-t)}")

    t2 = torch.tensor([1.0, 2.0, 3.0])
    print(f"  t2 = {t2}")
    print(f"  torch.exp(t2) = {torch.exp(t2).round(decimals=4)}   e^x")
    print(f"  torch.log(t2) = {torch.log(t2).round(decimals=4)}   ln(x)")
    print(f"  torch.log2(t2)= {torch.log2(t2).round(decimals=4)}  log2(x)")
    print(f"  torch.log10(t2)={torch.log10(t2).round(decimals=4)} log10(x)")

    # 原地操作 (in-place, 带下划线)
    t3 = torch.tensor([1.0, 2.0, 3.0])
    t3.mul_(10)  # 原地乘以10，不产生新张量
    print(f"  t3.mul_(10) 原地: {t3}  (注意: 有梯度的叶子节点不可用原地操作)")

def demo_reduction_ops():
    """
    归约运算: 将多个元素归约为一个（或更少）
    关键参数 dim: 沿哪个维度归约
      不指定 dim -> 全局归约，返回标量
      指定 dim   -> 沿该维度归约，输出少一个维度
    keepdim=True -> 保持维度数（被归约的维度变为1）
    """
    print(SEP)
    print("2. 归约运算: sum/mean/max/min/std/var（含 dim 参数）")
    print(SEP)

    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"  t shape={t.shape}:")
    print(f"  {t}")

    print(f"  sum()        = {t.sum():.1f}         全局求和")
    print(f"  sum(dim=0)   = {t.sum(dim=0)}  沿行归约->每列之和")
    print(f"  sum(dim=1)   = {t.sum(dim=1)}       沿列归约->每行之和")
    print(f"  mean()       = {t.mean():.4f}     全局均值")
    print(f"  mean(dim=0)  = {t.mean(dim=0)}  各列均值")
    print(f"  std(dim=1)   = {t.std(dim=1).round(decimals=4)}  各行标准差")
    print(f"  var(dim=1)   = {t.var(dim=1).round(decimals=4)}    各行方差")

    # max/min 全局版: 返回单个值
    print(f"  max()        = {t.max():.1f}   min()={t.min():.1f}")

    # keepdim 演示
    s_keep = t.sum(dim=1, keepdim=True)
    print(f"  sum(dim=1, keepdim=True) shape={s_keep.shape}:")
    print(f"  {s_keep}")
    print("  (keepdim 常用于广播: 如 (t - mean) / std 需要维度对齐)")

def demo_matmul_ops():
    """
    矩阵运算（维度要求严格，易出错）:
    torch.dot(a, b)   : 1D点积，返回标量，两向量长度须相同
    torch.mm(A, B)    : 2D矩阵乘法，shape (m,k)x(k,n)->(m,n)
    torch.matmul(A,B) : 通用矩阵乘法，自动广播（最推荐）
    A @ B             : matmul 的运算符简写
    torch.bmm(A, B)   : 批量矩阵乘法，(b,m,k)x(b,k,n)->(b,m,n)
                        两个 batch 维必须相同，不支持广播
    """
    print(SEP)
    print("3. 矩阵运算: matmul / @ / dot / mm / bmm")
    print(SEP)

    # dot: 1D 向量点积
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])
    print(f"  dot({v1.tolist()}, {v2.tolist()}) = {torch.dot(v1,v2):.1f}  (1*4+2*5+3*6=32)")

    # mm: 严格 2D 矩阵乘法
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = torch.mm(A, B)
    print(f"  mm: ({A.shape}) x ({B.shape}) -> {C.shape}")

    # matmul / @: 通用，支持广播
    D = torch.randn(2, 3, 4)
    E = torch.randn(2, 4, 5)
    F = torch.matmul(D, E)
    print(f"  matmul: {D.shape} x {E.shape} -> {F.shape}")
    print(f"  @ 运算符等价: {(D @ E).shape}")

    # bmm: 批量 2D 矩阵乘法（不支持广播，batch 维必须相等）
    batch_A = torch.randn(8, 3, 4)
    batch_B = torch.randn(8, 4, 5)
    batch_C = torch.bmm(batch_A, batch_B)
    print(f"  bmm: {batch_A.shape} x {batch_B.shape} -> {batch_C.shape}")
    print("  注意: bmm 不支持广播，两 batch 维必须完全相等")

def demo_comparison_ops():
    """
    比较运算: 返回 bool 类型张量（逐元素）
    eq / ne / gt / lt / ge / le 可用运算符 == != > < >= <=
    torch.max(t, dim) 带 dim 参数时返回 (values, indices)
      indices 即 argmax，常用于获取分类预测结果
    """
    print(SEP)
    print("4. 比较运算: eq/ne/gt/lt/ge/le 及 torch.max 返回 indices")
    print(SEP)

    t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    print(f"  t = {t.tolist()}")

    print(f"  t == 1   : {(t == 1).tolist()}   (eq)")
    print(f"  t != 1   : {(t != 1).tolist()}   (ne)")
    print(f"  t > 3    : {(t > 3).tolist()}    (gt)")
    print(f"  t < 4    : {(t < 4).tolist()}    (lt)")
    print(f"  t >= 4   : {(t >= 4).tolist()}   (ge)")
    print(f"  t <= 3   : {(t <= 3).tolist()}   (le)")

    # torch.max 两种用法
    max_val = torch.max(t)               # 不带 dim: 返回最大值标量
    print(f"  torch.max(t) = {max_val}")

    vals, idxs = torch.max(t, dim=0)     # 带 dim: 返回 (values, indices)
    print(f"  torch.max(t, dim=0): value={vals}, index={idxs}")

    # 分类预测中的常见用法
    logits = torch.tensor([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)    # 等价于 torch.max(probs,1).indices
    print(f"  logits shape={logits.shape}")
    print(f"  argmax(dim=1) -> 预测类别: {preds.tolist()}  (0=类0, 1=类1, 2=类2)")

def demo_utility_ops():
    """
    实用数学工具:
    clamp(min, max): 将值截断在 [min, max] 范围内（梯度截断常用）
    round():         四舍五入到最近整数
    floor():         向下取整
    ceil():          向上取整
    sign():          返回符号 (-1/0/+1)
    """
    print(SEP)
    print("5. 实用工具: clamp / round / floor / ceil / sign")
    print(SEP)

    t = torch.tensor([-3.7, -0.5, 0.0, 0.5, 1.4, 2.6, 10.0])
    print(f"  t = {t.tolist()}")

    clamped = torch.clamp(t, min=-2.0, max=2.0)
    print(f"  clamp(-2,2) = {clamped.tolist()}  (超出范围的值被截断)")
    print("  工程用途: 梯度截断防止梯度爆炸 grad.clamp(-1, 1)")

    rounded = torch.round(t)
    print(f"  round()   = {rounded.tolist()}")

    floored = torch.floor(t)
    print(f"  floor()   = {floored.tolist()}")

    ceiled = torch.ceil(t)
    print(f"  ceil()    = {ceiled.tolist()}")

    signed = torch.sign(t)
    print(f"  sign()    = {signed.tolist()}  (-1/0/1)")

    # clamp 只设置单边
    relu_manual = torch.clamp(t, min=0)  # 等价于 ReLU
    print(f"  clamp(min=0) = {relu_manual.tolist()}  (手动 ReLU)")


if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    demo_elementwise_ops()
    demo_reduction_ops()
    demo_matmul_ops()
    demo_comparison_ops()
    demo_utility_ops()
    print(SEP)
    print("所有数学运算 API 演示完毕！")
    print(SEP)