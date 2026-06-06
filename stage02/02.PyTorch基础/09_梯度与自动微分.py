# ==============================================================
# 文件：09_梯度与自动微分.py
# 主题：PyTorch 自动微分（Autograd）机制
# API：requires_grad / backward / grad / no_grad / detach
#       retain_graph / grad_fn / autograd.grad
# ==============================================================
import torch
import torch.nn as nn

SEP = "=" * 60

# ==============================================================
# 1. requires_grad  开启梯度追踪
# ==============================================================
print(SEP)
print("1. requires_grad")
print(SEP)
x = torch.tensor(3.0, requires_grad=True)
print("x = 3.0, requires_grad:", x.requires_grad)
# 也可以对已有张量开启
y = torch.tensor(2.0)
y.requires_grad_(True)
print("y = 2.0, requires_grad_(True):", y.requires_grad)
# 默认情况：普通张量不追踪梯度
z = torch.tensor(1.0)
print("z = 1.0 默认 requires_grad:", z.requires_grad)

# ==============================================================
# 2. grad_fn  计算图节点，记录产生该张量的运算
# ==============================================================
print()
print(SEP)
print("2. grad_fn")
print(SEP)
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2         # y = x^2
z = y * 3 + 1     # z = 3x^2 + 1
print("x.grad_fn:", x.grad_fn)     # None，叶子节点
print("y.grad_fn:", y.grad_fn)     # PowBackward
print("z.grad_fn:", z.grad_fn)     # AddBackward
# 叶子节点判断
print("x.is_leaf:", x.is_leaf)   # True
print("z.is_leaf:", z.is_leaf)   # False

# ==============================================================
# 3. backward  反向传播，计算梯度
# ==============================================================
print()
print(SEP)
print("3. backward")
print(SEP)
# 标量输出直接 backward()
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3          # y = x^3, dy/dx = 3x^2
y.backward()
print("y = x^3, x=2")
print("dy/dx = 3x^2 = 3*4 = 12, 实际:", x.grad.item())

# 多次 backward 需要 retain_graph=True
x2 = torch.tensor(3.0, requires_grad=True)
f = x2 ** 2 + 2 * x2
f.backward(retain_graph=True)   # 第一次
print("f = x^2+2x, x=3, df/dx = 2x+2 = 8, 实际:", x2.grad.item())
x2.grad.zero_()                 # 清零梯度
f.backward()                    # 第二次（已用 retain_graph）
print("再次 backward 结果:", x2.grad.item())

# 非标量输出需传入 gradient 参数（向量雅可比乘积）
x3 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y3 = x3 ** 2        # [1, 4, 9]
v = torch.ones(3)   # upstream gradient
y3.backward(v)
print("y=x^2, x=[1,2,3], 传入 v=ones(3), dy/dx:", x3.grad.tolist())

# ==============================================================
# 4. no_grad  临时关闭梯度追踪（推理阶段常用，节省内存）
# ==============================================================
print()
print(SEP)
print("4. no_grad")
print(SEP)
x = torch.tensor(2.0, requires_grad=True)
# 在 no_grad 上下文内，运算不建立计算图
with torch.no_grad():
    y = x * 3
print("with no_grad: y.requires_grad =", y.requires_grad)    # False
print("y.grad_fn =", y.grad_fn)                              # None

# 装饰器用法
@torch.no_grad()
def inference(model_out):
    return model_out.softmax(dim=-1)
out = torch.randn(2, 3, requires_grad=True)
result = inference(out)
print("@no_grad 装饰器: result.requires_grad =", result.requires_grad)

# ==============================================================
# 5. detach  从计算图中分离，共享数据但无梯度
# ==============================================================
print()
print(SEP)
print("5. detach")
print(SEP)
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y_detached = y.detach()
print("y.requires_grad:", y.requires_grad)
print("y_detached.requires_grad:", y_detached.requires_grad)
print("共享底层数据（修改 detach 会影响原张量）:", y.data_ptr() == y_detached.data_ptr())

# GAN / Encoder-Decoder 中常用：阻止梯度流入 encoder
encoder_out = torch.randn(4, 8, requires_grad=True)
stopped = encoder_out.detach()   # 阻断梯度回传
loss = stopped.sum()
# loss.backward()  # stopped.detach() 没有 grad_fn，无法 backward
print("encoder_out.grad 为 None（梯度被阻断）:", encoder_out.grad)   # None

# ==============================================================
# 6. autograd.grad  精确计算指定变量对指定输出的梯度
#    比 backward() 更灵活，常用于元学习、高阶梯度
# ==============================================================
print()
print(SEP)
print("6. autograd.grad")
print(SEP)
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x ** 2 * y + y ** 3   # z = x^2*y + y^3
# dz/dx = 2xy = 2*2*3 = 12
# dz/dy = x^2 + 3y^2 = 4 + 27 = 31
dz_dx, dz_dy = torch.autograd.grad(z, [x, y], create_graph=False)
print("z = x^2*y + y^3, x=2, y=3")
print("dz/dx = 2xy = 12, 实际:", dz_dx.item())
print("dz/dy = x^2+3y^2 = 31, 实际:", dz_dy.item())

# 高阶梯度：create_graph=True 保留计算图用于再次求导
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
dy_dx, = torch.autograd.grad(y, x, create_graph=True)
print("dy/dx = 3x^2 = 12, 实际:", dy_dx.item())
d2y_dx2, = torch.autograd.grad(dy_dx, x)
print("d^2y/dx^2 = 6x = 12, 实际:", d2y_dx2.item())

# ==============================================================
# 7. 综合示例：手动实现一步梯度下降（线性回归）
# ==============================================================
print()
print(SEP)
print("7. 综合示例：手动梯度下降（线性回归）")
print(SEP)
torch.manual_seed(42)
X = torch.randn(20, 1)
true_w, true_b = 2.5, 1.0
y = true_w * X + true_b + 0.1 * torch.randn(20, 1)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for epoch in range(50):
    pred = w * X + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

print(f"真实 w={true_w}, b={true_b}")
print(f"学到 w={w.item():.4f}, b={b.item():.4f}")
print()
print("09_梯度与自动微分.py 运行完毕!")
