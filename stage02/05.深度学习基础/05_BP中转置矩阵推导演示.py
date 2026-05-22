"""
验证反向传播中 W^T 的出现
核心：从标量展开 -> 矩阵形式，亲眼看到转置是怎么来的
"""
import torch
import numpy as np

# ============================================================
# 第一部分：手动展开标量链式法则，亲眼看到转置出现
# ============================================================
print("=" * 60)
print("第一部分：标量展开 -> 矩阵形式")
print("=" * 60)

W = np.array([[1., 2.],   # W: 2×2 (m=2, n=2)
              [3., 4.]])
x = np.array([5., 6.])    # x: 2×1 (n=2)

# 前向：y = W @ x
y = W @ x
print(f"\nW =\n{W}")
print(f"x = {x}")
print(f"y = W @ x = {y}")  # [17., 39.]

# 现在手动展开标量形式
# y₁ = w₁₁·x₁ + w₁₂·x₂ = 1·5 + 2·6 = 17
# y₂ = w₂₁·x₁ + w₂₂·x₂ = 3·5 + 4·6 = 39
print(f"\ny₁ = w₁₁·x₁ + w₁₂·x₂ = {W[0,0]}·{x[0]} + {W[0,1]}·{x[1]} = {y[0]}")
print(f"y₂ = w₂₁·x₁ + w₂₂·x₂ = {W[1,0]}·{x[0]} + {W[1,1]}·{x[1]} = {y[1]}")

# 假设上游梯度 ∂L/∂y
dL_dy = np.array([0.1, 0.2])
print(f"\n上游梯度 ∂L/∂y = {dL_dy}")

# 标量展开：x₁ 通过两条路影响 L
# ∂L/∂x₁ = ∂L/∂y₁ · ∂y₁/∂x₁ + ∂L/∂y₂ · ∂y₂/∂x₁
#         = ∂L/∂y₁ · w₁₁       + ∂L/∂y₂ · w₂₁
dL_dx1 = dL_dy[0] * W[0,0] + dL_dy[1] * W[1,0]  # 注意：W[1,0] 不是 W[0,1]！
dL_dx2 = dL_dy[0] * W[0,1] + dL_dy[1] * W[1,1]

print(f"\n--- 标量展开 ---")
print(f"∂L/∂x₁ = ∂L/∂y₁·w₁₁ + ∂L/∂y₂·w₂₁")
print(f"        = {dL_dy[0]}·{W[0,0]} + {dL_dy[1]}·{W[1,0]}")
print(f"        = {dL_dx1}")
print(f"∂L/∂x₂ = ∂L/∂y₁·w₁₂ + ∂L/∂y₂·w₂₂")
print(f"        = {dL_dy[0]}·{W[0,1]} + {dL_dy[1]}·{W[1,1]}")
print(f"        = {dL_dx2}")

# 矩阵形式：恰好 = Wᵀ @ ∂L/∂y
dL_dx_matrix = W.T @ dL_dy
print(f"\n--- 矩阵形式 ---")
print(f"Wᵀ @ ∂L/∂y = {dL_dx_matrix}")
print(f"标量展开结果 = [{dL_dx1}, {dL_dx2}]")
print(f"两者一致 ✓" if np.allclose(dL_dx_matrix, [dL_dx1, dL_dx2]) else "不一致 ✗")

# ============================================================
# 第二部分：非方阵（更一般的情况）
# ============================================================
print("\n" + "=" * 60)
print("第二部分：非方阵 W (3×2)，转置变成 (2×3)")
print("=" * 60)

W2 = np.array([[1., 2.],    # W2: 3×2 (m=3, n=2)
               [3., 4.],    #   3个输出，2个输入
               [5., 6.]])
x2 = np.array([7., 8.])     # x2: 2个输入

y2 = W2 @ x2
print(f"\nW (3×2) =\n{W2}")
print(f"x (2,) = {x2}")
print(f"y = W @ x (3,) = {y2}")

dL_dy2 = np.array([0.1, 0.2, 0.3])

# 手动标量展开
# ∂L/∂x₁ = dL_dy[0]·W[0,0] + dL_dy[1]·W[1,0] + dL_dy[2]·W[2,0]
#         = x₁ 通过 3 条路影响 y₁,y₂,y₃，全加起来
# ∂L/∂x₂ = dL_dy[0]·W[0,1] + dL_dy[1]·W[1,1] + dL_dy[2]·W[2,1]
dL_dx2_scalar = np.array([
    dL_dy2[0]*W2[0,0] + dL_dy2[1]*W2[1,0] + dL_dy2[2]*W2[2,0],
    dL_dy2[0]*W2[0,1] + dL_dy2[1]*W2[1,1] + dL_dy2[2]*W2[2,1],
])

# 矩阵形式
dL_dx2_matrix = W2.T @ dL_dy2

print(f"\n∂L/∂x (标量展开) = {dL_dx2_scalar}")
print(f"∂L/∂x = Wᵀ @ ∂L/∂y = {dL_dx2_matrix}")
print(f"两者一致 ✓" if np.allclose(dL_dx2_scalar, dL_dx2_matrix) else "不一致 ✗")

print(f"\n关键：W 是 3×2，Wᵀ 是 2×3")
print(f"  ∂L/∂y 是 (3,)，Wᵀ @ ∂L/∂y = (2,3) @ (3,) = (2,) = ∂L/∂x 的维度 ✓")

# ============================================================
# 第三部分：用 PyTorch 自动求导验证
# ============================================================
print("\n" + "=" * 60)
print("第三部分：PyTorch autograd 验证")
print("=" * 60)

W_t = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
x_t = torch.tensor([5., 6.], requires_grad=True)
y_t = W_t @ x_t
loss = (y_t * torch.tensor([0.1, 0.2])).sum()
loss.backward()

print(f"\nPyTorch ∂L/∂x = {x_t.grad.numpy()}")
print(f"手动   ∂L/∂x = {dL_dx_matrix}")
print(f"一致 ✓" if np.allclose(x_t.grad.numpy(), dL_dx_matrix) else "不一致 ✗")

# ============================================================
# 第四部分：维度推导——为什么必须是 Wᵀ
# ============================================================
print("\n" + "=" * 60)
print("第四部分：维度论证——为什么只能用 Wᵀ")
print("=" * 60)

print("""
前向传播：  y = W @ x
            y: (m,)  =  W: (m,n)  @  x: (n,)

反向传播：  ∂L/∂x = ?  @  ∂L/∂y
            ∂L/∂x: (n,)     ∂L/∂y: (m,)

问：? 必须是什么形状？
答：(n,) = ? @ (m,)  →  ? 必须是 (n, m)

而 W 是 (m, n)，Wᵀ 恰好是 (n, m)。

所以维度上唯一可能就是 Wᵀ。
这不是约定，是数学必然。
""")

# ============================================================
# 第五部分：Jacobian 视角（更深层的理解）
# ============================================================
print("=" * 60)
print("第五部分：Jacobian 视角")
print("=" * 60)

print("""
对于 y = Wx，Jacobian 矩阵 J = ∂y/∂x = W (m×n)

J 的第 i 行第 j 列 = ∂yᵢ/∂xⱼ = Wᵢⱼ

链式法则（向量版）：
  ∂L/∂x = Jᵀ · ∂L/∂y

为什么是 Jᵀ 而不是 J？

因为梯度是 Jacobian 的 VJP（Vector-Jacobian Product）：
  ∂L/∂xⱼ = Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂xⱼ)
           = Σᵢ vᵢ · Jᵢⱼ       (v = ∂L/∂y)
           = (Jᵀ · v)ⱼ

本质：梯度是把上游向量 v 乘到 Jacobian 的"列"上，
而矩阵乘向量默认乘"行"，所以要转置。
""")
