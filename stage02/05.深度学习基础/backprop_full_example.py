"""
完整反向传播示例：2层网络，从输入到Loss到梯度回传
每一步都有具体数字，标出 Wᵀ 出现的位置
"""
import torch
import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("完整反向传播示例：2 层全连接网络")
print("=" * 70)

# ============================================================
# 网络结构
# ============================================================
print("""
网络结构：

    x(2,) ──→ z₁=W₁@x+b₁ ──→ h₁=ReLU(z₁) ──→ z₂=W₂@h₁+b₂ ──→ h₂=ReLU(z₂) ──→ L=½(h₂-t)²
    输入       第1层线性          第1层激活          第2层线性          第2层激活         MSE损失

参数：
    W₁: (2×2)  第1层权重
    W₂: (1×2)  第2层权重
    b₁, b₂:    偏置（设为0简化）
    σ:         ReLU
""")

# ============================================================
# 参数设置（选了好算的数字）
# ============================================================
x = np.array([1., 2.])        # 输入
W1 = np.array([[1., 2.],      # 第1层权重 (2×2)
               [3., 4.]])
b1 = np.array([0., 0.])
W2 = np.array([[5., 6.]])     # 第2层权重 (1×2)
b2 = np.array([0.])
t = np.array([50.])           # 目标值

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

# ============================================================
# 前向传播（6 步）
# ============================================================
print("=" * 70)
print("▶ 前向传播（6 步，从 x 到 L）")
print("=" * 70)

# 步骤1: 线性变换 z₁ = W₁ @ x + b₁
z1 = W1 @ x + b1
print(f"\n步骤1: z₁ = W₁ @ x + b₁")
print(f"       z₁ = {W1} @ {x} + {b1}")
print(f"       z₁ = {z1}")

# 步骤2: 激活 h₁ = ReLU(z₁)
h1 = relu(z1)
print(f"\n步骤2: h₁ = ReLU(z₁)")
print(f"       h₁ = ReLU({z1})")
print(f"       h₁ = {h1}   （z₁ > 0，所以 ReLU 不截断）")

# 步骤3: 线性变换 z₂ = W₂ @ h₁ + b₂
z2 = W2 @ h1 + b2
print(f"\n步骤3: z₂ = W₂ @ h₁ + b₂")
print(f"       z₂ = {W2} @ {h1} + {b2}")
print(f"       z₂ = {z2}")

# 步骤4: 激活 h₂ = ReLU(z₂)
h2 = relu(z2)
print(f"\n步骤4: h₂ = ReLU(z₂)")
print(f"       h₂ = ReLU({z2})")
print(f"       h₂ = {h2}   （z₂ > 0，ReLU 不截断）")

# 步骤5: 损失 L = ½(h₂ - t)²
L = 0.5 * np.sum((h2 - t) ** 2)
print(f"\n步骤5: L = ½(h₂ - t)²")
print(f"       L = ½({h2} - {t})²")
print(f"       L = ½ × ({h2 - t})²")
print(f"       L = {L}")

# ============================================================
# 反向传播（6 步，从 L 回到 x）
# ============================================================
print("\n" + "=" * 70)
print("◀ 反向传播（6 步，从 L 回到 x）")
print("=" * 70)

# 步骤1: ∂L/∂h₂ —— Loss 对网络输出的梯度
dL_dh2 = h2 - t
print(f"\n步骤1: ∂L/∂h₂ = h₂ - t")
print(f"       ∂L/∂h₂ = {h2} - {t}")
print(f"       ∂L/∂h₂ = {dL_dh2}")
print(f"       解释：y₂ 和 L 的直接关系就是 MSE 公式，直接求导即得")

# 步骤2: ∂L/∂z₂ —— 穿过 ReLU
dL_dz2 = dL_dh2 * relu_grad(z2)
print(f"\n步骤2: ∂L/∂z₂ = ∂L/∂h₂ × ReLU'(z₂)")
print(f"       ∂L/∂z₂ = {dL_dh2} × {relu_grad(z2)}")
print(f"       ∂L/∂z₂ = {dL_dz2}")
print(f"       解释：ReLU 导数 = 1（因为 z₂ > 0）")

# 步骤3: ∂L/∂h₁ = W₂ᵀ @ ∂L/∂z₂  ★★★ Wᵀ 第一次出现 ★★★
dL_dh1 = W2.T @ dL_dz2
print(f"\n步骤3: ∂L/∂h₁ = W₂ᵀ @ ∂L/∂z₂    ★★★ Wᵀ 在这里出现 ★★★")
print(f"       W₂ = {W2}     (1×2)")
print(f"       W₂ᵀ = {W2.T}  (2×1)")
print(f"       ∂L/∂h₁ = {W2.T} @ {dL_dz2}")
print(f"       ∂L/∂h₁ = {dL_dh1}")
print(f"       解释：h₁(2,) 通过 W₂(1×2) 影响了 z₂(1,)")
print(f"             h₁ 的每个分量影响了 z₂，梯度要沿 W₂ 的【列】汇聚 → W₂ᵀ")

# 标量展开验证
print(f"\n       --- 标量展开验证 ---")
print(f"       h₁₁ 通过 w=5 影响 z₂，h₁₂ 通过 w=6 影响 z₂")
print(f"       ∂L/∂h₁₁ = w₂₁₁ × ∂L/∂z₂ = 5 × {dL_dz2[0]} = {5 * dL_dz2[0]}")
print(f"       ∂L/∂h₁₂ = w₂₁₂ × ∂L/∂z₂ = 6 × {dL_dz2[0]} = {6 * dL_dz2[0]}")
print(f"       结果 = [{5*dL_dz2[0]}, {6*dL_dz2[0]}]  和矩阵形式一致 ✓")

# 步骤4: ∂L/∂z₁ —— 穿过第1层 ReLU
dL_dz1 = dL_dh1 * relu_grad(z1)
print(f"\n步骤4: ∂L/∂z₁ = ∂L/∂h₁ × ReLU'(z₁)")
print(f"       ∂L/∂z₁ = {dL_dh1} × {relu_grad(z1)}")
print(f"       ∂L/∂z₁ = {dL_dz1}")

# 步骤5: ∂L/∂x = W₁ᵀ @ ∂L/∂z₁  ★★★ Wᵀ 第二次出现 ★★★
dL_dx = W1.T @ dL_dz1
print(f"\n步骤5: ∂L/∂x = W₁ᵀ @ ∂L/∂z₁    ★★★ Wᵀ 在这里又出现 ★★★")
print(f"       W₁ = {W1}   (2×2)")
print(f"       W₁ᵀ = {W1.T}  (2×2)")
print(f"       ∂L/∂x = {W1.T} @ {dL_dz1}")
print(f"       ∂L/∂x = {dL_dx}")

# 标量展开验证
print(f"\n       --- 标量展开验证 ---")
print(f"       x₁ 通过 w₁₁=1 和 w₂₁=3 影响了 z₁₁ 和 z₁₂")
dL_dx1_scalar = dL_dz1[0]*W1[0,0] + dL_dz1[1]*W1[1,0]
dL_dx2_scalar = dL_dz1[0]*W1[0,1] + dL_dz1[1]*W1[1,1]
print(f"       ∂L/∂x₁ = ∂L/∂z₁₁·w₁₁ + ∂L/∂z₁₂·w₂₁ = {dL_dz1[0]}×1 + {dL_dz1[1]}×3 = {dL_dx1_scalar}")
print(f"       ∂L/∂x₂ = ∂L/∂z₁₁·w₁₂ + ∂L/∂z₁₂·w₂₂ = {dL_dz1[0]}×2 + {dL_dz1[1]}×4 = {dL_dx2_scalar}")
print(f"       结果 = [{dL_dx1_scalar}, {dL_dx2_scalar}]  和矩阵形式一致 ✓")

# 步骤6: ∂L/∂W（顺带算权重的梯度）
dL_dW1 = np.outer(dL_dz1, x)
dL_dW2 = np.outer(dL_dz2, h1)
print(f"\n步骤6: ∂L/∂W（权重的梯度，用于更新参数）")
print(f"       ∂L/∂W₂ = ∂L/∂z₂ · h₁ᵀ = {dL_dz2} ⊗ {h1}")
print(f"       ∂L/∂W₂ = {dL_dW2}")
print(f"       ∂L/∂W₁ = ∂L/∂z₁ · xᵀ  = {dL_dz1} ⊗ {x}")
print(f"       ∂L/∂W₁ = {dL_dW1}")

# ============================================================
# PyTorch 验证
# ============================================================
print("\n" + "=" * 70)
print("✓ PyTorch autograd 验证")
print("=" * 70)

x_t = torch.tensor([1., 2.], requires_grad=True)
W1_t = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
W2_t = torch.tensor([[5., 6.]], requires_grad=True)
b1_t = torch.tensor([0., 0.], requires_grad=True)
b2_t = torch.tensor([0.], requires_grad=True)
t_t = torch.tensor([50.])

z1_t = W1_t @ x_t + b1_t
h1_t = torch.relu(z1_t)
z2_t = W2_t @ h1_t + b2_t
h2_t = torch.relu(z2_t)
L_t = 0.5 * torch.sum((h2_t - t_t) ** 2)
L_t.backward()

print(f"\n手动 ∂L/∂x  = {dL_dx}")
print(f"PyTorch      = {x_t.grad.numpy()}")
print(f"一致 ✓" if np.allclose(dL_dx, x_t.grad.numpy()) else "不一致 ✗")

print(f"\n手动 ∂L/∂W₁ = {dL_dW1}")
print(f"PyTorch      = {W1_t.grad.numpy()}")
print(f"一致 ✓" if np.allclose(dL_dW1, W1_t.grad.numpy()) else "不一致 ✗")

print(f"\n手动 ∂L/∂W₂ = {dL_dW2}")
print(f"PyTorch      = {W2_t.grad.numpy()}")
print(f"一致 ✓" if np.allclose(dL_dW2, W2_t.grad.numpy()) else "不一致 ✗")

# ============================================================
# 梯度流向总览
# ============================================================
print("\n" + "=" * 70)
print("梯度流向总览")
print("=" * 70)
print(f"""
前向（数据流）：
  x={x} → z₁={z1} → h₁={h1} → z₂={z2} → h₂={h2} → L={L}

反向（梯度流）：
  ∂L/∂h₂ = {dL_dh2}
       ↓ × ReLU'
  ∂L/∂z₂ = {dL_dz2}
       ↓ × W₂ᵀ  ← ★转置出现★
  ∂L/∂h₁ = {dL_dh1}
       ↓ × ReLU'
  ∂L/∂z₁ = {dL_dz1}
       ↓ × W₁ᵀ  ← ★转置出现★
  ∂L/∂x  = {dL_dx}

规律：每个线性层反向传播时，都用该层权重的转置！
      z = W @ h  →  ∂L/∂h = Wᵀ @ ∂L/∂z
""")
