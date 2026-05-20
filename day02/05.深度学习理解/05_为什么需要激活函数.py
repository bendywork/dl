"""
为什么需要激活函数？—— 用 XOR 问题直观演示

核心结论：
  1. 线性变换只能画直线，永远分不开 XOR 这种非线性问题
  2. 没有激活函数，堆多少层线性层都等价于一层
  3. 激活函数引入非线性，让网络能画曲线，从而解决非线性问题
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# XOR 数据：线性不可分的最经典例子
# ============================================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR: 相同为0，不同为1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


# ============================================================
# 实验1：没有激活函数的网络（2层线性 = 1层线性，学不动XOR）
# ============================================================
np.random.seed(42)
W1_no = np.random.randn(2, 4) * 0.5
b1_no = np.zeros(4)
W2_no = np.random.randn(4, 1) * 0.5
b2_no = np.zeros(1)
lr = 0.01

losses_no_act = []
for epoch in range(5000):
    # 前向：不加任何激活函数
    h = X @ W1_no + b1_no          # 第一层线性
    out = h @ W2_no + b2_no        # 第二层线性（还是线性的！）

    # MSE损失
    loss = np.mean((out.flatten() - y) ** 2)
    losses_no_act.append(loss)

    # 反向
    d_out = 2 * (out.flatten() - y) / 4
    d_W2 = h.T @ d_out.reshape(-1, 1)
    d_b2 = d_out.sum()
    d_h = d_out.reshape(-1, 1) @ W2_no.T
    d_W1 = X.T @ d_h
    d_b1 = d_h.sum(axis=0)

    W2_no -= lr * d_W2
    b2_no -= lr * d_b2
    W1_no -= lr * d_W1
    b1_no -= lr * d_b1

print("=" * 60)
print("实验1：无激活函数（2层纯线性）")
print(f"  最终损失: {losses_no_act[-1]:.4f}")
print(f"  预测值:   {out.flatten()}")
print(f"  目标值:   {y}")
print(f"  → 损失降不下去，根本学不会 XOR！")


# ============================================================
# 实验2：加 sigmoid 激活函数（同样的网络结构，但加了非线性）
# ============================================================
np.random.seed(42)
W1_act = np.random.randn(2, 4) * 0.5
b1_act = np.zeros(4)
W2_act = np.random.randn(4, 1) * 0.5
b2_act = np.zeros(1)

losses_with_act = []
for epoch in range(5000):
    # 前向：加了 sigmoid！
    z1 = X @ W1_act + b1_act
    h = sigmoid(z1)               # ← 关键区别：这里加了激活函数
    z2 = h @ W2_act + b2_act
    out = sigmoid(z2)             # 输出层也加 sigmoid

    # MSE损失
    loss = np.mean((out.flatten() - y) ** 2)
    losses_with_act.append(loss)

    # 反向
    d_loss = 2 * (out.flatten() - y) / 4
    d_out = d_loss * sigmoid_deriv(z2).flatten()
    d_W2 = h.T @ d_out.reshape(-1, 1)
    d_b2 = d_out.sum()
    d_h = d_out.reshape(-1, 1) @ W2_act.T
    d_z1 = d_h * sigmoid_deriv(z1)
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    W2_act -= lr * d_W2
    b2_act -= lr * d_b2
    W1_act -= lr * d_W1
    b1_act -= lr * d_b1

print("\n" + "=" * 60)
print("实验2：有 sigmoid 激活函数")
print(f"  最终损失: {losses_with_act[-1]:.4f}")
print(f"  预测值:   {out.flatten().round(4)}")
print(f"  目标值:   {y}")
print(f"  → 损失几乎降到0，成功学会 XOR！")


# ============================================================
# 可视化对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- 图1：损失曲线对比 ---
axes[0].plot(losses_no_act, label='No Activation (pure linear)', color='red')
axes[0].plot(losses_with_act, label='With Sigmoid', color='blue')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- 图2：无激活函数的决策边界 ---
ax = axes[1]
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', s=200, marker='o', label='y=0', zorder=5)
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', s=200, marker='o', label='y=1', zorder=5)

# 画无激活函数网络的决策区域
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
h_no = grid @ W1_no + b1_no
pred_no = (h_no @ W2_no + b2_no).flatten()
pred_no = pred_no.reshape(xx.shape)
ax.contourf(xx, yy, pred_no, levels=20, alpha=0.3, cmap='coolwarm')
ax.set_title('No Activation: Can Only Draw\nStraight Boundaries')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

# --- 图3：有 sigmoid 的决策边界 ---
ax = axes[2]
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', s=200, marker='o', label='y=0', zorder=5)
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', s=200, marker='o', label='y=1', zorder=5)

z1_grid = grid @ W1_act + b1_act
h_grid = sigmoid(z1_grid)
z2_grid = h_grid @ W2_act + b2_act
pred_act = sigmoid(z2_grid).flatten().reshape(xx.shape)
ax.contourf(xx, yy, pred_act, levels=20, alpha=0.3, cmap='coolwarm')
ax.set_title('With Sigmoid: Can Draw\nCurved Boundaries!')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/sunchengxin/PycharmProjects/dl/day02/05.深度学习理解/activation_comparison.png', dpi=150)
