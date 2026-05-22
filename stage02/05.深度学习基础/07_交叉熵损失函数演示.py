"""
交叉熵损失函数 — 完整案例代码
从信息论根源到 PyTorch 实践，每步有具体数字
"""
import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 第一部分：信息论根源 — 为什么叫"熵"
# ============================================================
print("=" * 70)
print("第一部分：信息论根源")
print("=" * 70)

# 信息量：事件越罕见，信息量越大
def information(p):
    """一个概率为 p 的事件的信息量: I(p) = -log₂(p)"""
    return -np.log2(p) if p > 0 else float('inf')

print("\n--- 信息量 I(p) = -log₂(p) ---")
for p in [1.0, 0.5, 0.25, 0.1, 0.01]:
    print(f"  p={p:<5} → I(p)={information(p):.2f} bits  {'(必然事件，无信息)' if p==1 else ''}")

# 熵：信息量的期望 — 一个分布的"平均不确定性"
def entropy(p_dist):
    """H(P) = -Σ P(x) · log₂(P(x))"""
    return -sum(p * np.log2(p) for p in p_dist if p > 0)

print("\n--- 熵 H(P) = -Σ P(x)·log₂(P(x)) ---")
print(f"  公平硬币 [0.5, 0.5]   H = {entropy([0.5, 0.5]):.2f} bits  (最大不确定性)")
print(f"  偏向硬币 [0.9, 0.1]   H = {entropy([0.9, 0.1]):.2f} bits  (不确定性降低)")
print(f"  必定硬币 [1.0, 0.0]   H = {entropy([1.0, 0.0]):.2f} bits  (完全确定)")
print("  → 分布越均匀，熵越大（越不确定）；分布越集中，熵越小")

# 交叉熵：用分布 Q 编码，但数据来自分布 P 的平均信息量
def cross_entropy(p_dist, q_dist):
    """H(P, Q) = -Σ P(x) · log₂(Q(x))"""
    return -sum(p * np.log2(q) for p, q in zip(p_dist, q_dist) if p > 0 and q > 0)

print("\n--- 交叉熵 H(P, Q) = -Σ P(x)·log₂(Q(x)) ---")
P = [0.9, 0.1]  # 真实分布（偏向正面）

Q1 = [0.9, 0.1]  # Q = P，完美预测
Q2 = [0.5, 0.5]  # Q 完全无知
Q3 = [0.1, 0.9]  # Q 完全反了

print(f"  真实分布 P = {P}")
print(f"  Q1={Q1} → H(P,Q1) = {cross_entropy(P, Q1):.4f}  (Q=P，最低)")
print(f"  Q2={Q2} → H(P,Q2) = {cross_entropy(P, Q2):.4f}  (Q 无知，更高)")
print(f"  Q3={Q3} → H(P,Q3) = {cross_entropy(P, Q3):.4f}  (Q 反了，最高)")
print(f"  H(P)     = {entropy(P):.4f}  (熵是交叉熵的下界)")
print("  → 交叉熵 ≥ 熵，等号在 Q=P 时成立。Q 离 P 越远，交叉熵越大")

# KL 散度：交叉熵 - 熵 = 额外代价
def kl_divergence(p_dist, q_dist):
    """KL(P||Q) = H(P,Q) - H(P)"""
    return cross_entropy(p_dist, q_dist) - entropy(p_dist)

print(f"\n--- KL 散度 = 交叉熵 - 熵 = 额外代价 ---")
print(f"  KL(P||Q1) = {kl_divergence(P, Q1):.4f}  (完美预测，无额外代价)")
print(f"  KL(P||Q2) = {kl_divergence(P, Q2):.4f}  (无知，有代价)")
print(f"  KL(P||Q3) = {kl_divergence(P, Q3):.4f}  (完全反了，代价巨大)")
print("  → 优化交叉熵 = 最小化 KL 散度 = 让 Q 靠近 P")

# ============================================================
# 第二部分：二分类交叉熵（Binary Cross-Entropy）
# ============================================================
print("\n" + "=" * 70)
print("第二部分：二分类交叉熵 BCE")
print("=" * 70)

print("""
公式：L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

y  = 真实标签 (0 或 1)
ŷ  = 模型预测概率 (0到1之间)

两种情况：
  y=1 (正类): L = -log(ŷ)     ŷ 越接近1，损失越小
  y=0 (负类): L = -log(1-ŷ)   ŷ 越接近0，损失越小
""")

# 具体数值案例
print("--- 数值案例 ---")
y_true = 1  # 真实标签：正类
for y_pred in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
    loss = -np.log(y_pred)
    print(f"  y=1, ŷ={y_pred:<5} → L = -log({y_pred}) = {loss:.4f}")

print()
y_true = 0  # 真实标签：负类
for y_pred in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    loss = -np.log(1 - y_pred)
    print(f"  y=0, ŷ={y_pred:<5} → L = -log(1-{y_pred}) = {loss:.4f}")

print("\n规律：预测越准，损失越接近0；预测越离谱，损失趋向∞")

# ============================================================
# 第三部分：多分类交叉熵（Categorical Cross-Entropy）
# ============================================================
print("\n" + "=" * 70)
print("第三部分：多分类交叉熵 CCE")
print("=" * 70)

print("""
公式：L = -Σᵢ yᵢ · log(ŷᵢ)

y  = one-hot 真实标签 (只有一个是1，其余0)
ŷ  = softmax 输出的概率分布

因为 y 是 one-hot，只有 yₖ=1，所以：
L = -log(ŷₖ)    ← 只看真实类别上的预测概率
""")

# 3分类案例：猫、狗、鸟
print("--- 3分类案例 ---")
classes = ['猫', '狗', '鸟']

# 真实标签：猫 → one-hot [1, 0, 0]
y = np.array([1, 0, 0])

# 不同预测
preds = {
    '自信正确': np.array([0.9, 0.05, 0.05]),
    '勉强正确': np.array([0.4, 0.3, 0.3]),
    '完全错误': np.array([0.1, 0.1, 0.8]),
}

for name, pred in preds.items():
    loss = -np.sum(y * np.log(pred))
    print(f"\n  {name}: ŷ = {pred}")
    print(f"    L = -log(ŷ_猫) = -log({pred[0]}) = {loss:.4f}")

# ============================================================
# 第四部分：为什么分类用交叉熵而不是 MSE
# ============================================================
print("\n" + "=" * 70)
print("第四部分：交叉熵 vs MSE（分类任务）")
print("=" * 70)

print("""
MSE 用于分类的两个致命问题：

1. 梯度消失：当预测 ŷ 接近 0 或 1 时，sigmoid/softmax 的梯度极小
   → MSE 的梯度 ∝ (ŷ-y) × σ'(z)，σ'(z) 在两端接近0
   → 交叉熵的梯度 ∝ (ŷ-y)，没有 σ'(z) 项，梯度始终有效

2. 惩罚不够：MSE 对"自信且错误"的惩罚不够狠
   → MSE 惩罚是二次的：(ŷ-y)²
   → 交叉熵惩罚是对数的：-log(ŷ)，错得越离谱惩罚越重（趋向∞）
""")

# 数值对比
print("--- 惩罚力度对比（y=1 正类）---")
print(f"  {'ŷ':<8} {'MSE=(ŷ-1)²':<15} {'BCE=-log(ŷ)':<15}")
for y_pred in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]:
    mse = (y_pred - 1) ** 2
    bce = -np.log(y_pred)
    print(f"  {y_pred:<8} {mse:<15.4f} {bce:<15.4f}")

print("\n  MSE 在 ŷ=0.001 时才 0.001，几乎没感觉")
print("  BCE 在 ŷ=0.001 时已达 6.9，惩罚力度天差地别")

# ============================================================
# 第五部分：PyTorch 实践
# ============================================================
print("\n" + "=" * 70)
print("第五部分：PyTorch 实践")
print("=" * 70)

# --- 二分类 BCE ---
print("\n--- 二分类: nn.BCELoss ---")
y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred = torch.tensor([0.9, 0.1, 0.8, 0.3])

bce_loss = nn.BCELoss()
loss = bce_loss(y_pred, y_true)
print(f"  y_true = {y_true.tolist()}")
print(f"  y_pred = {y_pred.tolist()}")
print(f"  BCE Loss = {loss.item():.4f}")

# 手动验证
manual = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
print(f"  手动计算 = {manual.item():.4f}  ✓")

# --- 多分类 CrossEntropyLoss ---
print("\n--- 多分类: nn.CrossEntropyLoss ---")
# 注意：PyTorch 的 CrossEntropyLoss = LogSoftmax + NLLLoss
# 输入是 logits（未经 softmax），标签是类别索引（不是 one-hot）

logits = torch.tensor([[2.0, 1.0, 0.1],    # 样本1: 倾向类别0
                       [0.5, 2.5, 0.3]])    # 样本2: 倾向类别1
target = torch.tensor([0, 1])               # 真实类别

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, target)
print(f"  logits = {logits.tolist()}")
print(f"  target = {target.tolist()}")
print(f"  CrossEntropyLoss = {loss.item():.4f}")

# 手动验证：先 softmax 再算交叉熵
softmax_probs = torch.softmax(logits, dim=1)
print(f"  softmax = {softmax_probs.detach().tolist()}")
manual_ce = -torch.log(softmax_probs[0, 0])  - torch.log(softmax_probs[1, 1])
print(f"  手动计算 = {manual_ce.item() / 2:.4f}  ✓")  # 默认 mean reduction

# ============================================================
# 第六部分：梯度对比 — 交叉熵为什么学得快
# ============================================================
print("\n" + "=" * 70)
print("第六部分：梯度对比")
print("=" * 70)

print("""
sigmoid + MSE 的梯度：
  ∂L/∂z = (ŷ - y) × ŷ(1-ŷ)    ← ŷ 接近0或1时，ŷ(1-ŷ)≈0，梯度消失！

sigmoid + BCE 的梯度：
  ∂L/∂z = ŷ - y               ← 没有 ŷ(1-ŷ) 项，梯度始终有效！
""")

# 数值对比
print("--- 梯度大小对比（y=1）---")
print(f"  {'ŷ':<8} {'MSE梯度':<20} {'BCE梯度':<15}")
for y_pred in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
    mse_grad = (y_pred - 1) * y_pred * (1 - y_pred)
    bce_grad = y_pred - 1
    print(f"  {y_pred:<8} {mse_grad:<20.6f} {bce_grad:<15.6f}")

print("\n  MSE 在 ŷ=0.01 时梯度 = -0.0099，几乎不动")
print("  BCE 在 ŷ=0.01 时梯度 = -0.99，强力纠正！")
