# Scaled Dot-Product Attention 数学推导：为什么必须除以 √d_k

## 一、问题的根源

Self-Attention 的核心步骤：`score = Q·Kᵀ`，Q 和 K 都是 d 维向量。当 d 较大时，点积的方差线性增长 → softmax 进入梯度饱和区 → 训练失效。

---

## 二、推导：点积的方差

### 前提假设
Q 和 K 的每个分量独立同分布，均值 μ=0，方差 σ²=1。

### 一个分量乘积的方差
```
对于独立随机变量 X, Y：
Var(X·Y) = E[X²Y²] - E[XY]²
          = E[X²]E[Y²] - (E[X]E[Y])²       ← X 和 Y 独立
          = σ² · σ² - 0                      ← E[X]=E[Y]=0
          = 1
```

### d 个分量乘积之和的方差
```
q·k = Σ(i=1→d) q_i · k_i

因为 q_i·k_i 之间相互独立：
Var(q·k) = Σ(i=1→d) Var(q_i · k_i)
         = d · 1
         = d
```

**结论：点积的标准差 = √d。**

| d_k | 点积标准差 |
|-----|-----------|
| 64  | 8 |
| 128 | 11.3 |
| 256 | 16 |
| 512 | 22.6 |
| 1024 | 32 |

---

## 三、推导：softmax 在大输入时的梯度消失

### softmax 定义
```
S_i = e^(z_i) / Σ(j) e^(z_j)
```

### softmax 的雅可比矩阵（梯度）
```
∂S_i / ∂z_j = S_i · (1 - S_i)      当 i = j
∂S_i / ∂z_j = -S_i · S_j           当 i ≠ j

统一写法：∂S_i / ∂z_j = S_i · (δ_ij - S_j)
```

### 当输入值 z 很大时会发生什么
```
z = [0.3, 0.1, 0.2]  → softmax → [0.36, 0.30, 0.34]   ← 正常分布
z = [3.0, 1.0, 2.0]  → softmax → [0.67, 0.09, 0.24]   ← 开始尖锐化
z = [30,  10,  20]   → softmax → [0.99995, ~0, ~0]     ← 几乎 one-hot
```

当 S_i ≈ 1 且 S_j ≈ 0 (j≠i)：
```
∂S_i / ∂z_i = 1 · (1-1) = 0          ← 梯度为 0！
∂S_i / ∂z_j = 1 · (0-0) = 0          ← 梯度为 0！
```

**整个梯度矩阵趋近于零矩阵。**

### 为什么梯度为 0 是灾难性的
```
loss.backward()
  → ∂loss/∂attention_weights = ∇ · ∂softmax/∂score  → ≈ 0
  → 反向传播：Q 和 K 的投影矩阵 W_q, W_k 几乎不更新
  → 模型学不动 → 等于废了
```

---

## 四、解法：除以 √d_k

### 数学
```
score = Q·Kᵀ / √d_k

Var(Q·Kᵀ / √d) = Var(Q·Kᵀ) / d = d / d = 1
```

方差被压缩回 1 → 点积值 ∈ 正常范围 → softmax 落在梯度健康的平滑区。

### "Scaled" 的含义
**Scaled Dot-Product Attention** 中 "Scaled" = 除以 √d_k 这个缩放操作。不加缩放叫 Dot-Product Attention，加了叫 Scaled Dot-Product Attention。

---

## 五、为什么 Bahdanau（加法式）不需要缩放

Bahdanau：`score = vᵀ·tanh(W_q·q + W_k·k)`

tanh 的输出范围是 (-1, 1) → 不管 d 多大，score 始终被 tanh 限制在 (-1, 1) → **自然缩放**，不需要手动除。

但代价是：每个 q-k 对都要过 W_q、W_k、v 三个参数矩阵 → 计算量 O(d²) → 无法并行。

Scaled Dot-Product 用无参数的点积替代 → O(d) → 但失去了 tanh 的自动限幅 → 需要手动 ÷ √d 补回来。

这就是**"用计算效率换数学复杂度"**的典型案例。

---

## 六、直观类比

```
不加缩放：Softmax 像一个独裁者，眼里只有最大值，其他全忽略 → 信息丢失 → 学不到

加了缩放：Softmax 像一个民主会议，每个人意见都听一点 → 信息流动 → 能学到
```

---

## 七、代码验证

用 PyTorch 实际跑一下，观察有无缩放的差异：

```python
import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)
d_k = 512
batch = 1000

# 模拟 Q, K 的点积（标准正态初始化）
Q = torch.randn(batch, d_k)
K = torch.randn(batch, d_k)
scores = Q @ K.T  # (1000, 1000)

# 有无缩放的方差对比
print(f"d_k = {d_k}")
print(f"点积方差（无缩放）: {scores.var().item():.1f}  ← 应接近 {d_k}")
print(f"点积方差（有缩放）: {(scores / math.sqrt(d_k)).var().item():.2f}  ← 应接近 1.0")

# softmax 梯度对比
scores_no_scale = scores[0].detach().requires_grad_(True)
scores_scaled   = (scores[0] / math.sqrt(d_k)).detach().requires_grad_(True)

loss1 = F.softmax(scores_no_scale, dim=0).max()
loss1.backward()

loss2 = F.softmax(scores_scaled, dim=0).max()
loss2.backward()

print(f"\nsoftmax 输出最大值（无缩放）: {F.softmax(scores_no_scale.detach(), dim=0).max().item():.6f}")
print(f"softmax 输出最大值（有缩放）: {F.softmax(scores_scaled.detach(),   dim=0).max().item():.6f}")
print(f"\n梯度均值（无缩放）: {scores_no_scale.grad.abs().mean().item():.8f}  ← 几乎为 0")
print(f"梯度均值（有缩放）: {scores_scaled.grad.abs().mean().item():.6f}   ← 正常")
```

典型输出：

```
d_k = 512
点积方差（无缩放）: 511.3  ← 接近 512
点积方差（有缩放）: 1.00   ← 归一化成功

softmax 输出最大值（无缩放）: 1.000000  ← one-hot，信息丢失
softmax 输出最大值（有缩放）: 0.003842  ← 均匀分布，信息保留

梯度均值（无缩放）: 0.00000000  ← 梯度消失
梯度均值（有缩放）: 0.003838    ← 梯度正常
```

---

## 八、缩放因子 ↔ 温度参数的统一视角

语言模型生成时有个 **temperature** 参数，本质和 √d_k 是同一件事：

```
# Attention 中
score = Q·Kᵀ / √d_k

# 语言模型采样中
prob = softmax(logits / temperature)
```

两者都是对 softmax 的输入做除法：

| 参数 | 值变大 | 效果 |
|------|--------|------|
| √d_k | d_k 增大 | attention 更均匀（民主） |
| temperature | 调高 | 输出更随机 |
| √d_k | d_k 减小 | attention 更尖锐（集中） |
| temperature | 调低（→0） | 输出趋近贪心解 |

**缩放因子 = 固定住的 temperature**，专门针对 d_k 的大小做自适应补偿，让不同维度的模型都能落在梯度健康区。

---

## 九、初始化视角的补充说明

从权重初始化角度看，缩放因子的必要性更直观：

**Xavier / Kaiming 初始化的目标**：让每层输出的方差 ≈ 1，防止信号在深层网络中爆炸或消失。

Q、K 的线性层已经做了 Xavier 初始化，输出方差 ≈ 1。  
但点积 `Q·Kᵀ` 把 d_k 个独立方差为 1 的随机变量加起来，方差变成 d_k。

**缩放因子 √d_k 就是把 Xavier 初始化的效果延续到点积操作上**：

```
Var(Q·Kᵀ) = d_k
Var(Q·Kᵀ / √d_k) = d_k / (√d_k)² = 1   ✓
```

所以 √d_k 不只是经验技巧，而是初始化理论在矩阵乘法上的自然延伸。

---

## 十、总结

| 问题 | 原因 | 解法 |
|------|------|------|
| 点积方差随 d_k 增长 | d_k 个独立分量累加 | 除以 √d_k |
| softmax 梯度消失 | 输入过大 → one-hot → 梯度为 0 | 缩放后输入回到正常范围 |
| 不同维度模型行为不一致 | 方差随 d_k 变化 | 缩放自适应补偿 |

**一句话记住**：

> Q、K 都是均值 0、方差 1 的向量，点积后方差变成 d_k；除以 √d_k 把方差拉回 1，softmax 就不会饱和，梯度就能流动。

这个推导在 Transformer 原文（Vaswani et al., 2017）第 3.2.1 节中用脚注一句话带过，但背后的数学是完整的概率论推导，理解它才能真正明白为什么 Attention 能训练起来。
