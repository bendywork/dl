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
