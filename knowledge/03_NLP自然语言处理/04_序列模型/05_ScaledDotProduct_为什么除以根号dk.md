# Scaled Dot-Product Attention 为什么除以 √d_k

## 一、问题起点

Attention 计算分数时：

```
score = Q · K^T
```

当 d_k（向量维度）很大时，这个分数会变得很大，导致 softmax 出现问题。

---

## 二、softmax 的极值问题

softmax 公式：

```
softmax(xi) = e^xi / (e^x1 + e^x2 + ... + e^xn)
```

当输入数值很大时：

```
scores = [120, 2, -5, 118]

e^120  ≈ 7.7 × 10^52  （极大）
e^2    = 7.4
e^-5   = 0.007
e^118  ≈ 1.0 × 10^51  （极大）

softmax → [0.88, 0.0001, 0.0001, 0.12]
```

**核心：e 的指数增长极快，大值完全压制小值，输出接近 one-hot。**

---

## 三、为什么 Q·K 的数值会变大

### 3.1 方差的基础知识

```
一个变量 x，方差=1，代表值大约在 [-1, 1] 之间波动

两个方差=1的独立变量相乘：a*b 的方差=1
两个方差=1的独立变量相加：a+b 的方差=2  ← 方差直接相加
```

### 3.2 Q·K 的方差推导

神经网络初始化时，Q 和 K 的每个元素近似满足：均值=0，方差=1

```
Q·K = q1*k1 + q2*k2 + ... + q_dk * k_dk
      ← 共 d_k 项相加

每一项 qi*ki 的方差 = 1
d_k 项相加后，总方差 = d_k × 1 = d_k
标准差 = √d_k
```

类比理解：

```
投1枚骰子，结果在 [1, 6]
投512枚骰子加起来，结果在 [512, 3072]  ← 范围大得多

d_k=512 → Q·K 结果标准差 ≈ 22.6，大部分值落在 [-22, 22]
```

---

## 四、完整因果链

```
d_k 很大（比如512）
  ↓
Q·K 是512个数相加，结果数值大（标准差≈22）
  ↓
大数值进 softmax，e 指数压制小值
  ↓
输出接近 one-hot，大部分位置权重≈0
  ↓
大部分位置梯度≈0，参数更新困难
  ↓
解决：除以 √d_k，把方差归1，数值回到正常范围
```

---

## 五、为什么是 √d_k 而不是别的数

```
Q·K 的标准差 = √d_k

除以标准差 = 把数据标准化到标准差=1
这是统计学里最经典的归一化操作

不是凑巧选了 √d_k，而是把方差归1的唯一最优解
```

---

## 六、数值对比（除以√d_k 前后）

除以之前：

```
scores = [15, 2, -5, 14]  （d_k=512时的典型范围）
softmax → [0.73, 0.000002, 0.0, 0.27]  接近 one-hot
```

除以 √512 ≈ 22.6 之后：

```
scores = [0.66, 0.09, -0.22, 0.62]
softmax → [0.30, 0.22, 0.16, 0.32]  分布平滑，每个位置都有梯度
```

---

## 七、PyTorch 代码

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    # Q: [bs, t_q, d_k]
    # K: [bs, t_k, d_k]
    # V: [bs, t_k, d_v]
    
    d_k = Q.size(-1)
    
    # 1. 计算相似度分数，除以 √d_k
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    # scores: [bs, t_q, t_k]
    
    # 2. softmax 归一化
    weights = F.softmax(scores, dim=-1)
    # weights: [bs, t_q, t_k]
    
    # 3. 加权求和
    out = torch.bmm(weights, V)
    # out: [bs, t_q, d_v]
    
    return out
```

---

## 八、总结一句话

> Q·K 点积的方差随维度 d_k 线性增大，除以 √d_k 是把方差归1的最优解，
> 防止 softmax 因极值而退化成 one-hot，保证梯度正常流动。
