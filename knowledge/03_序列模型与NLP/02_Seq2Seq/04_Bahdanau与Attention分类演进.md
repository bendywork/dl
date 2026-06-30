# Bahdanau Attention 与注意力机制分类演进

## 📌 核心问题
> Bahdanau Attention 是什么？Attention 有哪些分类？从传统时间步到注意力机制经历了什么？

---

## 🧠 Bahdanau Attention（Additive Attention / 加性注意力）

Bahdanau Attention 是 2014 年由 Dzmitry Bahdanau 等人在神经机器翻译（NMT）论文中首次提出的注意力机制，也是**第一个将 Attention 引入 seq2seq 的工作**。

### 核心公式

**对齐分数（Alignment Score）：**

$$\text{score}(s_{t-1}, h_i) = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$$

**注意力权重（Softmax 归一化）：**

$$\alpha_{ti} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_{j=1}^{T_x} \exp(\text{score}(s_{t-1}, h_j))}$$

**上下文向量（加权求和）：**

$$c_t = \sum_{i=1}^{T_x} \alpha_{ti} h_i$$

| 符号 | 含义 |
|------|------|
| $s_{t-1}$ | Decoder **上一时刻**的隐藏状态 |
| $h_i$ | Encoder 第 $i$ 个位置的隐藏状态 |
| $W_a, U_a, v_a$ | 可学习的参数矩阵 |
| $\alpha_{ti}$ | 第 $t$ 个解码步对第 $i$ 个编码位置的注意力权重 |
| $c_t$ | 当前解码步的上下文向量 |

### 为什么叫 "Additive"（加性）

因为核心运算 **$W_a s_{t-1} + U_a h_i$** 是**加法**，与 Luong Attention 的**乘法**（$s_t^T h_i$）形成对比。加性意味着 Query 和 Key 先各自线性变换再相加，而非直接做点积。

### Bahdanau vs Luong 对比

| 特性 | Bahdanau (2014) | Luong (2015) |
|------|-----------------|--------------|
| 对齐函数 | 加性（MLP 网络） | 乘法（点积 / 通用） |
| 注意力计算时机 | 先算 attention，再更新 decoder 状态 | 先更新 decoder 状态，再算 attention |
| Query 来源 | $s_{t-1}$（上一时刻状态） | $s_t$（当前时刻状态） |
| 计算复杂度 | 略高（需要 MLP 前向传播） | 更低（纯矩阵乘法） |
| Encoder | 双向 RNN | 可用单向也可用双向 |

---

## 🗂️ Attention 完整分类体系

### 一、按计算方式分类

```
Attention
├── 加性注意力 (Additive / Bahdanau)
│   └── score = vᵀ tanh(W·q + U·k)
│
├── 乘法注意力 (Multiplicative / Luong)
│   ├── 点积 (Dot):        score = qᵀ k
│   └── 通用 (General):    score = qᵀ W k
│
└── 缩放点积注意力 (Scaled Dot-Product)
    └── score = qᵀ k / √dₖ   ← Transformer 使用
```

**为什么要缩放？** 当 $d_k$ 很大时，点积值会很大，导致 softmax 梯度趋于 0。除以 $\sqrt{d_k}$ 保持方差稳定。

### 二、按注意力范围分类

| 类型 | 描述 | 特点 |
|------|------|------|
| **Global / Soft Attention** | 对所有输入位置计算注意力 | 可微，Bahdanau / Luong / Transformer 均属此类 |
| **Local / Hard Attention** | 只关注一个局部窗口或单个位置 | 不可微，需强化学习 / 采样技巧 |
| **Sparse Attention** | 只关注部分位置 | Longformer、BigBird，降低 $O(n^2)$ 复杂度 |

### 三、按 Query-Key-Value 来源分类

| 类型 | Q 来源 | K/V 来源 | 典型应用 |
|------|--------|---------|---------|
| **Self-Attention**（自注意力） | 同一序列 | 同一序列 | Transformer Encoder |
| **Cross-Attention**（交叉注意力） | Decoder | Encoder | Transformer Decoder |
| **Multi-Head Attention** | 多组 Q/K/V 并行 | — | Transformer 核心组件 |

### 四、按层级 / 结构分类

| 类型 | 代表工作 | 核心思想 |
|------|---------|---------|
| **Token-level** | 标准 seq2seq attention | 词级别加权 |
| **Sentence / Hierarchical** | Hierarchical Attention | 先句子编码，再做句子级 attention |
| **Graph Attention (GAT)** | GAT | 在图结构上对邻居节点做 attention |
| **Channel / Spatial** | SE-Net, CBAM | CV 中对通道或空间维度做 attention |

---

## 🔄 从传统时间步到注意力机制 — 演进五阶段

### 阶段一：Encoder-Decoder（无 Attention）— 信息瓶颈

```
"我 爱 你" → [Encoder RNN] → 固定长度向量 c → [Decoder RNN] → "I love you"
```

**核心问题：信息瓶颈**

无论输入多长，所有信息都被压缩到一个固定长度的上下文向量 $c$ 中。当句子变长：
- 翻译质量急剧下降
- BLEU 分数随句子长度增加而递减
- Decoder 无法为不同解码步提供不同的"关注点"

> 这是 seq2seq 的原罪 — 把变长序列强行压缩成定长向量。

### 阶段二：Bahdanau Attention（2014）— 动态上下文向量

```
"我 爱 你" → [Encoder Bi-RNN] → h₁, h₂, h₃
                                       ↓
                              [Decoder] ← 每个解码步动态加权 hᵢ
```

**核心突破：**

- 不再使用单一固定向量，而是**每个解码步动态生成一个加权上下文向量** $c_t$
- Decoder 在每个时间步可以"选择性关注"输入的不同部分
- Bi-RNN Encoder 捕获双向上下文
- 注意力权重可视化，提供可解释性
- 对齐信息被隐式学习出来

### 阶段三：Luong Attention（2015）— 多样化解法

在 Bahdanau 基础上的改进：

- 提供了 **Dot、General、Concat** 三种打分函数
- 计算顺序不同：先算 decoder 状态 $s_t$，再算 attention
- 引入 **input-feeding**：将 $c_t$ 和 decoder 状态拼接后作为下一时间步的输入，让模型"记住"之前的对齐决策
- 提出 **Local Attention**：只关注一个窗口，降低计算量

### 阶段四：Self-Attention & Transformer（2017）— 抛弃 RNN

```
抛弃 RNN，完全基于 Attention
     ↓
Q, K, V 来自同一序列 → Self-Attention
     ↓
Scaled Dot-Product + Multi-Head → Transformer
```

**关键转变：**

| 维度 | RNN + Attention | Transformer |
|------|----------------|-------------|
| 序列处理 | 串行（时间步递推） | 并行（矩阵乘法） |
| 长距离依赖 | Attention 辅助 RNN | 直接通过 Self-Attention |
| 串行步数 | $O(n)$ | $O(1)$ |
| 计算复杂度 | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| 位置信息 | 隐式（RNN 自带） | 显式（Positional Encoding） |

### 阶段五：高效 Attention（2019+）— 突破 $O(n^2)$

Transformer 的 $O(n^2)$ 复杂度催生了大量改进：

| 方法 | 代表工作 | 核心思想 |
|------|---------|---------|
| Sparse | Sparse Transformer | 只计算部分位置的 attention |
| Low-Rank | Linformer | 将 K/V 投影到低维空间 |
| Kernel | Performer | 通过核方法近似 softmax |
| Recurrence | Transformer-XL | 引入段级循环，跨段传递信息 |
| Locality | Longformer | 滑动窗口 + 全局 token 混合 |

---

## 💡 演进本质总结

```
传统 Seq2Seq (固定向量 c)
    │
    │  问题：信息瓶颈，长句退化
    ▼
Bahdanau Attention (动态加权 cₜ)
    │
    │  改进：多 Score 函数，全局/局部
    ▼
Luong Attention (更简洁，更多变体)
    │
    │  颠覆：扔掉 RNN，全靠 Attention
    ▼
Transformer / Self-Attention (并行，多头)
    │
    │  优化：降低 O(n²) 复杂度
    ▼
高效 Attention (稀疏、低秩、核方法...)
```

**本质变化**：从一个"被动压缩信息"的过程，演进为**"主动选择信息"**的过程。

Bahdanau 的伟大之处在于他第一个意识到：**Decoder 在每个时间步应该有权"看不同的东西"**，而不是被迫依赖一个僵死的上下文向量。这个洞察直接开启了现代 NLP 的注意力时代。
