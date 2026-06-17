# Attention 三型对比：演化关系与本质

## 一句话定调

**Attention 的骨架永远是 QKV，变的是"Q/K/V 来自哪"和"分数怎么算"。** 没有 QKV 就没有 Attention。

---

## 三型一览

| | Bahdanau (2014) | Scaled Dot-Product (2017) | Self-Attention (2017) |
|---|---|---|---|
| **论文** | Seq2Seq+Align | Attention Is All You Need | Attention Is All You Need |
| **Q 来源** | decoder 当前隐状态 | decoder 当前隐状态 | **序列自身的每个 token** |
| **K 来源** | encoder 所有输出 | encoder 所有输出 | **序列自身的每个 token** |
| **V 来源** | encoder 所有输出 | encoder 所有输出 | **序列自身的每个 token** |
| **分数公式** | `vᵀ·tanh(W_q·q + W_k·k)` 加法式 | `Q·Kᵀ / √d_k` 点积缩放式 | `Q·Kᵀ / √d_k` 点积缩放式 |
| **核心问题** | "生成当前词该看源句哪些位置？" | 同上 | **"句子内部词跟词之间有什么关系？"** |

## 1. Bahdanau Attention（加法式 Attention）

**2014 年 Bahdanau 在 Seq2Seq 翻译任务上提出的原始 Attention。**

### 想解决的问题

Seq2Seq 里，一个固定长度上下文向量 C 要承载整条输入序列的全部语义 → 长句翻不好。Bahdanau 的办法：decoder 每生成一个词，用自己当前状态去 encoder 输出的每个位置做匹配，产生该位置的注意力分数，然后按权重从 encoder 输出取信息。

### 分数计算（核心区别）

```
score(q, k) = vᵀ · tanh(W_q · q + W_k · k)
```

- `W_q` 和 `W_k` 是两个可训练的投影矩阵，把 q 和 k 投影到同一个空间再做加法
- `vᵀ` 把投影结果映射为一个标量分值
- 因为这个加法结构（q 和 k 先分别变，再加起来），被称为 **Additive Attention**（加法式注意力）

### 特征

- Q/K/V 各经过独立线性层 → 可训练参数多
- 非线性激活 tanh → 表达能力强
- 但算得慢：每个 query-key 对都要过一个小网络

## 2. Scaled Dot-Product Attention（点积缩放式，你学的这个）

**2017 年 Transformer 论文统一为矩阵乘法，把 attention 从 O(n²) 小网络算分变成了纯矩阵乘。**

### 公式

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

| 步骤 | 操作 | 含义 |
|------|------|------|
| ① | `Q·Kᵀ` | 点积求相似度 |
| ② | `÷ √d_k` | 防 softmax 梯度饱和 |
| ③ | `softmax` | 相似度 → 概率权重 |
| ④ | `·V` | 加权取信息 |

### 相对 Bahdanau 的变化

| | Bahdanau | Scaled Dot-Product |
|---|---|---|
| 分数计算 | `vᵀ·tanh(W_q·q + W_k·k)` | `q·kᵀ / √d_k` |
| 参数 | 有（W_q, W_k, v） | **无参数**（纯数学运算） |
| 速度 | 慢（每个对过小网络） | 快（矩阵乘一把算完） |
| 缩放 | 不需要（tanh 自带限幅） | **必须有**（点积方差随 d 增大） |

**关键洞察**：用"无参数的点积"替代"有参数的加法网络"，舍弃一点表达能力换来了极大的计算效率。这是能堆叠成 Transformer 的前提——加法式每层都有一堆参数，点积式 layer 越多越省。

## 3. Self-Attention（自注意力）

**Q 和 K 和 V 来自同一个序列自己，让序列内部互相"认识"。**

### 核心区别：Q/K/V 的来源

```
传统 Attention（Cross-Attention）：
    Q ← decoder 状态
    K, V ← encoder 输出序列
    → "解码器查询编码器"，跨序列对齐

Self-Attention：
    Q ← 序列自己（经过线性投影）
    K ← 序列自己（经过线性投影）
    V ← 序列自己（经过线性投影）
    → "序列自己查询自己"，内部相互理解
```

同一句话"我今天吃了苹果"：

| 方式 | 问题 |
|------|------|
| Cross-Attention | "生成'apple'这个词要看源句哪些位置？" |
| Self-Attention | **"'苹果'跟'吃'是什么关系？跟'我'是什么关系？"** |

一次 Self-Attention 计算后，序列里每个 token 不再是孤立的向量，而是**融入了整个句子上下文信息后的向量**——"苹果"知道自己是被"吃"的，"吃"知道主语是"我"。这就是 Self-Attention 能取代 RNN 的根基。

> 公式就是同一个 `softmax(QKᵀ/√d)V`，唯一区别是 **Q、K、V 都由输入序列 X 各乘一个投影矩阵得到**：`Q = X·W_q, K = X·W_k, V = X·W_v`。

## 三者的演化关系

```
Bahdanau (2014)
    │  提出"对齐"概念 → 每一步动态关注源序列
    │  问题：算得慢，每个 q-k 对要走加法网络
    │
    ↓ 简化分数计算
    │
Scaled Dot-Product (2017)
    │  用 Q·Kᵀ/√d 替代加法网络 → 纯矩阵乘，飞快
    │  问题：decode 时仍需一步步串行
    │
    ↓ 把 Q 的来源反转
    │
Self-Attention (2017)
    │  Q=K=V=序列自己 → 整个序列"内部自省"
    │  并行计算整个序列 → 扔掉 RNN，堆成 Transformer
```

本质上是一条线：**分数计算越来越简单（网络→点积），QKV 来源越来越抽象（跨序列→序列内）。**

## 回到你的代码

`04_Seq2Seq+Attention理解.py` 里：

```python
def qkv_attention_value(q, k, v):      # → 通用 QKV 框架（Scaled Dot-Product）
    score = q @ k.transpose(2,1) / √e
    alpha = softmax(score, dim=-1)
    return alpha @ v

def attention_value(encoder_output_value, decoder_state):
    return qkv_attention_value(          # → Cross-Attention
        q=decoder_state[:, None, :],    #   Q ← decoder 状态
        k=encoder_output_value,          #   K ← encoder 输出
        v=encoder_output_value           #   V ← encoder 输出
    )
```

- `qkv_attention_value` 是**通用 QKV 计算引擎**（Scaled Dot-Product）
- `attention_value` 是把 Q/K/V **按 Cross-Attention 来源组装**的壳
- 如果未来你把 Q 的输入换成 `input[:, None, :]`（序列自己），这同一个函数就能实现 Self-Attention

**QKV 是公式，Attention 的类型由"谁当 Q，谁当 K/V"决定。**

## 一句话记忆

> Bahdanau 发明了 Attention 的对齐概念 → Scaled Dot-Product 用点积提速让它能堆叠 → Self-Attention 让序列自己看自己，彻底扔掉 RNN，催生了 Transformer。
