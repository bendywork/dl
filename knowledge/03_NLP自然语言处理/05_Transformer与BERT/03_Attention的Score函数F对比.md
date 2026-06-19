# Attention 的 Score 函数 F：为什么数据量大时各种 F 效果等价

## 📌 核心问题
> Attention 里计算 Q 和 K 相关性分数的函数 F 有好几种（点积、加性、general），为什么在数据量增大时它们的效果基本等价？既然等价，为什么 Transformer 选了点积？

## 🌱 根源与动机

Attention 的核心是"算 Q 和 K 有多相关"，这个打分由 Score 函数 F 完成。F 的选择历史上有过几种方案，但实践中发现：**只要数据够多，选哪种 F 最终效果差不多。** 这背后有一个更深的原因——真正决定效果的不是 F 的形式，而是 Q、K 的表示空间学得好不好。

## 📐 Score 函数 F 的几种形式

| 名称 | 公式 | 可学参数 | 速度 |
|------|------|---------|------|
| **点积 (Dot-product)** | `F(Q,K) = Q·K` | 0 | 快 |
| **缩放点积 (Scaled)** | `F(Q,K) = Q·K / √d_k` | 0 | 快 |
| **加性 (Additive)** | `F(Q,K) = v^T·tanh(W_q·Q + W_k·K)` | 3个矩阵 | 慢 |
| **General** | `F(Q,K) = Q^T·W·K` | 1个矩阵 | 中 |

- 加性 Attention 是 Bahdanau（2014）最初用的，F 自带可学参数，理论更灵活
- 点积 Attention 是 Transformer（2017）用的，F 无参数，靠 W_q/W_k 把 Q/K 投影到合适空间

## 💡 关键理解：F 只打分，表示空间才是关键

### 灵活性从哪里来？

```
加性 Attention:  F 自带参数 (W_q, W_k, v) → F 本身就灵活
点积 Attention:  F 无参数，灵活性来自 W_q, W_k → 把 Q/K 投影到好空间
```

两种方式都能表达"Q 和 K 的相关性"，只是把灵活性放在了不同地方：
- 加性：灵活性在 F 内部
- 点积：灵活性在 F 外面的投影矩阵

### 为什么数据量增大时效果趋同？

```
数据少时：
  加性 F 靠自己学，有优势       → 效果略好
  点积 F 无参数，全靠 W_q/W_k
    但 W_q/W_k 还没学好         → 效果略差

数据多时：
  加性 F 学好了
  点积 F 的 W_q/W_k 也学好了
    → Q/K 投影到合适的表示空间
    → 点积也能算出准确的相关性
  → 两者效果趋同
```

**核心洞察：F 只负责"打分"，真正决定效果的是 Q 和 K 的表示空间。数据量大时，无论哪种 F，Q/K 的表示空间都被训练得足够好，F 用哪种形式都能算出准确分数。**

### 为什么选点积？—— 因为快

既然效果等价，就选最快的：

```
加性: tanh(W_q·Q + W_k·K) → 过神经网络，一个个算 → 慢
点积: Q @ K^T             → 一次矩阵乘法 → 快（GPU 并行吃满）
```

点积无额外参数、可全并行、省内存。**这就是 Transformer 用缩放点积的原因——不是因为它效果最好，而是它效果不差且最快。**

## 🔧 缩放 √d_k 的来由

点积有个隐患：d_k 越大，Q·K 的数值越大（向量维度多，加的项多），softmax 输入太大会饱和，梯度几乎为 0，训练不动。

```
点积:     Q·K           ← d_k 大时数值大，softmax 梯度消失
缩放点积: Q·K / √d_k    ← 压回去，恢复数值稳定性
```

加性 Attention 用 tanh 自然限制了输出范围，没有这个问题。**除以 √d_k 就是为了让"无参数的点积"在数值稳定性上追平"有参数的加性"。**

为什么是 √d_k 而不是 d_k？数学上，假设 Q、K 各分量独立、均值0、方差1，则 Q·K 的方差 = d_k，标准差 = √d_k。除以标准差能把方差拉回 1，这是最自然的归一化。

## 📊 完整公式对比

```
加性 Attention (Bahdanau 2014):
  score(Q, K) = v^T · tanh(W_q·Q + W_k·K)
  → 灵活性在 F 内部，慢但稳定

缩放点积 Attention (Transformer 2017):
  score(Q, K) = (Q·K) / √d_k
  → 灵活性在 W_q/W_k 投影，快但需缩放补救
```

## ⚠️ 易错点与常见误解

1. **以为 F 的形式决定效果好坏** → 真正决定效果的是 Q/K 的表示空间，F 只打分；数据够大时各种 F 趋同
2. **以为加性 Attention 比 Transformer 更先进** → 加性更早（2014），更灵活但更慢；Transformer（2017）用点积是工程优化
3. **以为点积不要 √d_k 也行** → d_k 大时数值爆炸，softmax 饱和，梯度消失；必须缩放
4. **以为 √d_k 是随便选的** → 它是 Q·K 的标准差，归一化到方差1，有数学依据
5. **以为效果等价就不需要选了** → 效果等价时选最快的，工程上点积是必然选择

## 🔗 知识延伸

- [[从RNN到Transformer架构演进]] — Bahdanau 用加性 Attention，Transformer 改用缩放点积，是演进的一环
- [[Self-Attention与Multi-Head-Attention]] — F 是 Self-Attention 内部计算 Q-K 分数的函数
- Multi-Head 的 d_k = d_model // n_heads → d_k 变小，缩放因子也变小，数值更稳定
- FlashAttention — 现代对点积 Attention 的工程优化，进一步加速

## 📚 参考资料

- Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014) — 加性 Attention
- Attention Is All You Need (Vaswani et al., 2017) — 论文 3.2.1 节明确对比点积 vs 加性，指出 dk 小时效果相近，点积更快
- 论文脚注：加性和点积"functionally similar in practice"
