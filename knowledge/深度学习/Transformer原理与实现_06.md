# Transformer 原理与实现

## 📌 核心问题
> RNN 顺序计算难并行、长程依赖弱，Transformer 如何用纯 Attention 替代 RNN？

## 🌱 根源与动机

Attention 已经证明能捕捉长程依赖，那能不能**去掉 RNN，完全用 Attention 建模序列**？

2017 年 "Attention Is All You Need" 给出答案：
- 用 **Multi-Head Self-Attention** 让序列内所有位置两两交互
- 用 **Position Encoding** 弥补没有位置顺序的缺陷
- **并行计算全部 token**，训练速度远超 RNN

## 📐 理论推导

### Self-Attention（缩放点积注意力）

```
Q = X·Wq,  K = X·Wk,  V = X·Wv

Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
```

- 除以 `√dₖ` 防止点积过大导致 softmax 梯度消失
- 每个 token 同时作为 query 去查询所有 key

### Multi-Head Attention

```
head_i = Attention(QWᵢq, KWᵢk, VWᵢv)
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wₒ
```

多头的意义：**同时关注不同子空间的信息**（如语法、语义、指代等）

### 位置编码（Sinusoidal）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/dmodel))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))
```

绝对位置 → 固定编码；相对位置 → 可通过线性变换表达

### Encoder Block

```
x → MultiHead Self-Attention → Add&Norm
  → FFN (两层线性+ReLU) → Add&Norm
```

### Decoder Block（多一个 Cross-Attention）

```
y → Masked Self-Attention → Add&Norm  (防止看到未来)
  → Cross-Attention(Q=decoder, K=V=encoder) → Add&Norm
  → FFN → Add&Norm
```

## 💡 关键理解

| 组件 | 作用 |
|------|------|
| Self-Attention | 序列内全局交互，O(n²d) |
| Masked Attention | Decoder 自回归，不看未来 |
| Cross-Attention | Decoder 查询 Encoder 输出（≈ Seq2Seq 的 Attention）|
| Add&Norm | 残差连接 + LayerNorm，稳定深层训练 |
| FFN | 位置独立的非线性变换，扩大表达能力 |

## 🔧 代码实现

对应代码：`knowledge/深度学习/代码实践/Transformer/`
- `01_Self-Attention理解.py` — 从零实现 Self-Attention

## ⚠️ 易错点与常见误解

1. **Self-Attention 没有位置信息**，必须加 Position Encoding；RNN 天然有顺序
2. **Decoder Masked Attention** 是下三角 mask，确保 t 时刻只能看 1~t
3. **Cross-Attention 的 Q 来自 Decoder，K/V 来自 Encoder**——不要搞反
4. **LayerNorm 在 Add 之后**（Pre-LN 变体是在 Add 之前，更稳定）
5. `√dₖ` 的 `dₖ` 是 head 维度，不是全模型维度

## 🔗 知识延伸

- [[Seq2Seq与Attention机制]] — Transformer 的 Cross-Attention 就是 Seq2Seq Attention 的泛化
- [[BERT]] — Encoder-only Transformer，双向预训练
- [[位置编码进阶]] — RoPE、ALiBi 是 Sinusoidal 的改进，现代 LLM 主流

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/10_Transformer.pdf`
- 原论文：Vaswani et al. 2017 "Attention Is All You Need"
