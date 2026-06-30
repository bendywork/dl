# 02 · Multi-Head Attention（MHA，多头注意力）

是 Transformer 论文 (2017) 的核心创新之一。不是凭空创造新机制，而是把 Scaled Dot-Product Attention **并行跑多份**。

## 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

head_i = Attention(Q·W_i^Q,  K·W_i^K,  V·W_i^V)
```

投影矩阵把 d_model 维的 Q/K/V 映射到 d_k = d_model / h 维子空间，各头在低维空间独立算 Attention，最后拼接回去。

## 为什么需要多头

| 问题 | 答案 |
|------|------|
| 一个头不够吗？ | 一个头只能学到一种"关注模式" |
| 多个头解决什么？ | 不同的头关注不同语义关系 |

```
Head 1: 学到"主语 ↔ 谓语"的语法关系
Head 2: 学到"形容词 ↔ 名词"的修饰关系
Head 3: 学到"代词 → 指代对象"的共指关系
Head 4: 学到位置相邻关系（局部依赖）
Head 5: 学到远距离句型结构关系
...
Head 8: 学到标点/分隔符的结构作用
```

8 个头不是"8 个人各自看一遍"，而是**8 个低维子空间并行算**，总计算量跟 1 个大头差不多（d_model/h 减小抵消了 head 数量增加）。

## 工程细节

```
d_model = 512
h = 8
d_k = d_v = 64

总计算量: O(8 × n² × 64) = O(n² × 512)
         ≈ 1 个 512 维大头的计算量

但 8 个头能学到 8 种不同的关注模式 → 表达能力远大于单头
```

## 工业应用

| 模型 | 配置 |
|------|------|
| Transformer 原版 | h=8, d_model=512 |
| BERT-base | h=12, d_model=768 |
| GPT-3 | h=96 (175B), d_model=12288 |
| LLaMA-7B | h=32, d_model=4096 |

**MHA 是所有 Transformer 的默认配置**。后续变体（MQA/GQA）是在 MHA 基础上做 KV 共享以减少推理代价。

## MHA 的推理解析瓶颈

每个 head 都有独立的 K 和 V，推理时 decode 阶段需要把所有 head 的 KV 拼起来做 Attention → 显存带宽被打满：

```
每生成 1 个 token → 读所有 head 的完整 KV cache → 带宽瓶颈
```

这就是 MQA 和 GQA 要解决的问题。
