# Seq2Seq 与 Attention 机制

## 📌 核心问题
> RNN 能处理序列，但如何从一个序列**映射到另一个不等长序列**？Attention 解决了什么痛点？

## 🌱 根源与动机

RNN/LSTM 输出一个固定长度的 hidden state，当输入序列很长时，早期信息会被"挤压"丢失——这就是 **长程依赖瓶颈**。

Seq2Seq 将问题拆成两个角色：
- **Encoder**：将变长输入压缩为一个上下文向量 `c`
- **Decoder**：以 `c` 为起点，逐步解码生成目标序列

瓶颈在于：**所有信息只靠一个向量 `c` 传递**，序列越长损失越大。

Attention 的思路：**解码每一步时，让 Decoder 动态地"回头看" Encoder 的所有输出**，而不是只看最后一个 hidden state。

## 📐 理论推导

### Seq2Seq 基本结构

```
Encoder:  x₁,x₂,...,xₙ  → h₁,h₂,...,hₙ  → c = hₙ
Decoder:  c → s₀ → y₁ → s₁ → y₂ → ... → yₘ
```

### Attention 计算步骤

1. **相似度打分**（多种方式，最常用点积）：
   ```
   eᵢⱼ = sⱼ₋₁ · hᵢ     (Luong: dot)
   eᵢⱼ = vᵀ tanh(W[sⱼ₋₁; hᵢ])  (Bahdanau: additive)
   ```

2. **Softmax 归一化**得到注意力权重：
   ```
   αᵢⱼ = softmax(eᵢⱼ)
   ```

3. **加权求和**得到上下文向量：
   ```
   cⱼ = Σᵢ αᵢⱼ · hᵢ
   ```

4. **Decoder 输入** = `[sⱼ₋₁, yⱼ₋₁, cⱼ]`

## 💡 关键理解

- Attention 本质是 **soft retrieval**：给一个 query，从 key-value 库中加权检索
- 权重 α 可视化后就是"对齐矩阵"——翻译任务中能清晰看出源语言与目标语言词语的对应关系
- **Teacher Forcing**：训练时 Decoder 用真实标签而非上一步预测作为输入，加速收敛但会有 exposure bias

## 🔧 代码实现

对应代码：`knowledge/深度学习/代码实践/Seq2Seq/`
- `01_Seq2Seq理解_v0.py` — 最简 Seq2Seq（无 Attention）
- `02_Seq2Seq理解_v1.py` — 加入 Teacher Forcing
- `03_Seq2Seq理解_v2.py` — 完整带 Padding Mask 版本
- `04_Seq2Seq+Attention理解_v0.py` — 加入 Bahdanau Attention

## ⚠️ 易错点与常见误解

1. **Attention 权重之和为 1**（softmax），但每个时间步的 c 不同，不是共享同一个
2. **Encoder 最后 hidden 和 Decoder 初始 hidden**：最后 hidden 作为 Decoder s₀，但之后每步都靠 Attention 重算 c，不是只用一次
3. **Teacher Forcing 只在训练时用**，推理时用模型自身输出
4. **Seq2Seq 输出用 `<EOS>` 标记结束**，推理时遇到 `<EOS>` 停止

## 🔗 知识延伸

- [[Transformer]] — Self-Attention 把 Attention 机制推广到所有位置之间，不依赖 RNN
- [[BERT]] — Encoder-only Transformer，双向 Attention
- 对联生成项目 `projects/TextGeneration/CoupletGeneration/` 用了 Seq2Seq + BERT4Torch

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/08_Seq2Seq.pdf`、`09_Attention.pdf`
