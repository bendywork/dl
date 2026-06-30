# 07 · Cross-Attention 与 Causal Attention

## Cross-Attention（交叉注意力）

### 核心：Q 和 K/V 来源不同

```
Cross-Attention:
  Q ← decoder 当前状态（"我(decoder)现在想找什么信息？"）
  K ← encoder 输出序列（"源句(encoder)每个位置有什么特征？"）
  V ← encoder 输出序列（"源句每个位置的实际内容是什么？"）
```

**数学上就是一个公式**：`Attention(Q_dec, K_enc, V_enc)`。区别只在谁当 Q，谁当 K/V。

### 工业应用

| 场景 | Q 来源 | K/V 来源 |
|------|--------|---------|
| Seq2Seq 翻译 | decoder 隐状态 | encoder 所有 token 输出 |
| Stable Diffusion | 噪声图像的 latent | 文本 prompt 的 embedding |
| DETR 目标检测 | 学到的 object query | 图像 backbone 特征图 |
| Whisper 语音识别 | decoder token | 音频 encoder 特征 |

### 跟 Self-Attention 的区别

| | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| Q/K/V | 全部同一序列 | Q 和 K/V 不同来源 |
| 谁关注谁 | 序列内部互相查 | 一个序列去查另一个序列 |
| 典型用途 | 理解（BERT）、生成（GPT） | 翻译、跨模态（图→文） |

## Causal Attention（因果注意力 / Masked Self-Attention）

### 核心：只能往左看

```
不加 mask:
  Q₁ Q₂ Q₃ Q₄
K₁ ✓  ✓  ✓  ✓      ← Q₁ 能看所有 K
K₂ ✓  ✓  ✓  ✓
K₃ ✓  ✓  ✓  ✓
K₄ ✓  ✓  ✓  ✓

加 causal mask:
  Q₁ Q₂ Q₃ Q₄
K₁ ✓  ✗  ✗  ✗      ← Q₁ 只能看 K₁
K₂ ✓  ✓  ✗  ✗      ← Q₂ 只能看 K₁ K₂（自己及左边）
K₃ ✓  ✓  ✓  ✗
K₄ ✓  ✓  ✓  ✓      ← Q₄ 能看全部（因为 4 是最后一个）
```

实现：计算完 score 后，把 Q_i 对应位置 j>i 的 score 设为 `-inf` → softmax 之后这些位置的注意力权重自动变成 0。

### 为什么需要因果 mask

GPT 的预训练任务 "预测下一个 token" 是左到右的生成任务——训练时**每个位置只能根据前面的 token 预测自己**，不能往后看。往后偷看未来 token 才是作弊，causal mask 是反作弊机制。

```
训练时: 给定 "我 今天 吃" → 预测 "了"
         加 mask 确保 "吃" 看不到 "了"

推理时: 一个一个生成，新 token 追加到序列末尾 → 新增 token 能看到所有已生成的内容
        不需要额外 mask（因为新 token 在所有旧 token 右边）
```

### 工业应用

| 模型/场景 | Attention 类型 |
|-----------|---------------|
| GPT 1/2/3/4 系列 | Causal Self-Attention（纯解码器） |
| LLaMA 1/2/3 | Causal Self-Attention |
| BERT | 无 mask 的 Self-Attention（双向） |
| Transformer 原版解码器 | Causal Self-Attention + Cross-Attention |

## 一句话

> Cross-Attention 是"我问你答"（decoder 问，encoder 答），Causal Attention 是"只能往回看"（防止偷窥未来 token）。
