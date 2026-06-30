# 04 · FlashAttention（闪存注意力）

Attention 的原始实现有个问题：**O(n²) 的中间矩阵要在 HBM（高带宽显存）和 SRAM 之间反复搬运**。FlashAttention 不改变数学模型，只改变 I/O 路径。

## 核心问题

```
标准 Attention 计算:
  Q·Kᵀ → [n, n] 矩阵 → 写回 HBM (80GB, ~1.5TB/s)
        → 从 HBM 读回 → softmax → 写回 HBM
        → 从 HBM 读回 → ·V → 写回 HBM

每一步都在 HBM ↔ SRAM 之间搬运一个 O(n²) 的矩阵
n=4096 → 4K×4K = 16M 个元素 → 128MB per layer
```

GPU 的 SRAM（片上缓存）只有 ~20MB，放不下 O(n²) 的 Attention 矩阵，所以只能放 HBM。**但 HBM 比 SRAM 慢 10 倍以上。** 计算瓶颈不在于乘加，在于搬运。

## FlashAttention 解法：Tiling + Recomputation

```
把 Q, K, V 切成小块，一块一块地在 SRAM 里算 attention，永远不把完整的 [n,n] 矩阵写回 HBM

块算法:
  for block_i in Q:
    for block_j in K:
      S_ij = Q_i · K_jᵀ            ← 在 SRAM 里算
      S_ij = S_ij / √d_k            ← 在 SRAM 里做
      P_ij = softmax(S_ij)          ← 在线 softmax（不需要全局 max）
      O_i += P_ij · V_j            ← 累加到输出
```

**效果**：完全消除了 O(n²) 矩阵的 HBM 读写。

## 三代演进

| 版本 | 年份 | 改进 |
|------|------|------|
| FlashAttention-1 | 2022 | Tiling + Recomputation，训练速度 ×2-4 |
| FlashAttention-2 | 2023 | 更好的 warp 调度 + 减少非矩阵乘操作，速度再 ×2 |
| FlashAttention-3 | 2024 | 支持 FP8 + Hopper 架构异步运行，速度再 ×1.5-2 |

## 工业应用

| 模型 | 使用 |
|------|------|
| GPT-4 | FlashAttention-2（训练加速 2-3×） |
| LLaMA 2/3 | FlashAttention-2 |
| Mistral | FlashAttention-2 + Sliding Window |
| Claude | FlashAttention（推测） |

PyTorch 2.0+ 内置：`F.scaled_dot_product_attention(q, k, v)`，自动走最优后端（FlashAttention / Memory-Efficient / Math）。

## 本质

> Standard Attention: O(n²) 显存读写 → 慢在搬运
> FlashAttention: 分块在 SRAM 里算完整 Attention → 快在省搬运
>
> 数学上完全等价，工程上颠覆了 Transformer 训练的经济性。
