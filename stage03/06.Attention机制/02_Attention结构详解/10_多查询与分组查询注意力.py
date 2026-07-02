"""
10_多查询与分组查询注意力 (MQA / GQA / MLA)
=============================================
KV Cache 是推理时的主要内存瓶颈. MQA/GQA/MLA 通过共享 KV 头来压缩 Cache.

目录:
  1. Multi-Head Attention (MHA)          → H 个独立 Q,K,V 头 (baseline)
  2. Multi-Query Attention (MQA)         → 所有 Q 头共享 1 组 K,V
  3. Grouped-Query Attention (GQA)       → G 组 K,V (1 < G < H)
  4. Multi-head Latent Attention (MLA)   → DeepSeek-V2, KV 压缩到低维 latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========== 1. Standard MHA (baseline) ==========
class MultiHeadAttention(nn.Module):
    """H 个独立的 Q,K,V 头. KV Cache 大小 = 2*H*d*L."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # TODO: Q,K,V,O projections
        pass
    # TODO: forward with optional kv_cache


# ========== 2. Multi-Query Attention (MQA) ==========
class MultiQueryAttention(nn.Module):
    """
    Shazeer 2019. 所有 Q 头共享 1 组 K,V.
    KV Cache: 2*d*L → 节省 H 倍.
    代价: 多头多样性降低, 质量略降.
    使用: PaLM, Falcon.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: Q: (num_heads * head_dim), K,V: (1 * head_dim)
        # TODO: K,V 投影输出维度是 head_dim 而非 num_heads * head_dim
        pass
    # TODO: forward — broadcast K,V to all heads


# ========== 3. Grouped-Query Attention (GQA) ==========
class GroupedQueryAttention(nn.Module):
    """
    Ainslie et al. 2023. G 组 K,V (1 < G < H).
    H//G 个 Q 头共享一组 K,V.
    在 MHA 质量和 MQA 效率之间取平衡.
    使用: LLaMA-2 70B (G=8, H=64), LLaMA-3, Mistral.
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads  # G
        self.head_dim = d_model // num_heads
        self.kv_groups = num_heads // num_kv_heads  # 每组 Q 头数
        # TODO: Q: num_heads * head_dim, K,V: num_kv_heads * head_dim
        pass
    # TODO: forward — repeat_interleave K,V 到 num_heads


# ========== 4. Multi-head Latent Attention (MLA) ==========
class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2 (2024). KV 压缩到低维 latent 空间, 推理时解压.
    核心:
      C_KV = x @ W_DKV           → 低维压缩 (latent_dim << d_model)
      K = C_KV @ W_UK, V = C_KV @ W_UV  → 解压
    Q 也做压缩: C_Q = x @ W_DQ → Q = C_Q @ W_UQ
    进一步节省 KV Cache, 且训练时只存 latent.
    DeepSeek-V2: d=5120, latent=512, 压缩比 ~10x.
    """
    def __init__(self, d_model: int, num_heads: int,
                 kv_lora_rank: int = 512, q_lora_rank: int = 1536):
        super().__init__()
        # TODO: W_DKV: down-projection for KV  (d_model → kv_lora_rank)
        # TODO: W_UK, W_UV: up-projection        (kv_lora_rank → num_heads*head_dim)
        # TODO: W_DQ, W_UQ: Q compression        (d_model → q_lora_rank → num_heads*head_dim)
        # TODO: RoPE 集成在解压后
        pass
    # TODO: forward(x, kv_cache) — 压缩 → 存 latent 到 cache → 解压后做 attention


if __name__ == "__main__":
    # TODO: compare_kv_cache_size — 对比 MHA/MQA/GQA/MLA 在 seq_len=4096 的 KB 数
    # TODO: quality_benchmark — 相同参数下对比 perplexity
    pass
