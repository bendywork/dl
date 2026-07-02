"""
12_跨注意力与门控注意力 (Cross-Attention & Gated Attention)
==============================================================
Self-Attention: Q,K,V 同源. Cross-Attention: Q 和 K,V 来自不同序列.
广泛用于: Encoder-Decoder, 多模态, 扩散模型 (Stable Diffusion).

目录:
  1. Cross-Attention              → Q(decoder) × K,V(encoder)
  2. Gated Cross-Attention        → 门控机制控制信息流
  3. Memory-Compressed Attention  → 压缩外部记忆
  4. Multi-Scale Cross-Attention  → 多尺度特征融合
  5. Deformable Attention (DETR)  → 可变形稀疏采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 1. Cross-Attention ==========
class CrossAttention(nn.Module):
    """
    标准 Encoder-Decoder Cross-Attention.
    - Self-Attention: Q=K=V=decoder_hidden
    - Cross-Attention: Q=decoder_hidden, K=V=encoder_hidden
    使用: Transformer NMT, T5, Stable Diffusion (text→image)
    """
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int):
        super().__init__()
        # TODO: Q 投影 (q_dim → num_heads*d_k)
        # TODO: K,V 投影 (kv_dim → num_heads*d_k) — 注意 K,V 也可以共享投影
        pass
    # TODO: forward(q, kv) — q 来自 decoder, kv 来自 encoder/condition
    # TODO: 对于 SD: q=latent_image, kv=text_embedding


# ========== 2. Gated Cross-Attention ==========
class GatedCrossAttention(nn.Module):
    """
    Alayrac et al. 2022 (Flamingo).
    在 cross-attention 后加 tanh gating, 控制外部信息的注入量.
    output = x + tanh(α) * CrossAttn(LN(x), context)
    门 α 初始化为 0 → 初始阶段仅使用预训练权重, 平滑引入新模态.
    """
    def __init__(self, q_dim, kv_dim, num_heads):
        super().__init__()
        # TODO: self.cross_attn = CrossAttention(...)
        # TODO: self.alpha = nn.Parameter(torch.zeros(1))  # 初始为0
        pass
    # TODO: forward(x, context) — x + tanh(alpha) * cross_attn(norm(x), context)


class GatedSelfAttention(nn.Module):
    """
    控制 self-attention 信息流. 用于:
    - 可学习的注意力 dropout
    - 稀疏门控: 只对重要 token 计算 attention
    """
    pass


# ========== 3. Memory-Compressed Attention ==========
class MemoryCompressedAttention(nn.Module):
    """
    Liu et al. 2018.
    将过去 context 压缩成固定大小的 memory slots.
    Mem = Conv1d(stride=k) @ past_KV  → 压缩 k 倍
    每层维护固定数量 memory, 复杂度与序列长度无关.
    """
    # TODO: compress(past_kv) — 用 1D 卷积 / 池化压缩
    # TODO: forward(x, memory) — 当前 token 同时注意 memory + 当前 context
    pass


# ========== 4. Multi-Scale Cross-Attention ==========
class MultiScaleCrossAttention(nn.Module):
    """
    对多尺度 feature map 做 cross-attention.
    使用: 目标检测 (Deformable DETR), 分割 (Mask2Former), 超分.
    每层对对应分辨率的 feature 做 attention.
    """
    # TODO: 对每个 scale 的 feature 独立做 cross-attn, 再融合


# ========== 5. Deformable Attention ==========
class DeformableAttention(nn.Module):
    """
    Zhu et al. 2020 (Deformable DETR).
    不计算全部 token 的 attention, 而是:
    1. 对每个 query 预测 K 个采样点坐标 (offset + 参考点)
    2. 从 feature map 的采样点位置插值取值
    3. 只对 K 个采样点做 attention (K << HW)
    复杂度: O(N_query * K * d)  对于图像通常是 O(HW * K * d)

    也适用于: 视频理解, 对稀疏采样帧做 attention.
    """
    def __init__(self, d_model, num_heads, num_points=4):
        super().__init__()
        # TODO: self.offset_proj = nn.Linear(d_model, num_heads*num_points*2)  # 2D offsets
        # TODO: self.attn_weights_proj = nn.Linear(d_model, num_heads*num_points)
        pass
    # TODO: forward(query, reference_points, feature_map)
    # 1. offsets = offset_proj(query) → (N, heads, points, 2)
    # 2. sampling_points = reference_points + offsets
    # 3. sampled_features = grid_sample(feature_map, sampling_points)
    # 4. attn = softmax(attn_weights_proj(query)) × sampled_features
    # 5. output = sum over points


if __name__ == "__main__":
    # TODO: test_cross_attn — decoder(seq=10) attends to encoder(seq=20)
    # TODO: test_gated — 验证 alpha 初始化为0时 cross-attn 不影响输出
    # TODO: visualize_deformable — 可视化采样点偏移 (类似论文 Fig.3)
    pass
