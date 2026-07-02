"""
13_特殊结构注意力 (Axial / Hierarchical / Graph / Set)
==========================================================
针对特定数据结构(图像, 图, 集合, 树)设计的注意力模式.

目录:
  1. Axial Attention            → 行列分离, 图像O(N*sqrt(N))
  2. Hierarchical Attention     → 金字塔结构, 粗细粒度兼顾
  3. Graph Attention (GAT)      → 图结构归约的注意力
  4. Set Attention (PMA/SAB)    → 置换不变集合级注意力
  5. Image/Video Transformer    → ViT, Swin, TimeSformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========== 1. Axial Attention ==========
class AxialAttention(nn.Module):
    """
    Ho et al. 2019.
    对 2D 图像: 分别在行列上做 attention.
    步骤:
      1. Row Attention:  对每行独立做 self-attn  → O(H * W^2) → 实际 W * O(W^2)
      2. Column Attention: 对每列独立做 self-attn → O(W * H^2)
    总复杂度: O(H*W*W + W*H*H) = O(N * max(H,W))

    使用: Axial Transformer, Axial-DeepLab (语义分割).
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Q,K,V projections
        pass
    # TODO: forward(x_2d) — x: (B, H, W, C)
    # 1. reshape → (B*H, W, C), attention, reshape back
    # 2. transpose → (B*W, H, C), attention, reshape back


# ========== 2. Hierarchical Attention ==========
class HierarchicalAttention(nn.Module):
    """
    金字塔式的多层注意力:
    1. 底层: fine-grained local attention (高分辨率)
    2. 上层: coarse-grained global attention (合并后低分辨率)
    每一层将 token 聚合(池化/卷积/Merge token), 减少token数.
    使用: Swin Transformer (窗口+shift窗口), Hiera, MViT.
    """
    # TODO: multi-scale token merging (e.g. patch merge 2x2→1)
    # TODO: 每层 attention + downsample


# ========== 3. Graph Attention (GAT) ==========
class GraphAttentionLayer(nn.Module):
    """
    Veličković et al. 2018 (GAT).
    节点只对邻居计算 attention (边决定连接).
    e_ij = LeakyReLU(a^T [W*h_i; W*h_j])  — additive score
    α_ij = softmax_j(e_ij)  — 仅邻居上归一化
    h_i' = σ(sum_j α_ij * W*h_j)

    关键区别: attention 仅在图的边上计算, 不是全连接.
    还有 GATv2 (动态注意力), GatedGAT.
    """
    # TODO: __init__(in_dim, out_dim, num_heads)
    # TODO: forward(x, edge_index) — edge_index: (2, num_edges)
    # TODO: 实现 multi-head 版本的 concatenation / averaging


# ========== 4. Set Attention (Set Transformer) ==========
class SetAttention(nn.Module):
    """
    Lee et al. 2019 (Set Transformer).
    输入是无序集合, 要求置换不变性.

    SAB (Set Attention Block):  集合内 self-attention (置换等变)
    ISAB (Induced SAB):         引入 K 个 inducing points 降低复杂度
    PMA (Pooling by Multihead Attention): 用 K 个可学习 seed queries 聚合集合 → 置换不变输出
    """
    # TODO: SAB — 标准 self-attention over set elements
    # TODO: ISAB — M_ISAB = SAB(seed_Q, SAB(S))  O(N*K) 复杂度
    # TODO: PMA — K seed vectors attend to N elements, 输出固定大小


# ========== 5. Image/Video Transformer 常用模式 ==========
class ViTAttention(nn.Module):
    """标准 ViT: patch embedding + position embedding + Transformer Encoder"""
    # TODO: patchify(image) → (N, P^2*C), 加 class_token + pos_embed
    pass

class SwinAttention(nn.Module):
    """Swin Transformer: shifted window, local attention in windows."""
    # TODO: window_partition → attention → window_reverse
    # TODO: shifted window masking for cyclic shift
    pass

class TimeSformerAttention(nn.Module):
    """
    视频 Transformer: 分离空间注意力和时间注意力.
    - Spatial: 每帧内独立 patches 做 attention
    - Temporal: 同一空间位置的不同帧做 attention
    """
    # TODO: divided space-time attention (两轮交替)
    pass


if __name__ == "__main__":
    # TODO: test_axial — 对比 Full(16384) vs Axial 的显存和速度
    # TODO: test_gat — 在 Cora/CiteSeer 数据集上验证
    # TODO: test_set — 验证 PMA 置换不变性: shuffle 输入, 输出相同
    pass
