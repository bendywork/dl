"""
09_低秩与核注意力 (Low-Rank & Kernel Attention)
================================================
通过数学近似将 O(n^2) 降到 O(n), 核心思路:
  - 低秩:   Key/Value 序列投影到固定低维空间
  - 核方法: 用核函数 φ 使得 softmax(QK^T)V ≈ φ(Q)(φ(K)^T V)
  - 注意: softmax 无法直接分解为内积积, 需要近似

目录:
  1. Linformer (Wang et al. 2020)         → 低秩投影 K,V
  2. Performer / FAVOR+ (2021)             → 正随机特征近似 softmax
  3. Linear Attention (cosFormer 等)       → 替换 softmax 为线性核
  4. Nyströmformer (2021)                  → Nyström 矩阵近似
  5. Softmax-Free Attention                → 去掉 softmax, 直接用线性激活
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ========== 1. Linformer ==========
class LinformerAttention(nn.Module):
    """
    核心: K,V 投影到低维 (seq_len → k_proj), k_proj << seq_len.
    K' = E @ K, V' = F @ V  其中 E,F: (k_proj, seq_len) 可学习/固定
    output = softmax(Q @ K'^T / sqrt(d)) @ V'
    复杂度: O(n*k*d) 当 k_proj << n 时近似线性
    """
    def __init__(self, d_model: int, num_heads: int, k_proj: int = 256,
                 share_kv_proj: bool = True):
        super().__init__()
        # TODO: Q,K,V projections
        # TODO: self.E = nn.Parameter(torch.randn(k_proj, max_seq_len))
        # TODO: self.F = nn.Parameter if not share_kv_proj else self.E
        pass
    # TODO: forward(x) — K' = E @ K, V' = F @ V, 然后标准 attention


# ========== 2. Performer / FAVOR+ ==========
class PerformerAttention(nn.Module):
    """
    Choromanski et al. 2021.
    用正随机特征 (Positive Random Features) 近似 softmax:
      softmax(QK^T) ≈ φ(Q) @ φ(K)^T
    其中 φ(x) = exp(||x||^2/2) / sqrt(m) * [exp(w_1^T x), ..., exp(w_m^T x)]
    w_i ~ N(0, I_d)

    计算顺序: φ(Q) @ (φ(K)^T @ V)  先算 K^T @ V: O(m*d*n)
    """
    def __init__(self, d_model, num_heads, num_features=256, ortho_features=True):
        super().__init__()
        # TODO: register random projection matrix: (d_model//heads, num_features)
        # TODO: ortho_features: 对投影矩阵做正交化(更稳定)
        pass
    # TODO: _feature_map(x) — 实现 φ(x) = exp(||x||^2/2) * exp(x@W) / sqrt(m)
    # TODO: forward(Q,K,V) — Q'=φ(Q), K'=φ(K), 先 KV=K'^T @ V, 再 Q'@KV
    # TODO: causal 版本 — 需要前缀和技巧: 累积 K'^T @ V


# ========== 3. Linear Attention (cosFormer) ==========
class CosFormerAttention(nn.Module):
    """
    Qin et al. 2022.
    用 cos 重加权 + ReLU 核替换 softmax:
      score = (Q cos(θ_i)) @ (K cos(θ_j))^T
      φ(x) = ReLU(x)  (或 ELU+1)
      θ_i = i * π / (2 * N)  — 位置相关的 cos 权重, 保持局部偏置
    """
    # TODO: 预计算 cos 权重
    # TODO: forward: Q' = ReLU(Q) * cos_weight; K' = ReLU(K) * cos_weight
    # TODO: output = Q' @ (K'^T @ V)  复杂度 O(n*d^2)


class FlowFormerAttention(nn.Module):
    """
    Flow Network based. 用流网络替代 softmax 做归一化.
    """
    pass


# ========== 4. Nyströmformer ==========
class NystromformerAttention(nn.Module):
    """
    Xiong et al. 2021.
    用 Nyström 方法近似 attention 矩阵:
      从序列中选 m 个 landmark, 用 landmark 间的 attention + landmark-token attention
      重构完整的 n×n attention 矩阵.
    复杂度: O(n*m*d)
    """
    # TODO: _select_landmarks(x) — 均值分割 / 随机 / K-Means
    # TODO: forward — 计算 landmark Q,K → 小矩阵 Q_l@K_l^T → 近似还原


# ========== 5. Softmax-Free Attention ==========
class SoftmaxFreeAttention(nn.Module):
    """
    SOFT (Lu et al. 2021): 用 Gaussian kernel 替代 softmax
    Linear Transformer (Katharopoulos 2020): ELU+1 特征映射, O(n*d^2)
    """
    # TODO: φ(x) = ELU(x) + 1  (保证正值)
    # TODO: KV_state = K'^T @ V; output = Q' @ KV_state / (Q' @ K_sum)


if __name__ == "__main__":
    # TODO: test_linformer — 对比 Linformer k_proj=128 与 full attention 的输出差异
    # TODO: test_performer — 验证近似误差 ||φ(Q)φ(K)^T - softmax(QK^T)||
    # TODO: benchmark — seq_len=[512,1024,2048,4096] 各方案速度内存对比
    pass
