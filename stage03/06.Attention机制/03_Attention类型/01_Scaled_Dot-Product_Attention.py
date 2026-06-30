# -*- coding: utf-8 -*-
"""
01 · Scaled Dot-Product Attention（放缩点积注意力）
====================================================
公式: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V
所有后续 Attention 变体的数学基础。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, heads, seq_len, d_k)
        mask:    (batch, 1, seq_len, seq_len)  or None
        """
        d_k = Q.size(-1)
        # ① Q·K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # ② mask (可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # ③ softmax → 权重
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        # ④ 加权求和
        output = torch.matmul(weights, V)
        return output, weights


if __name__ == '__main__':
    B, H, T, d_k = 2, 4, 10, 64
    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_k)

    attn = ScaledDotProductAttention(dropout=0.1)
    out, w = attn(Q, K, V)
    print(f"Input:  Q/K/V {Q.shape}")
    print(f"Output: {out.shape}")          # [2, 4, 10, 64]
    print(f"Weights: {w.shape}")           # [2, 4, 10, 10]
    print(f"Weight sum per query: {w.sum(dim=-1)[0,0,:3]}")  # 每行 ≈1.0
