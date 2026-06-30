# -*- coding: utf-8 -*-
"""
02 · Multi-Head Attention（多头注意力）
=========================================
Transformer 原版 Attention: 8 个头在不同子空间并行算，然后拼接。
公式: MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
      head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        # Q/K/V 投影 + 输出投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        # 投影 + 拆多头: (B,T,d) → (B,h,T,d_k)
        Q = self.W_q(Q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        # Scaled Dot-Product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        # 拼回头: (B,h,T,d_k) → (B,T,d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out), weights


if __name__ == '__main__':
    B, T, d_model = 2, 10, 512
    x = torch.randn(B, T, d_model)
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    out, w = mha(x, x, x)
    print(f"Input:  {x.shape}")        # [2, 10, 512]
    print(f"Output: {out.shape}")       # [2, 10, 512]
    print(f"Weights: {w.shape}")        # [2, 8, 10, 10]
    print(f"Params: {sum(p.numel() for p in mha.parameters()):,}")  # ~1M
