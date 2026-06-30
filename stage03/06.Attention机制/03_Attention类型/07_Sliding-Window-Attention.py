# -*- coding: utf-8 -*-
"""
07 · Sliding Window Attention（滑动窗口注意力）
=================================================
每个 token 只 attend 前后 w 个 token，复杂度 O(n·w)。
Mistral 7B 的核心优化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model=512, window_size=4, dropout=0.1):
        super().__init__()
        self.W = window_size
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # 全量算 scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)  # [B,T,T]
        # 构造窗口 mask
        idx = torch.arange(T, device=x.device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        mask = (dist <= self.W // 2).unsqueeze(0)                  # [1,T,T]
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.bmm(weights, V), weights


if __name__ == '__main__':
    B, T, d = 2, 12, 64
    x = torch.randn(B, T, d)
    sw = SlidingWindowAttention(d_model=d, window_size=4)
    out, w = sw(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    # 验证 window=4 时每个 token 最多关注 5 个位置
    per_token = (w[0] > 0).sum(dim=-1)
    print(f"Per-token attention count: {per_token}")
    print(f"Max per token: {per_token.max().item()}, Min: {per_token.min().item()}")
