# -*- coding: utf-8 -*-
"""
04 · Self-Attention（自注意力）
=================================
Q = K = V = 同一序列。序列内部每个 token 与所有 token 交互。
带 causal mask → GPT 风格；不带 mask → BERT 风格。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class SelfAttention(nn.Module):
    def __init__(self, d_model=512, causal=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device))  # 下三角
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.bmm(weights, V), weights


if __name__ == '__main__':
    B, T, d = 2, 8, 64
    x = torch.randn(B, T, d)

    # BERT 风格：双向
    sa_bi = SelfAttention(d, causal=False)
    out, w = sa_bi(x)
    print(f"BERT-style (bidirectional): out={out.shape}, weights={w.shape}")

    # GPT 风格：因果 mask
    sa_causal = SelfAttention(d, causal=True)
    out, w = sa_causal(x)
    print(f"GPT-style (causal):       out={out.shape}")
    # 验证上三角全是 0
    print(f"Upper-triangle all zero: {(w[:, :, 1:, :-1].triu(1) == 0).all().item()}")
