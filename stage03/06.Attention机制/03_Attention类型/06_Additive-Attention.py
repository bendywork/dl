# -*- coding: utf-8 -*-
"""
06 · Additive Attention（Bahdanau 加法注意力, 2014）
=====================================================
最早的 Attention 实现。不是 Q·K^T，而是用一个小网络算相关度。
公式: score(q,k) = v^T · tanh(W_q·q + W_k·k)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        query:  (B, query_dim)         — decoder 当前隐状态
        keys:   (B, T, key_dim)        — encoder 所有时刻输出
        values: (B, T, value_dim)
        返回: context (B, value_dim), weights (B, T)
        """
        q = self.W_q(query).unsqueeze(1)            # (B, 1, hidden)
        k = self.W_k(keys)                           # (B, T, hidden)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)           # (B, T)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # (B, V_dim)
        return context, weights


if __name__ == '__main__':
    B, T, d = 2, 10, 128
    dec_state = torch.randn(B, d)           # decoder 当前状态
    enc_out = torch.randn(B, T, d)          # encoder 输出

    attn = BahdanauAttention(query_dim=d, key_dim=d, hidden_dim=64)
    ctx, w = attn(dec_state, enc_out, enc_out)
    print(f"Decoder state: {dec_state.shape}")
    print(f"Encoder out:   {enc_out.shape}")
    print(f"Context:       {ctx.shape}")        # [2, 128]
    print(f"Weights:       {w.shape}")          # [2, 10]
    print(f"Weight sum:   {w.sum(dim=-1)}")     # [1.0, 1.0]
