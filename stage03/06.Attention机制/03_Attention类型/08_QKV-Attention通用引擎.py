# -*- coding: utf-8 -*-
"""
08 · QKV Attention 可视化 & 通用 QKV 引擎
============================================
对标 04_Seq2Seq+Attention理解.py 中 qkv_attention_value 的实现。
通用的 QKV 框架：指定谁当 Q/K/V 即可切换 Attention 类型。
"""
import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)


def qkv_attention(query, key, value):
    """
    Scaled Dot-Product Attention 通用引擎
    Q: (B, qt, d)   K: (B, kvt, d)   V: (B, kvt, d)
    """
    d = query.size(-1)
    score = torch.matmul(query, key.transpose(1, 2)) / np.sqrt(d)
    alpha = torch.softmax(score, dim=-1)
    return torch.matmul(alpha, value), alpha


# ===== 用同一个引擎切换 4 种 Attention 类型 =====
B, T, d = 1, 5, 64
seq = torch.randn(B, T, d)
dec_state = torch.randn(B, d)

print("1. Self-Attention: Q=K=V=序列自身")
out, w = qkv_attention(seq, seq, seq)
print(f"   Q={list(seq.shape)} K={list(seq.shape)} → out {list(out.shape)}\n")

print("2. Causal Self-Attention: 加 mask 让每个 token 只看左边")
mask = torch.tril(torch.ones(T, T)).unsqueeze(0)
d2 = seq.size(-1)
s = torch.matmul(seq, seq.transpose(1, 2)) / np.sqrt(d2)
s = s.masked_fill(mask == 0, float('-inf'))
out2 = torch.matmul(F.softmax(s, dim=-1), seq)
print(f"   Q={list(seq.shape)} + causal mask → out {list(out2.shape)}\n")

print("3. Cross-Attention: Q=decoder状态, K/V=序列")
out3, w3 = qkv_attention(dec_state.unsqueeze(1), seq, seq)
print(f"   Q={list(dec_state.unsqueeze(1).shape)} K={list(seq.shape)} → out {list(out3.shape)}\n")

print("4. Multi-Head 模拟: Q/K/V 各自先做线性投影再算 attention")
W_q = torch.randn(d, d) * 0.02
proj_q = seq @ W_q
out4, w4 = qkv_attention(proj_q, seq, seq)
print(f"   Q=projected {list(proj_q.shape)} K={list(seq.shape)} → out {list(out4.shape)}")
