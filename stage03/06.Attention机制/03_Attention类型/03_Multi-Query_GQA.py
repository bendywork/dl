# -*- coding: utf-8 -*-
"""
03 · Multi-Query & Grouped-Query Attention
=============================================
MQA: 所有 head 共享同一份 K/V，只有 Q 是多头（PaLM, Falcon）
GQA: head 分成 G 组，组内共享 K/V（LLaMA 2/3）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class MultiQueryAttention(nn.Module):
    """MQA: 1 份 K/V，h 份 Q"""
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.W_q = nn.Linear(d_model, d_model)               # h 份 Q
        self.W_k = nn.Linear(d_model, self.d_k)              # 1 份 K 共享
        self.W_v = nn.Linear(d_model, self.d_k)              # 1 份 V 共享
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)  # [B,h,T,d_k]
        K = self.W_k(x).unsqueeze(1)                                    # [B,1,T,d_k]
        V = self.W_v(x).unsqueeze(1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.W_o(out)


class GroupedQueryAttention(nn.Module):
    """GQA: G 组 K/V，h 份 Q（h 是 G 的倍数）"""
    def __init__(self, d_model=512, num_heads=8, num_groups=4, dropout=0.1):
        super().__init__()
        assert num_heads % num_groups == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.g = num_groups
        self.heads_per_group = num_heads // num_groups
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k * num_groups)
        self.W_v = nn.Linear(d_model, self.d_k * num_groups)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)    # [B,h,T,d_k]
        K = self.W_k(x).view(B, T, self.g, self.d_k).transpose(1, 2)    # [B,g,T,d_k]
        V = self.W_v(x).view(B, T, self.g, self.d_k).transpose(1, 2)
        # 每组 K/V 重复 heads_per_group 次
        K = K.repeat_interleave(self.heads_per_group, dim=1)              # [B,h,T,d_k]
        V = V.repeat_interleave(self.heads_per_group, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.W_o(out)


if __name__ == '__main__':
    B, T, d = 2, 10, 512
    x = torch.randn(B, T, d)

    mqa = MultiQueryAttention(d_model=512, num_heads=8)
    gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_groups=4)

    print("MQA KV params:", sum(p.numel() for n, p in mqa.named_parameters() if 'W_k' in n or 'W_v' in n))
    print("GQA KV params:", sum(p.numel() for n, p in gqa.named_parameters() if 'W_k' in n or 'W_v' in n))
    print(f"MHA KV params: {512*512*2:,}")  # 对比 MHA
    print(f"MQA out: {mqa(x).shape}")
    print(f"GQA out: {gqa(x).shape}")
