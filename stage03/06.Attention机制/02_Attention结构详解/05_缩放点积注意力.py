"""
05_缩放点积注意力 (Scaled Dot-Product Attention)
================================================
Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V

核心公式来源: 《Attention Is All You Need》(Vaswani et al., 2017)
为什么除以 sqrt(d_k): 当 d_k 较大时, Q @ K^T 的方差 ≈ d_k,
  导致 softmax 进入饱和区(梯度≈0), 缩放后方差≈1, 梯度正常.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Q,K,V 形状: (batch, heads, seq_len, d_k) 或 (batch, seq_len, d_model)
    mask 形状: (batch, 1, seq_len, seq_len) 可广播
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = None  # 每次forward根据d_k动态设置

    # TODO: 实现 forward(Q, K, V, mask=None)
    # 1. d_k = Q.size(-1); scale = 1 / sqrt(d_k)
    # 2. scores = Q @ K.transpose(-2, -1) * scale
    # 3. if mask is not None: scores.masked_fill_(mask == 0, -1e9)
    # 4. attn_weights = F.softmax(scores, dim=-1)
    # 5. attn_weights = self.dropout(attn_weights)
    # 6. output = attn_weights @ V
    # 7. 返回 output, attn_weights (返回权重便于可视化)
    pass


# ---- 对比实验: 无缩放 vs 有缩放 ----
class DotProductAttentionNoScale(nn.Module):
    """不加 sqrt(d_k) 的版本, 用于对比梯度分布"""
    # TODO: 实现同上但去掉 scale
    pass


# ---- 辅助函数: 可视化梯度分布 ----
def compare_gradient_distribution(d_k: int = 64, seq_len: int = 128):
    """
    TODO: 生成随机 Q,K,V, 分别用有缩放和无缩放计算,
         对比 softmax 输出的熵和 Q 的梯度分布
    """
    pass


# ---- 测试入口 ----
if __name__ == "__main__":
    # TODO: 测试 forward 形状正确性
    # TODO: 测试 mask 有效性(被mask位置权重≈0)
    # TODO: 运行梯度分布对比实验
    pass
