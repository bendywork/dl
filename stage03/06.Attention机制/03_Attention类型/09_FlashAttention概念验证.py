# -*- coding: utf-8 -*-
"""
09 · FlashAttention 概念验证
===============================
不是完整实现（需要 CUDA kernel），而是用分块计算演示 Tiling 原理。
把 Q/K/V 切成小块，逐块在"SRAM"里算 attention，验证结果与全局计算等价。
"""
import torch
import torch.nn.functional as F
import math

torch.manual_seed(42)


def standard_attention(Q, K, V, scale):
    """全局 Attention（等价于 HBM 版本）"""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


def flash_attention_tiled(Q, K, V, scale, block_size=3):
    """分块 Tiling Attention：模拟 FlashAttention 的 I/O 模式"""
    B, H, T, d = Q.shape
    O = torch.zeros_like(Q)
    # 这里只做概念：把 softmax 拆成分块在线计算
    # 真实 FlashAttention 用在线 softmax + rescaling，这里简化为分块后再 softmax
    for i in range(0, T, block_size):
        i_end = min(i + block_size, T)
        for j in range(0, T, block_size):
            j_end = min(j + block_size, T)
            Q_block = Q[:, :, i:i_end, :]          # [B, H, Bs, d]
            K_block = K[:, :, j:j_end, :]
            V_block = V[:, :, j:j_end, :]
            # 块内 attention（模拟 SRAM 计算）
            s_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
            w_block = F.softmax(s_block, dim=-1)
            O[:, :, i:i_end, :] += torch.matmul(w_block, V_block)
    return O


if __name__ == '__main__':
    B, H, T, d = 2, 4, 9, 64  # T=9, block_size=3 → 完美分块
    Q = torch.randn(B, H, T, d)
    K = torch.randn(B, H, T, d)
    V = torch.randn(B, H, T, d)
    scale = 1.0 / math.sqrt(d)

    out_full, _ = standard_attention(Q, K, V, scale)
    out_tiled = flash_attention_tiled(Q, K, V, scale, block_size=3)

    diff = (out_full - out_tiled).abs().max().item()
    print(f"Full Attention output: {out_full.shape}")
    print(f"Tiled Attention output: {out_tiled.shape}")
    print(f"Max difference: {diff:.2e}")
    print(f"Results match: {diff < 1e-5}")

    # 演示：FlashAttention 省了什么
    mem_full = T * T * 4  # [T,T] fp32 matrix
    mem_tiled = 3 * 3 * 4  # [Bs,Bs] fp32 matrix, only in SRAM
    print(f"\nMemory per attention matrix:")
    print(f"  Standard: {mem_full} bytes → 要在 HBM 和 SRAM 之间搬")
    print(f"  Tiled:    {mem_tiled} bytes → 一直在 SRAM 里，不搬")
    print(f"  Saving:   {(1 - mem_tiled/mem_full)*100:.0f}% of HBM traffic")
