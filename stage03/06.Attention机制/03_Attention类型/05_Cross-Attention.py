# -*- coding: utf-8 -*-
"""
05 · Cross-Attention（交叉注意力）
====================================
Q 来自 decoder，K/V 来自 encoder。解码器每步查询编码器的全部输出。
Seq2Seq Decoder、Stable Diffusion、DETR 的核心组件。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


class CrossAttention(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_hidden, encoder_output):
        """
        decoder_hidden: (B, T_dec, d_model)  ← Q
        encoder_output: (B, T_enc, d_model)  ← K, V
        """
        Q = self.W_q(decoder_hidden)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.bmm(weights, V), weights


class Seq2SeqDecoderBlock(nn.Module):
    """Decoder 子层：Causal Self-Attn → Cross-Attn → FFN"""
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 8, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(1)))
        tgt = self.norm1(tgt + tgt2)
        tgt2, _ = self.cross_attn(tgt, memory)
        tgt = self.norm2(tgt + tgt2)
        return tgt


if __name__ == '__main__':
    B, T_enc, T_dec, d = 2, 20, 8, 512
    enc_out = torch.randn(B, T_enc, d)
    dec_in = torch.randn(B, T_dec, d)

    # 纯 Cross-Attention
    ca = CrossAttention(d)
    out, w = ca(dec_in, enc_out)
    print(f"Cross-Attn: decoder {dec_in.shape} + encoder {enc_out.shape}")
    print(f"  → output {out.shape}, weights {w.shape}")
    print(f"  → each decoder pos attends {w.size(-1)} encoder positions")

    # Decoder Block
    block = Seq2SeqDecoderBlock(d)
    out = block(dec_in, enc_out)
    print(f"\nDecoderBlock output: {out.shape}")
