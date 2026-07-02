"""
06_Bahdanau与Luong注意力 (Seq2Seq时代的经典)
================================================
这两个是 Seq2Seq + Attention 时代的奠基工作, 在 Transformer 之前统治 NMT.

Bahdanau (Additive/Concat, 2015):
  score(h_dec, h_enc) = v^T * tanh(W @ [h_dec; h_enc])
  - 将 decoder hidden state 与每个 encoder hidden state 拼接后过 MLP
  - 论文: https://arxiv.org/abs/1409.0473

Luong (Multiplicative, 2015):
  三种 score 函数:
  - dot:   h_dec^T @ h_enc
  - general: h_dec^T @ W @ h_enc
  - concat: v^T @ tanh(W @ [h_dec; h_enc])  (同 Bahdanau)
  - 还区分了 global(全序列) 和 local(窗口) attention
  - 论文: https://arxiv.org/abs/1508.04025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Additive / Concat Attention — Seq2Seq 解码器的标准注意力.
    对每个 decoder hidden state, 计算其与所有 encoder states 的 score.
    """
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int):
        super().__init__()
        # TODO: W_enc = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        # TODO: W_dec = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        # TODO: v = nn.Linear(attn_dim, 1, bias=False)  -- 标量score
        pass

    # TODO: forward(encoder_outputs, decoder_hidden)
    # encoder_outputs: (batch, src_len, enc_hidden_dim)
    # decoder_hidden:   (batch, dec_hidden_dim)
    # 1. 扩展 dec_hidden 到 (batch, src_len, dec_hidden_dim)
    # 2. energy = tanh(W_enc(enc_out) + W_dec(dec_expanded))
    # 3. scores = v(energy).squeeze(-1)  → (batch, src_len)
    # 4. attn_weights = F.softmax(scores, dim=-1)
    # 5. context = sum(attn_weights * encoder_outputs) → (batch, enc_hidden_dim)
    # 返回 context, attn_weights


class LuongAttention(nn.Module):
    """
    Multiplicative Attention — 三种 score 模式.
    global: 对所有 encoder states 计算 attention
    local:  仅在窗口内计算 (预测对齐位置后滑动窗口)
    """
    def __init__(self, hidden_dim: int, score_mode: str = "general"):
        """
        score_mode: "dot" | "general" | "concat"
        """
        super().__init__()
        self.score_mode = score_mode
        # TODO: 仅 general 模式需要 Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # TODO: concat 模式需要 Wa + va 两个线性层
        pass

    # TODO: score(h_dec, h_enc) → (batch, src_len)
    # dot:     torch.bmm(h_dec.unsqueeze(1), h_enc.transpose(1,2)).squeeze(1)
    # general: torch.bmm(Wa(h_dec).unsqueeze(1), h_enc.transpose(1,2)).squeeze(1)
    # concat:  va(torch.tanh(Wa(torch.cat([h_dec_expanded, h_enc], -1)))).squeeze(-1)

    # TODO: forward(h_dec, h_enc, mode="global", window_size=None)
    # 计算 context vector 和 attention weights

    # TODO: _local_attention(h_dec, h_enc, window_size)  Luong 独有的局部注意力


# ---- Seq2Seq Decoder with Attention ----
class AttentiveDecoder(nn.Module):
    """
    把 Bahdanau/Luong attention 嵌入到 RNN decoder 中.
    典型用法: 每步 decoder 输出 → attention → context → 与 decoder state 拼接 → 预测 token
    """
    # TODO: __init__ 接收 attention 模块 + embedding + rnn + output_projection
    # TODO: forward(encoder_outputs, target_sequence, teacher_forcing_ratio)
    # TODO: 实现完整的 decode 循环, 返回 outputs + attention_weights


if __name__ == "__main__":
    # TODO: 测试 Bahdanau: 随机 encoder/decoder, 检查 context 形状
    # TODO: 测试 Luong 三种 score_mode
    # TODO: 对比 dot vs general vs concat 的参数量和速度
    pass
