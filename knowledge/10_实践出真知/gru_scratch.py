# -*- coding: utf-8 -*-
"""
GRU 结构 — 数学公式 + 参数量 + 手动实现
TODO: 按下面框架逐步补全

【公式】（2 个门，比 LSTM 少一个 Cell State）
    r_t = sigmoid(X_t @ W_ir.T + b_ir + H_{t-1} @ W_hr.T + b_hr)  # reset gate（重置门）
    z_t = sigmoid(X_t @ W_iz.T + b_iz + H_{t-1} @ W_hz.T + b_hz)  # update gate（更新门）
    n_t = tanh   (X_t @ W_in.T + b_in + r_t * (H_{t-1} @ W_hn.T + b_hn))  # new gate
    H_t = (1 - z_t) * n_t + z_t * H_{t-1}

【直觉理解】
    z_t（更新门）：控制"保留多少旧记忆"
        z_t → 1：H_t ≈ H_{t-1}，完全保留旧隐状态，忽略新输入
        z_t → 0：H_t ≈ n_t，完全用新候选值更新
    r_t（重置门）：控制"旧记忆对候选值的影响程度"
        r_t → 0：n_t 完全忽略旧隐状态，相当于重新开始

【参数量】
    W_ih: [3*hidden, input]   → 3 * input * hidden
    W_hh: [3*hidden, hidden]  → 3 * hidden * hidden
    b_ih, b_hh: [3*hidden]    → 3 * hidden * 2
    ──────────────────────────────────────────────
    总计: 3 * hidden * (input + hidden + 2)
    对比: RNN = 1×  LSTM = 4×  GRU = 3×（门更少，但保留了门控机制）

【与 LSTM 的关系】
    GRU 将 LSTM 的 forget/input gate 合并为 update gate z_t，
    取消了独立的 Cell State，用 H_t 一个向量同时承担长短期记忆。
    参数量更少，在序列不超长时效果与 LSTM 相当。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    """手动实现单步 GRU Cell"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # TODO: 定义权重
        # PyTorch 惯例：W_ih 按 r/z/n 顺序拼成 [3h, e]，W_hh [3h, h]
        pass

    def forward(self, x, h_prev):
        """
        x:      [bs, input_size]
        h_prev: [bs, hidden_size]
        返回 h_t: [bs, hidden_size]
        """
        # TODO:
        # gates_x = x @ W_ih.T + b_ih         → [bs, 3h]
        # r_x, z_x, n_x = gates_x.chunk(3, dim=-1)
        # gates_h = h_prev @ W_hh.T + b_hh    → [bs, 3h]
        # r_h, z_h, n_h = gates_h.chunk(3, dim=-1)
        #
        # r_t = sigmoid(r_x + r_h)
        # z_t = sigmoid(z_x + z_h)
        # n_t = tanh(n_x + r_t * n_h)          ← 注意：reset gate 只作用于 hidden 部分
        # h_t = (1 - z_t) * n_t + z_t * h_prev
        pass


class GRULayer(nn.Module):
    """手动实现单层 GRU（时间维度展开）"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        """
        x: [bs, t, input_size]
        返回:
            outputs: [bs, t, hidden_size]
            h_n:     [1, bs, hidden_size]
        """
        bs, t, _ = x.shape
        h = h0 if h0 is not None else torch.zeros(bs, self.hidden_size)
        outputs = []
        for i in range(t):
            # TODO: 调用 cell
            pass
        # TODO: stack, return
        pass


def compare_param_count():
    """对比 RNN / LSTM / GRU 参数量"""
    input_size, hidden_size = 128, 256
    rnn  = nn.RNN( input_size, hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    gru  = nn.GRU( input_size, hidden_size, batch_first=True)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print(f"RNN  参数量: {count(rnn):,}   理论: {hidden_size*(input_size+hidden_size+2):,}")
    print(f"GRU  参数量: {count(gru):,}   理论: {3*hidden_size*(input_size+hidden_size+2):,}")
    print(f"LSTM 参数量: {count(lstm):,}  理论: {4*hidden_size*(input_size+hidden_size+2):,}")


def verify_against_api():
    """从 nn.GRU 复制权重，验证手动实现输出一致"""
    # TODO
    pass


if __name__ == "__main__":
    compare_param_count()
    verify_against_api()
