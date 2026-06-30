# -*- coding: utf-8 -*-
"""
LSTM 结构 — 数学公式 + 参数量 + 手动实现
TODO: 按下面框架逐步补全

【公式】（4 个门，参数量是 RNN 的 4 倍）
    i_t = sigmoid(X_t @ W_ii.T + b_ii + H_{t-1} @ W_hi.T + b_hi)  # input gate
    f_t = sigmoid(X_t @ W_if.T + b_if + H_{t-1} @ W_hf.T + b_hf)  # forget gate
    g_t = tanh   (X_t @ W_ig.T + b_ig + H_{t-1} @ W_hg.T + b_hg)  # cell gate
    o_t = sigmoid(X_t @ W_io.T + b_io + H_{t-1} @ W_ho.T + b_ho)  # output gate

    C_t = f_t * C_{t-1} + i_t * g_t   # cell state（长期记忆）
    H_t = o_t * tanh(C_t)              # hidden state（短期记忆/输出）

【参数量】
    PyTorch 将 4 个门的 W 拼成一个大矩阵：
    W_ih: [4*hidden, input]   → 4 * input * hidden
    W_hh: [4*hidden, hidden]  → 4 * hidden * hidden
    b_ih, b_hh: [4*hidden]    → 4 * hidden * 2
    ──────────────────────────────────────────────
    总计: 4 * hidden * (input + hidden + 2)
    比较: RNN 总计 = hidden * (input + hidden + 2)
         LSTM  = 4 × RNN 参数量

【与 RNN 的本质区别】
    RNN 只有 H_t（短期记忆），容易梯度消失
    LSTM 额外引入 C_t（Cell State，长期记忆）
        C_t 通过加法传播，梯度可以直通，缓解梯度消失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """手动实现单步 LSTM Cell"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # TODO: 定义 W_ih [4h, e], W_hh [4h, h], b_ih [4h], b_hh [4h]
        #       PyTorch 惯例：将 i/f/g/o 四个门的权重拼成一个大矩阵
        pass

    def forward(self, x, h_prev, c_prev):
        """
        x:      [bs, input_size]
        h_prev: [bs, hidden_size]
        c_prev: [bs, hidden_size]
        返回: (h_t, c_t)  各 [bs, hidden_size]
        """
        # TODO:
        # gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh  → [bs, 4h]
        # 切分四个门: i, f, g, o = gates.chunk(4, dim=-1)
        # i_t = sigmoid(i), f_t = sigmoid(f), g_t = tanh(g), o_t = sigmoid(o)
        # c_t = f_t * c_prev + i_t * g_t
        # h_t = o_t * tanh(c_t)
        pass


class LSTMLayer(nn.Module):
    """手动实现单层 LSTM（时间维度展开）"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, state=None):
        """
        x: [bs, t, input_size]
        state: (h0, c0) 或 None（全零初始化）
        返回:
            outputs: [bs, t, hidden_size]
            (h_n, c_n): 最后时刻状态
        """
        bs, t, _ = x.shape
        if state is None:
            h = torch.zeros(bs, self.hidden_size)
            c = torch.zeros(bs, self.hidden_size)
        else:
            h, c = state[0].squeeze(0), state[1].squeeze(0)

        outputs = []
        for i in range(t):
            # TODO: 调用 cell，收集每步输出
            pass
        # TODO: stack, return
        pass


def verify_against_api():
    """从 nn.LSTM 复制权重，验证手动实现输出一致"""
    # TODO
    pass


if __name__ == "__main__":
    verify_against_api()
