# -*- coding: utf-8 -*-
"""
RNN 结构 — 数学公式 + 参数量 + 手动实现
TODO: 按下面框架逐步补全

【公式】
    H_t = tanh(X_t @ W_ih.T + b_ih + H_{t-1} @ W_hh.T + b_hh)

【参数量】
    W_ih: [hidden, input]   → input * hidden
    b_ih: [hidden]          → hidden
    W_hh: [hidden, hidden]  → hidden * hidden
    b_hh: [hidden]          → hidden
    ─────────────────────────
    总计: input*hidden + hidden*hidden + 2*hidden
          = hidden*(input + hidden + 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    """手动实现单步 RNN Cell"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: 定义 W_ih, W_hh, b_ih, b_hh
        pass

    def forward(self, x, h_prev):
        """
        x:      [bs, input_size]
        h_prev: [bs, hidden_size]
        返回 h: [bs, hidden_size]
        """
        # TODO: H_t = tanh(x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh)
        pass


class RNNLayer(nn.Module):
    """手动实现单层 RNN（在时间维度展开）"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None):
        """
        x:  [bs, t, input_size]
        返回:
            outputs: [bs, t, hidden_size]  每个时刻的隐状态
            h_n:     [1, bs, hidden_size]  最后时刻的隐状态
        """
        bs, t, _ = x.shape
        h = h0 if h0 is not None else torch.zeros(bs, self.hidden_size)
        outputs = []
        for i in range(t):
            # TODO: 调用 cell，收集每步输出
            pass
        # TODO: stack outputs, return (outputs, h_n)
        pass


def verify_against_api():
    """用 nn.RNN 官方 API 验证手动实现数值一致"""
    # TODO: 从 nn.RNN 复制权重到 RNNLayer，对比最大误差
    pass


if __name__ == "__main__":
    verify_against_api()
