# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/14 15:06
Create User : 19410
Desc : xxx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def tt_with_lstm01():
    bs, t, e = 1, 10, 4
    v = 3
    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=True,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(bs, t, e)
    print(f"LSTM输入的特征向量维度:{token_embs.shape}")

    # 调用lstm
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)
    print(f"LSTM每个时刻的输出特征向量:\n\t{lstm_output.shape}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_output[:, -1]}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_h_n}")
    print(f"LSTM最后一个时刻的状态向量ct:\n\t{lstm_c_n}")


def tt_with_lstm02():
    """
    LSTM过程公式拆解
    :return:
    """
    bs, t, e = 1, 5, 4
    v = 3

    # ──────────────────────────────────────────────────────
    # 关于 u / w 命名的理解：
    #
    # 课本公式通常写成：f_t = σ(W_f · [h_{t-1}, x_t])
    # 即把 h_{t-1} 和 x_t 拼接后用一个大矩阵 W_f 变换。
    #
    # 代码这里把这个"大W"拆成两个矩阵分别乘，完全等价：
    #   f_t = σ( x_t @ u_ft  +  h_{t-1} @ w_ft )
    #              ↑ u：作用在当前输入 x_t 上
    #                              ↑ w：作用在上一步输出 h_{t-1} 上
    #
    # u_xy 命名规则：u_[门名][t]，表示该门里处理"输入x"的权重
    # w_xy 命名规则：w_[门名][t]，表示该门里处理"隐状态h"的权重
    # ──────────────────────────────────────────────────────

    u_ft = nn.Parameter(torch.randn(e, v))  # 遗忘门：x_t 对应的权重矩阵 [e, v]
    w_ft = nn.Parameter(torch.randn(v, v))  # 遗忘门：h_{t-1} 对应的权重矩阵 [v, v]

    u_it = nn.Parameter(torch.randn(e, v))  # 输入门：x_t 对应的权重矩阵
    w_it = nn.Parameter(torch.randn(v, v))  # 输入门：h_{t-1} 对应的权重矩阵

    u_ct = nn.Parameter(torch.randn(e, v))  # 候选记忆（C̃_t）：x_t 对应的权重矩阵
    w_ct = nn.Parameter(torch.randn(v, v))  # 候选记忆（C̃_t）：h_{t-1} 对应的权重矩阵

    u_ot = nn.Parameter(torch.randn(e, v))  # 输出门：x_t 对应的权重矩阵
    w_ot = nn.Parameter(torch.randn(v, v))  # 输出门：h_{t-1} 对应的权重矩阵

    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=False,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # PyTorch 内部把四个门的权重合并存储：
    #   weight_ih_l0: [4v, e]，存的是 [i, f, g(候选), o] 四个门的 u 矩阵，.T 后变 [e, 4v]
    #   weight_hh_l0: [4v, v]，存的是 [i, f, g(候选), o] 四个门的 w 矩阵，.T 后变 [v, 4v]
    # split 按列切成 4 份，每份 v 列，分别对应 i/f/g/o 四个门
    u_it, u_ft, u_ct, u_ot = torch.split(lstm.weight_ih_l0.T, split_size_or_sections=v, dim=1)
    w_it, w_ft, w_ct, w_ot = torch.split(lstm.weight_hh_l0.T, split_size_or_sections=v, dim=1)

    # bs个样本，每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为e
    token_embs = torch.rand(bs, t, e)  # [bs, t, e]

    new_token_embs_list = []
    ht = torch.zeros((bs, v))  # h_0：初始隐藏状态，全零
    ct = torch.zeros((bs, v))  # c_0：初始长期记忆，全零
    for _t in range(t):
        xt = token_embs[:, _t, :]  # 取第 _t 步的输入，shape: [bs, e]

        # 遗忘门 f_t = σ(x_t @ u_ft + h_{t-1} @ w_ft)
        # 作用：决定上一步长期记忆 c_{t-1} 里哪些要忘掉（接近0忘，接近1留）
        ft = F.sigmoid(torch.matmul(xt, u_ft) + torch.matmul(ht, w_ft))

        # 输入门 i_t = σ(x_t @ u_it + h_{t-1} @ w_it)
        # 作用：决定候选记忆 cur_ct 里哪些要写入长期记忆（接近0不写，接近1写入）
        it = F.sigmoid(torch.matmul(xt, u_it) + torch.matmul(ht, w_it))

        # 候选记忆 C̃_t = tanh(x_t @ u_ct + h_{t-1} @ w_ct)
        # 作用：计算"想写入长期记忆的候选内容"（不是最终写入，要乘 it 才写）
        cur_ct = F.tanh(torch.matmul(xt, u_ct) + torch.matmul(ht, w_ct))

        # 输出门 o_t = σ(x_t @ u_ot + h_{t-1} @ w_ot)
        # 作用：决定长期记忆 c_t 里哪些信息输出成 h_t
        ot = F.sigmoid(torch.matmul(xt, u_ot) + torch.matmul(ht, w_ot))

        # 更新长期记忆 c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t
        # 遗忘门控制"保留多少旧记忆"，输入门控制"写入多少新内容"
        ct = ct * ft + cur_ct * it

        # 计算当前步输出 h_t = o_t ⊙ tanh(c_t)
        # 输出门决定把长期记忆里哪些内容暴露给外部
        ht = ot * F.tanh(ct)

        oi = ht[:, None]  # [bs, v] -> [bs, 1, v]，增加时间维方便后面拼接
        new_token_embs_list.append(oi)

    new_token_embs = torch.cat(new_token_embs_list, dim=1)  # 所有时刻拼接 [bs, t, v]
    print(new_token_embs.shape)

    # 与 PyTorch 官方 LSTM 对比，误差应接近 0（验证手动实现正确性）
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)
    print(torch.max(torch.abs(new_token_embs - lstm_output)))


if __name__ == '__main__':
    # tt_with_lstm01()
    tt_with_lstm02()
