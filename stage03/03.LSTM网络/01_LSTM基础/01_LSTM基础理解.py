# -*- coding: utf-8 -*-
"""
LSTM 基础理解案例
目标：用"记笔记"的直觉，理解 LSTM 四个门的作用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 直觉类比
# ─────────────────────────────────────────────
# 把 LSTM 想象成一个"带橡皮的笔记本"：
#
#   ct（细胞状态） = 笔记本内容（长期记忆）
#   ht（隐藏状态） = 当前正在看的那一页（短期记忆 / 输出）
#
#   遗忘门 ft  = 橡皮：决定把旧笔记擦掉多少
#   输入门 it  = 新信息的采纳比例：决定新内容写进去多少
#   候选值 c̃t = 新信息草稿：这一步想写什么
#   输出门 ot  = 翻到第几页：决定从笔记本里读出什么
# ─────────────────────────────────────────────


class ManualLSTMCell(nn.Module):
    """手动实现单步 LSTM Cell，完全透明四个门"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # 四组权重：输入 x → 门
        self.W_f = nn.Linear(input_size, hidden_size)   # 遗忘门
        self.W_i = nn.Linear(input_size, hidden_size)   # 输入门
        self.W_c = nn.Linear(input_size, hidden_size)   # 候选值
        self.W_o = nn.Linear(input_size, hidden_size)   # 输出门

        # 四组权重：上一步隐状态 h → 门
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        """
        x      : [bs, input_size]  当前时刻输入
        h_prev : [bs, hidden_size] 上一时刻隐状态
        c_prev : [bs, hidden_size] 上一时刻细胞状态
        """
        # 遗忘门：要不要把旧记忆擦掉？  → 0=全擦  1=全留
        ft = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))

        # 输入门：新信息要不要写进去？   → 0=不写  1=全写
        it = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))

        # 候选值：这一步想写什么内容？   → [-1, 1]
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        # 更新细胞状态：橡皮擦旧的 + 写入新的
        ct = ft * c_prev + it * c_tilde

        # 输出门：从笔记本里读哪部分出来？
        ot = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))

        # 最终隐状态输出
        ht = ot * torch.tanh(ct)

        return ht, ct


class ManualLSTM(nn.Module):
    """沿时间步展开 ManualLSTMCell"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x):
        """x: [bs, seq_len, input_size]"""
        bs, seq_len, _ = x.shape
        h = torch.zeros(bs, self.hidden_size)
        c = torch.zeros(bs, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h.unsqueeze(1))   # [bs, 1, hidden]

        return torch.cat(outputs, dim=1), (h, c)  # [bs, seq_len, hidden]


# ─────────────────────────────────────────────
# 案例 1：观察四个门的数值行为
# ─────────────────────────────────────────────
def demo_gate_values():
    print("=" * 55)
    print("案例1：观察四个门在一个时间步上的数值")
    print("=" * 55)

    input_size, hidden_size = 4, 3
    cell = ManualLSTMCell(input_size, hidden_size)

    x      = torch.randn(1, input_size)    # 一个样本，一个时间步
    h_prev = torch.zeros(1, hidden_size)
    c_prev = torch.zeros(1, hidden_size)

    with torch.no_grad():
        # 手动拆出每一步（只是为了打印，不影响 forward）
        ft      = torch.sigmoid(cell.W_f(x) + cell.U_f(h_prev))
        it      = torch.sigmoid(cell.W_i(x) + cell.U_i(h_prev))
        c_tilde = torch.tanh(   cell.W_c(x) + cell.U_c(h_prev))
        ot      = torch.sigmoid(cell.W_o(x) + cell.U_o(h_prev))
        ct      = ft * c_prev + it * c_tilde
        ht      = ot * torch.tanh(ct)

    print(f"遗忘门 ft   (0=全擦, 1=全留): {ft.squeeze().tolist()}")
    print(f"输入门 it   (0=不写, 1=全写): {it.squeeze().tolist()}")
    print(f"候选值 c̃t  [-1,1]:          {c_tilde.squeeze().tolist()}")
    print(f"输出门 ot   (0=不读, 1=全读): {ot.squeeze().tolist()}")
    print(f"细胞状态 ct:                 {ct.squeeze().tolist()}")
    print(f"隐状态  ht（输出）:           {ht.squeeze().tolist()}")


# ─────────────────────────────────────────────
# 案例 2：手动 LSTM 与 nn.LSTM 输出对齐验证
# ─────────────────────────────────────────────
def demo_manual_vs_pytorch():
    print("\n" + "=" * 55)
    print("案例2：手动实现 vs nn.LSTM 输出误差")
    print("=" * 55)

    bs, seq_len, input_size, hidden_size = 2, 6, 4, 3

    # PyTorch 官方 LSTM
    torch.manual_seed(42)
    ref_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=True)

    # 手动 LSTM，拷贝同一套权重
    manual = ManualLSTM(input_size, hidden_size)

    # 从 ref_lstm 中提取权重并拷贝到手动模块
    # nn.LSTM 将四组权重拼在一起：顺序是 i / f / g(c̃) / o
    with torch.no_grad():
        wih = ref_lstm.weight_ih_l0  # [4*H, E]
        whh = ref_lstm.weight_hh_l0  # [4*H, H]
        bih = ref_lstm.bias_ih_l0    # [4*H]
        bhh = ref_lstm.bias_hh_l0    # [4*H]

        H = hidden_size
        # PyTorch 顺序：i, f, g, o  → 我们顺序：f, i, c, o
        order = [1, 0, 2, 3]   # f=1, i=0, c̃=2, o=3
        linears_W = [manual.cell.W_f, manual.cell.W_i,
                     manual.cell.W_c, manual.cell.W_o]
        linears_U = [manual.cell.U_f, manual.cell.U_i,
                     manual.cell.U_c, manual.cell.U_o]

        for idx, pytorch_idx in enumerate(order):
            linears_W[idx].weight.copy_(wih[pytorch_idx*H:(pytorch_idx+1)*H])
            linears_W[idx].bias.copy_(
                bih[pytorch_idx*H:(pytorch_idx+1)*H] +
                bhh[pytorch_idx*H:(pytorch_idx+1)*H]
            )
            linears_U[idx].weight.copy_(whh[pytorch_idx*H:(pytorch_idx+1)*H])

    x = torch.randn(bs, seq_len, input_size)

    with torch.no_grad():
        ref_out, (ref_hn, ref_cn) = ref_lstm(x)
        man_out, (man_hn, man_cn) = manual(x)

    max_err = (ref_out - man_out).abs().max().item()
    print(f"输出序列最大误差: {max_err:.2e}  (应 < 1e-5)")
    print(f"最终 hn 最大误差: {(ref_hn.squeeze(0) - man_hn).abs().max().item():.2e}")
    print(f"最终 cn 最大误差: {(ref_cn.squeeze(0) - man_cn).abs().max().item():.2e}")
    if max_err < 1e-5:
        print("✅ 手动实现与 nn.LSTM 完全对齐")
    else:
        print("❌ 存在较大误差，检查权重复制逻辑")


# ─────────────────────────────────────────────
# 案例 3：简单序列预测 —— sin 波下一步
# ─────────────────────────────────────────────
def demo_sin_prediction():
    print("\n" + "=" * 55)
    print("案例3：用 LSTM 预测 sin 波下一个点")
    print("=" * 55)

    import math

    # 构造数据：用前 seq_len 个点预测第 seq_len+1 个点
    seq_len = 10
    n_samples = 200
    t = torch.linspace(0, 4 * math.pi, n_samples + seq_len)
    data = torch.sin(t)

    # 滑动窗口切片
    X = torch.stack([data[i:i+seq_len] for i in range(n_samples)]).unsqueeze(-1)  # [N, T, 1]
    Y = data[seq_len:n_samples+seq_len].unsqueeze(-1)                              # [N, 1]

    # 模型：LSTM + 线性头
    model = nn.Sequential()
    lstm_layer = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 16, batch_first=True)
            self.fc   = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # 取最后时刻

    model = LSTMRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.6f}")

    # 随机检查几个预测
    with torch.no_grad():
        sample_idx = [0, 50, 100, 150]
        print("\n真实值 vs 预测值（sin 波下一个点）：")
        for i in sample_idx:
            true_val = Y[i].item()
            pred_val = model(X[i:i+1]).item()
            print(f"  样本{i:3d}  真实={true_val:+.4f}  预测={pred_val:+.4f}  "
                  f"误差={abs(true_val-pred_val):.4f}")


if __name__ == '__main__':
    demo_gate_values()
    demo_manual_vs_pytorch()
    demo_sin_prediction()
