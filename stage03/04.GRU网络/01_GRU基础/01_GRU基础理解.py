# -*- coding: utf-8 -*-
"""
GRU 基础理解案例
目标：理解 GRU 相比 LSTM 的简化思路，以及两个门的作用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 直觉类比：GRU vs LSTM
# ─────────────────────────────────────────────
# LSTM 有三个门 + 独立的细胞状态 ct
# GRU  把它们合并，只剩两个门 + 一个隐状态 ht：
#
#   重置门 rt  = "选择性失忆"：决定上一步记忆中有多少影响候选值
#               → rt=0：候选值完全不受历史影响（全新开始）
#               → rt=1：候选值充分参考历史（完整延续）
#
#   更新门 zt  = "新旧混合比"：直接控制输出 ht 中新旧信息的比例
#               → zt=0：ht = 候选值（完全更新，丢弃旧记忆）
#               → zt=1：ht = h_{t-1}（完全保留，不学习新信息）
#
# 公式：
#   rt    = sigmoid(Wr·x  + Ur·h_{t-1})
#   zt    = sigmoid(Wz·x  + Uz·h_{t-1})
#   h̃t   = tanh(Wh·x  + Uh·(rt ⊙ h_{t-1}))   候选值
#   ht    = (1 - zt) ⊙ h̃t  +  zt ⊙ h_{t-1}   输出
# ─────────────────────────────────────────────


class ManualGRUCell(nn.Module):
    """手动实现单步 GRU Cell，完全透明两个门"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # 重置门
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # 更新门
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        # 候选值：注意 PyTorch GRU 的候选值 hh bias 需要在乘 r 之后再加
        # 所以 W_h 持有 bias_ih_n，U_h 持有 bias_hh_n（独立，不与 W_h 合并）
        self.W_h   = nn.Linear(input_size, hidden_size)             # 含 bias_ih_n
        self.U_h   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_hh_n = nn.Parameter(torch.zeros(hidden_size))        # bias_hh_n 独立

    def forward(self, x, h_prev):
        """
        x      : [bs, input_size]
        h_prev : [bs, hidden_size]
        """
        rt = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        zt = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        # 候选值：bias_hh_n 乘以 r 后再加
        h_tilde = torch.tanh(self.W_h(x) + rt * (self.U_h(h_prev) + self.b_hh_n))
        ht = (1 - zt) * h_tilde + zt * h_prev
        return ht


class ManualGRU(nn.Module):
    """沿时间步展开 ManualGRUCell"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ManualGRUCell(input_size, hidden_size)

    def forward(self, x):
        """x: [bs, seq_len, input_size]"""
        bs, seq_len, _ = x.shape
        h = torch.zeros(bs, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1), h  # [bs, seq_len, hidden], [bs, hidden]


# ─────────────────────────────────────────────
# 案例 1：观察两个门的数值
# ─────────────────────────────────────────────
def demo_gate_values():
    print("=" * 55)
    print("案例1：观察两个门在一个时间步上的数值")
    print("=" * 55)

    input_size, hidden_size = 4, 3
    cell = ManualGRUCell(input_size, hidden_size)

    x      = torch.randn(1, input_size)
    h_prev = torch.randn(1, hidden_size)   # 有一些历史记忆

    with torch.no_grad():
        rt      = torch.sigmoid(cell.W_r(x) + cell.U_r(h_prev))
        zt      = torch.sigmoid(cell.W_z(x) + cell.U_z(h_prev))
        h_tilde = torch.tanh(cell.W_h(x) + rt * (cell.U_h(h_prev) + cell.b_hh_n))
        ht      = (1 - zt) * h_tilde + zt * h_prev

    print(f"重置门 rt (0=失忆, 1=延续): {rt.squeeze().tolist()}")
    print(f"更新门 zt (0=全新, 1=全旧): {zt.squeeze().tolist()}")
    print(f"候选值 h̃t [-1,1]:          {h_tilde.squeeze().tolist()}")
    print(f"旧隐状态  h_prev:           {h_prev.squeeze().tolist()}")
    print(f"新隐状态  ht（输出）:        {ht.squeeze().tolist()}")

    # 直觉验证：zt 越大，ht 越接近 h_prev
    for i in range(hidden_size):
        blend = zt.squeeze()[i].item()
        print(f"  维度{i}: zt={blend:.2f} → "
              f"{'偏旧记忆' if blend > 0.5 else '偏新信息'}")


# ─────────────────────────────────────────────
# 案例 2：手动 GRU 与 nn.GRU 输出对齐验证
# ─────────────────────────────────────────────
def demo_manual_vs_pytorch():
    print("\n" + "=" * 55)
    print("案例2：手动实现 vs nn.GRU 输出误差")
    print("=" * 55)

    torch.manual_seed(0)
    bs, seq_len, input_size, hidden_size = 2, 6, 4, 3

    ref_gru = nn.GRU(input_size, hidden_size, batch_first=True, bias=True)
    manual  = ManualGRU(input_size, hidden_size)

    # nn.GRU 权重拼接顺序：r, z, n(h̃)
    with torch.no_grad():
        wih = ref_gru.weight_ih_l0   # [3*H, E]
        whh = ref_gru.weight_hh_l0   # [3*H, H]
        bih = ref_gru.bias_ih_l0
        bhh = ref_gru.bias_hh_l0
        H   = hidden_size

        # r=0, z=1, h̃=2
        gates = [(manual.cell.W_r, manual.cell.U_r, 0),
                 (manual.cell.W_z, manual.cell.U_z, 1),
                 (manual.cell.W_h, manual.cell.U_h, 2)]

        for W, U, idx in gates:
            W.weight.copy_(wih[idx*H:(idx+1)*H])
            U.weight.copy_(whh[idx*H:(idx+1)*H])

        # r 和 z 的 bias 可合并（不受 r 影响）
        manual.cell.W_r.bias.copy_(bih[0*H:1*H] + bhh[0*H:1*H])
        manual.cell.W_z.bias.copy_(bih[1*H:2*H] + bhh[1*H:2*H])
        # 候选值：bias_ih_n 放 W_h.bias，bias_hh_n 独立存放
        manual.cell.W_h.bias.copy_(bih[2*H:3*H])
        manual.cell.b_hh_n.copy_(bhh[2*H:3*H])

    x = torch.randn(bs, seq_len, input_size)

    with torch.no_grad():
        ref_out, ref_hn = ref_gru(x)
        man_out, man_hn = manual(x)

    max_err = (ref_out - man_out).abs().max().item()
    print(f"输出序列最大误差: {max_err:.2e}  (应 < 1e-5)")
    print(f"最终 hn 最大误差: {(ref_hn.squeeze(0) - man_hn).abs().max().item():.2e}")
    print("✅ 对齐成功" if max_err < 1e-5 else "❌ 存在误差，检查权重复制")


# ─────────────────────────────────────────────
# 案例 3：GRU vs LSTM 参数量对比
# ─────────────────────────────────────────────
def demo_param_count():
    print("\n" + "=" * 55)
    print("案例3：GRU vs LSTM 参数量对比")
    print("=" * 55)

    input_size, hidden_size = 64, 128

    gru  = nn.GRU( input_size, hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    gru_params  = count_params(gru)
    lstm_params = count_params(lstm)

    print(f"输入维度={input_size}, 隐藏维度={hidden_size}")
    print(f"GRU  参数量: {gru_params:,}   (3 组权重: r/z/h̃)")
    print(f"LSTM 参数量: {lstm_params:,}  (4 组权重: i/f/g/o)")
    print(f"GRU 比 LSTM 少 {(1 - gru_params/lstm_params)*100:.1f}% 参数")
    print()
    print("理论公式：")
    H, E = hidden_size, input_size
    print(f"  GRU  = 3 × (E×H + H×H + H) = 3 × ({E}×{H} + {H}×{H} + {H}) = {3*(E*H+H*H+H):,}")
    print(f"  LSTM = 4 × (E×H + H×H + H) = 4 × ({E}×{H} + {H}×{H} + {H}) = {4*(E*H+H*H+H):,}")


# ─────────────────────────────────────────────
# 案例 4：极端门值下的行为演示
# ─────────────────────────────────────────────
def demo_extreme_gates():
    print("\n" + "=" * 55)
    print("案例4：极端门值行为演示（直觉理解）")
    print("=" * 55)

    H = 4
    h_prev   = torch.ones(1, H) * 0.8   # 旧记忆
    h_tilde  = torch.ones(1, H) * (-0.5)  # 候选新值

    # 场景A：zt ≈ 1 → 完全保留旧记忆
    zt_high = torch.ones(1, H) * 0.99
    ht_A = (1 - zt_high) * h_tilde + zt_high * h_prev
    print(f"场景A: zt≈1（更新门全开/保留旧记忆）")
    print(f"  h_prev={h_prev[0,0]:.2f}, h_tilde={h_tilde[0,0]:.2f} → ht={ht_A[0,0]:.4f}（接近h_prev）")

    # 场景B：zt ≈ 0 → 完全采用新信息
    zt_low = torch.ones(1, H) * 0.01
    ht_B = (1 - zt_low) * h_tilde + zt_low * h_prev
    print(f"场景B: zt≈0（更新门关闭/全用新信息）")
    print(f"  h_prev={h_prev[0,0]:.2f}, h_tilde={h_tilde[0,0]:.2f} → ht={ht_B[0,0]:.4f}（接近h_tilde）")

    # 场景C：rt ≈ 0 → 候选值不受历史影响
    rt_zero = torch.zeros(1, H)
    h_tilde_no_history = torch.tanh(torch.zeros(1, H) + torch.zeros(1, H))  # Uh*(0*h_prev)≈0
    print(f"场景C: rt≈0（重置门关闭/候选值忽略历史）")
    print(f"  rt=0 → h_tilde 不再依赖 h_prev，相当于从零开始计算候选值")


if __name__ == '__main__':
    demo_gate_values()
    demo_manual_vs_pytorch()
    demo_param_count()
    demo_extreme_gates()
