import torch
import torch.nn.functional as F
import math

# ============================================================
# 单头注意力（Single-Head Attention）完整结构
#
# 结构图：
#   X ──→ W_Q ──→ Q ──┐
#   X ──→ W_K ──→ K ──┼──→ QK^T/√d_k ──→ Softmax ──→ ⊗V ──→ W_O ──→ output
#   X ──→ W_V ──→ V ──┘
#
# 与基础版的区别：多了最后的输出投影 W_O
#   W_O 的作用：把 attention 聚合后的结果再做一次线性变换
#   让模型有机会重组聚合后的信息，增加表达能力
# ============================================================

batch  = 2
seq    = 4    # 序列长度
d_model = 16  # 输入/输出维度（保持一致，方便残差连接）
d_k    = 8    # Q、K 的投影维度
d_v    = 8    # V 的投影维度

X = torch.randn(batch, seq, d_model)   # 输入，shape=(2,4,16)

# QKV 投影权重（无bias，和Transformer原论文一致）
W_Q = torch.nn.Linear(d_model, d_k, bias=False)
W_K = torch.nn.Linear(d_model, d_k, bias=False)
W_V = torch.nn.Linear(d_model, d_v, bias=False)
# 输出投影：把 d_v 维聚合结果投影回 d_model，维持残差连接的维度
W_O = torch.nn.Linear(d_v, d_model, bias=False)

# ============================================================
# 前向计算
# ============================================================
Q = X @ W_Q.weight.T   # (2, 4, 8)  每个位置"提问"
K = X @ W_K.weight.T   # (2, 4, 8)  每个位置"被问的标签"
V = X @ W_V.weight.T   # (2, 4, 8)  每个位置"携带的内容"

# Step1: 点积相似度
scores = Q @ K.transpose(-2, -1)          # (2, 4, 4)
# Step2: 缩放（防止 d_k 大时点积过大导致 softmax 饱和）
scores = scores / math.sqrt(d_k)          # (2, 4, 4)
# Step3: softmax 归一化 → 注意力权重
attn_w = F.softmax(scores, dim=-1)        # (2, 4, 4)  每行和=1
# Step4: 加权聚合 V
context = attn_w @ V                      # (2, 4, 8)
# Step5: 输出投影（单头独有，多头版本合并后才做）
output = context @ W_O.weight.T           # (2, 4, 16) 维度回到 d_model

print("===== 单头注意力 shape 流水线 =====")
print(f"X      : {X.shape}")        # (2,4,16)
print(f"Q/K/V  : {Q.shape}")        # (2,4,8)
print(f"scores : {scores.shape}")   # (2,4,4)
print(f"attn_w : {attn_w.shape}")   # (2,4,4)
print(f"context: {context.shape}")  # (2,4,8)
print(f"output : {output.shape}")   # (2,4,16) ← 与输入同维，可做残差
print(f"\nattn_w[0]（样本0注意力矩阵）:\n{attn_w[0].detach()}")
print(f"每行和: {attn_w[0].sum(dim=-1).detach()}  ← 必须全为1")
