import torch
import torch.nn.functional as F
import math

# ============================================================
# 多头注意力（Multi-Head Attention）完整结构
#
# 核心思想：用 h 个"头"并行做 Attention，每个头关注不同的特征子空间
#   类比：用多个视角同时看一句话
#     头1 可能学会关注"语法结构"
#     头2 可能学会关注"语义相似性"
#     头3 可能学会关注"位置关系"
#
# 实现方式：
#   不是真的准备 h 套独立的 W_Q/W_K/W_V
#   而是用一套大矩阵投影到 h*d_k 维，然后 reshape 成 h 份
#   最后把 h 个头的输出拼接，再过一个 W_O 输出投影
#
# 维度关系（Transformer 原论文惯例）：
#   d_k = d_v = d_model / num_heads
# ============================================================

batch     = 2
seq       = 4
d_model   = 16   # 总维度
num_heads = 4    # 头数
d_k       = d_model // num_heads   # 每头的 Q/K 维度 = 4
d_v       = d_model // num_heads   # 每头的 V  维度 = 4

X = torch.randn(batch, seq, d_model)   # (2, 4, 16)

# 一套大投影矩阵（投影到 num_heads * d_k = d_model）
# 等价于 h 个小 W_Q 拼在一起
W_Q = torch.nn.Linear(d_model, num_heads * d_k, bias=False)   # 16→16
W_K = torch.nn.Linear(d_model, num_heads * d_k, bias=False)   # 16→16
W_V = torch.nn.Linear(d_model, num_heads * d_v, bias=False)   # 16→16
W_O = torch.nn.Linear(num_heads * d_v, d_model, bias=False)   # 16→16

# ============================================================
# 拆头：reshape + transpose 把"头"维度提前
# ============================================================
Q_big = X @ W_Q.weight.T   # (2, 4, 16) = (batch, seq, num_heads*d_k)
K_big = X @ W_K.weight.T   # (2, 4, 16)
V_big = X @ W_V.weight.T   # (2, 4, 16)

# reshape: (batch, seq, num_heads*d_k) → (batch, seq, num_heads, d_k)
# transpose: → (batch, num_heads, seq, d_k)  ← 把头提到前面，方便并行
Q = Q_big.view(batch, seq, num_heads, d_k).transpose(1, 2)  # (2,4,4,4)
K = K_big.view(batch, seq, num_heads, d_k).transpose(1, 2)  # (2,4,4,4)
V = V_big.view(batch, seq, num_heads, d_v).transpose(1, 2)  # (2,4,4,4)

print("===== 多头注意力 =====")
print(f"X原始      : {X.shape}")       # (2,4,16)
print(f"Q_big投影  : {Q_big.shape}")   # (2,4,16)
print(f"Q拆头后    : {Q.shape}")       # (2,4,4,4) = (batch,heads,seq,d_k)
print("含义：第2维=4个头，第3维=4个位置，第4维=每头4维空间")

# ============================================================
# 每个头并行做 Scaled Dot-Product Attention
# Q/K/V 的 shape 都是 (batch, num_heads, seq, d_k)
# 矩阵乘自动在 batch 和 heads 两个维度上广播（批量并行）
# ============================================================
scores = Q @ K.transpose(-2, -1)          # (2, 4, 4, 4)
scores = scores / math.sqrt(d_k)          # (2, 4, 4, 4)
attn_w = F.softmax(scores, dim=-1)        # (2, 4, 4, 4)
context = attn_w @ V                      # (2, 4, 4, 4)

# ============================================================
# 合并头：把 num_heads 个头的输出拼回一个向量
# transpose 回 (batch, seq, num_heads, d_v)
# contiguous() 保证内存连续（transpose 不改变底层存储）
# view 合并最后两维 → (batch, seq, num_heads*d_v)
# ============================================================
context_cat = context.transpose(1, 2).contiguous()  # (2,4,4,4)→(2,4,4,4)
context_cat = context_cat.view(batch, seq, num_heads * d_v)  # (2,4,16)

# 输出投影 W_O：把拼接后的多头结果线性变换回 d_model
output = context_cat @ W_O.weight.T   # (2, 4, 16)

print(f"\nscores     : {scores.shape}")       # (2,4,4,4) batch/heads/seq/seq
print(f"attn_w     : {attn_w.shape}")        # (2,4,4,4)
print(f"context    : {context.shape}")       # (2,4,4,4)
print(f"合并头后   : {context_cat.shape}")   # (2,4,16)
print(f"output(W_O): {output.shape}")        # (2,4,16) ← 与输入同维
print("\n关键理解：")
print("  多头 = 把 d_model 切成 num_heads 份，每份独立做 Attention")
print("  每个头用不同的子空间捕捉不同的语义关系")
print("  最后 W_O 把多个视角的信息融合成一个表示")
