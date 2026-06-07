import torch
import torch.nn.functional as F
import math

# ============================================================
# Attention QKV 手推入门 Demo
#
# 核心问题：序列中每个位置在"回看"其他位置时，应该关注多少？
# Attention 用 Q/K/V 三个角色来回答这个问题：
#   Q (Query)  ：我想找什么？（提问者）
#   K (Key)    ：我有什么？  （被问者的标签）
#   V (Value)  ：我实际提供的内容（被问者的答案）
#
# 直觉类比：
#   你去图书馆找书（Q = 你的搜索关键词）
#   每本书的书名/索引就是 K
#   书的正文内容就是 V
#   Attention = 用 Q 匹配所有 K → 算出相关度 → 按相关度加权取 V
# ============================================================

# ---------- 超参数 ----------
batch_size = 2    # 批量大小
seq_len    = 4    # 序列长度（比如一句话有4个词）
d_model    = 8    # 每个词的嵌入维度（输入特征维度）
d_k        = 4    # Q 和 K 的投影维度（缩放点积用这个维度）
d_v        = 4    # V 的投影维度（输出维度）

# ---------- 模拟输入 ----------
# X 代表一批序列，比如 batch=2 句话，每句 4 个词，每词用 8 维向量表示
# shape = (batch_size, seq_len, d_model)
X = torch.randn(batch_size, seq_len, d_model)
print(f"输入 X shape:         {X.shape}")   # (2, 4, 8)

# ---------- QKV 投影权重（三个独立线性层，无bias） ----------
# 为什么要投影？
#   原始 X 维度 d_model 可能很大，投影到更小的 d_k/d_v
#   可以让模型在低维空间里做高效的相似度计算
W_Q = torch.nn.Linear(d_model, d_k, bias=False)  # (8 -> 4)
W_K = torch.nn.Linear(d_model, d_k, bias=False)  # (8 -> 4)
W_V = torch.nn.Linear(d_model, d_v, bias=False)  # (8 -> 4)

# ============================================================
# Step 1：计算 Q、K、V
# ============================================================
# X shape:     (batch, seq_len, d_model) = (2, 4, 8)
# W_Q.weight:  (d_k, d_model)            = (4, 8)
# X @ W_Q.T：  (2, 4, 8) @ (8, 4)       = (2, 4, 4)
#
# 含义：
#   Q[b, i, :] = 第 b 个样本、第 i 个位置的"提问向量"
#   K[b, j, :] = 第 b 个样本、第 j 个位置的"标签向量"
#   V[b, j, :] = 第 b 个样本、第 j 个位置的"内容向量"

Q = X @ W_Q.weight.T   # (batch, seq_len, d_k)  = (2, 4, 4)
K = X @ W_K.weight.T   # (batch, seq_len, d_k)  = (2, 4, 4)
V = X @ W_V.weight.T   # (batch, seq_len, d_v)  = (2, 4, 4)

print(f"Q shape: {Q.shape}")   # (2, 4, 4)
print(f"K shape: {K.shape}")   # (2, 4, 4)
print(f"V shape: {V.shape}")   # (2, 4, 4)

# ============================================================
# Step 2：计算 Attention Score = Q @ K^T / sqrt(d_k)
# ============================================================
# Q shape:    (batch, seq_len, d_k) = (2, 4, 4)
# K^T shape:  (batch, d_k, seq_len) = (2, 4, 4)   ← 对最后两维转置
# Q @ K^T:    (2, 4, 4) @ (2, 4, 4) = (2, 4, 4)
#   结果矩阵 [b, i, j] = 第 b 个样本中，位置 i 对位置 j 的"原始相关度"
#
# 为什么要除以 sqrt(d_k)？
#   点积结果随 d_k 增大而变大，数值过大会导致 softmax 梯度极小（饱和）
#   除以 sqrt(d_k) 把方差稳定回 ~1，保持梯度流畅
#   这就是 "Scaled" Dot-Product Attention 名字中 "Scaled" 的来源

scale = math.sqrt(d_k)                      # sqrt(4) = 2.0
raw_scores = Q @ K.transpose(-2, -1)        # (2, 4, 4)
scores = raw_scores / scale                  # (2, 4, 4)  缩放后

print(f"raw_scores shape: {raw_scores.shape}")  # (2, 4, 4)
print(f"scaled scores shape: {scores.shape}")   # (2, 4, 4)
print(f"scores[0]（第0个样本的注意力矩阵）:\n{scores[0].detach()}")

# ============================================================
# Step 3：Softmax → 得到注意力权重（每行和为1）
# ============================================================
# softmax 沿最后一维（dim=-1）做归一化
# 含义：对于位置 i，它关注所有位置 j 的权重加起来 = 1
# 直觉：把原始相关度分数变成"分配注意力资源的概率分布"
#
# attn_weights[b, i, j] = 位置 i 把多少比例的注意力放在位置 j 上

attn_weights = F.softmax(scores, dim=-1)    # (2, 4, 4)
print(f"\nattn_weights shape:  {attn_weights.shape}")
print(f"每行之和（应全为1）: {attn_weights[0].sum(dim=-1).detach()}")

# ============================================================
# Step 4：加权聚合 V → 得到最终 Attention 输出
# ============================================================
# attn_weights shape: (batch, seq_len, seq_len) = (2, 4, 4)
# V shape:            (batch, seq_len, d_v)      = (2, 4, 4)
# output = attn_weights @ V：(2, 4, 4) @ (2, 4, 4) = (2, 4, 4)
#
# 含义：
#   output[b, i, :] = 位置 i 按注意力权重，从所有位置的 V 中加权提取信息
#   权重越大的位置，贡献的 V 越多
#   这就是 Attention 的本质：用相关度加权混合信息

attn_output = attn_weights @ V              # (2, 4, 4)
print(f"\nattn_output shape:   {attn_output.shape}")   # (2, 4, 4)

# ============================================================
# 整体流程回顾（shape 变化一览）
# ============================================================
print("\n===== Shape 流水线 =====")
print(f"输入 X:          {X.shape}")           # (2, 4, 8)
print(f"Q = X @ W_Q.T:  {Q.shape}")            # (2, 4, 4)
print(f"K = X @ W_K.T:  {K.shape}")            # (2, 4, 4)
print(f"V = X @ W_V.T:  {V.shape}")            # (2, 4, 4)
print(f"scores=Q@K^T/√d:{scores.shape}")       # (2, 4, 4)
print(f"attn_weights:   {attn_weights.shape}") # (2, 4, 4)
print(f"output=w@V:     {attn_output.shape}")  # (2, 4, 4)
