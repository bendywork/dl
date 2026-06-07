import torch
import torch.nn.functional as F
import math

# ============================================================
# 专题：如何理解 Attention(Q,K,V) = Softmax(QK^T / sqrt(d_k)) · V
#
# 把公式拆成 4 步，每步都用可手算的小数字演示：
#   Step1: Q @ K^T          → 计算每对位置的"相关度原始分"
#   Step2: / sqrt(d_k)      → 缩放，防止梯度消失
#   Step3: Softmax(...)     → 归一化成"注意力概率分布"
#   Step4: ... @ V          → 按概率加权混合 V 的内容
# ============================================================

# 用极小的维度，让你能手算每一步
# 3个词（位置），每个词用2维向量表示，Q/K/V投影到2维
seq_len = 3   # 3个位置：词0、词1、词2
d_k     = 2   # Q和K的维度（决定点积空间大小）
d_v     = 2   # V的维度（输出空间大小）

# 手动构造 Q、K、V（不走线性层，直接给值，方便理解）
# 每行 = 一个词的向量
# Q[i] = 词i在"提问"时的表示
# K[j] = 词j在"被问"时的标签
# V[j] = 词j实际携带的语义内容

Q = torch.tensor([
    [1.0, 0.0],   # 词0的Query
    [0.0, 1.0],   # 词1的Query
    [1.0, 1.0],   # 词2的Query
])  # shape: (3, 2)

K = torch.tensor([
    [1.0, 0.0],   # 词0的Key
    [0.0, 1.0],   # 词1的Key
    [1.0, 1.0],   # 词2的Key
])  # shape: (3, 2)

V = torch.tensor([
    [10.0,  0.0],  # 词0的Value（内容：偏向第1维）
    [ 0.0, 10.0],  # 词1的Value（内容：偏向第2维）
    [ 5.0,  5.0],  # 词2的Value（内容：两维均衡）
])  # shape: (3, 2)

print("=" * 50)
print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

# ============================================================
# Step 1: 计算 Q @ K^T  →  原始相关度矩阵
# ============================================================
# Q shape:   (3, 2)
# K^T shape: (2, 3)   ← 对K转置，让维度对齐
# 结果 shape: (3, 3)  ← [i,j] = 词i的Query 与 词j的Key 的点积
#
# 点积本质：两个向量越"方向一致"，点积越大
#   Q[0]=[1,0] 与 K[0]=[1,0] 点积 = 1*1+0*0 = 1.0  （方向完全一致）
#   Q[0]=[1,0] 与 K[1]=[0,1] 点积 = 1*0+0*1 = 0.0  （方向垂直，毫无相关）
#   Q[2]=[1,1] 与 K[2]=[1,1] 点积 = 1+1     = 2.0  （方向一致且幅度大）

raw_scores = Q @ K.T    # (3, 3)
print("\n" + "=" * 50)
print("Step1: Q @ K^T (原始相关度):")
print(raw_scores)
print("解读：raw_scores[i][j] = 词i 对 词j 的原始注意力分数")
print("  词0对词0:", raw_scores[0,0].item(), "（自己问自己，方向一致）")
print("  词0对词1:", raw_scores[0,1].item(), "（方向垂直，完全不相关）")
print("  词2对词2:", raw_scores[2,2].item(), "（[1,1]·[1,1]=2，最高）")

# ============================================================
# Step 2: / sqrt(d_k)  →  缩放，稳定梯度
# ============================================================
# 问题：当 d_k 很大时，点积的绝对值会很大
#   比如 d_k=512，两个随机单位向量点积期望~0，但方差=d_k=512
#   这样 softmax 输入里最大值可能是 20+ → softmax 梯度趋近0（饱和）
# 解决：除以 sqrt(d_k)，把方差压回 1
#   数学推导：Var(q·k) = d_k·Var(q_i)·Var(k_i)
#             除以sqrt(d_k)后方差=1（假设各分量方差为1）

scale = math.sqrt(d_k)   # sqrt(2) ≈ 1.414
scaled_scores = raw_scores / scale

print("\n" + "=" * 50)
print(f"Step2: / sqrt(d_k={d_k}) = / {scale:.3f}")
print("缩放前:\n", raw_scores)
print("缩放后:\n", scaled_scores)
print("注意：数值比例不变，但绝对值变小，softmax不会饱和")

# ============================================================
# Step 3: Softmax  →  归一化成概率分布（每行和=1）
# ============================================================
# softmax(x_i) = exp(x_i) / sum(exp(x_j))
# 作用：把任意大小的分数 → 变成 [0,1] 的权重，且每行加和=1
# 直觉：把"谁更相关"的竞争结果，转化成"各分配多少注意力"的比例
#
# 极端情况帮助理解：
#   [10, 0, 0] → softmax ≈ [0.9999, 0.00005, 0.00005]  几乎全注意力给第0个
#   [0,  0, 0] → softmax =  [0.333,  0.333,  0.333]    注意力均匀分配

attn_weights = F.softmax(scaled_scores, dim=-1)   # 沿最后一维归一化

print("\n" + "=" * 50)
print("Step3: Softmax(scaled_scores)  →  注意力权重:")
print(attn_weights)
print("每行之和（必须=1）:", attn_weights.sum(dim=-1))
print("\n解读每一行（每个词把注意力如何分配给其他词）：")
for i in range(seq_len):
    vals = attn_weights[i].tolist()
    print(f"  词{i}: ", end="")
    for j in range(seq_len):
        print(f"关注词{j}={vals[j]:.3f}", end="  ")
    print()

# ============================================================
# Step 4: attn_weights @ V  →  加权聚合内容
# ============================================================
# attn_weights shape: (3, 3)
# V shape:            (3, 2)
# output shape:       (3, 2)
#
# output[i] = sum_j( attn_weights[i,j] * V[j] )
# 含义：词i的输出 = 所有词的Value，按注意力权重加权平均
# 直觉：你更关注谁，就从谁那里"借"更多信息

output = attn_weights @ V   # (3, 2)

print("\n" + "=" * 50)
print("Step4: attn_weights @ V  →  最终输出:")
print(output)
print("\n手动验证词0的输出（展示加权平均过程）：")
w = attn_weights[0]   # 词0的注意力权重，shape=(3,)
manual = w[0]*V[0] + w[1]*V[1] + w[2]*V[2]
print(f"  权重: {w.tolist()}")
print(f"  手算: {w[0]:.3f}*{V[0].tolist()} + {w[1]:.3f}*{V[1].tolist()} + {w[2]:.3f}*{V[2].tolist()}")
print(f"  结果: {manual.tolist()}")
print(f"  矩阵乘结果: {output[0].tolist()}")
print("  ✅ 两种算法结果一致，矩阵乘就是批量加权平均")

# ============================================================
# 整体流程汇总：一行公式 vs 四步分解
# ============================================================
print("\n" + "=" * 50)
print("公式：Attention(Q,K,V) = Softmax(Q·K^T / sqrt(d_k)) · V")
print("=" * 50)
print("Step1  Q @ K^T        → 原始相关度  shape:", raw_scores.shape)
print("Step2  / sqrt(d_k)    → 缩放稳定    shape:", scaled_scores.shape)
print("Step3  Softmax(dim=-1)→ 注意力权重  shape:", attn_weights.shape)
print("Step4  @ V            → 聚合输出    shape:", output.shape)
print()
print("每个词最终输出是 V 的加权平均，权重由 Q·K 相似度决定")
print("Q决定'我想要什么'，K决定'我能匹配什么'，V决定'我能给什么'")
