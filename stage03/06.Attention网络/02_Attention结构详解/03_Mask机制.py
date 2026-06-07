import torch
import torch.nn.functional as F
import math

# ============================================================
# Attention 中的两种 Mask（遮蔽）机制
#
# 为什么需要 Mask？
#   Attention 默认让每个位置都能"看到"所有位置
#   但有两种情况不允许这样：
#
# 1. Padding Mask（填充遮蔽）
#    问题：batch 内不同句子长度不同，短句用 <PAD> 补齐
#    目标：不让模型把注意力浪费在 <PAD> 位置上
#    方法：把 <PAD> 对应的 score 设为 -∞，softmax 后权重≈0
#
# 2. Causal Mask / Look-ahead Mask（因果遮蔽）
#    问题：Decoder 自回归生成时，不能"偷看"未来的词
#    目标：位置 i 只能看到位置 0..i，看不到 i+1..end
#    方法：把上三角（未来位置）的 score 设为 -∞
# ============================================================

batch  = 2
seq    = 5
d_k    = 4

# 模拟 scores（已经做完 QK^T/sqrt(d_k)，还没做 softmax）
scores = torch.randn(batch, seq, seq)

# ============================================================
# Mask 类型1：Padding Mask
# ============================================================
# 假设：batch 中两个样本的有效长度分别是 3 和 4（其余是PAD）
valid_lens = [3, 4]

# 构造 padding_mask：True 的位置表示"这是 PAD，需要遮蔽"
# shape: (batch, seq) → 扩展为 (batch, 1, seq) 用于广播到 (batch, seq, seq)
pad_mask = torch.zeros(batch, seq, dtype=torch.bool)
for i, vl in enumerate(valid_lens):
    pad_mask[i, vl:] = True   # vl 之后全是 PAD

pad_mask_3d = pad_mask.unsqueeze(1)   # (2, 1, 5) → 广播到 (2, 5, 5)
scores_pad = scores.masked_fill(pad_mask_3d, float('-inf'))

print("===== Padding Mask =====")
print(f"原始 scores[0]:\n{scores[0].detach()}")
print(f"\npad_mask[0]: {pad_mask[0].tolist()}  ← True=PAD位置")
print(f"\n遮蔽后 scores[0]:\n{scores_pad[0].detach()}")
print(f"\nsoftmax 后（PAD列权重≈0）:\n{F.softmax(scores_pad[0], dim=-1).detach()}")

# ============================================================
# Mask 类型2：Causal Mask（因果/上三角遮蔽）
# ============================================================
# 生成上三角矩阵（不含对角线），True = 未来位置，需要遮蔽
# torch.triu(ones, diagonal=1) 取严格上三角
causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

print("\n===== Causal Mask =====")
print("causal_mask（True=未来，不可见）:")
print(causal_mask.int())   # 打印0/1更直观
print("解读：行=当前位置，列=被关注位置")
print("  位置0只能看列0；位置2能看列0,1,2；位置4能看所有")

scores_causal = scores[0].masked_fill(causal_mask, float('-inf'))
attn_causal   = F.softmax(scores_causal, dim=-1)

print(f"\nscores[0] 遮蔽后:\n{scores_causal.detach()}")
print(f"\nsoftmax 后（上三角≈0，下三角正常）:\n{attn_causal.detach()}")

# ============================================================
# 完整带 Mask 的 Attention 函数（可复用模板）
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)
    V: (batch, heads, seq_k, d_v)
    mask: (batch, 1, seq_q, seq_k) 或 (seq_q, seq_k)，True=遮蔽
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    return attn_w @ V, attn_w

# 测试函数
Q_t = torch.randn(2, 4, 5, d_k)
K_t = torch.randn(2, 4, 5, d_k)
V_t = torch.randn(2, 4, 5, d_k)
cm  = causal_mask.unsqueeze(0).unsqueeze(0)   # (1,1,5,5) 广播

out, w = scaled_dot_product_attention(Q_t, K_t, V_t, mask=cm)
print("\n===== 完整带Mask Attention 函数测试 =====")
print(f"output shape: {out.shape}")   # (2,4,5,4)
print(f"attn_w shape: {w.shape}")     # (2,4,5,5)
print(f"下三角（含对角）有效，上三角≈0:\n{w[0,0].detach()}")
print("\n两种Mask总结：")
print("  Padding Mask  → 遮蔽PAD列，Encoder/Decoder均用")
print("  Causal  Mask  → 遮蔽上三角，仅Decoder自回归时用")
