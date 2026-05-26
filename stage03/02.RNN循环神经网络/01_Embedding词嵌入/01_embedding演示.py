"""
Embedding 词嵌入 — 从零理解

核心问题：文字没法直接输入神经网络，需要把词转成"有意义的数字向量"
关键思想：相似的词，向量空间里距离近（king - man + woman ≈ queen）
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'  # Mac 中文字体

IMG_DIR = os.path.join(os.path.dirname(__file__), '../../../../base/images')

# ─────────────────────────────────────────
# Part 1: Embedding 的本质 — 一张可学习的查找表
# ─────────────────────────────────────────

print("=" * 50)
print("Part 1: Embedding 本质 — 查找表")
print("=" * 50)

# 词表大小=5，每个词用3维向量表示
vocab_size = 5
embed_dim = 3

embedding = nn.Embedding(vocab_size, embed_dim)

print(f"词表大小: {vocab_size}")
print(f"嵌入维度: {embed_dim}")
print(f"Embedding 权重矩阵 shape: {embedding.weight.shape}")
print(f"权重矩阵（每行=一个词的向量）:\n{embedding.weight.data}\n")

# 输入词的索引（不是 one-hot，直接是整数）
word_indices = torch.tensor([0, 2, 4])       # 查第0、2、4个词的向量
result = embedding(word_indices)
print(f"查询词索引 {word_indices.tolist()} 的向量:\n{result}\n")

# ─────────────────────────────────────────
# Part 2: 手动理解 — Embedding 等价于 one-hot × 权重矩阵
# ─────────────────────────────────────────

print("=" * 50)
print("Part 2: Embedding = one-hot @ weight_matrix")
print("=" * 50)

# 手动构造 one-hot
idx = 2
one_hot = torch.zeros(vocab_size)
one_hot[idx] = 1.0
print(f"词索引 {idx} 的 one-hot: {one_hot}")

# one-hot @ 权重矩阵 = 直接取第 idx 行
manual = one_hot @ embedding.weight  # (vocab,) @ (vocab, embed_dim) = (embed_dim,)
direct = embedding(torch.tensor(idx))
print(f"one-hot × 矩阵结果:  {manual.detach()}")
print(f"Embedding 直接查询: {direct.detach()}")
print(f"两者是否相等: {torch.allclose(manual, direct)}\n")

# ─────────────────────────────────────────
# Part 3: 处理一个句子（序列输入）
# ─────────────────────────────────────────

print("=" * 50)
print("Part 3: 处理句子序列")
print("=" * 50)

# 模拟词表
vocab = {"<pad>": 0, "我": 1, "爱": 2, "深度": 3, "学习": 4}
sentence = ["我", "爱", "深度", "学习"]

# 文字 → 索引
indices = torch.tensor([vocab[w] for w in sentence])
print(f"句子: {sentence}")
print(f"索引: {indices}")

# 索引 → 向量（shape: seq_len × embed_dim）
embed_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4)
sentence_vectors = embed_layer(indices)
print(f"句子向量 shape: {sentence_vectors.shape}  (seq_len={len(sentence)}, embed_dim=4)")
print(f"句子向量:\n{sentence_vectors.detach()}\n")

# ─────────────────────────────────────────
# Part 4: Batch 输入（实际训练时的形状）
# ─────────────────────────────────────────

print("=" * 50)
print("Part 4: Batch 输入")
print("=" * 50)

# batch_size=2, seq_len=4
batch_input = torch.tensor([
    [1, 2, 3, 4],   # 句子1: 我 爱 深度 学习
    [1, 2, 0, 0],   # 句子2: 我 爱 <pad> <pad>
])

batch_embed = embed_layer(batch_input)
print(f"batch_input shape:  {batch_input.shape}   (batch=2, seq=4)")
print(f"batch_embed shape: {batch_embed.shape}  (batch=2, seq=4, embed=4)")

# ─────────────────────────────────────────
# Part 5: padding_idx — pad 的向量保持全零，不参与梯度
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("Part 5: padding_idx 的作用")
print("=" * 50)

pad_embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=4, padding_idx=0)
pad_vector = pad_embed(torch.tensor(0))  # <pad> 的向量
print(f"<pad> 向量 (padding_idx=0): {pad_vector.detach()}")
print("→ 全零，且反向传播时梯度为零，不影响其他词的学习\n")

# ─────────────────────────────────────────
# Part 6: 可视化 — 二维降维看词向量距离
# ─────────────────────────────────────────

print("=" * 50)
print("Part 6: 可视化词向量（PCA降至2D）")
print("=" * 50)

import numpy as np

words = ["<pad>", "我", "爱", "深度", "学习"]
# 使用训练好的模型，这里用随机初始化做演示
vectors = embed_layer.weight.detach().numpy()  # (5, 4)

# 简单 PCA 降到 2D
from numpy.linalg import svd
centered = vectors - vectors.mean(axis=0)
_, _, Vt = svd(centered)
proj = centered @ Vt[:2].T  # (5, 2)

plt.figure(figsize=(6, 5))
for i, word in enumerate(words):
    plt.scatter(proj[i, 0], proj[i, 1], s=100)
    plt.annotate(word, (proj[i, 0], proj[i, 1]), fontsize=12,
                 xytext=(5, 5), textcoords='offset points')
plt.title("词向量 PCA 2D 可视化（随机初始化）")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "08_RNN_Embedding词嵌入可视化.png"), dpi=120)
plt.show()
print("图已保存: base/images/08_RNN_Embedding词嵌入可视化.png")

print("\n✅ Embedding 演示完成！")
print("\n【核心总结】")
print("1. Embedding = 可学习的查找表，输入整数索引，输出浮点向量")
print("2. 等价于 one-hot × 权重矩阵，但更高效（避免稀疏乘法）")
print("3. shape 变化: (batch, seq_len) → (batch, seq_len, embed_dim)")
print("4. padding_idx 让 <pad> 向量保持全零且不更新")
