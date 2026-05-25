"""
NLP 序号化（Token Indexing）案例
目标：把原始文本转换成整数序列，供神经网络输入层使用。

核心步骤：
  1. 分词（tokenize）
  2. 构建词表（vocabulary）
  3. 词 → 索引（word2idx）
  4. 索引 → 词（idx2word，解码用）
  5. 句子对齐（padding / truncating）
"""

import torch
from torch.nn.utils.rnn import pad_sequence

# ── 1. 原始语料 ────────────────────────────────────────────────
corpus = [
    "我 喜欢 深度 学习",
    "自然 语言 处理 很 有趣",
    "深度 学习 改变 了 世界",
    "我 喜欢 自然 语言 处理",
]

# ── 2. 分词（这里以空格分词；实际项目用 jieba/HanLP）──────────────
# tokenized = [sentence.split() for sentence in corpus]
tokenized = [sentence.split() for sentence in corpus]
print("分词结果：")
for t in tokenized:
    print(" ", t)

# ── 3. 构建词表 ────────────────────────────────────────────────
# 特殊 token：<PAD>=0, <UNK>=1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# 统计所有出现过的词
all_words = [word for sentence in tokenized for word in sentence]
# 构建词典
# 排序保证可复现
unique_words = sorted(set(all_words))

word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

for word in unique_words:
    word2idx[word] = len(word2idx)

idx2word = {idx: word for word, idx in word2idx.items()}

print(f"\n词表大小（含特殊 token）：{len(word2idx)}")
print("词表：", word2idx)

# ── 4. 句子 → 索引序列 ─────────────────────────────────────────
def encode(sentence_tokens, word2idx, unk_idx=1):
    """把词 list 转成索引 list，未登录词用 <UNK> 替代"""
    return [word2idx.get(w, unk_idx) for w in sentence_tokens]

encoded = [encode(tokens, word2idx) for tokens in tokenized]
print("\n编码结果（索引序列）：")
for i, seq in enumerate(encoded):
    print(f"  句子{i+1}: {tokenized[i]}  →  {seq}")

# ── 5. 解码：索引 → 词 ─────────────────────────────────────────
def decode(indices, idx2word):
    return [idx2word.get(i, UNK_TOKEN) for i in indices]

print("\n解码验证（第1句）：", decode(encoded[0], idx2word))

# ── 6. Padding：把变长序列对齐到同一长度 ──────────────────────────
# 方式A：手动 padding 到固定最大长度
MAX_LEN = 6

def pad_or_truncate(seq, max_len, pad_idx=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))

padded_manual = [pad_or_truncate(seq, MAX_LEN) for seq in encoded]
print(f"\n手动 Padding（max_len={MAX_LEN}）：")
for seq in padded_manual:
    print(" ", seq)

# 方式B：用 PyTorch pad_sequence（自动对齐到 batch 内最长序列）
tensors = [torch.tensor(seq) for seq in encoded]
# pad_sequence 默认 batch_first=False，这里设 True 更直观
padded_torch = pad_sequence(tensors, batch_first=True, padding_value=0)
print("\nPyTorch pad_sequence 结果（shape:", padded_torch.shape, "）：")
print(padded_torch)

# ── 7. 构建 attention mask（1=真实 token，0=padding）─────────────
attention_mask = (padded_torch != 0).long()
print("\nAttention Mask：")
print(attention_mask)

# ── 8. 模拟查 Embedding（可直接送入 nn.Embedding）────────────────
VOCAB_SIZE = len(word2idx)
EMBED_DIM = 8

embedding = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
embedded = embedding(padded_torch)   # shape: (batch, seq_len, embed_dim)
print(f"\nEmbedding 输出 shape：{embedded.shape}")
print("第1句第1个 token 的向量：", embedded[0, 0].detach())
