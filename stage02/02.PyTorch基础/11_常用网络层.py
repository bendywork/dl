# ==============================================================
# 文件：11_常用网络层.py
# 主题：PyTorch 常用网络层
# API：Linear / Conv2d / Conv1d / RNN / LSTM / GRU
#       Embedding / BatchNorm1d/2d / Dropout / LayerNorm / MultiheadAttention
# ==============================================================
import torch
import torch.nn as nn

SEP = "=" * 60

# ==============================================================
# 1. Linear  全连接层
# ==============================================================
print(SEP)
print("1. nn.Linear")
print(SEP)
linear = nn.Linear(in_features=8, out_features=4, bias=True)
x = torch.randn(2, 8)
out = linear(x)
print("输入:", x.shape, "-> 输出:", out.shape)
print("weight shape:", linear.weight.shape)  # (out, in)
print("bias shape:", linear.bias.shape)
# 手动等价运算
manual = x @ linear.weight.T + linear.bias
print("手动计算与 Linear 一致:", torch.allclose(out, manual))

# ==============================================================
# 2. Conv2d  二维卷积（图像处理核心层）
# ==============================================================
print()
print(SEP)
print("2. nn.Conv2d")
print(SEP)
# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# 图像: batch=2, channels=3, H=32, W=32
img = torch.randn(2, 3, 32, 32)
feat = conv2d(img)
print("输入 (B,C,H,W):", img.shape)
print("输出 (B,C_out,H_out,W_out):", feat.shape)
print("weight (out,in,kH,kW):", conv2d.weight.shape)
total_params = sum(p.numel() for p in conv2d.parameters())
print("参数量:", total_params)

# 常见配置
conv_s2 = nn.Conv2d(3, 32, 3, stride=2, padding=1)  # 下采样一半
print("stride=2 下采样:", conv_s2(img).shape)
conv_d = nn.Conv2d(3, 16, 3, padding=2, dilation=2)  # 空洞卷积
print("dilation=2 空洞卷积:", conv_d(img).shape)

# ==============================================================
# 3. Conv1d  一维卷积（NLP/时序信号处理）
# ==============================================================
print()
print(SEP)
print("3. nn.Conv1d")
print(SEP)
# Conv1d(in_channels, out_channels, kernel_size)
# 输入 shape: (batch, channels, length)
conv1d = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
seq = torch.randn(4, 64, 20)  # batch=4, embed=64, seq_len=20
out = conv1d(seq)
print("输入 (B,C,L):", seq.shape)
print("输出 (B,C_out,L_out):", out.shape)

# ==============================================================
# 4. Embedding  词嵌入层
# ==============================================================
print()
print(SEP)
print("4. nn.Embedding")
print(SEP)
# Embedding(num_embeddings, embedding_dim)
vocab_size = 10000
embed_dim = 128
embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
# padding_idx=0 表示索引0的词向量始终为0（用于<PAD>）
token_ids = torch.tensor([[1, 23, 456, 0], [7, 89, 0, 0]])
embedded = embedding(token_ids)
print("token_ids shape:", token_ids.shape)
print("嵌入后 shape:", embedded.shape)
print("<PAD>(idx=0) 词向量为零:", (embedded[0, 3] == 0).all().item())
print("weight shape:", embedding.weight.shape)

# ==============================================================
# 5. RNN / LSTM / GRU
# ==============================================================
print()
print(SEP)
print("5. RNN / LSTM / GRU")
print(SEP)

# 输入约定：(seq_len, batch, input_size)  默认 batch_first=False
# 若 batch_first=True: (batch, seq_len, input_size)
seq_len, batch, input_size, hidden_size = 10, 4, 32, 64

# RNN
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
             num_layers=2, batch_first=True, dropout=0.1)
x_seq = torch.randn(batch, seq_len, input_size)
output_r, h_n = rnn(x_seq)
print("RNN:")
print("  output shape (B,T,H):", output_r.shape)
print("  h_n shape (num_layers,B,H):", h_n.shape)

# LSTM
lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
               num_layers=2, batch_first=True, dropout=0.1)
output_l, (h_n, c_n) = lstm(x_seq)
print("LSTM:")
print("  output shape:", output_l.shape)
print("  h_n shape:", h_n.shape)
print("  c_n shape:", c_n.shape)

# GRU
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
             num_layers=2, batch_first=True)
output_g, h_n_g = gru(x_seq)
print("GRU:")
print("  output shape:", output_g.shape)
print("  h_n shape:", h_n_g.shape)

# ==============================================================
# 6. BatchNorm1d / BatchNorm2d
# ==============================================================
print()
print(SEP)
print("6. BatchNorm1d / BatchNorm2d")
print(SEP)
# BatchNorm1d：用于全连接层后，输入 (B, C) 或 (B, C, L)
bn1d = nn.BatchNorm1d(num_features=16)
x_fc = torch.randn(8, 16)
print("BatchNorm1d 输入:", x_fc.shape, "-> 输出:", bn1d(x_fc).shape)

# BatchNorm2d：用于卷积层后，输入 (B, C, H, W)
bn2d = nn.BatchNorm2d(num_features=16)
x_conv = torch.randn(4, 16, 8, 8)
print("BatchNorm2d 输入:", x_conv.shape, "-> 输出:", bn2d(x_conv).shape)

# ==============================================================
# 7. LayerNorm
# ==============================================================
print()
print(SEP)
print("7. LayerNorm")
print(SEP)
# LayerNorm 对每个样本的指定维度归一化（不受 batch size 影响）
# Transformer 标准：normalized_shape = [embed_dim]
ln = nn.LayerNorm(normalized_shape=64)
x_ln = torch.randn(4, 10, 64)   # (B, T, D)
print("LayerNorm 输入:", x_ln.shape, "-> 输出:", ln(x_ln).shape)

# ==============================================================
# 8. Dropout
# ==============================================================
print()
print(SEP)
print("8. Dropout")
print(SEP)
dropout = nn.Dropout(p=0.5)  # 50% 置零，其余缩放 1/(1-p)
x_d = torch.ones(3, 6)
torch.manual_seed(0)
out_d = dropout(x_d)
print("Dropout(p=0.5) 输出（train 模式）:")
print(out_d)
dropout.eval()
print("eval 模式（不丢弃）:", dropout(x_d))

# ==============================================================
# 9. MultiheadAttention
# ==============================================================
print()
print(SEP)
print("9. MultiheadAttention")
print(SEP)
# MultiheadAttention(embed_dim, num_heads)
# 输入 query/key/value: (seq_len, batch, embed_dim)
embed_dim = 64
num_heads = 8
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)

T, B = 10, 4
query = torch.randn(T, B, embed_dim)
key   = torch.randn(T, B, embed_dim)
value = torch.randn(T, B, embed_dim)

attn_out, attn_weights = mha(query, key, value)
print("attn_out shape:", attn_out.shape)
print("attn_weights shape:", attn_weights.shape)

# padding_mask：忽略某些位置（如 <PAD>）
key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
key_padding_mask[:, -2:] = True   # 最后2个位置为 padding
out_masked, _ = mha(query, key, value, key_padding_mask=key_padding_mask)
print("带 padding_mask 的输出 shape:", out_masked.shape)

# causal mask（自回归模型）
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
out_causal, _ = mha(query, key, value, attn_mask=causal_mask)
print("因果注意力 (attn_mask) 输出 shape:", out_causal.shape)
print()
print("11_常用网络层.py 运行完毕!")
