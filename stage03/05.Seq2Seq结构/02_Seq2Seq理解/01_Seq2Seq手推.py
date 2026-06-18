import torch

# ============================================================
# Seq2Seq 手推实现
# 结构：Encoder（LSTM） + Decoder（LSTM） + 线性输出层
# 不使用任何高层 API，完全展开每个时间步的矩阵运算
# ============================================================

# ---------- 超参数 ----------
batch_size   = 2
enc_seq_len  = 5   # 编码器输入序列长度（源序列）
dec_seq_len  = 4   # 解码器输入序列长度（目标序列，teacher forcing）
input_size   = 8   # 输入特征维度
hidden_size  = 16  # 隐藏层维度
vocab_size   = 20  # 输出词表大小

# ---------- 模拟输入 ----------
# enc_input: 编码器输入，shape=(batch, enc_seq_len, input_size)
# dec_input: 解码器输入（teacher forcing），shape=(batch, dec_seq_len, input_size)
enc_input = torch.randn(batch_size, enc_seq_len, input_size)
dec_input = torch.randn(batch_size, dec_seq_len, input_size)

# ============================================================
# Encoder 权重定义
# 每个门：W_x (input_size->hidden_size) + W_h (hidden_size->hidden_size) + bias
# 门：forget(f) / input(i) / cell_cache(g) / output(o)
# ============================================================
enc_Wxf = torch.nn.Linear(input_size,  hidden_size, bias=False)
enc_Whf = torch.nn.Linear(hidden_size, hidden_size, bias=True)
enc_Wxi = torch.nn.Linear(input_size,  hidden_size, bias=False)
enc_Whi = torch.nn.Linear(hidden_size, hidden_size, bias=True)
enc_Wxg = torch.nn.Linear(input_size,  hidden_size, bias=False)
enc_Whg = torch.nn.Linear(hidden_size, hidden_size, bias=True)
enc_Wxo = torch.nn.Linear(input_size,  hidden_size, bias=False)
enc_Who = torch.nn.Linear(hidden_size, hidden_size, bias=True)

# 提取权重矩阵（转置后直接右乘，x @ W）
enc_wxf, (enc_whf, enc_bf) = enc_Wxf.weight.T, (enc_Whf.weight.T, enc_Whf.bias)
enc_wxi, (enc_whi, enc_bi) = enc_Wxi.weight.T, (enc_Whi.weight.T, enc_Whi.bias)
enc_wxg, (enc_whg, enc_bg) = enc_Wxg.weight.T, (enc_Whg.weight.T, enc_Whg.bias)
enc_wxo, (enc_who, enc_bo) = enc_Wxo.weight.T, (enc_Who.weight.T, enc_Who.bias)

# ============================================================
# Encoder 前向传播（逐时间步展开）
# ============================================================
enc_h = torch.zeros(batch_size, hidden_size)  # 初始隐藏状态
enc_c = torch.zeros(batch_size, hidden_size)  # 初始细胞状态

for t in range(enc_seq_len):
    x = enc_input[:, t, :]                          # (batch, input_size)

    f = torch.sigmoid(x @ enc_wxf + enc_h @ enc_whf + enc_bf)   # 遗忘门
    i = torch.sigmoid(x @ enc_wxi + enc_h @ enc_whi + enc_bi)   # 输入门
    g = torch.tanh(   x @ enc_wxg + enc_h @ enc_whg + enc_bg)   # 候选细胞
    o = torch.sigmoid(x @ enc_wxo + enc_h @ enc_who + enc_bo)   # 输出门

    enc_c = f * enc_c + i * g   # 更新细胞状态
    enc_h = o * torch.tanh(enc_c)  # 更新隐藏状态

# Encoder 最终输出，作为 context vector 传给 Decoder
context_h = enc_h  # (batch, hidden_size)
context_c = enc_c  # (batch, hidden_size)
print(f"Encoder context_h: {context_h.shape}")

# ============================================================
# Decoder 权重定义（结构与 Encoder 完全相同）
# ============================================================
dec_Wxf = torch.nn.Linear(input_size,  hidden_size, bias=False)
dec_Whf = torch.nn.Linear(hidden_size, hidden_size, bias=True)
dec_Wxi = torch.nn.Linear(input_size,  hidden_size, bias=False)
dec_Whi = torch.nn.Linear(hidden_size, hidden_size, bias=True)
dec_Wxg = torch.nn.Linear(input_size,  hidden_size, bias=False)
dec_Whg = torch.nn.Linear(hidden_size, hidden_size, bias=True)
dec_Wxo = torch.nn.Linear(input_size,  hidden_size, bias=False)
dec_Who = torch.nn.Linear(hidden_size, hidden_size, bias=True)

dec_wxf, (dec_whf, dec_bf) = dec_Wxf.weight.T, (dec_Whf.weight.T, dec_Whf.bias)
dec_wxi, (dec_whi, dec_bi) = dec_Wxi.weight.T, (dec_Whi.weight.T, dec_Whi.bias)
dec_wxg, (dec_whg, dec_bg) = dec_Wxg.weight.T, (dec_Whg.weight.T, dec_Whg.bias)
dec_wxo, (dec_who, dec_bo) = dec_Wxo.weight.T, (dec_Who.weight.T, dec_Who.bias)

# ============================================================
# Decoder 前向传播（Teacher Forcing：每步喂真实目标词）
# 初始状态 = Encoder 的最终 (h, c)，即 context vector
# ============================================================
dec_h = context_h  # (batch, hidden_size)
dec_c = context_c  # (batch, hidden_size)

dec_outputs = []   # 收集每步的隐藏状态输出

for t in range(dec_seq_len):
    x = dec_input[:, t, :]                           # (batch, input_size)

    f = torch.sigmoid(x @ dec_wxf + dec_h @ dec_whf + dec_bf)
    i = torch.sigmoid(x @ dec_wxi + dec_h @ dec_whi + dec_bi)
    g = torch.tanh(   x @ dec_wxg + dec_h @ dec_whg + dec_bg)
    o = torch.sigmoid(x @ dec_wxo + dec_h @ dec_who + dec_bo)

    dec_c = f * dec_c + i * g
    dec_h = o * torch.tanh(dec_c)

    dec_outputs.append(dec_h.unsqueeze(1))  # 保存每步输出

# ============================================================
# 输出层：将每步隐藏状态投影到词表空间
# ============================================================
# 拼接所有时间步输出，shape=(batch, dec_seq_len, hidden_size)
dec_all_outputs = torch.cat(dec_outputs, dim=1)

# 线性层：hidden_size -> vocab_size
out_linear = torch.nn.Linear(hidden_size, vocab_size)
# shape=(batch, dec_seq_len, vocab_size)
logits = dec_all_outputs @ out_linear.weight.T + out_linear.bias

print(f"Decoder all outputs:  {dec_all_outputs.shape}")  # (2, 4, 16)
print(f"Logits (before softmax): {logits.shape}")        # (2, 4, 20)

# 取 argmax 得到预测 token（推理时使用，训练时对 logits 计算交叉熵）
pred_tokens = logits.argmax(dim=-1)
print(f"Predicted token ids: {pred_tokens}")             # (2, 4)
