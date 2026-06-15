import torch

batch_size = 2
src_len = 5 # 源序列长度（编码器输入步数）
trg_len = 4 # 目标序列长度（解码器输入步数）
input_size = 128
hidden_size = 256
vocabulary_size = 100

encoding_input = torch.randn(batch_size, src_len, input_size)

# 编码器的lstm
# f
f_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
f_h_linear = torch.nn.Linear(hidden_size, hidden_size)
f_x_weight = f_x_linear.weight.T
f_h_weight, f_h_bias = f_h_linear.weight.T, f_h_linear.bias
# i
i_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
i_h_linear = torch.nn.Linear(hidden_size, hidden_size)
i_x_weight = i_x_linear.weight.T
i_h_weight, i_h_bias = i_h_linear.weight.T, i_h_linear.bias
# g
g_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
g_h_linear = torch.nn.Linear(hidden_size, hidden_size)
g_x_weight = g_x_linear.weight.T
g_h_weight, g_h_bias = g_h_linear.weight.T, g_h_linear.bias
# o
o_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
o_h_linear = torch.nn.Linear(hidden_size, hidden_size)
o_x_weight = o_x_linear.weight.T
o_h_weight, o_h_bias = o_h_linear.weight.T, o_h_linear.bias

# h
h_t = torch.zeros(batch_size, hidden_size)
# c
c_t = torch.zeros(batch_size, hidden_size)


for t in range(src_len):
    x_t = encoding_input[:, t, :]
    f_t = torch.sigmoid(x_t @ f_x_weight + h_t @ f_h_weight + f_h_bias)
    i_t = torch.sigmoid(x_t @ i_x_weight + h_t @ i_h_weight + i_h_bias)
    g_t = torch.tanh(x_t @ g_x_weight + h_t @ g_h_weight + g_h_bias)
    o_t = torch.sigmoid(x_t @ o_x_weight + h_t @ o_h_weight + o_h_bias)
    c_t = f_t * c_t + i_t * g_t
    h_t = o_t * torch.tanh(c_t)


context_h = h_t
context_c = c_t
print(f"Encoder context_h: {context_h.shape}")
print(f"Encoder context_c: {context_c.shape}")



# 解码器
# 解码器lstm
# f
f_x_e_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
f_h_e_linear = torch.nn.Linear(hidden_size, hidden_size)
f_x_e_weight = f_x_e_linear.weight.T
f_h_e_weight, f_h_e_bias = f_h_e_linear.weight.T, f_h_e_linear.bias
# i
i_x_e_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
i_h_e_linear = torch.nn.Linear(hidden_size, hidden_size)
i_x_e_weight = i_x_e_linear.weight.T
i_h_e_weight, i_h_e_bias = i_h_e_linear.weight.T, i_h_e_linear.bias
# g
g_x_e_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
g_h_e_linear = torch.nn.Linear(hidden_size, hidden_size)
g_x_e_weight = g_x_e_linear.weight.T
g_h_e_weight, g_h_e_bias = g_h_e_linear.weight.T, g_h_e_linear.bias
# o
o_x_e_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
o_h_e_linear = torch.nn.Linear(hidden_size, hidden_size)
o_x_e_weight = o_x_e_linear.weight.T
o_h_e_weight, o_h_e_bias = o_h_e_linear.weight.T, o_h_e_linear.bias

# 解码器 初始h_t
h_t = context_h
# 解码器 初始c_t
c_t = context_c
decoding_input = torch.randn(batch_size, trg_len, input_size)
output_linear = torch.nn.Linear(hidden_size, vocabulary_size)
outputs = []
for i in range(trg_len):
    x_t = decoding_input[:, i, :]  # ← 目标序列逐步喂入
    # 用解码器自己的权重做 LSTM 计算
    f_t = torch.sigmoid(x_t @ f_x_e_weight + h_t @ f_h_e_weight + f_h_e_bias)
    i_t = torch.sigmoid(x_t @ i_x_e_weight + h_t @ i_h_e_weight + i_h_e_bias)
    g_t = torch.tanh(x_t @ g_x_e_weight + h_t @ g_h_e_weight + g_h_e_bias)
    o_t = torch.sigmoid(x_t @ o_x_e_weight + h_t @ o_h_e_weight + o_h_e_bias)
    c_t = f_t * c_t + i_t * g_t
    h_t = o_t * torch.tanh(c_t)

    pred = output_linear(h_t)  # ← 就是这步！
    outputs.append(pred)

outputs = torch.stack(outputs, dim=1)  # (2, 4, 100)

