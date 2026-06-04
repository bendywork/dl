import torch

# LSTM 每个时间步的核心变量：
# x_t       : 当前时间步的输入向量，shape=(batch, input_size)
# h_t_prev  : 上一时间步的隐藏状态，shape=(batch, hidden_size)
# h_t       : 当前时间步的隐藏状态（输出），shape=(batch, hidden_size)
# c_t_prev  : 上一时间步的细胞状态（长期记忆），shape=(batch, hidden_size)
# c_t       : 当前时间步的细胞状态（更新后的长期记忆）
# c_cache   : 候选细胞状态（tanh门的输出，又叫 g_t）

# 你写的：
# f_t = torch.sigmoid(x_t @ w_ft.T + h_t_prev * w_ft + b_ft)
# ⚠️ 问题1：x_t 和 h_t_prev 应该各有独立权重矩阵，用同一个 w_ft 会混淆
# ⚠️ 问题2：h_t_prev * w_ft 是逐元素乘，但 h_t_prev 是向量、w_ft 是矩阵，应该用 @ 做矩阵乘
# f_t = torch.sigmoid(x_t @ w_xf.T + h_t_prev @ w_hf.T + b_f)

# 你写的：
# i_t = torch.sigmoid(x_t @ w_it.T + h_t_prev * w_it + b_it)
# ⚠️ 同上：h_t_prev * w_it 应改为 h_t_prev @ w_hi.T，权重名分开
# i_t = torch.sigmoid(x_t @ w_xi.T + h_t_prev @ w_hi.T + b_i)

# 你写的：
# c_cache = torch.tanh(x_t @ w_cct.T + h_t_prev * w_cct + b_cct)
# ⚠️ 同上：权重名统一，h_t_prev 用 @ 矩阵乘
# c_cache = torch.tanh(x_t @ w_xg.T + h_t_prev @ w_hg.T + b_g)

# 你写的：
# c_t = f_t @ c_t_prev + i_t @ c_cache
# ⚠️ 严重错误：f_t、c_t_prev、i_t、c_cache shape 都是 (batch, hidden_size)
#    这里是逐元素相乘（门控），不是矩阵乘法，用 * 不是 @
# c_t = f_t * c_t_prev + i_t * c_cache

# 你写的：
# o_t = torch.sigmoid(x_ot @ w_ot.T + h_t_prev * w_ot + b_ot)
# ⚠️ 问题1：x_ot 应为 x_t，没有专门的 x_ot 变量
# ⚠️ 问题2：h_t_prev * w_ot 同样应改为 @ 矩阵乘，权重名分开
# o_t = torch.sigmoid(x_t @ w_xo.T + h_t_prev @ w_ho.T + b_o)

# 你没写这步，但这是 LSTM 最后一步，必须有：
# h_t = o_t * torch.tanh(c_t)
# 含义：输出门控制从细胞状态中读出多少，tanh 把 c_t 压缩到 (-1, 1)
# h_t = o_t * torch.tanh(c_t)
batch_size = 10
t = 5
input_size = 128
hidden_size = 256
x_t = torch.randn(batch_size, t, input_size)
h_t_prev = torch.zeros(batch_size,hidden_size)
c_t_prev = torch.zeros(batch_size,hidden_size)
forget_x_input_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
w_xf = forget_x_input_linear.weight.T
forget_h_input_linear = torch.nn.Linear(hidden_size, hidden_size)
input_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
input_h_input_linear = torch.nn.Linear(hidden_size, hidden_size)
c_cache_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
c_cache_h_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
output_x_linear = torch.nn.Linear(input_size, hidden_size, bias=False)
output_h_linear = torch.nn.Linear(hidden_size, hidden_size, bias=True)
w_xi = input_x_linear.weight.T
w_xc_cache = c_cache_x_linear.weight.T
w_hf, bias_hf = forget_h_input_linear.weight.T, forget_h_input_linear.bias
w_hi, bias_hi = input_h_input_linear.weight.T, input_h_input_linear.bias
w_hc_cache, bias_hc_cache = c_cache_h_linear.weight.T, c_cache_h_linear.bias
w_xo = output_x_linear.weight.T
w_ho, bias_ho = output_h_linear.weight.T, output_h_linear.bias
# embedding_news = []
for t_index in range(t):
    x_t_index_input = x_t[:,t_index,:]
    forget_gate = torch.sigmoid(x_t_index_input @ w_xf + h_t_prev @ w_hf + bias_hf)
    c_t_cache = c_t_prev * forget_gate
    input_gate = torch.sigmoid(x_t_index_input @ w_xi + h_t_prev @ w_hi + bias_hi)
    c_cache = torch.tanh(x_t_index_input @ w_xc_cache + h_t_prev @ w_hc_cache + bias_hc_cache)
    c_t_current = input_gate * c_cache
    c_t = c_t_cache + c_t_current
    output_gate = torch.sigmoid(x_t_index_input @ w_xo + h_t_prev @ w_ho + bias_ho)
    c_t_prev = c_t
    h_t = output_gate * torch.tanh(c_t)
    h_t_prev = h_t