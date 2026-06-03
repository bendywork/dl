import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



def rnn():
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    input_linear = nn.Linear(e, 2 * e, dtype=dtype) # 输入的权重矩阵U
    hidden_linear = nn.Linear(2 * e, 2 * e, dtype=dtype) # 隐状态的权重矩阵W

    token_embeddings = torch.randn(bs, t, e, dtype=dtype) # 输入序列 [bs, t, e]
    token_new_embeddings = []

    input_w , input_b = input_linear.weight.T, input_linear.bias
    hidden_w , hidden_b = hidden_linear.weight.T, hidden_linear.bias
    hidden_pre = torch.zeros(bs, 2*e, dtype=dtype)
    for t_index in range(t):
        # 先拿到t时刻对应的embedding向量
        x_t = token_embeddings[:,t_index,:]
        x_trans = torch.matmul(x_t, input_w) + input_b
        h_trans = torch.matmul(hidden_pre, hidden_w) + hidden_b
        hidden_current = torch.tanh(x_trans + h_trans)
        hidden_pre = hidden_current
        token_new_embeddings.append(hidden_current[:,None])
    token_new_embeddings = torch.cat(token_new_embeddings, dim=1)
    


