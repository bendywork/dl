import torch
from torch import nn
from torch.nn import functional as F


def test_rnn():
    """
    1. 循环神经网络（RNN）提取特征向量
     优点：能够处理变长输入；能够捕捉长距离依赖关系；能够捕捉序列中的位置信息
     问题：存在梯度消失和梯度爆炸问题；计算效率较低；难以并行化训练
     解决方案：引入门控机制（如LSTM、GRU）来缓解梯度消失问题；使用残差连接和层归一化来缓解梯度爆炸问题；使用双向RNN和注意力机制来提高模型性能
     总结：RNN是一种强大的序列建模工具，但也存在一些挑战。通过引入门控机制、残差连接、层归一化、双向RNN和注意力机制等技术，可以有效地提升RNN的性能，并在自然语言处理等领域取得了显著的成果。
     后续：Transformer模型通过自注意力机制克服了RNN的一些限制，成为当前自然语言处理领域的主流模型架构。
     未来：随着深度学习技术的发展，可能会出现更多创新的模型架构，如基于图神经网络的序列建模方法，以及结合RNN和Transformer优势的新型混合模型。
     总之，RNN在序列建模方面具有重要意义，但也需要不断改进和创新，以适应不断变化的应用需求和挑战。
     参考资料：
      - RNN原理与实现：https://www.bilibili.com/video/BV1s4411H7Zt?p=1
      - LSTM原理与实现：https://www.bilibili.com/video/B
    """
    # RNN 公式：H_t = act(U * X_t + W * H_{t-1})
    #   X_t    : 当前时刻输入的 token 特征向量
    #   H_{t-1}: 上一时刻的隐状态（记忆）
    #   U      : 输入权重矩阵，负责对当前输入做线性变换
    #   W      : 隐状态权重矩阵，负责对上一时刻隐状态做线性变换
    #   act    : 激活函数（此处用 relu）

    bs, t, e = 1, 5, 128
    dtype = torch.float32

    # U 矩阵对应的线性层：将当前输入 X_t 从 e 维映射到 2e 维
    input_linear = nn.Linear(e, e *2, dtype=dtype)
    input_weight, input_bias = input_linear.weight.T, input_linear.bias


    # W 矩阵对应的线性层：将上一时刻隐状态 H_{t-1} 从 2e 维映射到 2e 维
    hidden_linear = nn.Linear(2 * e, e * 2, dtype=dtype)
    hidden_weight, hidden_bias = hidden_linear.weight.T, hidden_linear.bias

    token_embeddings = torch.randn(bs, t, e, dtype=dtype)  # 输入序列 [bs, t, e]
    token_new_embeddings = []
    h_prev = torch.zeros(bs, 2 * e, dtype=dtype)  # H_0 初始化为零向量 [bs, 2e]
    for index in range(t):
        # 取当前时刻 t 的输入 token 特征向量
        current_input = token_embeddings[:, index, :]                          # X_t [bs, e]

        # U * X_t：对当前输入做线性变换
        input_transform = torch.matmul(current_input, input_weight) + input_bias    # [bs, 2e]

        # W * H_{t-1}：对上一时刻隐状态做线性变换
        hidden_transform = torch.matmul(h_prev, hidden_weight) + hidden_bias        # [bs, 2e]

        # H_t = act(U * X_t + W * H_{t-1})：融合当前输入与历史记忆，得到当前隐状态
        h_cur = F.relu(input_transform + hidden_transform)                     # [bs, 2e]

        h_prev = h_cur                                                         # 当前隐状态作为下一时刻的 H_{t-1}
        token_new_embeddings.append(h_cur[:, None]) 
    


     # U 矩阵对应的线性层：将当前输入 X_t 从 e 维映射到 2e 维
    input_linear_back = nn.Linear(e, e *2, dtype=dtype)
    input_weight_back, input_bias_back = input_linear_back.weight.T, input_linear_back.bias
    # W 矩阵对应的线性层：将上一时刻隐状态 H_{t-1} 从 2e 维映射到 2e 维
    hidden_linear_back = nn.Linear(2 * e, e * 2, dtype=dtype)
    hidden_weight_back, hidden_bias_back = hidden_linear_back.weight.T, hidden_linear_back.bias
    h_prev_back = torch.zeros(bs, 2 * e, dtype=dtype)  # H_T 初始化为零向量 [bs, 2e]
    token_new_embeddings_back = []
    for index in range(t-1, -1, -1):
        # 取当前时刻 t 的输入 token 特征向量
        current_input = token_embeddings[:, index, :]                          # X_t [bs, e]

        # U * X_t：对当前输入做线性变换
        input_transform = torch.matmul(current_input, input_weight_back) + input_bias_back    # [bs, 2e]

        # W * H_{t+1}：对下一时刻隐状态做线性变换
        hidden_transform = torch.matmul(h_prev_back, hidden_weight_back) + hidden_bias_back        # [bs, 2e]

        # H_t = act(U * X_t + W * H_{t+1})：融合当前输入与历史记忆，得到当前隐状态
        h_cur_back = F.relu(input_transform + hidden_transform)                     # [bs, 2e]

        h_prev_back = h_cur_back                                                         # 当前隐状态作为下一时刻的 H_{t+1}
        token_new_embeddings_back.append(h_cur_back[:, None])
    
    result = torch.cat([torch.cat(token_new_embeddings, dim=1), torch.cat(token_new_embeddings_back, dim=1)], dim=-1)  # [bs, t, 4e]
    return result
