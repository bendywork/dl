from typing import List

import torch



def get_lstm_params(vocabulary_size:int, hidden_num:int, device:torch.device) -> List[torch.float32]:
    """
    获取LSTM的参数对象
    :param vocabulary_size:
    :param hidden_num:
    :param device:
    :return:
    """

    input_num = output_num = vocabulary_size

    def normal(shape):
        """
        生成最小高斯分布随机数目的是随机初始化权重
        :param shape:
        :return:
        """
        return torch.randn(size=shape, device= device) * 0.01

    def three_switch():
        return (normal((input_num, hidden_num)),
                normal((hidden_num, hidden_num)),
                torch.zeros((hidden_num,), device= device))

    W_xi, W_hi, b_i = three_switch()  # 输入门参数
    W_xf, W_hf, b_f = three_switch()  # 遗忘门参数
    W_xo, W_ho, b_o = three_switch()  # 输出门参数
    W_xc, W_hc, b_c = three_switch()  # 候选记忆元参数

    # 输出层参数
    W_hq = normal((hidden_num, output_num))
    b_q = torch.zeros(output_num, device=device)

    # 附加梯度
    params = [W_xi, W_hi, b_i,
              W_xf, W_hf, b_f,
              W_xo, W_ho, b_o,
              W_xc, W_hc,b_c,
              W_hq,
              b_q]

    for param in params:
        # 批量设置需要梯度计算
        param.requires_grad_(True)

    return params


def init_lstm_state(batch_size, num_hidden, device):
    """
    定义模型
    :param batch_size:
    :param num_hidden:
    :param device:
    :return:
    """
    return (torch.zeros((batch_size, num_hidden), device=device),
            torch.zeros((batch_size, num_hidden), device=device))


def lstm_forward(inputs, state, params):
    """
    inputs: [num_steps, batch_size, vocab_size] one-hot
    state: (H, C) 各自 [batch_size, num_hidden]
    """
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    H, C = state
    outputs = []

    for X in inputs:  # X: [batch_size, vocab_size]
        # 输入门
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        # 遗忘门
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        # 输出门
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        # 候选记忆
        C_tilde = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        # 更新细胞状态
        C = F * C + I * C_tilde
        # 更新隐藏状态
        H = O * torch.tanh(C)
        # 输出层：映射到词表
        Y = H @ W_hq + b_q  # [batch_size, vocab_size]
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


def train():
    vocab_size, num_hidden = 1000, 256
    device = torch.device('cpu')
    num_epochs, lr = 500, 1

    params = get_lstm_params(vocab_size, num_hidden, device)

    for epoch in range(num_epochs):
        state = init_lstm_state(batch_size=32, num_hidden=num_hidden, device=device)
        H, C = state

        # 模拟一批数据：5个时间步，batch=32，one-hot vocab_size=1000
        inputs = torch.randn(5, 32, vocab_size, device=device)
        targets = torch.randint(0, vocab_size, (5 * 32,), device=device)

        # 前向
        outputs, _ = lstm_forward(inputs, (H, C), params)
        # outputs: [num_steps * batch_size, vocab_size]

        loss = torch.nn.functional.cross_entropy(outputs, targets)

        # 清零梯度
        for param in params:
            if param.grad is not None:
                param.grad.zero_()

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # 手动 SGD 更新
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad

        if epoch % 50 == 0:
            print(f'epoch {epoch}, loss {loss.item():.4f}')


if __name__ == '__main__':
    train()