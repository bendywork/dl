import torch


def build_rnn():
    # 每次处理bs个样本，也就是bs个语句，每一个语句最大长度是t，包含t个token，然后每一个token被映射为128维度的向量
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    embeddings = torch.randn(bs, t, e)

    input = torch.nn.Linear(e, e * 2, bias=False, dtype=dtype)
    hidden = torch.nn.Linear(e*2, e*2, bias=True, dtype=dtype)

    input_w = input.weight.T
    hidden_w, hidden_b = hidden.weight.T, hidden.bias

    h_prev = torch.zeros(bs, 2*e)
    token_new_embeddings = []
    for t_index in range(t):
        # bs,e
        x_t = embeddings[:,t_index,:]
        h_trans = h_prev @ hidden_w + hidden_b
        x_trans = x_t @ input_w
        h_curr = torch.relu(x_trans + h_trans)
        h_prev = h_curr
        token_new_embeddings.append(h_curr[:,None])

    token_new_embeddings = torch.cat(token_new_embeddings, dim=1)
    print("shape is {}".format(token_new_embeddings.shape))

if __name__ == "__main__":
    build_rnn()