import torch
import torch.nn as nn


class config(object):
    VOCA_SIZE = 10000

    EMBEDDING_SIZE = 128

    NUM_LAYERS = 4

    HIDDEN_SIZE = 2 * EMBEDDING_SIZE

    DECODER_NUM_LAYERS = 2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.VOCA_SIZE, embedding_dim=config.EMBEDDING_SIZE)
        self.lstm_layer = nn.LSTM(
            input_size=config.EMBEDDING_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=False
        )
        self.ctx_feature_layer = nn.Sequential(
            nn.Linear(config.NUM_LAYERS * config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            nn.ReLU()
        )

    def forward(self, token_ids):
        embeddings = self.embed_layer(token_ids)
        outputs, state = self.lstm_layer(embeddings)
        # h0 t0 num_layer, bs, e
        h0, t0 = state
        h0_num_layer, h0_bs, h0_e = h0.shape
        h0 = torch.permute(h0, dims=(1, 0, 2))
        h0 = h0.reshape(h0_bs, -1)
        t0_num_layer, t0_bs, t0_e = t0.shape
        t0 = torch.permute(t0, dims=(1, 0, 2))
        t0 = t0.reshape(t0_bs, -1)
        feature_res = self.ctx_feature_layer((t0 + h0))
        return outputs, feature_res


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_layers):
        super().__init__()
        #  [bs, dt]
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
        # [bs, dt, hidden_size]
        # encoder_ctx: [bs, hidden_size]
        self.rnn_init_h0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        self.rnn_init_c0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        # self.rnn_init_h0_layers(encoder_ctx)  → h0
        # self.rnn_init_c0_layers(encoder_ctx)  → c0
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        # 这里做了一层适配，其实如果encoder做编码的时候直接mean的话就不需要了
        # self.ctx_adapter = nn.Linear(encoder_num_layer * hidden_size, hidden_size)
        # [bs, dt, hidden_size]
        self.classify = nn.Linear(hidden_size, out_features=vocabulary_size)

    def forward(self, token_ids, encoder_output, encoder_ctx):
        # encoder_ctx = self.ctx_adapter(encoder_ctx)
        bs, e = encoder_ctx.shape
        embedding = self.embedding(token_ids)
        bs, d_t, d_e = embedding.shape
        h0 = self.rnn_init_h0_layers(encoder_ctx)  # [bs, hidden] → [bs, num_layers*hidden]
        h0 = h0.reshape(bs, e, -1)  # [bs, hidden, num_layers]
        h0 = torch.permute(h0, dims=(2, 0, 1))  # [num_layers, bs, hidden]
        c0 = self.rnn_init_c0_layers(encoder_ctx)  # [bs, hidden] → [bs, num_layers*hidden]
        c0 = c0.reshape(bs, e, -1)  # [bs, hidden, num_layers]
        c0 = torch.permute(c0, dims=(2, 0, 1))  # [num_layers, bs, hidden]
        # self.lstm(c0 + h0)
        hcn = (h0, c0)  # 初始状态
        scores = []
        for _t in range(d_t):
            cur_token_embedding = embedding[:, _t:_t + 1, :]
            output, hcn = self.lstm(cur_token_embedding, hcn)
            score = self.classify(output)
            scores.append(score)

        return torch.concat(scores, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建 encoder 和 decoder
        self.encoder = Encoder()
        self.decoder = Decoder(vocabulary_size=config.VOCA_SIZE, hidden_size=config.HIDDEN_SIZE,
                               num_layers=config.DECODER_NUM_LAYERS)

    def forward(self, enc_token_ids, dec_token_ids):
        output, ctx = self.encoder(enc_token_ids)
        result = self.decoder(dec_token_ids, output, ctx)
        return result


def training():
    torch.manual_seed(24)
    model = Seq2Seq()
    print(model)
    print("=" * 60)

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 假数据: bs=1, encoder 输入 5 个 token, decoder 输入 6 个 token
    enc_token_ids = torch.tensor([[12, 35, 26, 34, 253]])         # [1, 5]
    dec_token_ids = torch.tensor([[3, 102, 235, 1523, 2132, 1243]])  # [1, 6]
    dec_target_ids = torch.tensor([[102, 235, 1523, 2132, 1243, 4]]) # [1, 6], 右移一位

    model.train()
    decoder_score = model(enc_token_ids, dec_token_ids)  # [1, 6, 10000]
    print(f"decoder_score shape: {decoder_score.shape}")

    # CrossEntropyLoss 需要 [bs, vocab, dt]
    loss = loss_fn(torch.permute(decoder_score, dims=(0, 2, 1)), dec_target_ids)
    print(f"loss: {loss}")


if __name__ == '__main__':
    training()
