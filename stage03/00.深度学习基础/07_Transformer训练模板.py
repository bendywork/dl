# -*- coding: utf-8 -*-
"""
Transformer 训练模板（NLP方向 · Encoder-Decoder 架构）
--------------------------------------------------------------
模型类型 : Transformer
适用场景 : 翻译、摘要、对话（现代 NLP 基准模型）
核心结构 : PositionalEncoding + Multi-Head Self-Attention + FFN + LayerNorm + 残差连接
关键革新 : 扔掉 RNN，纯 Attention 堆叠，训练可完全并行
--------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

torch.manual_seed(42)


class Seq2SeqDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx]


def collate_fn(batch):
    src, tgt = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


def build_demo_data(vocab_size=500, num_samples=500, max_len=10):
    src = [torch.randint(1, vocab_size, size=(torch.randint(3, max_len), ())) for _ in range(num_samples)]
    tgt = [torch.randint(1, vocab_size, size=(torch.randint(3, max_len), ())) for _ in range(num_samples)]
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == 0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_key_padding_mask = (tgt == 0)

        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(src_emb.size(-1)))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(tgt_emb.size(-1)))

        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_layer(output)


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_samples = 0, 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        dec_input = tgt[:, :-1]
        dec_target = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, dec_input)
        loss = loss_fn(logits.transpose(1, 2), dec_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
        total_samples += src.size(0)
    return total_loss / total_samples


def main():
    config = {
        'src_vocab_size': 500, 'tgt_vocab_size': 500,
        'd_model': 128, 'nhead': 4, 'num_encoder_layers': 2, 'num_decoder_layers': 2,
        'dim_feedforward': 256, 'batch_size': 32, 'lr': 1e-3, 'epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    src, tgt = build_demo_data(vocab_size=config['src_vocab_size'])
    split = int(len(src) * 0.8)
    train_loader = DataLoader(Seq2SeqDataset(src[:split], tgt[:split]),
                              batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = TransformerModel(
        config['src_vocab_size'], config['tgt_vocab_size'],
        config['d_model'], config['nhead'],
        config['num_encoder_layers'], config['num_decoder_layers'],
        config['dim_feedforward'],
    ).to(config['device'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(1, config['epochs'] + 1):
        loss = train_epoch(model, train_loader, loss_fn, optimizer, config['device'])
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | train loss {loss:.4f}")

    torch.save(model.state_dict(), 'best_transformer_model.pt')
    print(f"训练完成")


if __name__ == '__main__':
    main()
