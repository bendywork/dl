# -*- coding: utf-8 -*-
"""
RNN 循环神经网络训练模板（NLP方向 · 序列分类）
--------------------------------------------------------------
模型类型 : RNN (Vanilla RNN)
适用场景 : 短序列分类 / 序列标注
核心结构 : Embedding → RNN → Linear → 输出
局限性   : 长序列梯度消失/爆炸，长序列信息遗忘严重
--------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

torch.manual_seed(42)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.LongTensor([len(s) for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return padded, lengths, labels


def build_demo_data(vocab_size=3000, num_samples=800, num_classes=3):
    seqs = [torch.randint(1, vocab_size, size=(torch.randint(5, 30), ())) for _ in range(num_samples)]
    labels = torch.randint(0, num_classes, size=(num_samples,)).tolist()
    return seqs, labels


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn_layer = nn.RNN(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.classify_layer = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        embed = self.embed_layer(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, h_n = self.rnn_layer(packed)             # RNN 只返回一个状态
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)          # [bs, hidden*2]
        return self.classify_layer(h_cat)


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main():
    config = {
        'vocab_size': 3000, 'embedding_dim': 128, 'hidden_size': 128,
        'num_layers': 2, 'num_classes': 3, 'batch_size': 64, 'lr': 1e-3,
        'epochs': 20, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    seqs, labels = build_demo_data(vocab_size=config['vocab_size'], num_classes=config['num_classes'])
    split = int(len(seqs) * 0.8)
    train_loader = DataLoader(SequenceDataset(seqs[:split], labels[:split]),
                              batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(SequenceDataset(seqs[split:], labels[split:]),
                              batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = RNNModel(config['vocab_size'], config['embedding_dim'],
                     config['hidden_size'], config['num_layers'],
                     config['num_classes']).to(config['device'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, config['device'])
        valid_loss, valid_acc = validate(model, valid_loader, loss_fn, config['device'])
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'best_rnn_model.pt')
        print(f"Epoch {epoch:2d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"valid loss {valid_loss:.4f} acc {valid_acc:.4f}")
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")


if __name__ == '__main__':
    main()
