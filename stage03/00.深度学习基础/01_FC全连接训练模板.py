# -*- coding: utf-8 -*-
"""
全连接神经网络训练模板（NLP方向 · 文本分类）
--------------------------------------------------------------
模型类型 : 全连接 (MLP / Feed-Forward Network)
适用场景 : 文本分类、情感分析（仅用于短文本或已做池化的文本）
核心结构 : Embedding → Flatten/Pooling → Linear → ReLU → Linear → softmax
局限性   : token 之间无序列信息交互，无上下文融合能力
--------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

torch.manual_seed(42)


# 1. 数据准备
class TextDataset(Dataset):
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


# ============================================================
# 2. 模型定义
# ============================================================

class MLPModel(nn.Module):
    """全连接分类模型：Embedding → 池化 → Linear 堆叠"""

    def __init__(self, vocab_size, embedding_dim, hidden_sizes, num_classes, max_seq_len=30):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        input_dim = embedding_dim * max_seq_len
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_dim, hs))
            layers.append(nn.ReLU())
            input_dim = hs
        layers.append(nn.Linear(input_dim, num_classes))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x, lengths):
        embed = self.embed_layer(x)                       # [bs, t, e]
        # 零填充到固定长度，展平为单一向量
        padded = nn.functional.pad(
            embed, (0, 0, 0, self.max_seq_len - embed.size(1)), value=0
        )
        flat = padded.reshape(padded.size(0), -1)          # [bs, t*e]
        return self.fc_layers(flat)                        # [bs, num_classes]


# ============================================================
# 3. 训练 / 验证
# ============================================================

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


# ============================================================
# 4. 主流程
# ============================================================

def main():
    config = {
        'vocab_size': 3000, 'embedding_dim': 128, 'hidden_sizes': [256, 128],
        'num_classes': 3, 'batch_size': 64, 'lr': 1e-3,
        'epochs': 20, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    seqs, labels = build_demo_data(
        vocab_size=config['vocab_size'], num_classes=config['num_classes']
    )
    split = int(len(seqs) * 0.8)
    train_loader = DataLoader(
        SequenceDataset(seqs[:split], labels[:split]),
        batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        SequenceDataset(seqs[split:], labels[split:]),
        batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,
    )

    model = MLPModel(
        config['vocab_size'], config['embedding_dim'],
        config['hidden_sizes'], config['num_classes']
    ).to(config['device'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, config['device'])
        valid_loss, valid_acc = validate(model, valid_loader, loss_fn, config['device'])
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'best_mlp_model.pt')
        print(f"Epoch {epoch:2d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"valid loss {valid_loss:.4f} acc {valid_acc:.4f}")
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")


if __name__ == '__main__':
    main()
