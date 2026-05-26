# -*- coding: utf-8 -*-
"""
作业 20251207 - 第二题
文本分类：分词 + Token2Id + Embedding + RNN/LSTM/GRU + FC + CrossEntropyLoss + SGD

数据格式（text_classify）：每行 "文本\t标签"
数据路径：../../../base/datas/text_classify/train.csv  /  test.csv

使用方式：
    python hw2_text_classify_rnn.py --mode train
    python hw2_text_classify_rnn.py --mode infer --text "今天股市大涨" --topk 3
    # 切换数据集只需修改 CONFIG 中的路径与分隔符，其余代码无需改动
"""

import os
import json
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 全局配置 —— 切换数据集只改这里
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "../../../base/datas/text_classify")

CONFIG = dict(
    train_file=os.path.join(DATA_ROOT, "train.csv"),
    test_file=os.path.join(DATA_ROOT, "test.csv"),
    sep="\t",           # 字段分隔符
    text_col=0,         # 文本列下标
    label_col=1,        # 标签列下标
    has_header=False,   # 文件是否有表头

    # 模型保存目录
    save_dir=os.path.join(BASE, "checkpoints"),

    # 分词方式："char"（按字）| "jieba"（结巴词语级）
    tokenize_mode="char",

    # 超参数
    max_len=64,
    embed_dim=128,
    hidden_size=256,
    rnn_type="LSTM",    # RNN | LSTM | GRU
    num_layers=2,
    bidirectional=True,
    dropout=0.3,

    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu",
)


# ─────────────────────────────────────────────
# 1. 分词
# ─────────────────────────────────────────────
def tokenize(text: str, mode: str = "char") -> list:
    if mode == "char":
        return list(text.strip())
    elif mode == "jieba":
        import jieba
        return jieba.lcut(text.strip())
    else:
        return text.strip().split()


# ─────────────────────────────────────────────
# 2. 词表
# ─────────────────────────────────────────────
class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, token_lists: list, min_freq: int = 1):
        counter = Counter(t for tokens in token_lists for t in tokens)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens: list, max_len: int) -> list:
        ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))   # padding
        return ids

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        v = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v.word2idx = data["word2idx"]
        v.idx2word = {int(i): w for w, i in v.word2idx.items()}
        return v

    @property
    def size(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
def read_data(file_path: str, sep: str, text_col: int, label_col: int, has_header: bool):
    texts, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if has_header and i == 0:
                continue
            parts = line.strip().split(sep)
            if len(parts) <= max(text_col, label_col):
                continue
            texts.append(parts[text_col])
            labels.append(parts[label_col])
    return texts, labels


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, label2id: dict,
                 max_len: int, tokenize_mode: str):
        self.samples = []
        for text, label in zip(texts, labels):
            tokens = tokenize(text, tokenize_mode)
            ids = vocab.encode(tokens, max_len)
            lid = label2id[label]
            self.samples.append((torch.tensor(ids, dtype=torch.long),
                                  torch.tensor(lid, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# 4. 模型
# ─────────────────────────────────────────────
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 rnn_type="LSTM", num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))           # [bs, t, e]
        out, _ = self.rnn(emb)                          # [bs, t, h*dir]
        # mean pooling over time
        feat = out.mean(dim=1)                          # [bs, h*dir]
        logits = self.fc(self.dropout(feat))            # [bs, num_classes]
        return logits


# ─────────────────────────────────────────────
# 5. 训练
# ─────────────────────────────────────────────
def train():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    # 读数据
    train_texts, train_labels = read_data(
        cfg["train_file"], cfg["sep"], cfg["text_col"], cfg["label_col"], cfg["has_header"])

    # 词表 & 标签表
    vocab = Vocabulary()
    token_lists = [tokenize(t, cfg["tokenize_mode"]) for t in train_texts]
    vocab.build(token_lists)
    vocab.save(os.path.join(cfg["save_dir"], "vocab.json"))

    label_set = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    with open(os.path.join(cfg["save_dir"], "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False)

    print(f"词表大小: {vocab.size}  类别数: {len(label_set)}  训练样本: {len(train_texts)}")

    # DataLoader
    train_ds = TextDataset(train_texts, train_labels, vocab, label2id,
                           cfg["max_len"], cfg["tokenize_mode"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=False)

    # 模型
    model = TextRNN(
        vocab_size=vocab.size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=len(label_set),
        rnn_type=cfg["rnn_type"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch:02d}/{cfg['epochs']}  loss={avg_loss:.4f}  acc={acc:.4f}")

    # 保存模型
    model_path = os.path.join(cfg["save_dir"], "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")


# ─────────────────────────────────────────────
# 6. 推理
# ─────────────────────────────────────────────
def infer(text: str, topk: int = 3):
    cfg = CONFIG
    save_dir = cfg["save_dir"]
    device = torch.device(cfg["device"])

    # 加载词表
    vocab = Vocabulary.load(os.path.join(save_dir, "vocab.json"))
    with open(os.path.join(save_dir, "label2id.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(i): l for l, i in label2id.items()}
    num_classes = len(label2id)

    # 重建模型
    model = TextRNN(
        vocab_size=vocab.size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=num_classes,
        rnn_type=cfg["rnn_type"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt"),
                                     map_location=device))
    model.eval()

    # 处理输入
    tokens = tokenize(text, cfg["tokenize_mode"])
    ids = vocab.encode(tokens, cfg["max_len"])
    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)                    # [1, num_classes]
        probs = F.softmax(logits, dim=-1)[0] # [num_classes]

    topk = min(topk, num_classes)
    top_probs, top_ids = probs.topk(topk)

    print(f"\n输入文本: {text}")
    print(f"{'排名':<4} {'类别ID':<8} {'类别名称':<16} {'概率'}")
    print("-" * 45)
    for rank, (cid, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        print(f"  {rank:<4} {cid:<8} {id2label[cid]:<16} {prob:.4f}")

    return [{"rank": r + 1, "label_id": top_ids[r].item(),
             "label": id2label[top_ids[r].item()],
             "prob": top_probs[r].item()} for r in range(topk)]


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--text", type=str, default="今天的新闻很精彩")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        infer(args.text, args.topk)
