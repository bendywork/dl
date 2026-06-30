# -*- coding: utf-8 -*-
"""
作业 20251214 - 第二题
基于 LSTM/RNN/GRU/BERT 完成：
    A. 分词模型训练（字符级序列标注，BIO scheme）
    B. 推理（给定文本返回分词结果）
    C. 词性标注（POS tagging）最小改动扩展说明

─────────────────────────────────────────────────────────────
【数据格式】
训练数据 tokenizer_train.txt，每行一个已分词句子（用空格分隔词语）：
    人民 网 北京 1月 1日 电
→ 程序自动转换为字符级 BIO 标注序列：
    人/B 民/I 网/B 北/B 京/I 1/B 月/I 1/B 日/I 电/B
─────────────────────────────────────────────────────────────
【词性标注扩展（POS tagging）】
只需两处改动：
    1. 数据格式改为 "字/词性" 或 "词\t词性" 逐词标注；
    2. 将 BIO 标签集替换为词性标签集（B-NN、I-NN、B-VV、...），
       即换一个 label2id，其余代码（模型、训练、推理）完全不变。
─────────────────────────────────────────────────────────────
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "../../../base/datas")

CONFIG = dict(
    train_file=os.path.join(DATA_ROOT, "tokenizer_train.txt"),  # 每行：已分词句子
    save_dir=os.path.join(BASE, "checkpoints_tokenizer"),

    rnn_type="LSTM",   # LSTM | RNN | GRU | BERT（BERT 见下方 BertTagger）
    embed_dim=128,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    max_len=128,

    batch_size=32,
    epochs=20,
    lr=1e-3,
    device="cpu",
)

# BIO 标签（分词任务）
BIO_LABELS = ["<PAD>", "B", "I"]
# 词性标注时只需改这里（示例），其余不变：
# POS_LABELS = ["<PAD>", "B-NN", "I-NN", "B-VV", "I-VV", ...]


# ─────────────────────────────────────────────
# 数据处理：已分词句子 → BIO 字符序列
# ─────────────────────────────────────────────
def sentence_to_bio(segmented_sentence: str) -> Tuple[List[str], List[str]]:
    """
    输入：'人民 网 北京 电'
    输出：chars=['人','民','网',...], labels=['B','I','B',...]
    """
    chars, labels = [], []
    for word in segmented_sentence.strip().split():
        for i, ch in enumerate(word):
            chars.append(ch)
            labels.append("B" if i == 0 else "I")
    return chars, labels


def load_tokenizer_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    all_chars, all_labels = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chars, labels = sentence_to_bio(line)
            if chars:
                all_chars.append(chars)
                all_labels.append(labels)
    return all_chars, all_labels


# ─────────────────────────────────────────────
# 词表
# ─────────────────────────────────────────────
class CharVocab:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self):
        self.char2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2char = {0: self.PAD, 1: self.UNK}

    def build(self, char_lists: List[List[str]]):
        for chars in char_lists:
            for ch in chars:
                if ch not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[ch] = idx
                    self.idx2char[idx] = ch

    def encode(self, chars: List[str], max_len: int) -> List[int]:
        ids = [self.char2idx.get(c, 1) for c in chars[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        v = cls()
        with open(path, encoding="utf-8") as f:
            v.char2idx = json.load(f)
        v.idx2char = {int(i): c for c, i in v.char2idx.items()}
        return v

    @property
    def size(self):
        return len(self.char2idx)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class TokenizerDataset(Dataset):
    def __init__(self, char_lists, label_lists, vocab: CharVocab,
                 label2id: dict, max_len: int):
        self.samples = []
        for chars, labels in zip(char_lists, label_lists):
            x = torch.tensor(vocab.encode(chars, max_len), dtype=torch.long)
            # 标签也需要 padding（用 -100，CrossEntropyLoss 会自动忽略）
            l = [label2id[lb] for lb in labels[:max_len]]
            l += [-100] * (max_len - len(l))
            y = torch.tensor(l, dtype=torch.long)
            # 有效长度（用于推理时截断）
            length = min(len(chars), max_len)
            self.samples.append((x, y, length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# 模型（序列标注）
# ─────────────────────────────────────────────
class SeqLabelRNN(nn.Module):
    """
    字符级序列标注：每个字输出一个标签（B/I）
    与分类模型的唯一区别：最后的 FC 作用在每个时刻，而非对序列聚合。

    【词性标注最小改动】
        只需将 num_classes 对应的 label2id 换成词性标签集即可，
        模型结构、训练流程、推理代码完全不变。
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 rnn_type="LSTM", num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)
        direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction, num_classes)  # 作用于每个时刻
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))     # [bs, t, e]
        out, _ = self.rnn(emb)                    # [bs, t, h*dir]
        logits = self.fc(self.dropout(out))        # [bs, t, num_classes]
        return logits


# ─────────────────────────────────────────────
# BERT Tagger（选做）
# ─────────────────────────────────────────────
class BertTagger(nn.Module):
    """
    基于 transformers 库的 BERT 序列标注。
    需要：pip install transformers
    与 SeqLabelRNN 的接口完全相同（forward 返回 [bs, t, num_classes]），
    可以直接替换进训练/推理流程。
    """
    def __init__(self, bert_model_path: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, t]  token id（来自 BertTokenizer）
        attention_mask = (x != 0).long()
        out = self.bert(x, attention_mask=attention_mask).last_hidden_state  # [bs, t, 768]
        logits = self.fc(self.dropout(out))                                   # [bs, t, num_classes]
        return logits


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    char_lists, label_lists = load_tokenizer_data(cfg["train_file"])
    print(f"样本数: {len(char_lists)}")

    vocab = CharVocab()
    vocab.build(char_lists)
    vocab.save(os.path.join(cfg["save_dir"], "char_vocab.json"))

    label2id = {lb: i for i, lb in enumerate(BIO_LABELS)}
    id2label = {i: lb for lb, i in label2id.items()}
    with open(os.path.join(cfg["save_dir"], "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False)

    ds = TokenizerDataset(char_lists, label_lists, vocab, label2id, cfg["max_len"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = SeqLabelRNN(
        vocab_size=vocab.size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=len(label2id),
        rnn_type=cfg["rnn_type"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # ignore_index=-100 跳过 padding 位置的损失
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total = 0.0, 0
        for xb, yb, lens in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                         # [bs, t, num_classes]
            # reshape for loss: [bs*t, num_classes] vs [bs*t]
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        print(f"Epoch {epoch:02d}  loss={total_loss/total:.4f}")

    torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "model.pt"))
    print("模型已保存。")


# ─────────────────────────────────────────────
# 推理：文本 → 分词结果
# ─────────────────────────────────────────────
def infer(text: str) -> List[str]:
    cfg = CONFIG
    save_dir = cfg["save_dir"]
    device = torch.device(cfg["device"])

    vocab = CharVocab.load(os.path.join(save_dir, "char_vocab.json"))
    with open(os.path.join(save_dir, "label2id.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(i): lb for lb, i in label2id.items()}

    model = SeqLabelRNN(
        vocab_size=vocab.size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=len(label2id),
        rnn_type=cfg["rnn_type"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(save_dir, "model.pt"), map_location=device))
    model.eval()

    chars = list(text)
    x = torch.tensor([vocab.encode(chars, cfg["max_len"])], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)                        # [1, t, num_classes]
        pred_ids = logits[0].argmax(-1).tolist()  # [t]

    # BIO → 词列表
    words = []
    current = ""
    for ch, lid in zip(chars, pred_ids[:len(chars)]):
        label = id2label.get(lid, "B")
        if label == "B":
            if current:
                words.append(current)
            current = ch
        else:
            current += ch
    if current:
        words.append(current)

    print(f"输入: {text}")
    print(f"分词: {' / '.join(words)}")
    return words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--text", type=str, default="人民网北京1月1日电")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        infer(args.text)
