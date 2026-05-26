# -*- coding: utf-8 -*-
"""
作业 20251214 - 第一题
意图识别分类模型（LSTM）
与 20251207/hw2 结构完全一致，只需：
  1. CONFIG 中将 rnn_type 改为 "LSTM"
  2. 更换数据路径
这里独立放一份以便单独提交，核心逻辑与 hw2 共用。

数据格式：每行  文本 \t 意图标签
用法：
    python hw1_intent_classify_lstm.py --mode train
    python hw1_intent_classify_lstm.py --mode infer --text "帮我订明天去北京的机票" --topk 3
"""

import os
import sys

# 允许直接复用 hw2 里的通用组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../20251207"))
from hw2_text_classify_rnn import (
    tokenize, Vocabulary, TextDataset, TextRNN, read_data, infer as _infer
)

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# 仅需修改此处即可切换数据集
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "../../../base/datas/text_classify")

CONFIG = dict(
    train_file=os.path.join(DATA_ROOT, "train.csv"),
    test_file=os.path.join(DATA_ROOT, "test.csv"),
    sep="\t",
    text_col=0,
    label_col=1,
    has_header=False,

    save_dir=os.path.join(BASE, "checkpoints_lstm_intent"),

    tokenize_mode="char",
    max_len=64,
    embed_dim=128,
    hidden_size=256,
    rnn_type="LSTM",        # ← 关键：使用 LSTM
    num_layers=2,
    bidirectional=True,
    dropout=0.3,

    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu",
)


def train():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    train_texts, train_labels = read_data(
        cfg["train_file"], cfg["sep"], cfg["text_col"],
        cfg["label_col"], cfg["has_header"])

    vocab = Vocabulary()
    token_lists = [tokenize(t, cfg["tokenize_mode"]) for t in train_texts]
    vocab.build(token_lists)
    vocab.save(os.path.join(cfg["save_dir"], "vocab.json"))

    label_set = sorted(set(train_labels))
    label2id = {l: i for i, l in enumerate(label_set)}
    with open(os.path.join(cfg["save_dir"], "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False)

    print(f"词表: {vocab.size}  意图类别: {len(label_set)}  样本数: {len(train_texts)}")

    train_ds = TextDataset(train_texts, train_labels, vocab, label2id,
                           cfg["max_len"], cfg["tokenize_mode"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=False)

    model = TextRNN(
        vocab_size=vocab.size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_classes=len(label_set),
        rnn_type="LSTM",
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        print(f"Epoch {epoch:02d}  loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "model.pt"))
    print("模型已保存。")


def infer(text: str, topk: int = 3):
    """复用 hw2 的 infer，传入当前 CONFIG"""
    import importlib
    import hw2_text_classify_rnn as m
    old_cfg = m.CONFIG.copy()
    m.CONFIG.update(CONFIG)
    result = _infer(text, topk)
    m.CONFIG.update(old_cfg)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--text", type=str, default="帮我设置一个明天早上八点的闹钟")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        # 直接调用本文件内的推理逻辑（与 hw2 共用，但 save_dir 指向 checkpoints_lstm_intent）
        cfg = CONFIG
        import torch.nn.functional as F
        from hw2_text_classify_rnn import Vocabulary, TextRNN, tokenize

        save_dir = cfg["save_dir"]
        vocab = Vocabulary.load(os.path.join(save_dir, "vocab.json"))
        with open(os.path.join(save_dir, "label2id.json"), encoding="utf-8") as f:
            label2id = json.load(f)
        id2label = {int(i): l for l, i in label2id.items()}

        device = torch.device(cfg["device"])
        model = TextRNN(vocab.size, cfg["embed_dim"], cfg["hidden_size"],
                        len(label2id), "LSTM", cfg["num_layers"],
                        cfg["bidirectional"], cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(
            os.path.join(save_dir, "model.pt"), map_location=device))
        model.eval()

        tokens = tokenize(args.text, cfg["tokenize_mode"])
        ids = vocab.encode(tokens, cfg["max_len"])
        x = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=-1)[0]
        top_probs, top_ids = probs.topk(min(args.topk, len(id2label)))

        print(f"\n输入: {args.text}")
        print(f"{'排名':<4} {'类别ID':<8} {'类别名':<16} 概率")
        print("-" * 45)
        for r, (cid, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
            print(f"  {r:<4} {cid:<8} {id2label[cid]:<16} {prob:.4f}")
