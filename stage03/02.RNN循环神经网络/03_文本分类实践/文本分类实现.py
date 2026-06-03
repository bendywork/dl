# -*- coding: utf-8 -*-
"""
文本分类实现 —— RNN 实战复习
整体流程：数据解析加载 → 分词器 → Dataset/DataLoader → 模型 → 训练评估 → 持久化
"""

import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_file, obj):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# 分词方法
# ─────────────────────────────────────────────

def split_text_to_tokens_with_jieba(text: str) -> List[str]:
    return jieba.lcut(text)


def split_text_to_tokens_with_char(text: str) -> List[str]:
    return list(text)


def split_text_to_tokens(text: str) -> List[str]:
    """文本 → token列表，eg: "从这里怎么回家" → ["从", "这里", "怎么", "回家"]"""
    tokens = split_text_to_tokens_with_jieba(text)
    # tokens = split_text_to_tokens_with_char(text)
    return [t.lower() for t in tokens]


# ─────────────────────────────────────────────
# 构建 Token / Label 词典
# ─────────────────────────────────────────────

def build_vocab(train_csv: str, token2ids_path: str, label2ids_path: str):
    df = pd.read_csv(train_csv, sep="\t", header=None, names=["text", "label"])

    token2cnt, label2cnt, text_lens = {}, {}, []
    for _, row in df.iterrows():
        text, label = row["text"].strip(), row["label"].strip()
        tokens = split_text_to_tokens(text)
        for t in tokens:
            token2cnt[t] = token2cnt.get(t, 0) + 1
        label2cnt[label] = label2cnt.get(label, 0) + 1
        text_lens.append(len(tokens))

    print(f"总Token数量: {len(token2cnt)}")
    print(f"总标签数量: {len(label2cnt)}  {label2cnt}")

    # 可视化 token 频次分布（只看出现 <10 次的）
    _plot_distribution([v for v in token2cnt.values() if v < 10], "Token Count Distribution", "Token Count")
    _plot_distribution(text_lens, "Text Length Distribution", "Text Length")

    # 过滤低频 token 构建词典
    token2ids = {"<PAD>": 0, "<UNK>": 1}
    for token, cnt in token2cnt.items():
        if cnt >= 3:
            token2ids[token] = len(token2ids)

    label2ids = {label: idx for idx, label in enumerate(label2cnt)}

    save_json(token2ids_path, token2ids)
    save_json(label2ids_path, label2ids)
    return token2ids, label2ids


def _plot_distribution(data, title, xlabel):
    mu, sigma = np.mean(data), np.std(data)
    plt.figure(figsize=(8, 5))
    _, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black")
    x = np.linspace(min(bins), max(bins), 100)
    plt.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)),
             "r-", lw=2, label="curve")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 分词器
# ─────────────────────────────────────────────

@dataclass
class TokenizerOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    label: Optional[str] = None
    label_id: Optional[int] = None


class Tokenizer:
    def __init__(self, token2ids: Dict[str, int], label2ids: Dict[str, int],
                 unk_token="<UNK>", pad_token="<PAD>"):
        self.token2ids = token2ids
        self.label2ids = label2ids
        self.unk_token_id = token2ids[unk_token]
        self.pad_token_id = token2ids[pad_token]

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        tokens = split_text_to_tokens(text)
        token_ids = [self.token2ids.get(t, self.unk_token_id) for t in tokens]
        label_id = self.label2ids[label] if label is not None else None
        return TokenizerOutput(text, tokens, token_ids, label, label_id)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TextClassifyDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: Tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        out = self.tokenizer(self.texts[index], self.labels[index])
        return {
            "text": out.text,
            "tokens": out.tokens,
            "token_ids": torch.tensor(out.token_ids, dtype=torch.int64),
            "token_masks": torch.ones(len(out.token_ids), dtype=torch.float32),
            "label": out.label,
            "label_id": torch.tensor(out.label_id, dtype=torch.int64),
        }

    def __len__(self):
        return len(self.texts)


# ─────────────────────────────────────────────
# DataLoader（动态 padding）
# ─────────────────────────────────────────────

def build_collect_fn(pad_token_id: int):
    def _collect_fn(batch):
        max_len = max(len(item["token_ids"]) for item in batch)
        b_text, b_tokens, b_token_ids, b_masks, b_label, b_label_id = [], [], [], [], [], []
        for item in batch:
            b_text.append(item["text"])
            b_tokens.append(item["tokens"])
            b_label.append(item["label"])
            b_label_id.append(item["label_id"])

            ids, masks = item["token_ids"], item["token_masks"]
            pad = max_len - len(ids)
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), pad_token_id, dtype=ids.dtype)])
                masks = torch.cat([masks, torch.zeros(pad, dtype=masks.dtype)])
            b_token_ids.append(ids)
            b_masks.append(masks)

        return {
            "text": b_text,
            "tokens": b_tokens,
            "label": b_label,
            "label_id": torch.stack(b_label_id),
            "token_ids": torch.stack(b_token_ids),
            "token_masks": torch.stack(b_masks),
        }
    return _collect_fn


def build_dataloader(ds: TextClassifyDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collect_fn(ds.tokenizer.pad_token_id),
    )


# ─────────────────────────────────────────────
# 快速验证入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TRAIN_CSV = "../../../base/datas/text_classify/train.csv"
    TOKEN2IDS = "./output/text_classify/token2ids.json"
    LABEL2IDS = "./output/text_classify/label2ids.json"

    # 第一次运行：构建并保存词典（已生成，无需重复执行）
    # token2ids, label2ids = build_vocab(TRAIN_CSV, TOKEN2IDS, LABEL2IDS)

    # 后续运行：直接加载
    token2ids = load_json(TOKEN2IDS)
    label2ids = load_json(LABEL2IDS)

    tokenizer = Tokenizer(token2ids, label2ids)
    print(tokenizer("怎么从这里回家", "Travel-Query"))

    df = pd.read_csv(TRAIN_CSV, sep="\t", header=None, names=["text", "label"])
    ds = TextClassifyDataset(df.text.values, df.label.values, tokenizer)
    dl = build_dataloader(ds, batch_size=4, shuffle=True)

    for batch in dl:
        print("token_ids shape:", batch["token_ids"].shape)
        print("token_masks shape:", batch["token_masks"].shape)
        print("label_id:", batch["label_id"])
        break
