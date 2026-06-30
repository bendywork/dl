# -*- coding: utf-8 -*-
"""
作业 20251207 - 第一题
不依赖任何第三方库，纯 Python 原生实现：
    1. 序号化（Token Indexing）
    2. 文本/Token 哑编码（One-Hot）
    3. 词袋法（Bag of Words）
    4. TF-IDF
"""

import math
from collections import Counter


# ─────────────────────────────────────────────
# 工具：简单空格分词（不依赖 jieba）
# ─────────────────────────────────────────────
def tokenize(text: str) -> list:
    return text.strip().split()


# ─────────────────────────────────────────────
# 1. 序号化
# ─────────────────────────────────────────────
class TokenIndexer:
    """
    构建词表，将词序列转换为整数 id 序列。
    PAD=0, UNK=1，其余词按首次出现顺序编号。
    """
    PAD, UNK = '<PAD>', '<UNK>'

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build_vocab(self, corpus: list):
        """corpus: list of token lists"""
        for tokens in corpus:
            for token in tokens:
                if token not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token

    def encode(self, tokens: list) -> list:
        return [self.word2idx.get(t, 1) for t in tokens]

    def decode(self, ids: list) -> list:
        return [self.idx2word.get(i, self.UNK) for i in ids]

    def pad_or_truncate(self, ids: list, max_len: int) -> list:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))

    @property
    def vocab_size(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 2. 哑编码 One-Hot
# ─────────────────────────────────────────────
class OneHotEncoder:
    """
    每个 token → 长度为 vocab_size 的 0/1 向量。
    句子用所有 token one-hot 组成的矩阵表示，shape=(seq_len, vocab_size)。
    """
    def __init__(self, word2idx: dict):
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)

    def encode_token(self, token: str) -> list:
        vec = [0] * self.vocab_size
        idx = self.word2idx.get(token)
        if idx is not None:
            vec[idx] = 1
        return vec

    def encode_sentence(self, tokens: list) -> list:
        """返回 shape=(seq_len, vocab_size) 的矩阵（list of list）"""
        return [self.encode_token(t) for t in tokens]


# ─────────────────────────────────────────────
# 3. 词袋法 Bag of Words
# ─────────────────────────────────────────────
class BagOfWords:
    """
    句子 → vocab_size 维向量，每维是对应词在句子中出现的次数。
    与 One-Hot 的区别：BOW 对 seq 维度做了求和聚合，丢失顺序信息。
    """
    def __init__(self, word2idx: dict):
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)

    def encode(self, tokens: list) -> list:
        vec = [0] * self.vocab_size
        for t in tokens:
            idx = self.word2idx.get(t)
            if idx is not None:
                vec[idx] += 1
        return vec


# ─────────────────────────────────────────────
# 4. TF-IDF
# ─────────────────────────────────────────────
class TFIDF:
    """
    TF(t, d)  = 词 t 在文档 d 中出现次数 / 文档 d 总词数
    IDF(t)    = log( (N + 1) / (df(t) + 1) ) + 1   # 平滑版本
    TF-IDF(t, d) = TF(t, d) * IDF(t)
    """
    def __init__(self):
        self.idf: dict = {}
        self.word2idx: dict = {}
        self.vocab_size: int = 0

    def fit(self, corpus_tokens: list):
        """
        corpus_tokens: list of list of str
        统计 IDF 并建词表。
        """
        N = len(corpus_tokens)
        # 统计每个词出现在多少篇文档中（df）
        df = Counter()
        vocab = set()
        for tokens in corpus_tokens:
            vocab.update(tokens)
            for t in set(tokens):   # 每篇文档只算一次
                df[t] += 1

        # 建词表
        self.word2idx = {w: i for i, w in enumerate(sorted(vocab))}
        self.vocab_size = len(self.word2idx)

        # 计算 IDF（平滑）
        for word, idx in self.word2idx.items():
            self.idf[word] = math.log((N + 1) / (df.get(word, 0) + 1)) + 1

    def transform(self, tokens: list) -> list:
        """单个文档 → TF-IDF 向量"""
        vec = [0.0] * self.vocab_size
        total = len(tokens)
        if total == 0:
            return vec
        tf_map = Counter(tokens)
        for word, cnt in tf_map.items():
            idx = self.word2idx.get(word)
            if idx is None:
                continue
            tf = cnt / total
            idf = self.idf.get(word, 0.0)
            vec[idx] = tf * idf
        return vec

    def fit_transform(self, corpus_tokens: list) -> list:
        """一次性拟合并变换所有文档"""
        self.fit(corpus_tokens)
        return [self.transform(tokens) for tokens in corpus_tokens]


# ─────────────────────────────────────────────
# 验证
# ─────────────────────────────────────────────
def _show_vec(name: str, vec: list, top: int = 8):
    nz = [(i, v) for i, v in enumerate(vec) if v != 0]
    print(f"  {name}: 非零位 {nz[:top]} {'...' if len(nz) > top else ''}")


def main():
    corpus_raw = [
        "我 喜欢 深度 学习",
        "自然 语言 处理 很 有趣",
        "深度 学习 改变 了 世界",
        "我 喜欢 自然 语言 处理",
    ]
    corpus_tokens = [tokenize(s) for s in corpus_raw]

    print("=" * 60)
    print("【1】序号化")
    indexer = TokenIndexer()
    indexer.build_vocab(corpus_tokens)
    print(f"  词表大小: {indexer.vocab_size}")
    for tokens in corpus_tokens:
        ids = indexer.encode(tokens)
        padded = indexer.pad_or_truncate(ids, max_len=6)
        print(f"  {tokens} → ids={ids} → padded={padded}")

    print()
    print("=" * 60)
    print("【2】哑编码 One-Hot")
    ohe = OneHotEncoder(indexer.word2idx)
    for tokens in corpus_tokens[:2]:
        mat = ohe.encode_sentence(tokens)
        print(f"  句子 {tokens}  shape=({len(mat)}, {len(mat[0])})")
        for t, vec in zip(tokens, mat):
            nz = vec.index(1) if 1 in vec else -1
            print(f"    '{t}' → one-hot[{nz}]=1  (其余为0)")

    print()
    print("=" * 60)
    print("【3】词袋法 Bag of Words")
    bow = BagOfWords(indexer.word2idx)
    for tokens in corpus_tokens:
        vec = bow.encode(tokens)
        _show_vec(f"{tokens}", vec)

    print()
    print("=" * 60)
    print("【4】TF-IDF")
    tfidf = TFIDF()
    vecs = tfidf.fit_transform(corpus_tokens)
    idx2word = {v: k for k, v in tfidf.word2idx.items()}
    for i, (tokens, vec) in enumerate(zip(corpus_tokens, vecs)):
        nz = sorted([(idx2word[j], round(v, 4)) for j, v in enumerate(vec) if v > 0],
                    key=lambda x: -x[1])
        print(f"  doc{i+1} {tokens}")
        print(f"    TF-IDF: {nz}")


if __name__ == '__main__':
    main()
