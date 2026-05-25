# -*- coding: utf-8 -*-
"""
TF-IDF 简单案例
TF-IDF = TF (词频) × IDF (逆文档频率)
"""

import numpy as np
from collections import Counter

# ── 1. 原始语料 ────────────────────────────────────────────────
corpus = [
    "我 喜欢 深度 学习",
    "自然 语言 处理 很 有趣",
    "深度 学习 改变 了 世界",
    "我 喜欢 自然 语言 处理",
]

# ── 2. 分词 ────────────────────────────────────────────────────
tokenized = [sentence.split() for sentence in corpus]
print("分词结果：")
for i, tokens in enumerate(tokenized):
    print(f"  文档{i+1}: {tokens}")

# ── 3. 构建词汇表 ──────────────────────────────────────────────
all_words = [word for sentence in tokenized for word in sentence]
vocab = sorted(set(all_words))
word2idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)
print(f"\n词汇表 ({vocab_size} 个词): {vocab}")

# ── 4. 计算 TF (Term Frequency - 词频) ─────────────────────────
def compute_tf(tokens):
    """
    TF = 词在当前文档中出现的次数 / 当前文档的总词数
    """
    tf_dict = Counter(tokens)
    total_words = len(tokens)
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / total_words
    return tf_dict

# ── 5. 计算 IDF (Inverse Document Frequency - 逆文档频率) ──────
def compute_idf(tokenized, word2idx):
    """
    IDF = log(总文档数 / 包含该词的文档数)
    """
    num_docs = len(tokenized)
    # 统计每个词出现在多少个文档中
    doc_freq = np.zeros(vocab_size)
    for tokens in tokenized:
        unique_words = set(tokens)  # 每个文档中只计一次
        for word in unique_words:
            if word in word2idx:
                idx = word2idx[word]
                doc_freq[idx] += 1

    # 计算 IDF（加1平滑，避免除0）
    idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1
    return idf

idf = compute_idf(tokenized, word2idx)
print(f"\nIDF 向量: {idf.round(3)}")

# ── 6. 计算每个文档的 TF-IDF ───────────────────────────────────
def compute_tfidf(tokens, word2idx, idf):
    """
    TF-IDF = TF × IDF
    """
    tf_dict = compute_tf(tokens)

    # 创建 TF-IDF 向量
    tfidf_vector = np.zeros(len(idf))
    for word, tf in tf_dict.items():
        if word in word2idx:
            idx = word2idx[word]
            tfidf_vector[idx] = tf * idf[idx]

    return tfidf_vector

# 对所有文档计算 TF-IDF
tfidf_matrix = np.array([compute_tfidf(tokens, word2idx, idf) for tokens in tokenized])

print(f"\nTF-IDF 矩阵形状: {tfidf_matrix.shape}")
print("\n各文档的 TF-IDF 向量:")
for i, (tokens, tfidf) in enumerate(zip(tokenized, tfidf_matrix)):
    print(f"\n文档{i+1}: {' '.join(tokens)}")
    # 打印非零值
    nonzero_idx = np.nonzero(tfidf)[0]
    for idx in nonzero_idx:
        word = vocab[idx]
        print(f"  {word}: {tfidf[idx]:.4f}")

# ── 7. 使用 sklearn 验证 ──────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
# sklearn 需要空格分隔的字符串
corpus_str = [' '.join(tokens) for tokens in tokenized]
sklearn_tfidf = vectorizer.fit_transform(corpus_str)

print(f"\n\nSklearn TF-IDF 矩阵:\n{sklearn_tfidf.toarray().round(4)}")
print(f"词汇表: {vectorizer.get_feature_names_out()}")
