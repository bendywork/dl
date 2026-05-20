# ── 1. 原始语料 ────────────────────────────────────────────────
import numpy
import numpy as np

corpus = [
    "我 喜欢 深度 学习",
    "自然 语言 处理 很 有趣",
    "深度 学习 改变 了 世界",
    "我 喜欢 自然 语言 处理",
]

# ── 2. 分词（这里以空格分词；实际项目用 jieba/HanLP）──────────────
# tokenized = [sentence.split() for sentence in corpus]
tokenized = [sentence.split() for sentence in corpus]
# print("分词结果：")
# for t in tokenized:
#     print(" ", t)

# resolve dict
all_words = [word for sentence in tokenized for word in sentence]
final_words = sorted(set(all_words))
print(final_words)
word2idx = {word: idx for idx, word in enumerate(final_words)}

standard_len = len(final_words)

# 将原语句转为索引
def encode_numpy(sentence_tokens, word2idx, vocab_size):
    one_hots = np.zeros((len(sentence_tokens), vocab_size), dtype=np.int32)
    print(sentence_tokens)
    for i, w in enumerate(sentence_tokens):
        if w in word2idx:
            idx = word2idx[w]
            one_hots[i, idx] = 1
    # result = numpy.sum(one_hots, axis=0)
    # print(result)
    return one_hots

# BagOfWord
# def encode_numpy(sentence_tokens, word2idx, vocab_size):
#     one_hots = np.zeros((len(sentence_tokens), vocab_size), dtype=np.int32)
#     print(sentence_tokens)
#     for i, w in enumerate(sentence_tokens):
#         if w in word2idx:
#             idx = word2idx[w]
#             one_hots[i, idx] = 1
#     result = numpy.sum(one_hots, axis=0)
#     # print(result)
#     return result

encoded = [encode_numpy(tokens, word2idx, standard_len) for tokens in tokenized]
print(encoded)



