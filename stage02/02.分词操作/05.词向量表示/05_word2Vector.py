from gensim.models import Word2Vec

sentences = [
    ["我", "喜欢", "自然语言处理"],
    ["自然语言处理", "是", "人工智能", "的", "重要", "分支"],
    ["词向量", "可以", "表示", "词语", "的", "语义"],
    ["我", "喜欢", "深度学习"],
    ["深度学习", "是", "机器学习", "的", "子集"],
    ["词向量", "是", "自然语言处理", "的", "基础"],
]

model = Word2Vec(
    sentences,
    vector_size=50,  # 词向量维度
    window=3,        # 上下文窗口大小
    min_count=1,     # 最低词频
    epochs=100,
    sg=1,            # 1=Skip-Gram, 0=CBOW
)

# 查看词向量
word = "自然语言处理"
print(f"'{word}' 的词向量（前10维）：")
print(model.wv[word][:10])

# 找相似词
print(f"\n与 '{word}' 最相似的词：")
for w, score in model.wv.most_similar(word, topn=3):
    print(f"  {w}: {score:.4f}")

# 词语相似度
print(f"\n'深度学习' 和 '机器学习' 的相似度：")
print(f"  {model.wv.similarity('深度学习', '机器学习'):.4f}")

# 保存 & 加载
model.save("word2vec.model")
loaded = Word2Vec.load("word2vec.model")
print(f"\n模型加载成功，词表大小：{len(loaded.wv)}")
