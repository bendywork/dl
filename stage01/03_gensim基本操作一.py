#!/usr/bin/env python
# coding: utf-8

# In[1]:


    # 安装方式: pip install gensim==4.3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
import numpy as np
import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# ### 字典演示

# In[2]:


import jieba
text1 = "我是来自湖南张家界的小明，我喜好大海\n我从事IT相关工作\n我讨厌夏天"
text2 = "计算机视觉和自然语言我比较喜好自然语言的内容"
text3 = "我不想上班，我想出去玩"

# 正常的文本构建(针对每个文本进行分词)
docs = []
for text in [text1, text2, text3]:
    docs.append(list(jieba.lcut(text.replace('\n', ''))))
print(docs)

# In[3]:


word_dct = {}
for doc in docs:
    for word in doc:
        word_dct[word] = word_dct.get(word, len(word_dct))
print(f"总单词的数目:{len(word_dct)}")
print(word_dct)

# In[4]:


# 构建词典
dct = Dictionary(docs)
print(f"去重后单词数目/词典大小:{len(dct)}")
print(f"当前词典为:{dct}")

# In[5]:


dct.save_as_text('./datas/a.txt')
print(dct.token2id['上班'])
print(dct.token2id['我'])
print(dct[22])
print(dct[0])

# In[5]:


n = len(dct) + 1 # 词典大小 = 实际词典大小 + 1
text4 = "我是来自北京的小明"
text4_words = list(jieba.lcut(text4.replace('\n', '')))
result = list(np.asarray(dct.doc2idx(text4_words, unknown_word_index=-1)) + 1)
print(text4_words)
print("序号化结果:")
print(result)
print("OneHot结果:")
result2 = [[0] * n for _ in range(len(result))]
for i,_id in enumerate(result):
    if _id != -1:
        result2[i][_id] = 1
print(result2)
print("词袋法结果:")
result3 = list(np.sum(np.asarray(result2), 0))
print(result3)

# In[6]:


text4 = "我是来自北京的小明，我喜好玩游戏"
text4_words = list(jieba.lcut(text4.replace('\n', '')))
result = dct.doc2idx(text4_words)
print(text4_words)
print("序号化结果:")
print(result)
print("OneHot结果:")
result2 = [[0] * len(dct) for _ in range(len(result))]
for i,_id in enumerate(result):
    if _id != -1:
        result2[i][_id] = 1
print(result2)
print("词袋法结果:")
result3 = list(np.sum(np.asarray(result2), 0))
print(result3)
result4 = dct.doc2bow(text4_words) # 词袋法的封装
print(result4)

# In[7]:


corpus = [dct.doc2bow(doc) for doc in docs] # 针对每个文档进行词袋法的转换
model = TfidfModel(corpus=corpus)
print("维度大小:{}".format(np.shape(model[corpus[0]])))
model[corpus[0]]

# # 一、加载数据(数据预处理)

# In[5]:


# 加载数据
with open('./datas/text8', 'r', encoding='utf-8') as reader:
    content = reader.read()
# 划分单词，并转换为二进制形式
words = list(map(lambda word: word.encode("utf-8"), filter(lambda t: t.strip(), content.split(" "))))
total_words = len(words)
print("总单词数目:{}".format(total_words))
print("【前10个单词】:{}".format(words[:10]))
# 将其转换为文档的形式(必须， 也就是一个文档存在多个单词)
# 模拟的方式：模拟多个文档
word_per_doc = 10000
docs = []
for i in range(total_words // word_per_doc + 1):
    # 获取索引
    start_idx = i * word_per_doc
    end_idx = start_idx + word_per_doc
    # 获取对应的单词列表
    tmp_words = words[start_idx:end_idx]
    # 保存
    if len(tmp_words) > 0:
        docs.append(tmp_words)
print("总文档数目:{}".format(len(docs)))

# # 二、构建词典

# In[7]:


# 构建词典
# docs中必须是文档，文档内必须是一个一个的单词
# eg: docs --> list(list(str)) --> [['a', 'bv', 'c'], ['a', 'c'], ['d', 'f', ' f']]
dct = Dictionary(docs)
print(f"词典大小:{len(dct)}")
print(f"{len(dct.token2id)}")

# # 三、BOW词袋法转换

# In[10]:


docs[0]

# In[8]:


# 做一个词袋法转换(以dct中找到的单词作为特征属性，以文本中出现的数量作为特征值)
corpus = [dct.doc2bow(doc) for doc in docs]
print(corpus)

# # 四、TF-IDF构建

# In[9]:


model = TfidfModel(corpus=corpus)

# # 五、TF-IDF应用

# In[13]:


print("维度大小:{}".format(np.shape(model[corpus[0]])))
model[corpus[0]]

# In[14]:


# 针对其它字符串进行词向量转换
others = [
    ['my', 'name', 'name', 'is', 'gerry'], # 新的第一条文本
    ['my', 'name', 'is', 'xiaoming'] # 新的第二条文本
]
other_corpus = [dct.doc2bow(line) for line in others] # 新文本做词袋法处理
vectors = model[other_corpus] # TF-IDF模型转换
for vector in vectors:
    print(vector)

# In[16]:


dct.token2id['gerry']

# In[ ]:



