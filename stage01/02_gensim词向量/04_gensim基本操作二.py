#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.test.utils import common_texts
from gensim.models import TfidfModel, LdaModel, LsiModel
from gensim.corpora import Dictionary

# In[2]:


common_texts

# # 一、数据加载

# In[3]:


# 构建字典
common_dictionary = Dictionary(common_texts)
# 各个文本对应的词袋法的值
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
print("原始数据:\n{}".format(common_texts))
print("\n词袋法后的值:\n{}".format(common_corpus))
print(f"文本数目:{len(common_texts)}")
print(f"去重后单词数目:{len(common_dictionary)}")

# In[4]:


other_texts = [
    ['computer', 'time', 'graph'],  # 文本1
    ['survey', 'response', 'eps'],  # 文本2
    ['human', 'system', 'computer']  # 文本3
]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
print("测试数据对应的词袋法的值:\n{}".format(other_corpus))

# # 二、TF-IDF Model

# In[6]:


# 模型构建
model = TfidfModel(corpus=common_corpus)

# In[7]:


# 预测
vectors = model[other_corpus]
for vector in vectors:
    print(vector)

# # 三、LDA Model

# In[17]:


# 模型构建&训练
model = LdaModel(common_corpus, num_topics=5)

# In[18]:


# 模型保存
model.save('./datas/lda_model.pkl')

# In[19]:


# 模型加载
lda = LdaModel.load('./datas/lda_model.pkl')

# In[20]:


# 模型结果获取(文本向量)
vectors = lda[other_corpus]
for vector in vectors:
    print(vector)

# In[21]:


lda.get_topics() # 单词对应的主题向量矩阵

# In[11]:


# 更新模型（在当前模型基础上继续更新模型参数）
lda.update(other_corpus)

# In[12]:


# 更新后模型结果获取(文本向量)
vectors = lda[other_corpus]
for vector in vectors:
    print(vector)

# # 四、Other

# 官网文档：https://radimrehurek.com/gensim/apiref.html

# In[15]:


print("各个单词对应的主题向量:")
word_embedding_tabel = model.get_topics().T
print(type(word_embedding_tabel))
print(word_embedding_tabel.shape)
print(word_embedding_tabel)

# In[14]:


common_dictionary.token2id

# In[ ]:



