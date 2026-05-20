#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules & set up logging
import logging
import os

import numpy as np

import gensim
from gensim.models import word2vec

import jieba.analyse
import jieba

# In[2]:


# set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[3]:


sentence_file_path = './datas/in_the_name_of_people.txt'
word_file_path = './datas/cut_chars_of_in_the_name_of_people.txt'
model_file_path1 = './datas/gensim_char2vec1.w2v'
model_file_path2 = './datas/gensim_char2vec2.bin'
model_file_path3 = './datas/gensim_char2vec3_{}.npy'

# ## 一、分词

# In[4]:


with open(word_file_path,'w', encoding='utf-8') as writer:
    with open(sentence_file_path, 'r', encoding='utf-8') as reader:
        # 加载所有数据
        content = reader.read()
        
        # 分词 --> 以每个字作为独立的词
        content = list(content)
        
        # 合并结果
        result = ' '.join(content)
        
        # 结果输出
        writer.write(result)
print("Done!!!")

# ## 二、Gensim Word2Vec构建

# #### 训练方式一

# In[5]:


# 每行数据加载
print(f"文件路径:{word_file_path}")
sentences = word2vec.LineSentence(word_file_path) 

# 训练Word2Vec模型
"""
classgensim.models.word2vec.Word2Vec(
    sentences=None, corpus_file=None, 
    vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
    sample=0.001, seed=1, workers=3, min_alpha=0.0001, 
    sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, 
    hashfxn=<built-in function hash>, epochs=5, 
    null_word=0, trim_rule=None, sorted_vocab=1, 
    batch_words=10000, compute_loss=False, 
    callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True)
sg: 1(Skip-gram) 0(CBOW)
hs: 1(hierarchical softmax) 0(negative)
negative: 当hs为0的时候，给定负样本数目，给定为0表示不采用负采样
"""
model = word2vec.Word2Vec(sentences, hs = 1,min_count = 1,window = 3,vector_size = 100)

# #### 训练方式二

# In[6]:


# 每行数据加载
sentences = word2vec.LineSentence(word_file_path) 

# 训练Word2Vec模型
model = word2vec.Word2Vec(hs = 1,min_count = 1,window = 9,vector_size = 100)

# 构建词典
model.build_vocab(sentences)

# 模型训练
model.train(sentences, total_examples=model.corpus_count, epochs=5)

# ## 三、Word2Vec应用

# ### 0. 获取Word2Vec模型相关属性

# In[7]:


print("【词汇数目】: {}".format(len(model.wv.key_to_index)))
print("【转换的稠密的特征向量维度数目,每个单词转换的向量维度大小】: {}".format(model.wv.vector_size))
print("【单词到id的映射关系】: \n{}".format(model.wv.key_to_index))

# ### 1. 获取相似度最高的K个演员

# In[7]:


# 夹角余弦相似度
req_count = 20
for key in model.wv.similar_by_word('沙', topn =100):
    req_count -= 1
    print(key[0], key[1])
    if req_count == 0:
        break;

# ### 2. 获取单词之间的相似度

# In[8]:


# 夹角余弦相似度
print(model.wv.similarity('沙', '瑞'))

# In[9]:


# 夹角余弦相似度
print(model.wv.similarity('沙', '金'))

# ### 3. 获取单词的词向量

# In[10]:


v1 = model.wv.get_vector("提")
print(v1.shape)
print(v1)

# In[11]:


model.wv['提']

# In[12]:


# 异常：不存在"明"这个单词
# model.wv.get_vector("明")

# In[13]:


# 首先判断是否存在单词，如果存在，就返回，否则单词直接过滤
word = "明"
# word = "康"
if word in model.wv:
    print("【向量】:\n{}".format(model.wv[word]))
else:
    print("【单词不存在】!!!")

# ## 四、模型持久化&模型恢复加载

# ### 方式一：
# 直接使用save API进行模型持久化

# #### 持久化

# In[14]:


model.save(model_file_path1)

# #### 加载

# In[15]:


# 直接基于路径加载
model2 = word2vec.Word2Vec.load(model_file_path1)
print(model2)

v1 = model2.wv.get_vector("提")
print(v1.shape)
print(v1)

# ### 方式二：
# 保存为二进制词向量

# #### 持久化

# In[16]:


model.wv.save_word2vec_format(model_file_path2,binary=True)

# #### 加载

# In[17]:


# 加载模型
model2 = gensim.models.KeyedVectors.load_word2vec_format(model_file_path2,binary=True)
print(model2)

# 应用模型
v1 = model2.get_vector("提")
print(v1.shape)
print(v1)

# In[18]:


# 加载模型
model2 = gensim.models.KeyedVectors.load_word2vec_format('./datas/vectors.bin',
                                                         binary=True)
print(model2)

# 应用模型
v1 = model2.get_vector("酒")
print(v1.shape)
print(v1)

# ### 方式三：
# 直接使用NumPy API保存词向量信息

# #### 持久化

# In[19]:


# 获取词向量
norm_word_embeddings = model.wv.get_normed_vectors()
word_embeddings = model.wv.vectors
# 获取词典(词典到idx的映射)
vocab_2_index = list(map(lambda k: (k, model.wv.key_to_index[k]), model.wv.key_to_index))
print(np.shape(norm_word_embeddings), np.shape(word_embeddings), np.shape(vocab_2_index))
# 数据保存
np.save(model_file_path3.format("norm_embedding"), norm_word_embeddings)
np.save(model_file_path3.format("embedding"), word_embeddings)
np.save(model_file_path3.format("vocab_2_index"), vocab_2_index)

# #### 加载

# In[20]:


# 加载数据
norm_word_embeddings = np.load(model_file_path3.format("norm_embedding"))
word_embeddings = np.load(model_file_path3.format("embedding"))
vocab_2_index = np.load(model_file_path3.format("vocab_2_index"))

# 字典转换
vocab_2_index = dict(map(lambda t:(t[0], int(t[1])), vocab_2_index))

# 获取数据
word = "提"
index = vocab_2_index[word]
v1 = word_embeddings[index]
print(v1.shape)
print(v1)

#  

#  
