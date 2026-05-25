#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.test.utils import common_texts
print(common_texts)

# # 一、Doc2Vec

# In[2]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# In[3]:


# 拼接数据
documents = [TaggedDocument(doc, [i]) for i,doc in enumerate(common_texts)]
# 模型训练 --> 针对每个单词训练好对应的单词向量以及结构中的决策分类相关的特征向量
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=3)

# In[4]:


# 预测文本对应向量的时候，实际上是基于训练好的单词向量(冻结固定)，然后反向传播更新待预测文本/文档对应的特征向量
vector = model.infer_vector(["system", "response"])
print("【Doc2Vec结果】:\n{}".format(vector))

# # 二、FastText

# In[5]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[6]:


from gensim import utils
from gensim.models.fasttext import FastText

word_file_path = '../../../base/datas/resources/cut_words_of_in_the_name_of_people.txt'
class MyData(object):
    def __iter__(self):
        path = word_file_path
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                yield list(utils.tokenize(line))

# 模型构建
model = FastText(vector_size=4, window=3, min_count=1, sentences=MyData(), epochs=10)

# In[7]:


# 夹角余弦相似度
req_count = 5
for key in model.wv.similar_by_word('沙瑞金', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break;

# In[ ]:



