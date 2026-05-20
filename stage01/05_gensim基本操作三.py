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

# In[4]:


# set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[5]:


sentence_file_path = './datas/in_the_name_of_people.txt'
word_file_path = './datas/cut_words_of_in_the_name_of_people.txt'
model_file_path1 = './datas/gensim_word2vec1.w2v'
model_file_path2 = './datas/gensim_word2vec2.bin'
model_file_path3 = './datas/gensim_word2vec3_{}.npy'

# ## 一、分词

# In[10]:


# 人民的名义 小说分词
jieba.suggest_freq('沙瑞金',True)
jieba.suggest_freq('田国富',True)
jieba.suggest_freq('高育良',True)
jieba.suggest_freq('侯亮平',True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)
jieba.suggest_freq('京州市', True)
jieba.suggest_freq('副市长', True)
jieba.suggest_freq('赵德汉',True)
jieba.suggest_freq('H大学',True)
jieba.suggest_freq('H省',True)
jieba.suggest_freq('政法系', True)
jieba.suggest_freq(('哥', '求'), True)

word_cnt = {}
total_word_cnt = 0

with open(word_file_path,'w', encoding='utf-8') as writer:
    with open(sentence_file_path, 'r', encoding='utf-8') as reader:
        # 加载所有数据
        content = reader.read()
        
        # 分词
        content = jieba.lcut(content)

        # 统计各个单词出现的次数
        for word in content:
            word_cnt[word] = word_cnt.get(word, 0) + 1
            total_word_cnt += 1
        
        # 合并结果
        result = ' '.join(content)
        
        # 结果输出
        writer.write(result)
print("Done!!!")

# In[24]:


## 计算单词的词频(获取最多的十个)

word_freq = []
for word in word_cnt.keys():
    word_freq.append([word, word_cnt[word] / total_word_cnt])
sorted_word_freq = sorted(word_freq, key=lambda t: t[1], reverse=True)
sorted_word_freq[:100]

# In[20]:




# ## 二、Gensim Word2Vec构建

# #### 训练方式一

# In[27]:


print(f"针对低频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.0000001), 0)}")
print(f"针对低频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.0001), 0)}")
print(f"针对高频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.00011), 0)}")
print(f"针对高频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.0002), 0)}")
print(f"针对高频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.0011), 0)}")
print(f"针对高频词 -> 删除概率:{max(1 - np.sqrt(0.0001 / 0.02), 0)}")

# ![image.png](attachment:a7a77a12-9ee5-49ab-b8fe-774aa64d4b9c.png)

# In[28]:


# 每行数据加载
print(f"文件路径:{word_file_path}")
sentences = word2vec.LineSentence(word_file_path) 

# 训练Word2Vec模型
"""
class gensim.models.word2vec.Word2Vec(
    sentences=None, corpus_file=None, 
    vector_size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
    sample=0.001, seed=1, workers=3, min_alpha=0.0001, 
    sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, 
    hashfxn=<built-in function hash>, epochs=5, 
    null_word=0, trim_rule=None, sorted_vocab=1, 
    batch_words=10000, compute_loss=False, 
    callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True)

sentences=None,  给定训练数据对应的文本对象
corpus_file=None, 词汇表对应的文件，如果不给定的话，直接采用训练数据构建词典/词汇表
vector_size=100, 期望单词映射的维度大小
alpha=0.025, 学习率
window=5, 窗口大小，窗口大小 = 周边词 + 1
min_count=5, 针对词频小于5的单词直接删除    
sample: 0.001 二次重采样时候的系数t --> 在训练前对所有的token进行遍历处理 --> 随机丢弃 --> 单词词频高的被丢弃的可能性就高，词频低的丢弃的可能性就低 ---> 缓解低频词学习/更新少的情况
sg=0, 0表示CBOW、1表示Skip-Gram
hs=0, 0表示负采样 1表示哈夫曼树(层次Softmax)
negative=5, 仅在hs=0的时候生效，负采样的类别数量；当设定为0的时候，表示不进行负采样
ns_exponent=0.75, 负采样过程中，计算词频过程中的系数
cbow_mean=1, 0表示cbow中求和，1表示cbow中求均值

"""
model = word2vec.Word2Vec(sentences,sg=1, hs = 1,min_count = 1,window = 11,vector_size = 100)

# #### 训练方式二

# In[31]:


# 每行数据加载
sentences = word2vec.LineSentence(word_file_path) 

# 创建Word2Vec模型
model = word2vec.Word2Vec(hs = 1,min_count = 5,window = 5,vector_size = 100)

# 构建词典
model.build_vocab(sentences)

# 模型训练
model.train(sentences, total_examples=model.corpus_count, epochs=5)

# ## 三、Word2Vec应用
# - PS: Word2Vec的模型训练完成后，所有的使用操作都是基于单词向量的使用

# ### 0. 获取Word2Vec模型相关属性

# In[32]:


print("【词汇数目】: {}".format(len(model.wv.key_to_index)))
print("【转换的稠密的特征向量维度数目,每个单词转换的向量维度大小】: {}".format(model.wv.vector_size))
print("【单词到id的映射关系】: \n{}".format(model.wv.key_to_index))

# ### 1. 获取相似度最高的K个演员

# In[33]:


# 夹角余弦相似度
req_count = 10
for (word, sim) in model.wv.similar_by_word('沙瑞金', topn =100):
    if len(word)==3:  # 为了过滤一下
        req_count -= 1
        print(word, sim)
        if req_count == 0:
            break;

# In[34]:


# 夹角余弦相似度
req_count = 10
for (word, sim) in model.wv.similar_by_word('沙瑞金', topn =100):
    if len(word)==4:  # 为了过滤一下
        req_count -= 1
        print(word, sim)
        if req_count == 0:
            break;

# In[38]:


import jieba.posseg as pseg

jieba.add_word('沙瑞金',10,'nr')
jieba.add_word('田国富',10,'nr')
jieba.add_word('高育良',10,'nr')
jieba.add_word('侯亮平',10,'nr')
jieba.add_word('钟小艾', 10,'nr')
jieba.add_word('陈岩石', 10,'nr')
jieba.add_word('欧阳菁', 10,'nr')
jieba.add_word('易学习', 10,'nr')
jieba.add_word('王大路', 10,'nr')
jieba.add_word('蔡成功', 10,'nr')
jieba.add_word('孙连城', 10,'nr')
jieba.add_word('季昌明', 10,'nr')
jieba.add_word('丁义珍', 10,'nr')
jieba.add_word('郑西坡', 10,'nr')
jieba.add_word('赵东来', 10,'nr')
jieba.add_word('高小琴', 10,'nr')
jieba.add_word('赵瑞龙', 10,'nr')
jieba.add_word('林华华', 10,'nr')
jieba.add_word('陆亦可', 10,'nr')
jieba.add_word('刘新建', 10,'nr')
jieba.add_word('刘庆祝', 10,'nr')
jieba.add_word('京州市', 10,'nr')
jieba.add_word('副市长', 10,'nr')
jieba.add_word('赵德汉',10,'nr')
jieba.add_word("谢谢您", None, "o")


tmp01 = model.wv.similar_by_word('沙瑞金', topn=100)
tmp01 = dict(tmp01)
for k in tmp01.keys():
    jieba.suggest_freq(k, True)
tmp02 = ' '.join(list(tmp01.keys()))
words = pseg.lcut(tmp02)
for word,flag in words:
    if flag == 'nr':
        print(word, tmp01[word])


# In[36]:


tmp02

# ### 2. 获取单词之间的相似度

# In[39]:


# 夹角余弦相似度
print(model.wv.similarity('沙瑞金', '高育良'))

# In[42]:


# 夹角余弦相似度
v1 = model.wv.get_vector("沙瑞金")
v2 = model.wv.get_vector("高育良")
np.sum(v1 * v2) / (np.sqrt(np.sum(np.power(v1, 2))) * np.sqrt(np.sum(np.power(v2, 2))))

# ### 3. 获取单词的词向量

# In[43]:


v1 = model.wv.get_vector("提拔")
print(v1.shape)
print(v1)

# In[44]:


model.wv['提拔']

# In[45]:


# 异常：不存在"小明"这个单词
model.wv.get_vector("小明")

# In[47]:


# 首先判断是否存在单词，如果存在，就返回，否则单词直接过滤
word = "小明"
#word = "李达康"
if word in model.wv:
    print("【向量】:\n{}".format(model.wv[word]))
else:
    print("【单词不存在】!!!")

# ## 四、模型持久化&模型恢复加载

# ### 方式一：
# 直接使用save API进行模型持久化

# #### 持久化

# In[48]:


model.save(model_file_path1)

# #### 加载

# In[49]:


# 直接基于路径加载
model1 = word2vec.Word2Vec.load(model_file_path1)
print(model1)

v1 = model1.wv.get_vector("提拔")
print(v1.shape)
print(v1)

# ### 方式二：
# 保存为二进制词向量

# #### 持久化

# In[55]:


model.wv.save_word2vec_format(model_file_path2, binary=True)

# #### 加载

# In[56]:


# 加载模型
model2 = gensim.models.KeyedVectors.load_word2vec_format(model_file_path2, binary=True)
print(model2)

# 应用模型
v1 = model2.get_vector("提拔")
print(v1.shape)
print(v1)

# In[52]:


# 加载模型
model2 = gensim.models.KeyedVectors.load_word2vec_format('./datas/vectors.bin',
                                                         binary=True)
print(model2)

# 应用模型
v1 = model2.get_vector("酒店")
print(v1.shape)
print(v1)

# ### 方式三：
# 直接使用NumPy API保存词向量信息

# #### 持久化

# In[53]:


# 获取词向量
norm_word_embeddings = model.wv.get_normed_vectors()  # L2-norm归一化转换后的单词向量
word_embeddings = model.wv.vectors  # 归一化转换前的单词向量
# 获取词典(词典到idx的映射)
vocab_2_index = list(map(lambda k: (k, model.wv.key_to_index[k]), model.wv.key_to_index))
print(np.shape(norm_word_embeddings), np.shape(word_embeddings), np.shape(vocab_2_index))
# 数据保存
np.save(model_file_path3.format("norm_embedding"), norm_word_embeddings)
np.save(model_file_path3.format("embedding"), word_embeddings)
np.save(model_file_path3.format("vocab_2_index"), vocab_2_index)

# #### 加载

# In[54]:


# 加载数据
norm_word_embeddings = np.load(model_file_path3.format("norm_embedding"))
word_embeddings = np.load(model_file_path3.format("embedding"))
vocab_2_index = np.load(model_file_path3.format("vocab_2_index"))

# 字典转换
vocab_2_index = dict(map(lambda t:(t[0], int(t[1])), vocab_2_index))

# 获取数据
word = "提拔"
index = vocab_2_index[word]
v1 = word_embeddings[index]
print(v1.shape)
print(v1)

#  

#  

# # 五、扩展：直接从文件中读取数据来进行模型训练

# In[57]:


from gensim import utils
from gensim.models.word2vec import Word2Vec


class MyData(object):
    def __iter__(self):
        path = word_file_path
        with open(path, 'r', encoding='utf-8') as reader:
            for line in reader:
                yield list(utils.tokenize(line))

# 模型构建
model_ = Word2Vec(hs = 1,min_count = 1,window = 5,vector_size = 100, sentences=MyData())

# In[58]:


# 夹角余弦相似度
req_count = 5
for key in model_.wv.similar_by_word('沙瑞金', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break;

# # 六、扩展：可视化

# In[59]:


from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    #vectors = np.asarray(model.wv.vectors)
    #labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings
    vectors = np.asarray(model.wv.vectors)[:1000]
    labels = np.asarray(model.wv.index_to_key)[:1000]  # fixed-width numpy strings
    print(vectors.shape)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels



def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random
    
    plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 100)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))


# In[60]:


x_vals, y_vals, labels = reduce_dimensions(model)

# In[61]:


plot_with_matplotlib(x_vals, y_vals, labels)

# In[ ]:



