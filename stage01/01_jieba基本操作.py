#!/usr/bin/env python
# coding: utf-8

# In[1]:


#引入jieba模块, 默认安装pip install jieba
import jieba

# 基本使用
word_list = jieba.cut("欢迎来到NLP自然语言的世界!")
print(type(word_list))
print("【基本应用】: {}".format(" / ".join(word_list)))

# ## 一、分词

# In[2]:


# 导入包
import jieba

word_list = jieba.cut('我来到湖南国防科技大学', cut_all=True)
print("【全模式】: {}".format(" /".join(word_list)))

word_list = jieba.cut('我来到湖南国防科技大学')
print("【精确模式】: {}".format(" /".join(word_list)))

word_list = jieba.cut_for_search('我来到湖南国防科技大学')
print("【搜索引擎模式】: {}".format(" /".join(word_list)))


word_list = jieba.cut("我在台电大厦上班", HMM=False)
print("【仅词典模式】: {}".format(" /".join(word_list)))

word_list = jieba.cut("我在台电大厦上班", HMM=True)
print("【HMM新词发现模式】: {}".format(" /".join(word_list)))

# ### API区别说明
# - jieba.cut：返回的是一个迭代器对象
# - jieba.lcut：返回的是一个list集合

# In[3]:


cut_word_list = jieba.cut('我来到湖南国防科技大学')
print("【cut API返回的数据类型】: {}".format(type(cut_word_list)))
print("【cut API返回结果】: {}".format(' /'.join(cut_word_list)))
print("【cut API返回结果<再次获取>】: {}".format(' /'.join(cut_word_list)))

print("")

lcut_word_list = jieba.lcut('我来到湖南国防科技大学')
print("【lcut API返回的数据类型】: {}".format(type(lcut_word_list)))
print("【lcut API返回结果】: {}".format(' /'.join(lcut_word_list)))
print("【lcut API返回结果<再次获取>】: {}".format(' /'.join(lcut_word_list)))


# ## 二、自定义词典

# ### 载入词典
# ```
# jieba.load_userdict(filename): 加载给定文件filename中定义的单词
# ```
# - 每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒；
# - 词性详见：<a href="#ext1">jieba词性说明</a>

# In[4]:


# 仅词典匹配
word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=False)
print("【载入词典前<无HMM>】: {}".format('/'.join(word_list)))

# 对于连续单独成词的文本，使用HMM继续分词
word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=True)
print("【载入词典前<有HMM>】: {}".format('/'.join(word_list)))

# In[5]:


# 加载词典
jieba.load_userdict('./datas/word_dict.txt')

# In[6]:


with open('./datas/word_dict.txt', 'r', encoding='utf-8') as reader:
    for line in reader:
        print(line.strip())

# In[7]:


word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=False)
print("【载入词典后<无HMM>】: {}".format('/'.join(word_list)))

word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=True)
print("【载入词典后<有HMM>】: {}".format('/'.join(word_list)))

# ### 动态调整词典
# - 使用```add_word(word, freq=None, tag=None)```和```del_word(word)```可以在程序中动态修改词典
# - 使用```suggest_freq(segment, tune=True)```可以调节单个词语的词频，使其能或者不能被分出来

# In[8]:


word_list = jieba.cut('如果放到post中将出错。', HMM=False)
print("【不启动HMM+不添加分词】: {}".format('/'.join(word_list)))

# In[9]:


jieba.suggest_freq('中', tune=False)

# In[10]:


jieba.suggest_freq(('中', '将'), tune=True)

# In[11]:


jieba.suggest_freq('中', tune=False)

# In[12]:


word_list = jieba.cut('如果放到post中将出错。', HMM=False)
print("【不启动HMM+添加分词】: {}".format('/'.join(word_list)))

# ## 三、关键词抽取

# In[13]:


import jieba.analyse

# In[14]:


# https://mbd.baidu.com/newspage/data/landingsuper?context=%7B%22nid%22%3A%22news_10141364001547692745%22%7D&n_type=0&p_from=1
sentence = """
新华社澳门8月1日电（方钊、郭鑫）1日上午，解放军驻澳门部队在新口岸军营隆重举行升国旗仪式和“八一”招待会，庆祝中国人民解放军建军92周年。
8时，驻澳部队威武的仪仗队护送国旗，步伐铿锵走向升旗台。军乐队奏响雄壮的《中华人民共和国国歌》，驻澳门部队官兵在司令员徐良才和政委孙文举带领下整齐列队，面向国旗庄严敬礼，目送鲜艳的五星红旗冉冉升起，献上深情祝福。
“八一”招待会于11时举行，主礼嘉宾与各界来宾一同观看了纪录片《濠江战旗别样红》，全面了解驻军进澳门20年来履行防务情况。驻澳门部队司令员徐良才和澳门特区行政长官崔世安致辞。
徐良才深情回顾了中国人民解放军的光辉历程。他表示，今年是中国人民解放军进驻澳门20周年，驻军自进驻之日起，就坚定不移地贯彻“一国两制”伟大方针，坚定不移地遵守澳门基本法和驻军法，坚定不移地维护澳门繁荣稳定，始终视国家和民族利益高于一切，始终遵守澳门现行社会制度，尊重和支持特区政府依法施政，积极参加社会公益事业，把澳门同胞当亲人。
徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托，深入贯彻习近平强军思想，坚持政治建军、服务大局，坚持任务牵引、练兵备战，坚持依法从严、锤炼作风，部队履行防务能力稳步提升。驻军部队的建设发展，离不开特区各界和澳门同胞的关心，离不开中联办、外交公署等中央驻澳机构的支持，特别是特区政府为驻军有效履行防务创造了良好环境和条件，对此致以衷心的感谢和崇高的敬意。
崔世安向驻澳门部队官兵致以节日的祝贺，对驻军一直以来对特区发展的有力支持表示感谢。他表示，20年来，驻澳部队与澳门特区同呼吸、共命运，视驻地为故乡，把居民当亲人，支持特区政府依法施政，积极开展多元化的、丰富多彩的爱民活动，主动参与献血、植树等社会公益活动；与特区政府合办“澳门青年学生军事夏令营”，培养青年“爱国爱澳”的核心价值；在防灾救灾工作上，以高度的责任感，大力支持特区政府。事实证明，驻澳部队是维护“一国两制”的重要力量，是维护澳门繁荣稳定的重要基石，为澳门特区各项事业的进步作出了不懈的努力和巨大的贡献。
全国政协副主席何厚铧、中央政府驻澳门联络办公室主任傅自应、外交部驻澳门特派员公署特派员沈蓓莉、驻澳部队政委孙文举、澳门特区立法会主席高开贤等，以及澳门特区政府、中央驻澳机构、澳区全国人大代表、政协委员、社团、高校、往届军事夏令营学生代表等300余人出席了招待会。
"""

# ### 基于TF-IDF算法抽取关键词
# - ```jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())```
#     - 功能：关键词提取
#     - 参数说明：
#         - sentence：待提取的文本
#         - topK：返回多少个TF/IDF权重最大的关键词，默认为20个
#         - withWeight：是否返回关键词的权重值，默认为False，表示不返回
#         - allowPOS: 仅提取制定词性的词，默认为空，表示不筛选
# - ```jieba.analyse.set_idf_path(file_name)```
#     - 功能：自定义单词逆文件频率的值
#     - 参数说明：
#         - file_name: 本地磁盘文件路径，文件内容为各个单词的逆向文件频率，每行一个单词，两部分构成，第一部分为单词，第二部分为逆向文件频率，中间用空格隔开
#     - 参考：[idf.txt.big](https://github.com/fxsjy/jieba/blob/master/extra_dict/idf.txt.big)
# - ```jieba.analyse.set_stop_words(file_name)```
#     - 功能：自定义停止词
#     - 参数说明：
#         - file_name: 本地磁盘文件路径, 每行一个停止词
#     - 参考：[stop_words.txt](https://github.com/fxsjy/jieba/blob/master/extra_dict/stop_words.txt)

# In[15]:


jieba.analyse.extract_tags(sentence,topK=10)

# In[15]:


# 简单去看，内部就是计算一个TF-IDF = TF * IDF的值，然后排序
jieba.analyse.extract_tags(sentence,topK=10,withWeight=True)

# In[17]:


jieba.analyse.extract_tags(sentence,topK=10,withWeight=True, 
                           allowPOS=('n', 'ns','vn', 'a'))

# In[18]:


# 设置自定义IDF文件
jieba.analyse.set_idf_path('./datas/idf.txt.big')
# 设置自定义停止词
jieba.analyse.set_stop_words('./datas/stop_words.txt')
# 再进行关键词提取
jieba.analyse.extract_tags(sentence,topK=10,withWeight=True, 
                           allowPOS=('n', 'ns','vn', 'a'))

# ### 基于TextRank算法的关键词抽取
# - ```jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False) ```
#     - 功能：关键词提取
#     - 参数说明：
#         - sentence：待提取的文本
#         - topK：返回多少个TF/IDF权重最大的关键词，默认为20个
#         - withWeight：是否返回关键词的权重值，默认为False，表示不返回
#         - allowPOS: 仅提取制定词性的词，默认不为空，表示进行筛选
#         - withFlag：是否返回单词的词性值，默认为False，表示不返回(仅返回单词)
# - NOTE: 参考[TextRank: Bringing Order into Texts](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)论文

# In[19]:


jieba.analyse.textrank(sentence=sentence, topK=10)

# In[20]:


jieba.analyse.textrank(sentence=sentence, topK=10, withWeight=True, 
                       allowPOS=('n', 'ns','vn', 'v', 'a'), withFlag=True)

# In[21]:


jieba.analyse.textrank(sentence=sentence, topK=5, withWeight=True, 
                       allowPOS=('n','nr'), withFlag=True)

#  

# ## 四、词性标注
# - jieba中的分词词性说明详见: <a href="#ext1">jieba词性列表</a>

# In[16]:


import jieba.posseg as pseg
sentence = "我觉得人工智能未来的发展非常不错"
sentence = "姚明的身高"
sentence = "姚明的职业是什么"
# sentence = "成龙的身高是多少"
# 分词+词性标注
words = pseg.cut(sentence)

# 输出
print("%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))

# In[23]:


import jieba.posseg as pseg
sentence = "徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托。"
# 分词+词性标注
words = pseg.cut(sentence)

# 输出
print("%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))

# In[24]:


import jieba.posseg as pseg
sentence = "徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托。"

# 添加词典
jieba.add_word('习主席', 2, 'nr')

# 分词+词性标注
words = pseg.cut(sentence)

# 输出
print("%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))

# ## 五、并行分词
# - 原理：将目标文本按行分割后，把各行文本分配到多个Python进程中进行并行分词，然后归并结果。速度比单线程的快3~5倍。
# - 基于Python自带的multiprocessing模块，暂时不支持windows
# - 基本用法：
#     - jieba.enable_parallel(4)
#         - 开启并行分词模式，参数为并行进程数，可选
#     - jieba.disable_parallel()
#         - 关闭并行分词模式
# - NOTE: **当同时使用并行分词和自定义词典的时候，要求将自定义词典放到并行分词之前做。**

# In[25]:


import jieba

# NotImplementedError: jieba: parallel mode only supports posix system
# jieba.enable_parallel()

content = '我是小明\n我是小明'
words = jieba.cut(content)
print(' / '.join(content))

# ### 扩展一：<a name='ext1'>jieba词性说明</a>
# 详见: <a href='http://ictclas.nlpir.org/nlpir/html/readme.htm'>ICTCLAS汉语词性标注集</a>

# |词性符号|词性名称|描述说明|
# |:-|:-|:-|
# |n|名词||
# |nr|人名||
# |ns|地名||
# |nt|机构团体名||
# |nz|其它专名||
# |t|时间词||
# |s|处方词||
# |f|方位词||
# |v|动词||
# |vd|副动词|直接做状语的动词，动词和副词放到一起|
# |vn|名动词|具有名词功能的动词，动词和名词放到一起|
# |a|形容词||
# |ad|副形词|直接作状语的形容词。形容词和副词放到一起。|
# |an|名形词|具有名词功能的形容词。形容词和名词放到一起|
# |b|区别词||
# |z|状态词||
# |r|代词||
# |rr|人称代词||
# |rz|指示代词||
# |ry|疑问代词||
# |m|数词||
# |q|量词||
# |d|副词||
# |p|介词||
# |c|连词||
# |u|助词||
# |e|叹词||
# |eng|英语||
# |y|语气词||
# |o|拟声词||
# |h|前缀||
# |k|后缀||
# |i|成语||
# |l|习用语|临时的词语|
# |q|量词||
# |w|标点符号||
# |x|字符串|符号、未知词性等描述|

# In[ ]:




# In[ ]:



