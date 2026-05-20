#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 安装方式： pip install pyhanlp -i https://pypi.tuna.tsinghua.edu.cn/simple/
# note: 仅支持python 3.7 + python 3.8
import pyhanlp
from pyhanlp import *

# 支持通过jpype调用java代码
import jpype
from jpype import *

import time

# In[2]:


sentence1 = "你好，欢迎使用HanLP汉语处理包！"
sentence2 = "下雨天地面积水"
sentence3 = "我新造一个词叫幻想乡你能识别并标注正确词性吗？"
sentence4 = "我的希望是希望张晚霞的背影被晚霞映红"
sentence5 = "微软公司於1975年由比爾·蓋茲和保羅·艾倫創立，18年啟動以智慧雲端、前端為導向的大改組。"
sentence6 = "wo shi lai zi hu nan de xiao ming"
sentence7 = "罗伯特和王老师带着学生们去郊游了，但是谭飞和莉莉由于身体原因没有去！"
sentence8 = "普京和川普两位总统使用四川普通话进行了沟通"
sentence9 = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰"
sentences = [
    sentence1,
    sentence2,
    sentence3,
    sentence4,
    sentence5,
    sentence6,
    sentence7,
    sentence8,
    sentence9
]

# # 一、分词、词性标注
# - HanLP中默认在分词的时候，自带词性标注功能

# ## 1.1 基本分词/标准分词
# 基于最短路径方式进行分词，要求内存至少120M以上；功能类似jieba分词

# In[4]:


# 最基本的分词
print(HanLP.segment(sentence1))

# In[4]:


print("%8s\t%8s" % ("【单词】", "【词性】"))
for term in HanLP.segment(sentence1):
    print("%8s\t%8s" % (term.word, term.nature))

# In[5]:


for sentence in sentences:
    print(HanLP.segment(sentence))

# ## 1.2 NLP分词
# - com.hankcs.hanlp.tokenizer.NLPTokenizer：同时执行词性标注和命名实体识别；底层为感知机模型；模型训练来自9970万字的大型综合语料库。

# In[6]:


# 加载类
nlp_tokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")

# In[7]:


# 基本用法
print(nlp_tokenizer.segment(sentence1))

# In[8]:


print("%8s\t%8s" % ("【单词】", "【词性】"))
for term in nlp_tokenizer.segment(sentence1):
    print("%8s\t%8s" % (term.word, term.nature.toString()))

# In[9]:


for sentence in sentences:
    print(nlp_tokenizer.segment(sentence))

# In[10]:


print("%8s\t%8s" % ("【单词】", "【词性】"))
for term in nlp_tokenizer.analyze(sentence4):
    print("%8s\t%8s" % (term.getValue(), term.getLabel()))

# ## 1.3 CRF分词
# - com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer：对新词的识别能力比较强，但是开销大。

# In[11]:


# 加载类
CRFLexicalAnalyzer = SafeJClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
# 构造对象
crf = CRFLexicalAnalyzer()

# In[12]:


# 基本用法
print(crf.segment(sentence1))

# In[13]:


for sentence in sentences:
    print(crf.segment(sentence))

# In[14]:


print("%8s\t%8s" % ("【单词】", "【词性】"))
for term in crf.analyze(sentence4):
    print("%8s\t%8s" % (term.value, term.label))

# In[15]:


for sentence in sentences:
    print(crf.analyze(sentence))

# ## 1.4 极速分词
# - com.hankcs.hanlp.tokenizer.SpeedTokenizer：词典的最长分词，速度极快，精度一般，在i7-6700K上跑出4500万字每秒的速度

# In[16]:


# 加载类
SpeedTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.SpeedTokenizer")

# In[17]:


# 设置属性，关闭词性显示
HanLP.Config.ShowTermNature = False
# 开始极速分词
print(SpeedTokenizer.segment(sentence7))

# 计算时间
start = time.time()
pressure = 100000
for i in range(pressure):
    SpeedTokenizer.segment(sentence7)
cost_time = time.time() - start
print("SpeedTokenizer分词的速度::%.2f字每秒" % 
      (len(sentence) * pressure / cost_time))
HanLP.Config.ShowTermNature = True

# ## 1.5 自定义词典
# - com.hankcs.hanlp.dictionary.CustomDictionary：CustomDictionary是全局用户自定义词典，可以随时增删，影响所有分词器；代码中添加的单词，不会进行保存。
# - 可以在hanlp.properties中配置多个用户自定义词典信息，eg：```CustomDictionaryPath=data/dictionary/custom/CustomDictionary.txt; 用户词典1路径;用户词典2路径;```
#     - 每行一个单词，格式为: 单词 词性A A的词频 词性B B的词频...
#     - HanLP中全局默认词性为名词n
#     - 如果词典路径后面空格接词性，表示当前词典中的所有单词默认词性均为该值，比如：全国地名大全.txt ns; 那么表示这个文件中的所有单词默认词性为ns。
#     - HanLP主要通过统计分词来实现，故添加的单词不一样会识别出来。

# In[18]:


result = HanLP.segment(sentence9)
result = ' '.join(map(lambda term: term.word, result))
print("【添加词典前】：\n{}".format(result))

# In[19]:


# 添加自定义单词
CustomDictionary.add("攻城狮") # 动态增加
CustomDictionary.insert("白富美", "nz 1024") # 强行插入
# CustomDictionary.remove("攻城狮") # 删除词语
CustomDictionary.add("单身狗", "nz 1024 n 1") # 动态增加
print(CustomDictionary.get("单身狗"))

# In[20]:


result = HanLP.segment(sentence9)
result = ' '.join(map(lambda term: term.word, result))
print("【添加词典后】：\n{}".format(result))

#  

#  

# # 二、中国人名识别
# - 在分词器中默认开启中国人名的识别。比如HanLP.segment中使用的分词器
# - 可以通过下列代码强调调用人名识别
# - 建议使用NLP或者CRF词法分析器来提取人名

# In[21]:


sentences = [
       "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。",
       "武大靖创世界纪录夺冠，中国代表团平昌首金",
       "区长庄木弟新年致辞",
       "朱立伦：两岸都希望共创双赢 习朱历史会晤在即",
       "陕西首富吴一坚被带走 与令计划妻子有交集",
       "据美国之音电台网站4月28日报道，8岁的凯瑟琳·克罗尔（凤甫娟）和很多华裔美国小朋友一样，小小年纪就开始学小提琴了。她的妈妈是位虎妈么？",
       "凯瑟琳和露西（庐瑞媛），跟她们的哥哥们有一些不同。",
       "王国强、高峰、汪洋、张朝阳光着头、韩寒、小四",
       "张浩和胡健康复员回家了",
       "王总和小丽结婚了",
       "编剧邵钧林和稽道青说",
       "这里有关天培的有关事迹",
       "龚学平等领导说,邓颖超生前杜绝超生"]

# In[22]:


for sentence in sentences:
    term_list = HanLP.segment(sentence)
    print(term_list)

# In[23]:


# enableNameRecognize：开启/关闭中文名的命名实体识别, 默认为True，表示开启
segment = HanLP.newSegment().enableNameRecognize(True);
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

# In[24]:


# 加载类
CRFLexicalAnalyzer = SafeJClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
# 构造对象
crf = CRFLexicalAnalyzer()

# 分词然后查看词性
for sentence in sentences:
    term_list = crf.analyze(sentence)
    print(term_list)

print("=" * 50)

# 分词然后查看词性
for sentence in sentences:
    term_list = crf.analyze(sentence)
    flag = False
    for word in term_list:
        if word.label == 'nr':
            print("{}".format(word.getValue()), end='\t')
            flag = True
    if flag:
        print("")

# # 三、音译人名识别
# - 目前分词默认开始音译人名的识别，不用手动开启；下列代码仅为强调；
# - 目前仅支持HMM的viterbi算法来识别，也就是默认方式下识别。

# In[25]:


sentences = [
        "一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。",
        "世界上最长的姓名是简森·乔伊·亚历山大·比基·卡利斯勒·达夫·埃利奥特·福克斯·伊维鲁莫·马尔尼·梅尔斯·帕特森·汤普森·华莱士·普雷斯顿。",
    ]

# In[26]:


for sentence in sentences:
    term_list = HanLP.segment(sentence)
    print(term_list)

# In[27]:


# enableTranslatedNameRecognize：开启/关闭英译名的命名实体识别, 默认为True，表示开启
segment = HanLP.newSegment().enableTranslatedNameRecognize(True);
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

# In[28]:


# 修改词性(从识别的nr修改为nt的词性)
CustomDictionary.add("亚马逊", "nt 2")
for sentence in sentences:
    term_list = HanLP.segment(sentence)
    print(term_list)

print("=" * 50)

# 分词然后查看词性
for sentence in sentences:
    term_list = HanLP.segment(sentence)
    flag = False
    for term in term_list:
        if term.nature.toString() == 'nrf':
            print("{}".format(term.word), end='\t')
            flag = True
    if flag:
        print("")

# In[29]:


# 不支持音译名识别(CRF)
# 加载类
CRFLexicalAnalyzer = SafeJClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")
# 构造对象
crf = CRFLexicalAnalyzer()
# 分词然后查看词性
for sentence in sentences:
    term_list = crf.analyze(sentence)
    print(term_list)

# # 四、日本名识别
# - 标准分词器默认关闭日本名识别，需要手动开启

# In[30]:


sentences =[
       "北川景子参演了林诣彬导演的《速度与激情3》",
       "林志玲亮相网友:确定不是波多野结衣？",
       "龟山千广和近藤公园在龟山公园里喝酒赏花",
    ]

# In[31]:


# enableJapaneseNameRecognize: 开启/关闭日本名识别功能，默认为False，表示关闭
segment = HanLP.newSegment().enableJapaneseNameRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

# # 五、地名识别
# - 可以通过词典来识别地名
# - 标准分类器中地名识别是关闭的，需要手动开启
# - 建议直接使用NLP感知器词法分析器来提取地名（同时提取地名和机构名）

# 使用NLP词法解析器来实现地名的提取

# In[32]:


sentences = [
    "武胜县新学乡政府大楼门前锣鼓喧天",
    "蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"
]

# In[33]:


def parse_word(word, result=[]):
    if hasattr(word, "innerList"):
        for  inner_word in word.innerList:
            parse_word(inner_word, result)
    result.append([word.getValue(), word.getLabel()])
    return result

# In[34]:


# 加载类
nlp_tokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
for sentence in sentences:
    term_list = nlp_tokenizer.analyze(sentence)
    print(term_list)
print("=" * 50)
for sentence in sentences:
    word_list = nlp_tokenizer.analyze(sentence)
    result = []
    for word in word_list:
        result = parse_word(word, result)
    print(result)
    print()

# In[35]:


print("开启地名识别前:")
for sentence in sentences:
    print(HanLP.segment(sentence))

# In[36]:


print("开启地名识别后:")
segment = HanLP.newSegment().enablePlaceRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

# # 六、机构名识别
# - 可以通过词典来识别机构名
# - 标准分类器中机构名识别是关闭的，需要手动开启
# - 建议直接使用NLP感知器词法分析器来提取机构名

# 使用NLP词法解析器来实现地名的提取

# In[37]:


sentences = [
    "我在上海小鱼科技有限公司兼职工作，",
    "我经常在台电大厦的台川喜宴餐厅吃饭，",
    "偶尔去地中海影城看电影。",
    "武胜县新学乡政府大楼门前锣鼓喧天",
    "蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"
]

# In[38]:


def parse_word(word, result=[]):
    if hasattr(word, "innerList"):
        for  inner_word in word.innerList:
            parse_word(inner_word, result)
    result.append("%s/%s" % (word.getValue(), word.getLabel()))
    return result

# In[39]:


# 加载类
nlp_tokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
for sentence in sentences:
    term_list = nlp_tokenizer.analyze(sentence)
    print(term_list)
print("=" * 50)
for sentence in sentences:
    word_list = nlp_tokenizer.analyze(sentence)
    result = []
    for word in word_list:
        result = parse_word(word, result)
    print(result)
    print()

# In[40]:


print("开启机构名识别前:")
for sentence in sentences:
    print(HanLP.segment(sentence))

# In[41]:


print("开启机构名识别后:")
segment = HanLP.newSegment().enableOrganizationRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

# # 七、开启所有命名实体的识别
# - 当前版本仅支持：人名、地名、机构名识别
# - 可以考虑使用NLP感知器词法分析器

# In[42]:


sentences = [
    "微软的比尔盖茨、Facebook的扎克伯格相约来到湖南张家界市永定区合作桥乡八家村永红组进行捐赠活动，在张德彪等人的陪同下，一起在 嘟嘟乐餐馆的醉仙阁餐厅享受了中国美食，席间北川景子等影星也参与了",
    "我经常在台电大厦的台川喜宴餐厅吃饭，",
    "偶尔去地中海影城看电影。",
    "武胜县新学乡政府大楼门前锣鼓喧天",
    "蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"
]

# In[43]:


def parse_word(word, result=[]):
    if hasattr(word, "innerList"):
        for  inner_word in word.innerList:
            parse_word(inner_word, result)
    result.append("%s/%s" % (word.getValue(), word.getLabel()))
    return result

# In[44]:


# 加载类
nlp_tokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
for sentence in sentences:
    term_list = nlp_tokenizer.analyze(sentence)
    print(term_list)
print("=" * 50)
for sentence in sentences:
    word_list = nlp_tokenizer.analyze(sentence)
    result = []
    for word in word_list:
        result = parse_word(word, result)
    print(result)
    print()

# In[45]:


print("开启识别前:")
for sentence in sentences:
    print(HanLP.segment(sentence))

# In[46]:


print("开启识别后:")
segment = HanLP.newSegment().enableAllNamedEntityRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

#  

# # 八、关键词抽取
# - 默认为TextRank关键词抽取: HanLP.extractKeyword
# - 可选：com.hankcs.hanlp.mining.word.TfIdfCounter（TFIDF关键词抽取）

# In[47]:


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

# In[48]:


print("【默认TextRank关键词抽取】")
words = HanLP.extractKeyword(sentence, 20)
print(words)

# In[49]:


# 必须加入停止词(不建议使用)
print("【TFIDF关键词抽取】")
# class加载
tfidf_class = SafeJClass("com.hankcs.hanlp.mining.word.TfIdfCounter")
# 对象创建
tfidf = tfidf_class()
# 抽取
words = tfidf.getKeywords(sentence, 20)
print(words)

# # 九、自动摘要
# - 默认为TextRankSentence抽取: HanLP.extractSummary

# In[50]:


# https://mbd.baidu.com/newspage/data/landingsuper?context=%7B%22nid%22%3A%22news_10141364001547692745%22%7D&n_type=0&p_from=1
# sentence = """
# 算法可大致分为基本算法、数据结构的算法、数论算法、计算几何的算法、图的算法、动态规划以及数值分析、加密算法、排序算法、检索算法、随机化算法、并行算法、厄米变形模型、随机森林算法。
# 算法可以宽泛的分为三类，
# 一，有限的确定性算法，这类算法在有限的一段时间内终止。他们可能要花很长时间来执行指定的任务，但仍将在一定的时间内终止。这类算法得出的结果常取决于输入值。
# 二，有限的非确定算法，这类算法在有限的时间内终止。然而，对于一个（或一些）给定的数值，算法的结果并不是唯一的或确定的。
# 三，无限的算法，是那些由于没有定义终止定义条件，或定义的条件无法由输入的数据满足而不终止运行的算法。通常，无限算法的产生是由于未能确定的定义终止条件。
# """
sentence = """
新华社澳门8月1日电（方钊、郭鑫）1日上午，解放军驻澳门部队在新口岸军营隆重举行升国旗仪式和“八一”招待会，庆祝中国人民解放军建军92周年。
8时，驻澳部队威武的仪仗队护送国旗，步伐铿锵走向升旗台。军乐队奏响雄壮的《中华人民共和国国歌》，驻澳门部队官兵在司令员徐良才和政委孙文举带领下整齐列队，面向国旗庄严敬礼，目送鲜艳的五星红旗冉冉升起，献上深情祝福。
“八一”招待会于11时举行，主礼嘉宾与各界来宾一同观看了纪录片《濠江战旗别样红》，全面了解驻军进澳门20年来履行防务情况。驻澳门部队司令员徐良才和澳门特区行政长官崔世安致辞。
徐良才深情回顾了中国人民解放军的光辉历程。他表示，今年是中国人民解放军进驻澳门20周年，驻军自进驻之日起，就坚定不移地贯彻“一国两制”伟大方针，坚定不移地遵守澳门基本法和驻军法，坚定不移地维护澳门繁荣稳定，始终视国家和民族利益高于一切，始终遵守澳门现行社会制度，尊重和支持特区政府依法施政，积极参加社会公益事业，把澳门同胞当亲人。
徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托，深入贯彻习近平强军思想，坚持政治建军、服务大局，坚持任务牵引、练兵备战，坚持依法从严、锤炼作风，部队履行防务能力稳步提升。驻军部队的建设发展，离不开特区各界和澳门同胞的关心，离不开中联办、外交公署等中央驻澳机构的支持，特别是特区政府为驻军有效履行防务创造了良好环境和条件，对此致以衷心的感谢和崇高的敬意。
崔世安向驻澳门部队官兵致以节日的祝贺，对驻军一直以来对特区发展的有力支持表示感谢。他表示，20年来，驻澳部队与澳门特区同呼吸、共命运，视驻地为故乡，把居民当亲人，支持特区政府依法施政，积极开展多元化的、丰富多彩的爱民活动，主动参与献血、植树等社会公益活动；与特区政府合办“澳门青年学生军事夏令营”，培养青年“爱国爱澳”的核心价值；在防灾救灾工作上，以高度的责任感，大力支持特区政府。事实证明，驻澳部队是维护“一国两制”的重要力量，是维护澳门繁荣稳定的重要基石，为澳门特区各项事业的进步作出了不懈的努力和巨大的贡献。
全国政协副主席何厚铧、中央政府驻澳门联络办公室主任傅自应、外交部驻澳门特派员公署特派员沈蓓莉、驻澳部队政委孙文举、澳门特区立法会主席高开贤等，以及澳门特区政府、中央驻澳机构、澳区全国人大代表、政协委员、社团、高校、往届军事夏令营学生代表等300余人出席了招待会。
"""

# In[51]:


print("【默认TextRankSentence摘要抽取】")
summary_sentences = HanLP.extractSummary(sentence,5)
for summary_sentence in summary_sentences:
    print(summary_sentence)


# # 十、短语提取
# - 默认为MutualInformationEntropyPhraseExtractor抽取: HanLP.extractPhrase

# In[52]:


sentence = """
算法工程师 
算法（Algorithm）是一系列解决问题的清晰指令，也就是说，能够对一定规范的输入，在有限时间内获得所要求的输出。如果一个算法有缺陷，或不适合于某个问题，执行这个算法将不会解决这个问题。不同的算法可能用不同的时间、空间或效率来完成同样的任务。一个算法的优劣可以用空间复杂度与时间复杂度来衡量。算法工程师就是利用算法处理事物的人。 
1职位简介 
算法工程师是一个非常高端的职位； 
专业要求：计算机、电子、通信、数学等相关专业； 
学历要求：本科及其以上的学历，大多数是硕士学历及其以上； 
语言要求：英语要求是熟练，基本上能阅读国外专业书刊； 
必须掌握计算机相关知识，熟练使用仿真工具MATLAB等，必须会一门编程语言。 
2研究方向 
视频算法工程师、图像处理算法工程师、音频算法工程师 通信基带算法工程师 
3目前国内外状况 
目前国内从事算法研究的工程师不少，但是高级算法工程师却很少，是一个非常紧缺的专业工程师。算法工程师根据研究领域来分主要有音频/视频算法处理、图像技术方面的二维信息算法处理和通信物理层、雷达信号处理、生物医学信号处理等领域的一维信息算法处理。 
另外数据挖掘、互联网搜索算法也成为当今的热门方向。 
算法工程师逐渐往人工智能方向发展。
"""
# sentence = """
# 新华社澳门8月1日电（方钊、郭鑫）1日上午，解放军驻澳门部队在新口岸军营隆重举行升国旗仪式和“八一”招待会，庆祝中国人民解放军建军92周年。
# 8时，驻澳部队威武的仪仗队护送国旗，步伐铿锵走向升旗台。军乐队奏响雄壮的《中华人民共和国国歌》，驻澳门部队官兵在司令员徐良才和政委孙文举带领下整齐列队，面向国旗庄严敬礼，目送鲜艳的五星红旗冉冉升起，献上深情祝福。
# “八一”招待会于11时举行，主礼嘉宾与各界来宾一同观看了纪录片《濠江战旗别样红》，全面了解驻军进澳门20年来履行防务情况。驻澳门部队司令员徐良才和澳门特区行政长官崔世安致辞。
# 徐良才深情回顾了中国人民解放军的光辉历程。他表示，今年是中国人民解放军进驻澳门20周年，驻军自进驻之日起，就坚定不移地贯彻“一国两制”伟大方针，坚定不移地遵守澳门基本法和驻军法，坚定不移地维护澳门繁荣稳定，始终视国家和民族利益高于一切，始终遵守澳门现行社会制度，尊重和支持特区政府依法施政，积极参加社会公益事业，把澳门同胞当亲人。
# 徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托，深入贯彻习近平强军思想，坚持政治建军、服务大局，坚持任务牵引、练兵备战，坚持依法从严、锤炼作风，部队履行防务能力稳步提升。驻军部队的建设发展，离不开特区各界和澳门同胞的关心，离不开中联办、外交公署等中央驻澳机构的支持，特别是特区政府为驻军有效履行防务创造了良好环境和条件，对此致以衷心的感谢和崇高的敬意。
# 崔世安向驻澳门部队官兵致以节日的祝贺，对驻军一直以来对特区发展的有力支持表示感谢。他表示，20年来，驻澳部队与澳门特区同呼吸、共命运，视驻地为故乡，把居民当亲人，支持特区政府依法施政，积极开展多元化的、丰富多彩的爱民活动，主动参与献血、植树等社会公益活动；与特区政府合办“澳门青年学生军事夏令营”，培养青年“爱国爱澳”的核心价值；在防灾救灾工作上，以高度的责任感，大力支持特区政府。事实证明，驻澳部队是维护“一国两制”的重要力量，是维护澳门繁荣稳定的重要基石，为澳门特区各项事业的进步作出了不懈的努力和巨大的贡献。
# 全国政协副主席何厚铧、中央政府驻澳门联络办公室主任傅自应、外交部驻澳门特派员公署特派员沈蓓莉、驻澳部队政委孙文举、澳门特区立法会主席高开贤等，以及澳门特区政府、中央驻澳机构、澳区全国人大代表、政协委员、社团、高校、往届军事夏令营学生代表等300余人出席了招待会。
# """

# In[53]:


print("【默认MutualInformationEntropyPhraseExtractor短语抽取】")
phrases = HanLP.extractPhrase(sentence, 10)
for phrase in phrases:
    print(phrase)

# # 十一、拼音转换

# #### 汉字转拼音

# In[54]:


sentence = sentence4
print("【中文转换为拼音】")
print("原文:{}".format(sentence))
print(HanLP.convertToPinyinList(sentence))

# #### 汉字转拼音
# - 仅获取具体拼音

# In[55]:


sentence = sentence4
print("【中文转换为拼音】")
print("原文:{}".format(sentence))
pinyins = HanLP.convertToPinyinList(sentence)
print("【转换结果】:")
for pinyin in pinyins:
    print("拼音:{}, 音调:{}".format(pinyin.getPinyinWithoutTone(), pinyin.getTone()))

# #### 转换获取首字母

# In[56]:


first_char = HanLP.convertToPinyinFirstCharString(sentence, " ", True)
print(first_char)

# # 十二、简繁体转换

# In[57]:


sentence = "用笔记本电脑写程序，并使用打印机打印出来"
print("【简体中文转换为繁体中文】")
print("【原文】:{}".format(sentence))
print("【转换】:{}".format(HanLP.convertToTraditionalChinese(sentence)))

# In[58]:


sentence = "以後等妳當上皇后，就能買士多啤梨慶祝了"
print("【繁体中文转换为简体中文】")
print("【原文】:{}".format(sentence))
print("【转换】:{}".format(HanLP.convertToSimplifiedChinese(sentence)))

# # 十三、依存句法分析
# - 内部默认使用NeuralNetworkDependencyParser来进行句法分析，对外API为：HanLP.parseDependency

# In[59]:


print("【最简单的使用】")
# 序号、当前词语/标点的原型或词干、当前词语/标点的原型或词干、当前词语的词性（粗粒度）、当前词语的词性（细粒度）、-、
# 和谁的关系(当前词语的中心词)、当前词语与中心词的依存关系、-、-
print(HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"))

# In[60]:


# 句法分析返回的最终对象的属性方法
dir(next(HanLP.parseDependency("小明把苹果吃了").iterator()))

# In[61]:


print("【依存关系遍历】")
sentence = HanLP.parseDependency("小明把苹果吃了")
for word in sentence.iterator():  # 通过dir()可以查看sentence的方法
    # 当前单词 --> 关系 --> 上一个单词
    print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
print()

# 也可以直接拿到数组，任意顺序或逆序遍历
word_array = sentence.getWordArray()
for word in word_array:
    print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
print()

# 还可以直接遍历子树，从某棵子树的某个节点一路遍历到虚根
CoNLLWord = SafeJClass("com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord")
head = word_array[0]
while head.HEAD:
    if (head == CoNLLWord.ROOT):
        print(head.LEMMA)
    else:
        print("%s --(%s)--> " % (head.LEMMA, head.DEPREL))
    head = head.HEAD
if (head == CoNLLWord.ROOT):
    print(head.LEMMA)

# In[ ]:



