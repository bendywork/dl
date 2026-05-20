import jieba
import jieba.analyse
import jieba.posseg as pseg

# ============================================================
# 一、分词
# ============================================================

# 基本使用
str = "我喜欢学习深度学习，我正在学习jieba分词"
res = jieba.cut(str)
print("【基本应用】: {}".format(" ".join(res)))

# 全模式
word_list = jieba.cut('我来到湖南国防科技大学', cut_all=True)
print("【全模式】: {}".format(" ".join(word_list)))

# 精确模式
word_list = jieba.cut('我来到湖南国防科技大学')
print("【精确模式】: {}".format(" ".join(word_list)))

# 搜索引擎模式
word_list = jieba.cut_for_search('我来到湖南国防科技大学')
print("【搜索引擎模式】: {}".format(" ".join(word_list)))

# 仅词典模式（不启用HMM）
word_list = jieba.cut("我在台电大厦上班", HMM=False)
print("【仅词典模式】: {}".format(" ".join(word_list)))

# HMM新词发现
word_list = jieba.cut("我在台电大厦上班", HMM=True)
print("【HMM新词发现模式】: {}".format(" ".join(word_list)))

# cut vs lcut 返回类型区别
# cut 返回迭代器，lcut 返回 list
cut_word_list = jieba.cut('我来到湖南国防科技大学')
print("【cut 返回类型】: {}".format(type(cut_word_list)))
print("【cut 结果】: {}".format(' /'.join(cut_word_list)))
print("【cut 再次获取（已耗尽）】: {}".format(' /'.join(cut_word_list)))

lcut_word_list = jieba.lcut('我来到湖南国防科技大学')
print("【lcut 返回类型】: {}".format(type(lcut_word_list)))
print("【lcut 结果】: {}".format(' /'.join(lcut_word_list)))
print("【lcut 再次获取】: {}".format(' /'.join(lcut_word_list)))


# ============================================================
# 二、自定义词典
# ============================================================

# 载入词典前
word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=False)
print("【载入词典前<无HMM>】: {}".format('/'.join(word_list)))

word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=True)
print("【载入词典前<有HMM>】: {}".format('/'.join(word_list)))

# 加载自定义词典（文件路径相对于运行目录）
jieba.load_userdict('../datas/word_dict.txt')

word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=False)
print("【载入词典后<无HMM>】: {}".format('/'.join(word_list)))

word_list = jieba.cut('外卖送餐公司中饿了么是你值得信赖的选择', HMM=True)
print("【载入词典后<有HMM>】: {}".format('/'.join(word_list)))

# 动态调整词频：suggest_freq
word_list = jieba.cut('如果放到post中将出错。', HMM=False)
print("【不启动HMM+不添加分词】: {}".format('/'.join(word_list)))

jieba.suggest_freq(('中', '将'), tune=True)

word_list = jieba.cut('如果放到post中将出错。', HMM=False)
print("【不启动HMM+添加分词】: {}".format('/'.join(word_list)))


# ============================================================
# 三、关键词抽取
# ============================================================

sentence = """
新华社澳门8月1日电（方钊、郭鑫）1日上午，解放军驻澳门部队在新口岸军营隆重举行升国旗仪式和"八一"招待会，庆祝中国人民解放军建军92周年。
8时，驻澳部队威武的仪仗队护送国旗，步伐铿锵走向升旗台。军乐队奏响雄壮的《中华人民共和国国歌》，驻澳门部队官兵在司令员徐良才和政委孙文举带领下整齐列队，面向国旗庄严敬礼，目送鲜艳的五星红旗冉冉升起，献上深情祝福。
"八一"招待会于11时举行，主礼嘉宾与各界来宾一同观看了纪录片《濠江战旗别样红》，全面了解驻军进澳门20年来履行防务情况。驻澳门部队司令员徐良才和澳门特区行政长官崔世安致辞。
徐良才深情回顾了中国人民解放军的光辉历程。他表示，今年是中国人民解放军进驻澳门20周年，驻军自进驻之日起，就坚定不移地贯彻"一国两制"伟大方针，坚定不移地遵守澳门基本法和驻军法，坚定不移地维护澳门繁荣稳定，始终视国家和民族利益高于一切，始终遵守澳门现行社会制度，尊重和支持特区政府依法施政，积极参加社会公益事业，把澳门同胞当亲人。
"""

# TF-IDF 关键词提取
print("\n【TF-IDF 关键词】:")
print(jieba.analyse.extract_tags(sentence, topK=10))

print("\n【TF-IDF 带权重】:")
print(jieba.analyse.extract_tags(sentence, topK=10, withWeight=True))

print("\n【TF-IDF 指定词性】:")
print(jieba.analyse.extract_tags(sentence, topK=10, withWeight=True,
                                  allowPOS=('n', 'ns', 'vn', 'a')))

# 自定义 IDF 和停止词
jieba.analyse.set_idf_path('../datas/idf.txt.big')
jieba.analyse.set_stop_words('../datas/stop_words.txt')
print("\n【TF-IDF 自定义IDF+停止词】:")
print(jieba.analyse.extract_tags(sentence, topK=10, withWeight=True,
                                  allowPOS=('n', 'ns', 'vn', 'a')))

# TextRank 关键词提取
print("\n【TextRank 关键词】:")
print(jieba.analyse.textrank(sentence=sentence, topK=10))

print("\n【TextRank 带权重+词性过滤】:")
print(jieba.analyse.textrank(sentence=sentence, topK=10, withWeight=True,
                              allowPOS=('n', 'ns', 'vn', 'v', 'a'), withFlag=True))

print("\n【TextRank 仅人名+地名】:")
print(jieba.analyse.textrank(sentence=sentence, topK=5, withWeight=True,
                              allowPOS=('n', 'nr'), withFlag=True))


# ============================================================
# 四、词性标注
# ============================================================

sentence = "姚明的职业是什么"
words = pseg.cut(sentence)
print("\n%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))

sentence = "徐良才说，近年来，驻军官兵时刻牢记习主席重要嘱托。"
words = pseg.cut(sentence)
print("\n%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))

# 动态添加词语到词典
jieba.add_word('习主席', 2, 'nr')
words = pseg.cut(sentence)
print("\n【添加'习主席'词条后】:")
print("%8s\t%8s" % ("【单词】", "【词性】"))
for word, flag in words:
    print("%8s\t%8s" % (word, flag))


# ============================================================
# 五、并行分词（仅 posix 系统支持，Windows 不支持）
# ============================================================

# jieba.enable_parallel(4)   # 开启并行，参数为进程数
# jieba.disable_parallel()   # 关闭并行

content = '我是小明\n我是小明'
words = jieba.cut(content)
print("\n【并行分词示例】: {}".format(' / '.join(words)))
