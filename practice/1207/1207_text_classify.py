# 一、基于python原生代码(不依赖任何第三方库的情况)
# 实现一下序号化转换、文本 / Token哑编码转换、文本词袋法转换、文本TF - IDF转换的实现
# 二、结合分词、token向量化转换 + 机器学习(LR) / 深度学习，完成datas / text_classify文件夹中的文本分类模型的训练
# 完成：
# 训练: 如果替换一个数据集，如何在最小改动的情况下，完成新数据集的模型训练
# 推理：在训练完成的基础上，最终支持给定任何一个文本，返回该文本对应的类别id、类别名称以及预测所属概率，支持返回TopK的结果；
import math
import os.path
import pandas as pd

# 第一块：纯 Python 实现四种文本向量化
#
# 1. 序号化：每个词映射到一个整数 id
# 2. 哑编码（One-Hot）：每个词/token 用一个全零向量，对应位置置1
# 3. 词袋法（BoW）：统计每个词在文本中出现的次数
# 4. TF-IDF：在词袋基础上加权，降低高频无意义词的权重

def text_parse_2_dict(filePath : str):
    pass


# 解析csv格式的文件
def __parse_file(filePath : str):

    if not os.path.exists(filePath):
        raise Exception("The given file doesn't exist")

    data_frame = pd.read_csv(filepath_or_buffer=filePath, sep="\t", header=None)
    text_data = data_frame[0]
    total_docs = len(text_data)
    print("Total docs: ", total_docs)
    # print(text_data)

    result = text_data.str.split(" ")
    my_set = set()
    word_doc_freq = {}
    for item_lst in result:
        unique_list = set(item_lst)
        for item in unique_list:
            my_set.add( item)
            word_doc_freq[item] = word_doc_freq.get(item, 0) + 1
    # lst = list(my_set)
    lst = ["<PAD>", "<UNK>"] + list(my_set)
    # lst.insert(0, "<PAD>")
    # lst.insert(1, "<UNK>")
    id2word = {i: word for i, word in enumerate(lst)}
    word2id = {word: i for i, word in id2word.items()}
    # print(len(lst))
    # print(len(word2id))
    return id2word, word2id, total_docs, word_doc_freq

# 序号化
def word_2_id(text, word2idMap):
    result = []
    str_list = text.split(" ")
    for item in str_list:
        if item in word2idMap:
            result.append(word2idMap[item])
        else:
            result.append(word2idMap["<UNK>"])
    return result


# OneHot
def word_2_one_hot(text, word2idMap):
    str_list = text.split(" ")
    result_lst = []
    for item in str_list:
        result = [0] * len(word2idMap)
        if item in word2idMap:
            result[word2idMap[item]] = 1
        else:
            result[word2idMap["<UNK>"]] = 1
        result_lst.append(result)
    return result_lst

# 词袋法
def word_2_bag(text, word2idMap):
    str_list = text.split(" ")
    # result_lst = []
    result = [0] * len(word2idMap)
    for item in str_list:
        if item in word2idMap:
            result[word2idMap[item]] += 1
        else:
            result[word2idMap["<UNK>"]] += 1
    return result

# TF-IDF
def word_2_tf_idf(text, word2idMap, doc_num, word_doc_freq):
    str_list = text.split(" ")
    result = [0] * len(word2idMap)
    count_map = {}
    for item in str_list:
        if item in word2idMap:
            count_map[item] = count_map.get(item, 0) + 1
            tf = count_map[item] / len(str_list)
            idf = math.log(doc_num / (word_doc_freq[item] + 1))
            result[word2idMap[item]] = tf * idf
        else:
            result[word2idMap["<UNK>"]] = 1
    return result

if __name__ == '__main__':
    # __parse_file("/Users/scx/data/codeData/python/dl/base/datas/text_classify/train_tokens.csv")
    id2word, word2id, doc_num, word_doc_freq = __parse_file("/Users/scx/data/codeData/python/dl/base/datas/text_classify/train_tokens.csv")
    # print(word_2_id("双鸭山 到 淮阴 的 汽车票", word2id))
    # print(word_2_id("一个不存在的词", word2id))  # 应该返回 UNK 的 id=1
    # one_hot = word_2_one_hot("双鸭山 到 淮阴", word2id)
    # print(len(one_hot))  # 应该是 3（3个词）
    # print(len(one_hot[0]))  # 应该是词表大小
    # print(sum(one_hot[0]))  # 应该是 1（只有一个位置是1）
    # bow = word_2_bag("双鸭山 到 双鸭山", word2id)
    # print(sum(bow))        # 应该是 3（3个词）
    # print(max(bow))  # 应该是 2（出现了两次）
    pass


