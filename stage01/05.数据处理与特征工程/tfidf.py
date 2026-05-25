# 词袋法&TFIDF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

arr1 = [
    "This is spark, spark sql a every good",
    "Spark Hadoop Hbase",
    "This is sample",
    "This is anthor example anthor example",
    "spark hbase hadoop spark hive hbase hue oozie",
    "hue oozie spark"
]
arr2 = [
    "this is a sample a example",
    "this cd is another another sample example example",
    "spark Hbase hadoop Spark hive hbase"
]

# 相当于词袋法
count = CountVectorizer(min_df=0.1, dtype=np.float64, ngram_range=(0, 1))
df1 = count.fit_transform(arr1)
print(df1)
print(df1.toarray())
print(count.get_stop_words())
print(count.get_feature_names_out())
print("转换另外的文档数据")
print(count.transform(arr2).toarray())

# 基于TF的值(词袋法)，做一个IDF的转换
tfidf_t = TfidfTransformer()
df2 = tfidf_t.fit_transform(df1)
print(df2.toarray())
print("转换另外的文档数据")
print(tfidf_t.transform(count.transform(arr2)).toarray())

# 相当TF+IDF(先做词袋法再做IDF转换)
tfidf_v = TfidfVectorizer(min_df=0.0, dtype=np.float64)
df3 = tfidf_v.fit_transform(arr1)
print(df3.toarray())
print(tfidf_v.get_feature_names_out())
print(tfidf_v.get_stop_words())
print("转换另外的文档数据")
print(tfidf_v.transform(arr2).toarray())
