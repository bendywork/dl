# 加载数据
# 导入内置的鸢尾花数据
import numpy
from sklearn.datasets import load_iris

iris = load_iris()
# 定义数据、标签
X = iris.data
y = iris.target

# 划分训练集&测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=12,
                                                    stratify=y,
                                                    test_size=0.3)
print(X_train.shape)
# print(X_train[:5])
#
# 特征二值化
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=3)
binarizer.fit(X_train)

X_train_bin = binarizer.transform(X_train)
# print(X_train_bin)

X_test_bin = binarizer.transform(X_test)
# print(X_test_bin)

# 设置每一列特征的阈值
# binarizer = Binarizer(threshold=[3,2,3,1])
# binarizer.fit(X_train)
#
# X_train_bin = binarizer.transform(X_train)
# print(X_train_bin)

np_arr = numpy.asarray([3,2,3,1], dtype=float)
X_train_bin = numpy.where(X_train > np_arr, 1, 0)
X_test_bin = numpy.where(X_test > np_arr, 1, 0)
print(X_test_bin)
print('---' * 200)

# 分箱
from sklearn.preprocessing import KBinsDiscretizer

# 参数说明：
# n_bins: int or array-like, shape (n_features,) (default=5)
#         产生的箱数。如果n_bins <2，则引发ValueError。
# encode: {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
#         用于编码转换结果的方法。
#         - onehot: 使用一键编码对转换后的结果进行编码，然后返回一个稀疏矩阵
#         - onehot-dense: 用一个热编码对转换后的结果进行编码，并返回一个密集数组
#         - ordinal: 返回编码为整数值的bin标识符
# strategy: {'uniform', 'quantile', 'kmeans'}, (default='quantile')
#           用于定义箱子宽度的策略。
#           - uniform: 每个特征中的所有箱子具有相同的宽度
#           - quantile: 每个特征中的所有存储箱都具有相同的点数
#           - kmeans: 每个bin中的值都具有一维k均值聚类的最接近中心

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_train_kb = est.fit_transform(X_train)
# print(X_train_kb)

X_test_kb = est.transform(X_test)
# print(X_test_kb)

# 每个特征不同分箱数目
est = KBinsDiscretizer(n_bins=[3,2,4,3], encode='ordinal', strategy='uniform')
X_train_kb = est.fit_transform(X_train)
# print(X_train_kb)

X_test_kb = est.transform(X_test)
# print(X_test_kb)
#
# 哑编码（One-Hot）
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1],
     ['Female', 3],
     ['Female', 2]]
enc.fit(X)

print(enc.transform(X))
print(enc.transform(X).toarray())
print(enc.categories_)
print(enc.transform([['Female',1], ['Male',4]]).toarray())  # 4不在训练数据里面
print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]])) # 将转换后的特征向量反向恢复
print(enc.get_feature_names_out(['gender', 'group']))

# drop参数的作用
drop_enc = OneHotEncoder(drop='first',handle_unknown='ignore').fit(X)
print(drop_enc.categories_)
print(drop_enc.transform([['Female', 1], ['Male',4]]).toarray())

drop_binary_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore').fit(X)
print(drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray())

# 使用pandas的API进行哑编码转换
import pandas as pd

a = pd.DataFrame([
    ['a', 1, 2],
    ['b', 1, 1],
    ['a', 2, 1],
    ['c', 1, 2],
    ['c', 1, 2]
], columns=['c1', 'c2', 'c3'])
a = pd.get_dummies(a)
print(a)

# 思考：有什么区别？？？？
