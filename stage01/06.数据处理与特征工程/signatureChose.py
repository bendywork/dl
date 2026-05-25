# 特征选择
import numpy as np
import warnings

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

X = np.array([
    [0, 2, 0, 3],
    [0, 1, 4, 3],
    [0.1, 1, 1, 3],
    [1, 2, 3, 1],
    [2, 3, 4, 3]
], dtype=np.float32)
Y = np.array([1, 2, 1, 2, 1])

# 方差选择法
# 基于方差选择最优的特征属性
variance = VarianceThreshold(threshold=0.7)
print(variance)
variance.fit(X)
print("各个特征属性的方差为:")
print(variance.variances_)
print('-----------------')
print(variance.transform(X))

# 相关系数法
sk1 = SelectKBest(f_regression, k=2)
sk1.fit(X, Y)
print(sk1)
print('------------')
print(sk1.scores_)
print('------------')
print(sk1.transform(X))

# 卡方检验
# 使用chi2的时候要求特征属性的取值为非负数
sk2 = SelectKBest(chi2, k=2)
sk2.fit(X, Y)
print(sk2)
print(sk2.scores_)
print(sk2.transform(X))

# Wrapper-递归特征消除法
# 基于特征消去法做的特征选择
estimator = LogisticRegression()
selector = RFE(estimator, step=2, n_features_to_select=3)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)
print(selector.transform(X))

# Embedded【嵌入法】-基于惩罚项的特征选择法
X2 = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3., 1.4, 0.2],
    [-6.2, 0.4, 5.4, 2.3],
    [-5.9, 0., 5.1, 1.8]
], dtype=np.float64)
Y2 = np.array([0, 0, 2, 2])
estimator = LogisticRegression(penalty='l2', C=0.1)
sfm = SelectFromModel(estimator, threshold=0.09)
sfm.fit(X2, Y2)
print(sfm.transform(X2))
print("系数:")
print(sfm.estimator_.coef_)
