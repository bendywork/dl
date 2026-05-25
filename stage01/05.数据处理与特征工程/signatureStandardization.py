# 加载数据
# 导入内置的鸢尾花数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler  # 区间缩放法（归一化）

iris = load_iris()
# 定义数据、标签
X = iris.data
y = iris.target

# 划分训练集&测试集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=12,
                                                    stratify=y,
                                                    test_size=0.3)
print(X_train.shape)


# 归一化
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)  # 对训练集进行fit

# 原始特征属性的最大值
print(scaler.data_max_)
# 原始特征属性的最小值
print(scaler.data_min_)
# 原始特征属性的取值范围大小(最大值-最小值)
print(scaler.data_range_)

X_train_scaler = scaler.transform(X_train) # 转换训练集  fit_transform：fit和transform
print(X_train_scaler[:5])
print(X_train_scaler.min(axis=0))
print(X_train_scaler.max(axis=0))

X_test_scaler = scaler.transform(X_test)
print(X_test_scaler[:5])
print(X_test_scaler.min(axis=0))
print(X_test_scaler.max(axis=0))

# 标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # 对训练集进行fit

# 计算每个特征属性的均值
print(scaler.mean_)
# n_samples
print(scaler.n_samples_seen_)
# 输出每个特征属性的方差
print(scaler.scale_)

# 对训练集做一个数据转换
X_train_scaler = scaler.transform(X_train)
print(X_train_scaler[:5])
print(X_train_scaler.mean(axis=0))
print(X_train_scaler.var(axis=0))

# 对测试集做一个数据转换
X_test_scaler = scaler.transform(X_test)
print(X_test_scaler[:5])
print(X_test_scaler.mean(axis=0))
print(X_test_scaler.var(axis=0))
