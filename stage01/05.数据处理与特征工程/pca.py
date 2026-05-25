# PCA降维
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

X = np.array([
    [5.1, 3.5, 1.4, 0.2, 1, 23],
    [4.9, 3., 1.4, 0.2, 2.3, 2.1],
    [-6.2, 0.4, 5.4, 2.3, 2, 23],
    [-5.9, 0., 5.1, 1.8, 2, 3]
], dtype=np.float64)

# n_components: 给定降低到多少维度，但是要求该值必须小于等于样本数目/特征数目，如果给定的值大于，那么会选择样本数目/特征数目中最小的那个作为最终的特征数目
# whiten：是否做一个白化的操作，在PCA的基础上，对于特征属性是否做一个标准化
pca = PCA(n_components=0.9, whiten=False)
pca.fit(X)
print(pca.mean_)
print(pca.components_)  # 特征向量
print(pca.transform(X))
