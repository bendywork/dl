# ============================================================
# 章节：NumPy 基础
# 来源：p13_numpy.pdf
# 内容：ndarray 创建、属性、基本运算、广播机制、索引与切片、
#        高阶索引、bool 索引、常用操作、矩阵运算、比较函数、
#        数学函数、统计函数、随机函数等
# ============================================================

import numpy as np


# === ndarray 创建 — 标量创建 0 维数组 ===
num = 789
arr = np.array(num)
print(num)
print(arr)
print(type(num))
print(type(arr))


# === ndarray 创建 — 一维列表转数组 ===
lst = [6, 7, 1, 0, 9, 8]
arr = np.array(lst)
print(lst)
print(arr)


# === ndarray 创建 — 二维列表转数组，查看常用属性 ===
lst = [[6, 7, 1], [0, 9, 8]]
arr = np.array(lst)
print(lst)
print(arr)
print(arr.ndim)       # 秩（维度数量）
print(arr.dtype)      # 数据类型
print(arr.itemsize)   # 每个元素字节数
print(arr.shape)      # 形状
print(arr.size)       # 元素总个数


# === ndarray 创建 — 三维列表转数组 ===
lst = [[[6, 7], [1, 0], [9, 8]]]
arr = np.array(lst)
print(lst)
print(arr)


# === ndarray 创建 — np.arange ===
print(np.arange(3))
print(np.arange(3.0))
print(np.arange(3, 7))
print(np.arange(3, 7, 2))
print(np.arange(7, 3, -2))
print(np.arange(3, 7, 0.5))


# === ndarray 创建 — np.linspace ===
print(np.linspace(1, 50))
print(np.linspace(1, 10, num=10))
print(np.linspace(1, 10, num=10, dtype=np.int32))


# === 基本运算 — 逐元素算术与比较运算（数组与标量）===
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a + 2)
print(a - 2)
print(a * 2)
print(a / 2)
print(a < 4)
print(a > 3)


# === 基本运算 — 逐元素算术与比较运算（两个同形数组）===
b = np.array([[2, 2], [2, 1], [1, 1]])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a < b)
print(a > b)


# === 广播机制 — 后缘维度相同或某维度为 1 时可广播 ===
a = np.arange(24).reshape((2, 3, 4))
b = np.arange(12).reshape((3, 4))
c = np.arange(4).reshape((1, 4))
d = np.arange(4).reshape(4)
e = np.arange(12).reshape((1, 3, 4))
f = np.arange(6).reshape((2, 3, 1))
g = np.arange(2).reshape((2, 1, 1))
h = np.arange(2).reshape((1, 2, 1, 1))
i = np.arange(10).reshape((5, 2, 1, 1))

print((a + b).shape)
print((a + c).shape)
print((a + d).shape)
print((a + e).shape)
print((a + f).shape)
print((a + g).shape)
print((a + h).shape)
print((a + i).shape)


# === 索引和切片 — 序列索引与切片（视图特性）===
lst = [6, 8, 9, 1, 3]
arr = np.array(lst)
print(lst)
print(arr)

item_lst = lst[2]
part_lst = lst[2:3]
item_arr = arr[2]
part_arr = arr[2:3]   # 返回视图
print(item_lst)
print(part_lst)
print(item_arr)
print(part_arr)

lst[2] = 99
arr[2] = 99
print(item_lst)
print(part_lst)
print(item_arr)
print(part_arr)       # 动态性：视图随原数组改变


# === 索引和切片 — 多维数组针对各轴的索引与切片 ===
lst = [[[6, 7, 5, 1],
        [2, 9, 8, 0],
        [3, 4, 2, 8]],

       [[4, 5, 2, 3],
        [2, 9, 7, 1],
        [9, 5, 6, 7]]]

arr = np.array(lst)   # shape: (2, 3, 4)
print(lst[1][0][2])
print(arr[1][0][2])
print(arr[1, 0, 2])

print(lst[1:2][:1])
print(arr[1:2][:1])
print(arr[1:2, :1])

print(lst[1][::2][0])
print(arr[1][::2][0])
print(arr[1, ::2, 0])


# === 数组的高阶索引 — 整数列表作为索引 ===
x = np.arange(24).reshape((3, 2, 4))
print(x)

# 可以理解为 x[2], x[0], x[0] 构成一个更高维度的数组
print(x[[2, 0, 0]])

# 可以理解为 x[2,0], x[0,0], x[1,1] 构成一个更高维度的数组
print(x[[2, 0, 1], [0, 0, 1]])

# 可以理解为 x[2,0,1], x[0,0,2], x[1,1,3] 构成一个更高维度的数组
print(x[[2, 0, 1], [0, 0, 1], [1, 2, 3]])

# 基本索引和高阶索引组合时会发生广播，下面三个是等价的
print(x[0, [0, 0, 1], [1, 2, 3]])
print(x[[0], [0, 0, 1], [1, 2, 3]])
print(x[[0, 0, 0], [0, 0, 1], [1, 2, 3]])

# 下面三个也是等价的
print(x[0, [0, 0, 1], 2])
print(x[[0], [0, 0, 1], [2]])
print(x[[0, 0, 0], [0, 0, 1], [2, 2, 2]])

# 切片在高阶索引一侧，按轴顺序定 shape 即可
print(x[::2, [0, 0, 1], [3, 0, 2]])   # shape: (2, 3)
print(x[[2, 0, 1], [1, 0, 1], ::2])   # shape: (3, 2)

# 切片两侧都有高阶索引时，高阶索引在前，切片在后
print(x[[2, 0, 1], 1:, [3, 0, 2]])    # shape: (3, 1)


# === 数组的高阶索引 — bool 数组作为索引（最后一个维度）===
x = np.arange(24).reshape((3, 2, 4))
print(x)

bool_list = [[[True, False, True, False],
              [False, True, False, True]],
             [[True, False, True, False],
              [False, True, False, True]],
             [[True, False, True, False],
              [False, True, False, True]]]

print(x[np.array(bool_list)])

# x > 13 得到一个 shape 为 (3, 2, 4) 的 bool 数组
print(x[x > 13])


# === 数组的高阶索引 — bool 索引针对 1 轴 ===
bool_list = [[True, False],
             [False, True],
             [True, False]]

print(x[np.array(bool_list)])


# === 数组的高阶索引 — bool 索引针对 0 轴 ===
bool_list = [True, False, True]
print(x[np.array(bool_list)])


# === 常用操作 — np.reshape ===
arr1 = np.arange(6).reshape((2, 1, 3))
arr2 = np.reshape(arr1, 6)
arr3 = np.reshape(arr1, -1)
arr4 = np.reshape(arr1, (-1,))
print(arr2)   # [0 1 2 3 4 5]
print(arr3)   # [0 1 2 3 4 5]
print(arr4)   # [0 1 2 3 4 5]

arr1 = np.arange(24)
arr2 = np.reshape(arr1, (2, 2, -1, 2))
print(arr2.shape)   # (2, 2, 3, 2)


# === 常用操作 — ndarray.flatten() 扁平化 ===
a = np.array([[1, 2], [3, 4]])
print(a.flatten())


# === 常用操作 — ndarray.T 转置 ===
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.T)

a = np.arange(24).reshape((2, 3, 4))
print(a.T.shape)


# === 常用操作 — np.swapaxes 交换两个轴 ===
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.swapaxes(a, 0, 1))

a = np.arange(24).reshape((2, 3, 4))
print(np.swapaxes(a, 0, 2).shape)


# === 常用操作 — np.transpose 通过 axes 排列 shape ===
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.transpose(a))

a = np.arange(24).reshape((2, 3, 4))
print(np.transpose(a, (1, 0, 2)).shape)


# === 常用操作 — np.concatenate 沿现有轴连接数组 ===
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate((a, b), axis=0))
print(np.concatenate((a, b.T), axis=1))
print(np.concatenate((a, b), axis=None))


# === 常用操作 — np.stack 沿新轴连接数组 ===
a1 = np.arange(6).reshape((2, 3))
a2 = np.arange(10, 16).reshape((2, 3))
a3 = np.arange(20, 26).reshape((2, 3))
a4 = np.arange(30, 36).reshape((2, 3))
print(np.stack((a1, a2, a3, a4)).shape)
print(np.stack((a1, a2, a3, a4), axis=1).shape)
print(np.stack((a1, a2, a3, a4), axis=2).shape)


# === 矩阵运算 — np.dot 点积 ===
a = [1, 2, 3]
b = [1, 0, 2]
print(np.dot(a, b))

a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.dot(a, b))


# === 矩阵运算 — np.matmul 矩阵乘法（@ 操作符）===
a = np.array([[1, 0],
              [0, 1]])
b = np.array([[4, 1],
              [2, 2]])
print(np.matmul(a, b))
print(a @ b)


# === 比较函数 — np.greater（>）===
print(np.greater([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.greater(a, b))
print(a > b)


# === 比较函数 — np.greater_equal（>=）===
print(np.greater_equal([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.greater_equal(a, b))
print(a >= b)


# === 比较函数 — np.less（<）===
print(np.less([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.less(a, b))
print(a < b)


# === 比较函数 — np.less_equal（<=）===
print(np.less_equal([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.less_equal(a, b))
print(a <= b)


# === 比较函数 — np.equal（==）===
print(np.equal([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.equal(a, b))
print(a == b)


# === 比较函数 — np.not_equal（!=）===
print(np.not_equal([4, 2], [2, 2]))

a = np.array([[4, 2], [3, 1]])
b = np.array([[2, 2]])
print(np.not_equal(a, b))
print(a != b)


# === 数学函数 — np.sin 正弦 ===
print(np.sin(np.pi / 2))
print(np.sin(np.array((0, 30, 90)) * np.pi / 180))


# === 数学函数 — np.cos 余弦 ===
print(np.cos(np.pi / 2))
print(np.cos(np.array((0, 60, 90)) * np.pi / 180))


# === 数学函数 — np.tan 正切 ===
print(np.tan(-np.pi))
print(np.tan(np.array((0, 180)) * np.pi / 180))


# === 数学函数 — np.arcsin 反正弦 ===
print(np.arcsin(1))
print(np.arcsin(np.array([0.5, -0.5])))


# === 数学函数 — np.arccos 反余弦 ===
print(np.arccos(-1))
print(np.arccos(np.array([0.5, 1])))


# === 数学函数 — np.arctan 反正切 ===
print(np.arctan(1))
print(np.arctan(np.array([0, -1])))


# === 数学函数 — np.floor 向下取整 ===
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
print(np.floor(a))


# === 数学函数 — np.ceil 向上取整 ===
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
print(np.ceil(a))


# === 数学函数 — np.exp 指数（e 的 x 次方）===
# e 的 0 次方、e 的 1 次方、e 的 2 次方
print(np.exp([0, 1, 2]))


# === 数学函数 — np.log 自然对数 ===
print(np.log([1, np.e, np.e**2]))


# === 数学函数 — np.log2 以 2 为底的对数 ===
x = np.array([1, 2, 2**4])
print(np.log2(x))


# === 数学函数 — np.log10 以 10 为底的对数 ===
print(np.log10([1e-15, 1000]))


# === 统计函数 — np.max 最大值 ===
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
print(np.max(arr1))
print(np.max(arr1, axis=0))
print(np.max(arr1, axis=1))


# === 统计函数 — np.min 最小值 ===
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
print(np.min(arr1))
print(np.min(arr1, axis=0))
print(np.min(arr1, axis=1))


# === 统计函数 — np.mean 均值 ===
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
print(np.mean(arr1))
print(np.mean(arr1, axis=0))
print(np.mean(arr1, axis=1))


# === 统计函数 — np.var 方差 ===
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
print(np.var(arr1))
print(np.var(arr1, axis=0))
print(np.var(arr1, axis=1))


# === 统计函数 — np.std 标准差 ===
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
print(np.std(arr1))
print(np.std(arr1, axis=0))
print(np.std(arr1, axis=1))


# === 统计函数 — np.prod 乘积 ===
# 默认 axis=None 计算所有元素的乘积
print(np.prod([1, 2, 3, 4]))
print(np.prod([[1, 2], [3, 4]]))

print(np.prod([1, 2, 3, 4], initial=5))

print(np.prod([[1, 2], [3, 4]], axis=1))
print(np.prod([[1, 2], [3, 4]], axis=0))

print(np.prod([[1, 2], [3, 4]], axis=1, keepdims=True))
print(np.prod([[1, 2], [3, 4]], axis=0, keepdims=True))


# === 统计函数 — np.sum 求和 ===
# 默认 axis=None 计算所有元素的和
print(np.sum([1, 2, 3, 4]))
print(np.sum([[1, 2], [3, 4]]))

print(np.sum([1, 2, 3, 4], initial=5))

print(np.sum([[1, 2], [3, 4]], axis=1))
print(np.sum([[1, 2], [3, 4]], axis=0))

print(np.sum([[1, 2], [3, 4]], axis=1, keepdims=True))
print(np.sum([[1, 2], [3, 4]], axis=0, keepdims=True))


# === 查找函数 — np.nonzero 非零元素的索引 ===
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(x)
print(np.nonzero(x))
print(x[np.nonzero(x)])

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a > 3)
print(np.nonzero(a > 3))
print(a[np.nonzero(a > 3)])


# === 查找函数 — np.where 条件选择 / 返回索引 ===
a = np.arange(10)
print(np.where(a < 5, a, 10 * a))

# 三个参数：condition 成立返回 x，不成立返回 y
print(np.where([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]]))

# 只传第一个参数：返回符合条件的元素的索引
a = np.array([2, 4, 6, 8, 10])
print(np.where(a > 5))


# === 查找函数 — np.argwhere 按元素分组的非零元素索引 ===
x = np.arange(6).reshape(2, 3)
print(x)
print(x > 1)
print(np.argwhere(x > 1))


# === 查找函数 — np.maximum 逐元素最大值 ===
print(np.maximum([2, 3, 4], [1, 5, 2]))
print(np.maximum([[2, 3], [4, 5]], [[1, 5], [2, 6]]))


# === 查找函数 — np.minimum 逐元素最小值 ===
print(np.minimum([2, 3, 4], [1, 5, 2]))
print(np.minimum([[2, 3], [4, 5]], [[1, 5], [2, 6]]))


# === 查找函数 — np.argmax 最大值的索引 ===
a = np.arange(6).reshape(2, 3) + 10
print(a)

# 没有指定轴，则数组扁平化处理
print(np.argmax(a))

print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))


# === 查找函数 — np.argmin 最小值的索引 ===
a = np.arange(6).reshape(2, 3) + 10
print(a)

# 没有指定轴，则数组扁平化处理
print(np.argmin(a))

print(np.argmin(a, axis=0))
print(np.argmin(a, axis=1))


# === 随机函数 — np.random.normal 正态分布采样 ===
print(np.random.normal(3, 2.5, size=(2, 4)))


# === 随机函数 — np.random.randint 随机整数 ===
print(np.random.randint(2, size=10))         # 等价于下一行
print(np.random.randint(0, 2, size=10))      # 等价于上一行
print(np.random.randint(1, 4, size=(2, 3)))


# === 随机函数 — np.random.uniform 均匀分布采样 ===
print(np.random.uniform(2, size=10))         # 等价于下一行
print(np.random.uniform(0, 2, size=10))      # 等价于上一行
print(np.random.uniform(1, 4, size=(2, 3)))


# === 随机函数 — np.random.permutation 随机排列 ===
print(np.random.permutation(6))

arr1 = np.array([0, 1, 2, 3, 4, 5])
print(np.random.permutation(arr1))

arr2 = np.arange(10).reshape(5, 2)
print(np.random.permutation(arr2))


# === 随机函数 — np.random.seed 随机数种子 ===
np.random.seed(3)
print(np.random.uniform(1, 2, size=4))

np.random.seed(5)
print(np.random.uniform(1, 2, size=4))

np.random.seed(3)
print(np.random.uniform(1, 2, size=4))   # 与第一次 seed(3) 结果相同

np.random.seed()
print(np.random.uniform(1, 2, size=4))   # 不固定种子，每次不同


# === END p13 ===
