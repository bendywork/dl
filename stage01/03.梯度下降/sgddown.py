# encoding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 创建训练数据集
# 假设训练学习一个线性函数y = 2.33x
EXAMPLE_NUM = 100  # 训练总数
BATCH_SIZE = 10  # mini_batch训练集大小
TRAIN_STEP = 150  # 训练次数
LEARNING_RATE = 0.0001  # 学习率
X_INPUT = np.arange(EXAMPLE_NUM) * 0.1  # 生成输入数据X
Y_OUTPUT_CORRECT = 5 * X_INPUT  # 生成训练正确输出数据


# 构造训练的函数
def train_func(X, K):
    result = K * X
    return result


# BGD (Batch Gradient Descent)
# 参数初始化值
k_BGD = 0.0
# 记录迭代数据用于作图
k_BGD_RECORD = []
for step in range(TRAIN_STEP):
    SUM_BGD = 0
    for index in range(len(X_INPUT)):
        # 损失函数J(K)=1/(2m)*sum(KX-y_true)^2
        # J(K)的梯度 = (KX-y_true)*X
        SUM_BGD += (train_func(X_INPUT[index], k_BGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    # 这里实际上要对SUM_BGD求均值也就是要乘上个1/m 但是 LEARNING_RATE*1/m1 还是一个常数 所以这里就直接用一个常数表示
    k_BGD -= LEARNING_RATE * SUM_BGD
    k_BGD_RECORD.append(k_BGD)

print("BGD迭代次数:", len(k_BGD_RECORD))

# SGD (Stochastic Gradient Descent)
k_SGD = 0.0
k_SGD_RECORD = []
for step in range(TRAIN_STEP):
    index = np.random.randint(len(X_INPUT))
    SUM_SGD = (train_func(X_INPUT[index], k_SGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_SGD -= LEARNING_RATE * SUM_SGD
    k_SGD_RECORD.append(k_SGD)

# MBGD (Mini-Batch Gradient Descent)
k_MBGD = 0.0
k_MBGD_RECORD = []
for step in range(TRAIN_STEP):
    SUM_MBGD = 0
    index_start = np.random.randint(len(X_INPUT) - BATCH_SIZE)
    for index in np.arange(index_start, index_start + BATCH_SIZE):
        SUM_MBGD += (train_func(X_INPUT[index], k_MBGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_MBGD -= LEARNING_RATE * SUM_MBGD
    k_MBGD_RECORD.append(k_MBGD)

# 作图比较三种方法
plt.figure(figsize=(10, 6))
plt.plot(np.arange(TRAIN_STEP), np.array(k_BGD_RECORD), label='BGD')
plt.plot(np.arange(TRAIN_STEP), k_SGD_RECORD, label='SGD')
plt.plot(np.arange(TRAIN_STEP), k_MBGD_RECORD, label='MBGD')
plt.legend()
plt.ylabel('K')
plt.xlabel('step')
plt.title('Comparison of BGD, SGD, and MBGD')
plt.show()

# SGD with more iterations
k_SGD_extended = 0.0
k_SGD_EXTENDED_RECORD = []
for step in range(TRAIN_STEP * 20):
    index = np.random.randint(len(X_INPUT))
    SUM_SGD = (train_func(X_INPUT[index], k_SGD_extended) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_SGD_extended -= LEARNING_RATE * SUM_SGD
    k_SGD_EXTENDED_RECORD.append(k_SGD_extended)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(TRAIN_STEP * 20), k_SGD_EXTENDED_RECORD, label='SGD')
plt.legend()
plt.ylabel('K')
plt.xlabel('step')
plt.title('SGD with Extended Iterations')
plt.show()

print("\n说明:")
print("BGD：可以在迭代步骤上可以快速接近最优解，但是其时间消耗相对其他两种是最大的，因为每一次更新都需要遍历完所有数据。")
print("\nSGD：参数更新是最快的，因为每遍历一个数据都会做参数更新，但是由于没有遍历完所有数据，所以其路线不一定是最佳路线，甚至可能会反方向巡迹，不过其整体趋势是往最优解方向行进的，随机速度下降还有一个好处是有一定概率跳出局部最优解，而BGD会直接陷入局部最优解。")
print("\nMBGD：以上两种都是MBGD的极端，MBGD是中庸的选择，保证参数更新速度的前提下，用过小批量又增加了其准备度，所以大多数的梯度下降算法中都会使用到小批量梯度下降。")
