
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



# SGD (Stochastic Gradient Descent)
k_SGD = 0.0
k_SGD_RECORD = []
for step in range(TRAIN_STEP):
    index = np.random.randint(len(X_INPUT))
    SUM_SGD = (train_func(X_INPUT[index], k_SGD) - Y_OUTPUT_CORRECT[index]) * X_INPUT[index]
    k_SGD -= LEARNING_RATE * SUM_SGD
    k_SGD_RECORD.append(k_SGD)



plt.figure(figsize=(10, 6))
plt.clf()
plt.plot(np.arange(TRAIN_STEP), k_SGD_RECORD, label='SGD')
plt.legend()
plt.ylabel('K')
plt.xlabel('step')
plt.title('SGD')
plt.show()