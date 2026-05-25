import platform

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
# 解决中文显示问题 - 根据操作系统选择字体
system_name = platform.system()
if system_name == 'Darwin':  # macOS
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
elif system_name == 'Windows':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
else:  # Linux
    mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

mpl.rcParams['axes.unicode_minus'] = False



# ============================================
# 一维梯度下降示例
# ============================================
def gradient_descent_1d():
    """一维函数的梯度下降法实现"""

    # 目标函数
    def f1(x):
        return 0.5 * (x - 0.25) ** 2

    # 导函数
    def h1(x):
        return 0.5 * 2 * (x - 0.25)

    # 使用梯度下降法求解
    gd_x = []
    gd_y = []
    x = 4  # 初始化的x
    alpha = 0.1  # 步长越小，可能需要的迭代次数就要愈多
    f_change = f1(x)
    f_current = f_change
    gd_x.append(x)
    gd_y.append(f_current)
    iter_num = 0

    while f_change > 1e-10 and iter_num < 100:  # 停止迭代的条件 差值小于1e-10或者迭代次数大于等于50次
        iter_num += 1
        x = x - alpha * h1(x)
        tmp = f1(x)
        f_change = np.abs(f_current - tmp)
        f_current = tmp
        gd_x.append(x)
        gd_y.append(f_current)

    print(u"最终结果为:(%.5f, %.5f)" % (x, f_current))
    print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
    print(gd_x)

    # 构建数据
    X = np.arange(-4, 4.5, 0.05)
    Y = np.array(list(map(lambda t: f1(t), X)))

    # 画图
    plt.figure(facecolor='w')
    plt.plot(X, Y, 'r-', linewidth=2)
    plt.plot(gd_x, gd_y, 'bo--', linewidth=2)
    plt.title(u'函数$y=0.5 * (θ - 0.25)^2$; \n学习率:%.3f; 最终解:(%.3f, %.3f);迭代次数:%d' % (alpha, x, f_current, iter_num))
    plt.show()


# ============================================
# 二维梯度下降示例 1
# ============================================
def gradient_descent_2d_example1():
    """二维函数的梯度下降法实现 - 示例1"""

    # 目标函数
    def f2(x, y):
        return 0.6 * (x + y) ** 2 - x * y

    # 偏导数
    def hx2(x, y):
        return 0.6 * 2 * (x + y) - y

    def hy2(x, y):
        return 0.6 * 2 * (x + y) - x

    # 使用梯度下降法求解
    GD_X1 = []
    GD_X2 = []
    GD_Y = []

    x1 = 4
    x2 = 4
    alpha = 0.1
    f_change = f2(x1, x2)
    f_current = f_change
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)

    iter_num = 0
    while f_change > 1e-10 and iter_num < 100:
        iter_num += 1
        prex1 = x1
        prex2 = x2
        x1 = x1 - alpha * hx2(prex1, prex2)
        x2 = x2 - alpha * hy2(prex1, prex2)

        tmp = f2(x1, x2)
        f_change = np.abs(f_current - tmp)

        f_current = tmp
        GD_X1.append(x1)
        GD_X2.append(x2)
        GD_Y.append(f_current)

    print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f_current))
    print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
    print(GD_X1)

    # 构建数据
    X1 = np.arange(-4, 4.5, 0.2)
    X2 = np.arange(-4, 4.5, 0.2)
    X1, X2 = np.meshgrid(X1, X2)
    Y = np.array(list(map(lambda t: f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
    Y.shape = X1.shape

    # 画图
    fig = plt.figure(facecolor='w')
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
    ax.plot(GD_X1, GD_X2, GD_Y, 'ro--')

    ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
    plt.show()


# ============================================
# 二维梯度下降示例 2
# ============================================
def gradient_descent_2d_example2():
    """二维函数的梯度下降法实现 - 示例2"""

    # 目标函数
    def f2(x, y):
        return 0.15 * (x + 0.5) ** 2 + 0.25 * (y - 0.25) ** 2 + 0.35 * (1.5 * x - 0.2 * y + 0.35) ** 2

    # 偏导数
    def hx2(x, y):
        return 0.15 * 2 * (x + 0.5) + 0.25 * 2 * (1.5 * x - 0.2 * y + 0.35) * 1.5

    def hy2(x, y):
        return 0.25 * 2 * (y - 0.25) - 0.25 * 2 * (1.5 * x - 0.2 * y + 0.35) * 0.2

    # 使用梯度下降法求解
    GD_X1 = []
    GD_X2 = []
    GD_Y = []
    x1 = 4
    x2 = 4
    alpha = 0.01
    f_change = f2(x1, x2)
    f_current = f_change
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)

    iter_num = 0
    while f_change > 1e-10 and iter_num < 100:
        iter_num += 1
        prex1 = x1
        prex2 = x2
        x1 = x1 - alpha * hx2(prex1, prex2)
        x2 = x2 - alpha * hy2(prex1, prex2)

        tmp = f2(x1, x2)
        f_change = np.abs(f_current - tmp)
        f_current = tmp
        GD_X1.append(x1)
        GD_X2.append(x2)
        GD_Y.append(f_current)

    print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x1, x2, f_current))
    print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
    print(GD_X1)

    # 构建数据
    X1 = np.arange(-4, 4.5, 0.2)
    X2 = np.arange(-4, 4.5, 0.2)
    X1, X2 = np.meshgrid(X1, X2)
    Y = np.array(list(map(lambda t: f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
    Y.shape = X1.shape

    # 画图
    fig = plt.figure(facecolor='w')
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
    ax.plot(GD_X1, GD_X2, GD_Y, 'ko--')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(u'函数;\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
    plt.show()


if __name__ == '__main__':
    # 运行三个梯度下降示例
    print("=" * 50)
    print("示例1: 一维梯度下降")
    print("=" * 50)
    gradient_descent_1d()

    print("\n" + "=" * 50)
    print("示例2: 二维梯度下降 (示例1)")
    print("=" * 50)
    gradient_descent_2d_example1()

    print("\n" + "=" * 50)
    print("示例3: 二维梯度下降 (示例2)")
    print("=" * 50)
    gradient_descent_2d_example2()
