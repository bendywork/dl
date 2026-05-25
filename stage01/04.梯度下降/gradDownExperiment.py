from typing import List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import platform


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


class GradDown:

    def __init__(self):
        self.theta = None
        self.alpha = 0.01
        self.inner_func = None
        self.X =  None
        self.f_current = None
        self.inner_iter = 0

    def gradient_descent(self, X, alpha=0.01, max_iter=100) -> tuple[List[int], List[int]]:

        if not X:
            raise ValueError("X can not be None")

        self.X = X

        self.inner_iter = max_iter

        def target_func(theta):
            return 0.5 * (theta - 0.25) ** 2

        self.inner_func = target_func

        def target_func_grad(theta):
            return 0.5 * 2 * (theta - 0.25)

        gradient_x = []
        gradient_y = []

        if not alpha:
            self.alpha = alpha
        else:
            self.alpha = 0.1

        f_change = target_func(self.X)
        self.f_current = f_change
        gradient_x.append(self.X)
        gradient_y.append(self.f_current)
        iter_num = 0

        while f_change > 1e-10 and iter_num < self.inner_iter:
            iter_num += 1
            self.X = self.X - self.alpha * target_func_grad(self.X)
            temp = target_func(self.X)
            f_change = np.abs(self.f_current - temp)
            self.f_current = temp
            gradient_x.append(self.X)
            gradient_y.append(self.f_current)

        print(u"最终结果为:(%.5f, %.5f)" % (self.X, self.f_current))
        print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
        print(gradient_x)
        return gradient_x, gradient_y


    def paint(self, g_x=None, g_y=None):
        if g_y is None:
            g_y = []
        if g_x is None:
            g_x = []

        here_x = np.arange(-4, 4.5, 0.05)
        here_y = np.array(list(map(lambda t: self.inner_func(t), here_x)))

        plt.figure(facecolor='w')
        plt.plot(here_x, here_y, 'r-', linewidth=2)
        plt.plot(g_x, g_y, 'bo--', linewidth=2)
        plt.title(u'函数$y=0.5 * (θ - 0.25)^2$; \n学习率:%.3f; 最终解:(%.3f, %.3f);迭代次数:%d' % (self.alpha, self.X, self.f_current, self.inner_iter))
        plt.show()


if __name__ == '__main__':
    gd = GradDown()
    gradient_x , gradient_y = gd.gradient_descent(X = 4, alpha = 0.1)
    gd.paint(g_x=gradient_x, g_y=gradient_y)
