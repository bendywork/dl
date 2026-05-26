# ============================================================
# 章节：Matplotlib —— Python 2D 绘图库
# 涵盖：折线图、画布、坐标轴设置、图例、文字说明、
#        散点图、条形图、imshow、子图、画中图、保存图像
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# plt.plot() — 折线图基础
# ============================================================

# === 折线图：基本参数示例 ===
# color / linestyle / linewidth / marker / markerfacecolor / markersize

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.plot(x, y, color='skyblue', linestyle='-.', linewidth=2,
         marker='h', markerfacecolor='gold', markersize=8)
plt.show()


# ============================================================
# plt.figure() — 创建画布
# ============================================================

# === 隐式创建 figure（自动创建，多条线画在同一画布）===

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 2
y2 = x ** 2
y3 = np.sin(x)

plt.plot(x, y1, color='gold')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='green')
plt.show()


# === 显式创建 figure（多个画布分开显示）===

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 2
y2 = x ** 2
y3 = np.sin(x)

plt.figure()          # 画布1：y1 和 y2
plt.plot(x, y1, color='gold')
plt.plot(x, y2, color='red')

plt.figure()          # 画布2：y3
plt.plot(x, y3, color='green')
plt.show()


# === plt.figure() 常用参数：num / figsize / dpi / facecolor ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure(num=3, figsize=(7, 3), dpi=72, facecolor="red")
plt.plot(x, y)

plt.figure(num="画布二", figsize=(7, 3), dpi=72, facecolor="green")
plt.plot(x, y)
plt.show()


# ============================================================
# 中文 / 负号显示问题
# ============================================================

# === 修复中文和负号显示乱码 ===
# 将以下两行放到 matplotlib 使用的最前面即可

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 设置坐标轴
# ============================================================

# === 坐标轴标签：xlabel / ylabel ===

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.xlabel("这是x轴")
plt.ylabel("这是y轴", fontsize=14)
plt.show()


# === 坐标轴刻度：xticks / yticks ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()   # 画布1：自定义 y 轴刻度标签
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=["a", "b", "c", "d", "e"])

plt.figure()   # 画布2：去掉 x 轴刻度（轴线保留）
plt.plot(x, y)
plt.xticks(ticks=[])

plt.figure()   # 画布3：关闭整个坐标体系
plt.plot(x, y)
plt.axis("off")

plt.figure()   # 画布4：自定义 x 轴刻度（只传 ticks）
plt.plot(x, y)
new_xticks = np.linspace(-4, 4, 9)
plt.xticks(ticks=new_xticks)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1])
plt.show()


# === 坐标边框颜色：gca() + spines ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=["a", "b", "c", "d", "e"])

ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['left'].set_color('red')
ax.spines['bottom'].set_color('green')
plt.show()


# === 指定边框为坐标轴：set_ticks_position() ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=['a', 'b', 'c', 'd', 'e'])

ax = plt.gca()
ax.spines['right'].set_color('skyblue')
ax.spines['top'].set_color('blue')
ax.spines['left'].set_color('red')
ax.spines['bottom'].set_color('green')
# 上边框作为 x 轴
ax.xaxis.set_ticks_position('top')
# 右边框作为 y 轴
ax.yaxis.set_ticks_position('right')
plt.show()


# === 移动坐标边框：set_position() ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=['a', 'b', 'c', 'd', 'e'])

ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['left'].set_color('red')
ax.spines['bottom'].set_color('green')
# 左边框移动到 x=0 的位置
ax.spines['left'].set_position(('data', 0))
# 底边框移动到 y=-0.1 的位置
ax.spines['bottom'].set_position(('data', -0.1))
plt.show()


# ============================================================
# plt.legend() — 创建图例
# ============================================================

# === 图例：通过 label 参数和 legend() 联动 ===

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = np.sin(x)

plt.figure()
plt.plot(x, y1, color='blue', label='直线')
plt.plot(x, y2, color='green', label='曲线')
plt.legend(loc='lower right', fontsize=14, frameon=True, edgecolor='red', facecolor='yellow')


# === 图例：手动指定 labels ===

plt.figure()
plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='green')
plt.legend(labels=['直线', '曲线'])


# === 图例：指定 handles 控制柄 ===

plt.figure()
line1, = plt.plot(x, y1, color='blue', label='直线')
line2, = plt.plot(x, y2, color='green', label='曲线')
print(line1)
print(line2)
# 只为第一条线创建图例，且标签改为'线条1'
plt.legend(handles=[line1, ], labels=['线条1', ])
plt.show()


# ============================================================
# plt.text() — 文字说明
# ============================================================

# === 在图中添加文字 ===

x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.text(x=1.1, y=0.6, s="y=sinx", size=16, color="red")
plt.show()


# ============================================================
# plt.scatter() — 散点图
# ============================================================

# === 散点图：两组数据 ===

x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)
x2 = np.random.normal(0, 1, 100)
y2 = np.random.normal(0, 1, 100)

plt.scatter(x1, y1, s=90, c='green', marker='D', alpha=0.2, linewidths=2, edgecolors='red')
plt.scatter(x2, y2, s=90, c='yellow', marker='D', alpha=0.8, linewidths=2, edgecolors='black')
plt.show()


# ============================================================
# plt.bar() — 条形图
# ============================================================

# === 条形图：正负对称条形图 ===

x = np.arange(1, 11)
h1 = np.random.randint(20, 35, 10)
h2 = np.random.randint(15, 40, 10)

plt.figure()
plt.bar(x, +h1, bottom=0.5)
plt.bar(x, -h2, bottom=-0.5)
for i in range(len(x)):
    plt.text(x[i], h1[i] + 0.5, h1[i], ha='center')
    plt.text(x[i], -h2[i] - 0.5, h2[i], va='top', ha='center')
plt.yticks(ticks=range(-40, 31, 10), labels=[40, 30, 20, 10, 0, 10, 20, 30])


# === 条形图：并列条形图 ===

plt.figure()
x = np.arange(1, 25, 2.5)
h1 = np.random.randint(20, 35, 10)
h2 = np.random.randint(15, 40, 10)

plt.bar(x, h1, align='edge')
plt.bar(x - 0.4, h2)
for i in range(len(x)):
    plt.text(x[i] + 0.4, h1[i], h1[i], size=8, ha='center')
    plt.text(x[i] - 0.4, h2[i], h2[i], size=8, ha='center')
plt.xticks(ticks=x)
plt.show()


# ============================================================
# plt.imshow() — 数据转图像
# ============================================================

# === imshow：显示二维数组为图像 ===

plt.figure()
data = np.array([[0, 50, 200], [200, 100, 0], [0, 150, 200]])
plt.imshow(data)
# plt.imshow(data, cmap="Greys")
# plt.imshow(data, cmap="Greys", alpha=0.3)
plt.show()


# ============================================================
# plt.subplot() — 创建子图
# ============================================================

# === 子图：2×2 布局 ===

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
y3 = np.sin(x)
y4 = np.tan(x)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x, y1)

plt.subplot(2, 2, 2)
plt.plot(x, y2)

plt.subplot(2, 2, 3)
plt.plot(x, y3)

plt.subplot(2, 2, 4)
plt.plot(x, y4)


# === 子图：混合行列布局 ===

plt.figure()
# 2行1列，第1个子图
plt.subplot(2, 1, 1)
plt.plot(x, y1)

# 2行3列，第4个子图
plt.subplot(2, 3, 4)
plt.plot(x, y2)

# 2行3列，第5个子图
plt.subplot(2, 3, 5)
plt.plot(x, y3)

# 2行3列，第6个子图
plt.subplot(2, 3, 6)
plt.plot(x, y4)

plt.show()


# ============================================================
# plt.axes() — 画中图（图中图）
# ============================================================

# === 画中图：通过指定相对位置和宽高定制多个坐标轴 ===

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2
y3 = np.sin(x)

plt.figure()

# [left, bottom, width, height]，均为相对画布的比例
plt.axes([0.1, 0.1, 0.8, 0.8])
plt.title('直线')
plt.plot(x, y1)

plt.axes([0.2, 0.6, 0.25, 0.25])
plt.title('抛物线')
plt.plot(x, y2)

plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(x, y3)
plt.title('余弦曲线')

plt.show()


# ============================================================
# plt.savefig() — 保存图像
# ============================================================

# === 动态绘图并保存最终图像 ===

y2 = []
y1 = np.linspace(-3, 3, 50)
for i in y1 ** 2:
    y2.append(i)
    plt.clf()           # 清除上一次的数据
    plt.plot(y2)
    plt.pause(0.1)      # 暂停 0.1 秒

plt.savefig('../../base/images/14_matplotlib_动态绘图示例.png')   # 保存图像到指定路径

# === END p15 ===
