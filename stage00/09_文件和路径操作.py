# ============================================================
# 章节：文件和路径操作
# 内容：os 模块路径操作 + 文件读写 + with 语句
# ============================================================

import os
import time

# ---- 知识点：os.getcwd() 返回当前工作目录 ----
print(os.getcwd())


# ---- 知识点：os.listdir(path) 返回目录下所有文件/文件夹名的列表 ----
cwd = os.getcwd()
print(os.listdir(cwd))


# ---- 知识点：os.makedirs() 递归创建目录 ----
# exist_ok=False（默认）时，目标目录已存在会引发 FileExistsError
# 演示用，实际运行前确保 ./dir1 不存在，否则会报错
# os.makedirs('./dir1/dir2/dir3')


# ---- 知识点：os.path.basename(path) 返回路径最后一级名称（通常是文件名） ----
print(os.path.basename('./dir3/dir2/dir1/a.txt'))


# ---- 知识点：os.path.dirname(path) 返回路径的目录名 ----
print(os.path.dirname('./dir3/dir2/dir1/a.txt'))


# ---- 知识点：os.path.split(path) 将路径分割为 (dirname, basename) 元组 ----
print(os.path.split('./dir3/dir2/dir1/a.txt'))


# ---- 知识点：os.path.splitext(path) 将路径中扩展名分割出来，返回元组 ----
print(os.path.splitext('./dir3/dir2/dir1/a.txt'))


# ---- 知识点：os.path.exists(path) 判断路径是否存在 ----
p = r'D:\PythonFiles\p01.py'
print(os.path.exists(p))


# ---- 知识点：os.path.isfile(path) 判断路径是否为文件 ----
print(os.path.isfile("./dir3/dir2/dir1/a.txt"))


# ---- 知识点：os.path.isdir(path) 判断路径是否为目录 ----
print(os.path.isdir("./dir3/dir2/dir1"))


# ---- 知识点：os.path.join(path, *paths) 智能拼接一个或多个路径部分 ----
p1 = 'D:\\PythonFiles\\'
p2 = r'dir1\dir2\dir3'
p3 = 'p01.py'
print(os.path.join(p1, p2, p3))


# ============================================================
# 文件读写
# ============================================================

# ---- 知识点：open() 以默认模式（只读 'r'）打开文件，返回迭代器对象 ----
# 每次迭代返回文件中的一行数据
# 注意：以下代码需要 exam.txt 文件存在才能运行
# file = open('./exam.txt')
# print(next(file))
# for i in file:
#     print(i)


# ---- 知识点：file.read(size) 从文件中读取至多 size 个字符 ----
# size 为负值或 None 时读取至 EOF
# 需要 t01.txt 文件存在
# with open(r"./t01.txt") as file:
#     print(file.read(5))   # 读取前 5 个字符
#     print(file.read(2))   # 再读取 2 个字符
#     print(file.read())    # 读取剩余全部内容


# ---- 知识点：file.write(s) 将字符串写入文件，返回写入的字符数 ----
# mode='a' 表示追加模式
# with open(r"./t01.txt", mode='a') as file:
#     num = file.write('\nhello baby')
#     print(num)


# ---- 知识点：file.flush() 立即刷新缓冲区，将数据写入文件 ----
# file.close() 关闭文件时会自动刷新缓冲区
# file = open(r"./t01.txt", mode='a')
# file.write('\n123456789')
# time.sleep(5)  # 文件需要等到关闭文件时才会把数据从缓冲区写入文件
# file.close()   # 关闭文件，自动刷新缓冲区，数据才写入文件

# file = open(r"./t01.txt", mode='a')
# file.write('\n123456789')
# file.flush()   # 刷新缓冲区，数据立刻写入文件
# time.sleep(5)
# file.close()


# ---- 知识点：file.close() 关闭文件；关闭后再操作会引发 ValueError ----
# file = open(r'./t01.txt')
# print(file.read())
# file.close()
# file.read()  # 引发 ValueError


# ---- 知识点：file.seek(offset) 移动文件指针到指定位置 ----
# with open('./exam.txt', mode='w+') as file:
#     file.write('hello\nworld')
#     file.seek(7)           # 移动指针到第 7 个字节
#     print(file.read())     # 'world'


# ============================================================
# with 语句
# ============================================================

# ---- 知识点：传统写法 —— 若 open 之后 close 之前发生异常，文件可能不被关闭 ----
# file = open(r'./t01.txt', mode='w')
# file.write('hello world')
# ...
# file.close()


# ---- 知识点：try/finally 写法 —— 确保 close 一定被执行 ----
# file = open(r'./t01.txt', mode='w')
# try:
#     file.write('hello world')
#     ...
# finally:
#     file.close()


# ---- 知识点：with 语句 —— 最简洁优雅的写法，自动关闭文件 ----
# with 块结束后，无论是否发生异常，文件都会被自动关闭
# with open(r'./t01.txt', mode='w') as file:
#     file.write('hello world')
#     ...


# === END p11 ===
