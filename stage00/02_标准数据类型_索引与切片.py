# ============================================================
# 章节：标准数据类型、索引 & 切片
# 来源：p04_标准数据类型，索引 & 切片.pdf
# Python 3 中有六种标准数据类型：
#   数字（Number）、字符串（String）、列表（List）、
#   元组（Tuple）、字典（Dictionary）、集合（Set）
# ============================================================


# === 数字类型 — int() 转换 ===

print(int())       # 0
print(int(3))      # 3
print(int(-3))     # -3
print(int(3.99))   # 3  （直接截断，不四舍五入）
print(int(-3.99))  # -3
print(int(True))   # 1
print(int(False))  # 0
print(int('12'))   # 12
print(int('-12'))  # -12

# 当传的是字符串时, 必须是整数形式的
# int('12.1')  # ValueError


# === 数字类型 — float() 转换 ===

print(float())        # 0.0
print(float(3))       # 3.0
print(float(-3))      # -3.0
print(float(3.99))    # 3.99
print(float(-3.99))   # -3.99
print(float(True))    # 1.0
print(float(False))   # 0.0
print(float('12'))    # 12.0
print(float('-12'))   # -12.0
print(float('12.1'))  # 12.1


# === 数字类型 — bool() 转换 ===

"""
数字0, 0.0, 0j, False,
空字符串, 空列表, 空元组,
空字典, 空集合, 关键字None
...
以上这些数据bool判定为False,
其它通常判定为True
"""
print(bool())        # False
print(bool(0))       # False
print(bool(0.0))     # False
print(bool(0j))      # False
print(bool(False))   # False
print(bool(''))      # False
print(bool([]))      # False
print(bool(()))      # False
print(bool({}))      # False
print(bool(set()))   # False
print(bool(None))    # False
print(bool(' '))     # True
print(bool('None'))  # True
print(bool('False')) # True


# === 数字类型 — complex() 转换 ===

print(complex())           # 0j
print(complex(3.2, 4))     # (3.2+4j)
print(complex(3.2))        # (3.2+0j)
print(complex('3.2'))      # (3.2+0j)
print(complex("3.2+4j"))   # (3.2+4j)


# === 字符串 — 定义方式 ===

s1 = '这是一个单行字符串'

s2 = "这是一个单行字符串"

s3 = '''这是一个
多行字符串'''

s4 = """这是一个
多行字符串"""

""" 这是一个多行注释,
它会被解释器无视 """


# === 字符串 — str() 转换 ===

print(str())       # ''
print(str(1234))   # '1234'
print(str(-1.23))  # '-1.23'


# === 字符串 — 转义字符与 Raw 字符串 ===

print('https:\\www.example.com\nuxy\tngj')
print(r'https:\\www.example.com\nuxy\tngj')
print(R'https:\\www.example.com\nuxy\tngj')


# === 字符串 — % 格式化 ===

print('它说它叫%s, 今年%d岁, 每天睡%f小时!' % ('旺财', 2, 8.5))

# %.nf 表示精确到小数点后n位
print('今天买了%s斤青菜, %s元/斤, 花了%.2f元!' % (3.5, 2.59, 3.5*2.59))


# === 字符串 — format() 方法格式化 ===

name = '旺财'
age1 = 2
age2 = 3

# 传位置参数, 实参按照从左往右的顺序传入占位符{}
print('它说它叫{}, 它今年{}岁, 它宝宝{}个月了!'.format(name, age1, age2))

# 传关键字参数
print('它说它叫{n}, 它今年{a1}岁, 它宝宝{a2}个月了!'.format(a1=age1, n=name, a2=age2))

# 根据实参的下标传参
print('它说它叫{1}, 它今年{0}岁, 它宝宝{2}个月了!'.format(age1, name, age2))

# {:.nf} 表示精确到小数点后n位
print('今天买了{}斤青菜, {}元/斤, 花了{:.2f}元!'.format(3.5, 2.59, 3.5*2.59))


# === 字符串 — f-string 格式化 ===

name = '旺财'
age1 = 2
age2 = 3

print(f'它说它叫{name},\n它{age1}岁,\n它宝宝{age2}个月了!')
print(fr'它说它叫{name},\n它{age1}岁,\n它宝宝{age2}个月了!')

# {:.nf} 表示精确到小数点后n位
print(f'今天买了{3.5}斤青菜, {2.59}元/斤, 花了{3.5*2.59:.2f}元!')


# === 字符串方法 — replace() ===

s = "Line1 Line2 Line4"

# 用 "b" 替换所有的 "Li"
print(s.replace("Li", "b"))

# 用 "b" 替换 "Li" 最多2次
print(s.replace("Li", "b", 2))


# === 字符串方法 — strip() ===

# 删除字符串两边的空白符
str1 = ' \thello wrold h \n'
print(str1.strip())

# 删除字符串两边的'o'字符
str2 = "ooho hello wrold"
print(str2.strip('o'))

# 删除字符串两边的'c','w','o','m'字符
str3 = 'www.example.com'
print(str3.strip("cwom"))


# === 字符串方法 — startswith() ===

str1 = "hello world"
print(str1.startswith("h"))
print(str1.startswith("he"))
print(str1.startswith("wo"))
print(str1.startswith("wo", 6))
print(str1.startswith(("wo", "h")))


# === 字符串方法 — endswith() ===

str1 = "hello world"
print(str1.endswith("d"))
print(str1.endswith("ld"))
print(str1.endswith("lo"))
print(str1.endswith("lo", 1, 5))
print(str1.endswith(("d", "lo")))


# === 字符串方法 — isdigit() ===

string = '1234'
print(string.isdigit())  # True

string = '-123'
print(string.isdigit())  # False

string = '1.23'
print(string.isdigit())  # False


# === 字符串方法 — split() ===

s = " Line1  \nLine2   \tLine3"

print(s.split('Li'))
print(s.split(' '))
print(s.split())
print(s.split('Li', 2))


# === 字符串方法 — join() ===

s = '-.'

s1 = 'hello world'
print(s.join(s1))

s2 = ['1', '2', '3', '4']
print(s.join(s2))

s3 = ('1', '2', '3', '4')
print(s.join(s3))

# 字典作为iterable, 只有键参与迭代
s4 = {'height': 175, 'weight': 65}
print(s.join(s4))

s5 = {'5', 'hello', '789', 'world'}
print(s.join(s5))


# === 字符串方法 — count() ===

s = "hello world"
print(s.count('l'))
print(s.count('l', 3))
print(s.count('l', 3, 6))
print(s.count('l', 4, 6))


# === 字符串方法 — find() / rfind() / index() / rindex() ===

s = 'hello world'

print(s.find('l'))
print(s.rfind('l'))
print(s.find('lo'))
print(s.rfind('lo'))

print(s.index('l'))
print(s.rindex('l'))
print(s.index('lo'))
print(s.rindex('lo'))

print(s.find('ol'))   # -1
print(s.rfind('ol'))  # -1


# === 字符串方法 — 大小写转换 ===

s = '你好hELlo wo?rLD世界TuP'
print(s.capitalize())
print(s.title())
print(s.upper())
print(s.lower())
print(s.swapcase())


# === 列表 — 定义 ===

list0 = []
list1 = ['China', 1997, 2000]
list2 = [1, 2, 3, 4, 5]
list3 = ["a", "b", "c", "d"]
list4 = ['red', 'green', 'blue', 'yellow', 'white', 'black']


# === 列表 — 通过索引修改单个元素 ===

lst = [567, 'hello', 78.9, 'world', False]

"""
针对一个元素:
格式:  lst[index] = object
"""
lst[2] = 9.87
lst[3] = 'dlrow'
print(lst)


# === 列表 — 通过切片修改多个元素 ===

lst = [567, 'hello', 78.9, 'world', False]

"""
针对多个元素:
格式:  lst[start: end: step] = iterable
"""
# 1 vs 1
lst[2:3] = [9.87]

# n vs n
lst[2:4] = [9.87, 'dlrow']

# step为1, 可以 1 vs n
lst[2:3] = [7, 8, 9]

# step为1, 可以 n vs m
lst[2:4] = [1, 2, 3]
lst[1:4] = [1, 2]

# step为1, 可以 1 vs 0
lst[2:3] = []

# step为1, 可以 n vs 0
lst[1:4] = []

# step不为1, 只能 n vs n
lst = [567, 'hello', 78.9, 'world', False]
lst[1::2] = ['a', 'b']

# 插入一个元素
lst = [567, 'hello', 78.9, 'world', False]
lst[0:0] = ['a']
lst[1:1] = ['b']
lst[len(lst):] = ['c']

# 插入多个元素
lst = [1, 2, 3, 4, 5]
lst[0:0] = ['a', 'b', 'c']
lst[1:1] = ['d', 'f']
lst[len(lst):] = ['x', 'y', 'z']
print(lst)


# === 列表 — list() 转换 ===

print(list())
print(list("hello"))
print(list((1, 2, 3)))

# 字典作为一个iterable, 只有键参与迭代
print(list({1: 2, 3: 4}))
print(list({'a', 'b', 'c', 789, 456}))


# === 列表方法 — append() ===

lst = [1, 2, 3]

lst.append(4)
print(lst)

lst.append([5, 6])
print(lst)


# === 列表方法 — extend() ===

lst = [1, 2, 3]

lst.extend([5, 6])
print(lst)


# === 列表方法 — insert() ===

lst = [1, 2, 3, 4]
lst.insert(1, ['a', 'b'])
print(lst)


# === 列表方法 — sort() 与内置函数 sorted() ===

lst = [1, 2, -5, -3]
# 升序排序
lst.sort()
print(lst)

lst = [1, 2, -5, -3]
# 降序排序
lst.sort(reverse=True)
print(lst)

# chr(i) 返回Unicode码位为指定整数的字符
# ord(c) 返回指定字符对应的Unicode码位
print(chr(97))   # 'a'
print(ord('a'))  # 97

# 字符串在大小比较时是逐个字符进行比较的（根据字符在编码表里的位置）
lst = ['10', '2', '1', '-3', '101']
lst.sort()
print(lst)

# abs(number) 内置函数，返回number的绝对值
print(abs(9))     # 9
print(abs(9.87))  # 9.87
print(abs(0))     # 0
print(abs(-9))    # 9
print(abs(-9.87)) # 9.87
print(abs(True))  # 1
print(abs(False)) # 0
print(abs(3+4j))  # 求模, 5.0

"""
对lst中的元素按照绝对值的大小降序排序

把lst中的每个元素依次作为实参传递给key所指定的函数去调用, 即:
abs(1), abs(2), abs(-5), abs(-3)
返回值分别为: 1, 2, 5, 3
根据返回值的大小对原数据进行排序
"""
lst = [1, 2, -5, -3]
lst.sort(key=abs, reverse=True)
print(lst)


# === 内置函数 sorted() ===

lst = [1, 2, -5, -3]

# 升序排序
print(sorted(lst))

# 降序排序
print(sorted(lst, reverse=True))

# 对lst中的元素按照绝对值的大小降序排序
print(sorted(lst, key=abs, reverse=True))

# 对字符串排序
print(sorted('hello world'))


# === 列表方法 — reverse() ===

lst = [1, 3, 5, 2]
lst.reverse()  # inplace
print(lst)

lst = [1, 3, 5, 2]
print(lst[::-1])  # copy（切片方式，不改变原列表）


# === 列表方法 — count() ===

lst = [1, 2, 3, '23', [2, 4]]
print(lst.count(2))  # 1


# === 列表方法 — index() ===

lst = [1, 2, 3, 2, '23', [2, 4]]
print(lst.index(2))
# lst.index(2, 4)  # ValueError


# === 列表方法 — pop() ===

lst = [567, 'hello', True, False, 456]
print(lst.pop(1))  # 'hello'
print(lst)         # [567, True, False, 456]


# === 列表方法 — remove() ===

lst = [1, 2, 4, 2, 3, 3]

lst.remove(2)
lst.remove(2)
print(lst)


# === 列表方法 — copy() ===

lst = [567, 'hello', True, False, 456]
new_lst = lst.copy()
print(new_lst)


# === 列表方法 — clear() ===

lst = [567, 'hello', True, False, 456]
lst.clear()
print(lst)  # []


# === 元组 — 定义 ===

# 空元组
tup = ()

# 元组中只有一个元素时, 逗号不能省略
tup = (789,)

# 这不是元组，仍为数字789
tup = (789)

# 封包
tup = 'China', 1997, 2000

tup = ('China', 1997, 2000)

# 元组是不可变的, 但其中的可变成员仍然可以被改变
tup = (456, 'hello', ([789, 'world'],))
tup[-1][0][0] = 987
print(tup)


# === 元组 — tuple() 转换 ===

print(tuple())
print(tuple("hello"))
print(tuple([1, 2, 3]))

# 字典作为一个iterable, 只有键参与迭代
print(tuple({1: 2, 3: 4}))
print(tuple({'a', 'b', 'c', 789, 456}))


# === 元组方法 — count() / index() ===

tup = (1, 2, 3, 2, '23', [2, 4])
print(tup.count(2))

tup = (1, 2, 3, 2, '23', [2, 4])
print(tup.index(2))
# print(tup.index(2, 4))  # ValueError


# === 字典 — 多种创建方式 ===

# ① 直接在空字典里写键值对
d = {'name': 'Tom', 'age': 28}
print(d)

# ② 定义空字典，再添加键值对
d = {}
d['name'] = 'Tom'
d['age'] = 28
print(d)

# ③ 把键值对作为关键字参数传入
d = dict(name='Tom', age=28)
print(d)

# ④ 用可迭代对象来构建字典
d = dict([('name', 'Tom'), ('age', 28)])
print(d)

# ⑤ 用 zip 映射结构来构建字典
d = dict(zip(['name', 'age'], ['Tom', 28]))
print(d)

print(dict())
print(dict(one=1, two=2, three=3))
print(dict(zip(['one', 'two', 'three'], [1, 2, 3])))
print(dict([('one', 1), ('two', 2), ('three', 3)]))


# === 内置函数 zip() ===

# 迭代器一定是iterable
# 迭代器如果耗尽, 则无法继续迭代
res = zip('abcd', [4, 5, 7, 1])
print(list(res))
print(tuple(res))  # ()  已耗尽

res = zip('abcd', [4, 5, 7])
print(tuple(res))

res = zip('abcd', [4, 5, 7])
# next(iterator) 内置函数, 返回迭代器的下一个元素
print(next(res))
print(next(res))
print(next(res))

res = zip('abcd')
print(list(res))

res = zip()
print(list(res))


# === 字典 — 访问与修改 ===

d = {'Name': 'Tom', 'Age': 7, 'Class': 'First'}

print(d['Name'])
print(d['Age'])
# 如果指定的键不存在, 则报错
# d['Gender']  # KeyError

d = {'Name': 'Tom', 'Age': 7, 'Class': 'First'}

# 修改指定键所对应的值
d['Name'] = 'Tony'
d['Age'] = 8
print(d)

# 如果指定的键不存在, 则新增该键值对
d['Gender'] = 'male'
print(d)


# === 字典方法 — keys()（视图对象） ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
view_keys = d.keys()
print(view_keys)

# 修改字典后，视图同步变化
d['weight'] = 59
print(view_keys)


# === 字典方法 — values()（视图对象） ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
view_values = d.values()
print(view_values)

# 修改字典后，视图同步变化
d['weight'] = 59
print(view_values)


# === 字典方法 — items()（视图对象） ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
view_items = d.items()
print(view_items)

# 修改字典后，视图同步变化
d['weight'] = 59
print(view_items)


# === 字典方法 — get() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
print(d.get('age'))
print(d.get('weight'))
print(d.get('weight', '该键不存在'))


# === 字典方法 — update() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
d.update(age=18, weight=59)
d.update({'age': 18, 'weight': 59})
d.update(zip(['age', 'weight'], [18, 59]))
d.update([('age', 18), ('weight', 59)])
print(d)


# === 字典方法 — pop() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
print(d.pop('height'))
print(d)

print(d.pop('weight', None))


# === 字典方法 — popitem() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
print(d.popitem())
print(d)


# === 字典方法 — setdefault() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
print(d.setdefault('age'))         # 键存在，返回已有值

print(d.setdefault('weight'))      # 键不存在，返回None并新增
print(d)

print(d.setdefault('gender', 'male'))  # 键不存在，返回default并新增
print(d)


# === 字典方法 — copy() / clear() ===

d = {'name': 'Tom', 'age': 15, 'height': 162}
new_d = d.copy()
print(new_d)

d = {'name': 'Tom', 'age': 15, 'height': 162}
d.clear()
print(d)


# === 集合 — 定义 ===

# 空集合（必须用set()，{} 是空字典）
s = set()
print(s)

# 空字典
d = {}
print(d)

s = {789, 456, "hello", (135,), 'world'}
print(s)


# === 集合 — set() 转换 ===

print(set())
print(set("hello"))
print(set([1, 2, 3]))
print(set((1, 2, 3)))
# 字典作为一个iterable, 只有键参与迭代
print(set({1: 2, 3: 4}))


# === 集合方法 — update() ===

s = '12'
lst = [1, '2']
d = {1: '1', 2: '2'}

set1 = {'1', '2', 1, 3}
set1.update(s, lst, d)
print(set1)


# === 集合方法 — add() ===

s = {1, 2, 3}
s.add("hello world")
print(s)


# === 集合方法 — remove() / discard() ===

s = {1, 2, 3, 4}
s.remove(3)
print(s)

s = {1, 2, 3, 4}
s.discard(3)
s.discard(3)  # 不存在时不报错
s.discard(3)
print(s)


# === 集合方法 — pop() ===

s = {'1', '2', 'hello', 789}
print(s.pop())
print(s)


# === 集合方法 — copy() / clear() ===

set1 = {'1', '2', 1, 3}
set2 = set1.copy()
print(set2)

s = {'1', '2', 'hello', 789}
s.clear()
print(s)


# === 序列索引 ===

string = "Hello 1牛3 Python"
print(string[7])
print(string[-9])

# 索引超出范围时, 会报错
# print(string[16])   # IndexError
# print(string[-17])  # IndexError

lst = [567, 'hello', True, False, 456]
print(lst[1])
print(lst[-4])

tup = (567, 'hello', True, False, 456)
print(tup[1])
print(tup[-4])


# === 序列切片 ===

string = 'Hello 1牛3 Python'

"""
正向索引和反向索引都可以使用
步长默认为1, 取连续的数据
"""
print(string[7: 11])   # '牛3 P'
print(string[-9: -5])  # '牛3 P'
print(string[7: -5])   # '牛3 P'
print(string[-9: 11])  # '牛3 P'

""" 步长为2, 取数据时要隔一个再取 """
print(string[7: 14: 2])   # '牛 yh'

""" 步长为3, 取数据时要隔两个再取 """
print(string[7: 14: 3])   # '牛Ph'

""" 步长为负数, 表示从右往左取数据 """
print(string[10: 6: -1])  # 'P 3牛'

""" 步长为-2, 表示从右往左隔一个取数据 """
print(string[13: 6: -2])  # 'hy 牛'

""" 步长为正数, start没有指定, 默认为0 """
print(string[: 3])         # 'Hel'
print(string[0: 3])

""" 步长为负数, start没有指定, 默认为-1 """
print(string[: 12: -1])    # 'noh'
print(string[-1: 12: -1])

""" 步长为正数, end没有指定, 默认为len(string) """
print(string[13:])          # 'hon'
print(string[13:len(string)])

""" 步长为负数, end没有指定, 默认为-len(string)-1 """
print(string[2::-1])        # 'leH'
print(string[2:-len(string)-1:-1])

""" 把该序列复制一份 """
print(string[:])

""" 把该序列倒过来 """
print(string[::-1])

""" start到end是从左往右，但step表示从右往左 → 空序列 """
print(string[1: 3: -1])    # ''


# === 索引降维 vs 切片不降维 ===

""" 类比0维数据 """
item1 = 1
item2 = 2
item3 = 3
item4 = 4
item5 = 5
item6 = 6
item7 = 7
item8 = 8
item9 = 9

""" 类比1维数据 """
lst1 = [item1, item2, item3]
lst2 = [item4, item5, item6]
lst3 = [item7, item8, item9]

# 对1维数据索引，结果为0维数据（降维）
print(lst1[0])  # 1
print(lst2[1])  # 5
print(lst3[2])  # 9

# 无论怎么切片，维度保持不变
print(lst1[::2])      # [1, 3]
print(lst2[1:2])      # [5]
print(lst3[::2][1:2]) # [9]

""" 类比2维数据 """
lst4 = [lst1, lst2, lst3]

# 对2维数据索引，结果为1维数据（降维）
print(lst4[0])  # [1, 2, 3]
print(lst4[1])  # [4, 5, 6]
print(lst4[2])  # [7, 8, 9]

# 并且每索引一次，降低一次维度
print(lst4[0][1])  # 2

# 无论怎么切片，维度保持不变
print(lst4[::2])       # [[1, 2, 3], [7, 8, 9]]
print(lst4[1:2])       # [[4, 5, 6]]
print(lst4[::2][1:2])  # [[7, 8, 9]]


# === 内置函数 len() ===

print(len('abcd'))
print(len([1, 2, 3, 4]))
print(len((1, 2, 3, 4)))


# === del 语句 ===

# del 不是直接删除数据，而是解除对应的引用；
# 当该数据的引用计数为0时，才会被Python自动回收。

lst1 = [567, 'hello', 456, [912, 923], 'world']
lst2 = lst1

# 解除 lst1 对列表的引用，lst2 仍指向该对象
del lst1
print(lst2)   # [567, 'hello', 456, [912, 923], 'world']

# 删除索引1的元素 'hello'
del lst2[1]
print(lst2)   # [567, 456, [912, 923], 'world']

# 同时删除索引0和索引2（先删0后删2，注意每次删除后索引会变化）
del lst2[0], lst2[2]
print(lst2)   # [456, [912, 923]]

# 用切片删除，步长为2，[:2:2] 删索引0
del lst2[:2:2]
print(lst2)   # [[912, 923]]

# 删除子列表中的第一个元素（lst2[0] 是 [912, 923]）
del lst2[0][0]
print(lst2)   # [[923]]

# 清空列表
del lst2[:]
print(lst2)   # []

# === END p04 ===
