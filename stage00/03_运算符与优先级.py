# ============================================================
# 章节名称：运算符 & 优先级 (p05)
# ============================================================


# === 算术运算符 ===
# 运算符：+ 加，- 减，* 乘，/ 除，% 取模（求余数），** 幂，// 整除（相当于 / 的结果再向下取整）

a = 5
b = 2
print(a + b)   # 7
print(a - b)   # 3
print(a * b)   # 10
print(a / b)   # 2.5
print(a ** b)  # 25
print(a // b)  # 2
print(a % b)   # 1
print(-15 % 4) # 1


# === 比较运算符 ===
# 运算符：== 等于，!= 不等于，> 大于，< 小于，>= 大于或等于，<= 小于或等于
# 判断两个对象值的大小关系，返回布尔值：True，False

a = 456
b = 456
c = 789
print(a == b)  # True
print(a != c)  # True
print(c > a)   # True
print(b < c)   # True
print(a >= b)  # True
print(a <= b)  # True


# === 赋值运算符 ===
# 运算符：= 简单赋值，+= 加法赋值，-= 减法赋值，*= 乘法赋值，
#         /= 除法赋值，%= 取模赋值，**= 幂赋值，//= 取整赋值

a = 3
c = a + 2
print(c)   # 5

c += a
print(c)   # 8   结果等于 c = c + a

c -= a
print(c)   # 5   结果等于 c = c - a

c *= a
print(c)   # 15  结果等于 c = c * a

c /= a
print(c)   # 5.0  结果等于 c = c / a

c %= a
print(c)   # 2.0  结果等于 c = c % a

c **= a
print(c)   # 8.0  结果等于 c = c ** a

c //= a
print(c)   # 2.0  结果等于 c = c // a


# === 增强赋值（inplace 与普通赋值的区别） ===
# 增强赋值在条件符合的情况下（如：操作数是一个可变数据）会以 inplace 的方式来进行处理，
# 而普通赋值则会以新建的方式进行处理。

lst1 = [1, 2]
lst2 = [3, 4, 5]
print(id(lst1))   # 原 id
lst1 += lst2      # inplace 操作，id 不变
print(id(lst1))   # 与上面相同
print(lst1)       # [1, 2, 3, 4, 5]

lst1 = [1, 2]
lst2 = [3, 4, 5]
print(id(lst1))   # 原 id
lst1 = lst1 + lst2  # 普通赋值，创建新对象，id 改变
print(id(lst1))   # 与上面不同
print(lst1)       # [1, 2, 3, 4, 5]

a = [1, 2, 3, 4]
b = [4, 3, 2, 1]
c = a
print(id(a))
print(id(b))
print(id(c))  # id(c) == id(a)，c 和 a 是同一对象


# === +、* 的拼接操作（字符串、列表、元组） ===
# +、+=、*、*= 还支持字符串、列表、元组的拼接操作

# 字符串拼接
str1 = 'hello '
str2 = 'world'
print(str1 + str2)   # hello world
str1 += str2
print(str1)          # hello world

str1 = 'hello '
print(str1 * 3)      # hello hello hello
str1 *= 3
print(str1)          # hello hello hello

# 列表拼接
lst1 = [1, 2]
lst2 = [3, 4, 5]
print(lst1 + lst2)   # [1, 2, 3, 4, 5]
lst1 += lst2
print(lst1)          # [1, 2, 3, 4, 5]

lst1 = [1, 2]
print(lst1 * 3)      # [1, 2, 1, 2, 1, 2]
lst1 *= 3
print(lst1)          # [1, 2, 1, 2, 1, 2]

# 元组拼接
tup1 = (1, 2)
tup2 = (3, 4, 5)
print(tup1 + tup2)   # (1, 2, 3, 4, 5)
tup1 += tup2
print(tup1)          # (1, 2, 3, 4, 5)

tup1 = (1, 2)
print(tup1 * 3)      # (1, 2, 1, 2, 1, 2)
tup1 *= 3
print(tup1)          # (1, 2, 1, 2, 1, 2)


# === 基本序列赋值 ===
# 格式：a, b, c, ... = iterable
# 将 iterable 的元素分别赋值给对应变量，元素和变量个数需要一致

a, b = 3, 4
print(a, b)           # 3 4

a, b, c = [3, 4, 5]
print(a, b, c)        # 3 4 5

a, b, c, d = '你好吗?'
print(a, b, c, d)     # 你 好 吗 ?


# === 多目标赋值 ===
# 将一个对象同时赋值给多个变量

a = b = c = 999
print(id(a))
print(id(b))
print(id(c))  # 三者 id 相同，指向同一对象

a = b = c = [1, 2, 3]
print(id(a))
print(id(b))
print(id(c))  # 三者 id 相同，指向同一列表对象

b.append(4)
print(a)  # [1, 2, 3, 4]  a、b、c 均受影响
print(b)  # [1, 2, 3, 4]
print(c)  # [1, 2, 3, 4]


# === 逻辑运算符 ===
# and：左边 bool 判定为 False，返回左边；否则返回右边
# or ：左边 bool 判定为 True，返回左边；否则返回右边
# not：判定为 False 返回 True；判定为 True 返回 False

a = 2
b = 'hello'
c = []
d = 0

print(c and a)  # []
print(a and c)  # []
print(d and c)  # 0
print(c and d)  # []
print(a and b)  # 'hello'
print(b and a)  # 2

print(a or c)   # 2
print(c or a)   # 2
print(b or a)   # 'hello'
print(a or b)   # 2
print(c or d)   # 0
print(d or c)   # []

print(not a)    # False
print(not b)    # False
print(not c)    # True
print(not d)    # True

# 优先级：not > and > or
print(b and not a or c)  # []


# === 短路机制 ===
# 在逻辑表达式中，由于 and 和 or 的特点，表达式中的部分内容可能不会执行

a = 0
b = 1
c = ()

print(c and b / c)  # ()   c 为空元组，bool 为 False，and 短路，不计算 b/c（避免除以空元组错误）
print(b or a + c)   # 1    b 为 True，or 短路，不计算 a+c（避免类型错误）
# b and a + c  # 如果执行会 Error：int + tuple 类型不匹配


# === all() 和 any() ===
# all(iterable)：如果 iterable 的所有元素 bool 判定都为 True，则返回 True；iterable 为空也返回 True
# any(iterable)：如果 iterable 中存在至少一个元素 bool 判定为 True，则返回 True；iterable 为空返回 False

tup = ('0', ' ', 'None', 'False', '[]')
print(all(tup))   # True  （非空字符串 bool 判定均为 True）
print(all([]))    # True  （空 iterable）

tup = (0, '', None, False, [])
print(any(tup))   # False （所有元素 bool 判定均为 False）
print(any([]))    # False （空 iterable）


# === 成员运算符 ===
# in：在其中，not in：不在其中
# 判断某个对象是否为指定 iterable 的元素，返回布尔值：True，False

string = 'hello world'
print('e' in string)      # True
print('lo' in string)     # True
print('ol' not in string) # True

lst = [True, False, [2, 3], 4]
print(1 in lst)            # True  （1 == True）
print(0 in lst)            # True  （0 == False）
print(4 in lst)            # True
print(2 not in lst)        # True  （2 不是列表元素，[2,3] 是元素）
print(3 not in lst)        # True

d = {1: 2, 0: 4}
print(True in d)           # True  （True == 1，与键 1 相等）
print(False in d)          # True  （False == 0，与键 0 相等）
print(2 not in d)          # True  （2 不是键）
print(4 not in d)          # True  （4 不是键）


# === 身份运算符 ===
# is：类似于判断 id(a) == id(b)，判断两个标识符是否引用同一对象
# is not：类似于判断 id(a) != id(b)
# 返回布尔值：True，False

a = 256
b = 256
print(a == b)          # True
print(a is b)          # True   （Python 对小整数 [-5, 256] 做缓存，id 相同）
print(id(a) == id(b))  # True

a = 257
b = 257
print(a == b)          # True
print(a is b)          # False  （超出小整数缓存范围，id 不同）
print(id(a) == id(b))  # False

a = [257]
b = [257]
print(a == b)          # True   （值相等）
print(a is b)          # False  （不同对象，id 不同）
print(id(a) == id(b))  # False


# === 运算符优先级（从高到低） ===
# **                          指数
# * / % //                    乘，除，求余数和取整除
# + -                         加法、减法
# <= < > >=                   比较运算符
# == !=                       等于运算符
# %= /= //= -= += *= **=      赋值运算符
# is  is not                  身份运算符
# in  not in                  成员运算符
# not and or                  逻辑运算符
# =                           简单赋值运算符

# === END p05 ===
