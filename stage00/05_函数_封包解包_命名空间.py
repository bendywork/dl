# ============================================================
# 章节：函数 / 封包 & 解包 / 命名空间 & 作用域
# 来源：p07_函数，封包 & 解包，命名空间 & 作用域.pdf
# ============================================================


# === 定义函数 ===

def plus(num):
    print(num + 1)

""" 调用函数 """
plus(2)  # 3
plus(5)  # 6

f = plus

print(plus)
print(f)

f(2)  # 3
f(5)  # 6


# === return 用法 ===

# 返回一个对象
def add1(left, right):
    res = left + right
    return res

def add2(left, right):
    return left + right

# 返回多个对象, 自动打包成一个元组
def add3(left, right):
    res1 = left + right
    res2 = left * right
    return res1, res2

def add4(left, right):
    return left + right, left * right

# return None
def add5(left, right):
    print(left + right)
    return

# return None
def add6(left, right):
    pass

print(add1(3, 4))
print(add2(3, 4))
print(add3(3, 4))
print(add4(3, 4))
print(add5(3, 4))
print(add6(3, 4))


# === 参数传递：传不可变对象 ===

def func(b):
    print(id(a), a)
    print(id(b), b)

a = 789
func(a)


# === 参数传递：传可变对象（引用传递，不修改） ===

def func(b):
    print(id(a), a)
    print(id(b), b)

a = [789]
func(a)


# === 参数传递：传可变对象（引用传递，修改内容） ===

def func(b):
    b.append(345)

a = [789]
func(a)
print(a)


# === 必需参数 ===

def func(a, b):
    print(a - b)

func(3, 4)
func(3, b=4)
func(a=3, b=4)


# === 位置参数 ===

def func(a, b):
    print(a - b)

func(3, 4)  # -1
func(4, 3)  # 1


# === 关键字参数 ===

def func(a, b):
    print(a - b)

func(a=3, b=4)  # -1
func(b=4, a=3)  # -1
func(3, b=4)    # -1


# === 默认参数 ===

def func(a, b=4):
    print(a - b)

func(3)     # -1
func(3, 5)  # -2


# === 不定长参数：*args ===

def func(*args):
    print(args)

func()
func(3, 1, 4, 6)


# === 不定长参数：**kwargs ===

def func(**kwargs):
    print(kwargs)

func()
func(a=3, b=2, c=4)


# === 不定长参数：*args + **kwargs ===

def func(*args, **kwargs):
    print(args)
    print(kwargs)

func()
func(1, 2, a=3, b=4)


# === 特殊参数：/ 和 * 限制传参形式 ===

def func(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
    pass

func(1, 2, 3, kwd1=4, kwd2=5)
func(1, 2, pos_or_kwd=3, kwd1=4, kwd2=5)


# === 匿名函数（lambda） ===

print((lambda: 'It just returns a string')())

f = lambda: 'It just returns a string'
print(f())

(lambda x, y, z: print(x + y + z))(1, 2, 3)

f = lambda x, y, z: print(x + y + z)
f(1, 2, 3)

tup = (8, 5, -9, 6, 2)
print(sorted(tup, key=lambda x: -x if x < 0 else x))


# === 封包（Packing） ===

tup = 345, 'hello', 789
print(tup)


# === 解包：赋值过程中的解包 ===

a, b, c = [4, 3, 'a']
print(a)  # 4
print(b)  # 3
print(c)  # 'a'

a, *b, c = 'hello'
print(a)  # 'h'
print(b)  # ['e', 'l', 'l']
print(c)  # 'o'

a, *b, c = 'he'
print(a)  # 'h'
print(b)  # []
print(c)  # 'e'

*a, = 'hel'
print(a)  # ['h', 'e', 'l']

_, *b, _ = [4, 3, 5, 7]
print(b)  # [3, 5]


# === 解包：函数传参中的 * 解包（可迭代对象 → 位置参数） ===

def func(a, b, c):
    print(a, b, c)

"""
在函数传实参时, *iterable可以将
该iterable解包成位置参数
"""
tup = (1, 2, 3)
func(*tup)  # 等价于func(1, 2, 3)

d = {'a': 1, 'b': 2, 'c': 3}
func(*d)  # 等价于func('a', 'b', 'c')


# === 解包：函数传参中的 ** 解包（字典 → 关键字参数） ===

"""
在函数传参时, **dict可以将
该dict解包成关键字参数
"""
func(**d)  # 等价于func(a=1, b=2, c=3)


# === 命名空间：locals() 和 globals() ===

import this

def func1(arg1, arg2):
    num = 666
    print(locals())  # 返回局部命名空间

def func2(arg1, arg2):
    num = 777
    print(locals())

num = 111
func1(222, 333)
func2(444, 555)
print(globals())  # 返回全局命名空间

# 在全局作用域, locals()等价于globals()
print(locals())


# === 作用域：局部作用域（Local） ===

def func(x, y):
    """
    函数内部区域可以直接访问该函数所对应的局部命名空间,
    所以该区域为 局部作用域(Local)
    """
    a = 3
    b = 4
    print(x, y, a, b)

func(1, 2)


# === 作用域：闭包函数外的函数中（Enclosing） ===

def outer(a):
    """
    在inner函数的外部且在outer函数的内部区域,
    可以直接访问outer所对应的局部命名空间,
    所以该区域为 闭包函数外的函数中(Enclosing)
    """
    b = 2
    def inner(c):
        """ 局部作用域 """
        return a + b + c
    return inner

""" 全局作用域 """
print(outer(1)(3))


# === 作用域：全局作用域（Global） ===

def func():
    pass

"""
函数外部区域可以直接访问该模块所对应的全局命名空间,
所以该区域为 全局作用域(Global)
"""
a = 3
b = 4
print(a, b)


# === global 和 nonlocal ===

def outer():
    global a, b
    a, b, c, d = 3, 4, 5, 6
    print(a, b)

    def inner():
        global a, b
        nonlocal c, d
        a, b, c, d = 7, 8, 9, 0

    inner()
    print(c, d)

a, b = 1, 2
outer()
print(a, b)


# === END p07 ===
