# ============================================================
# 章节：错误 & 异常（p09）
# 内容：语法错误、异常处理、抛出异常、断言
# ============================================================


# --------------------------------------------------------
# 知识点：try ... except ...（捕获所有异常）
# 如果 try 子句没有异常，则不执行 except
# 如果 try 子句发生异常，跳过剩余部分，执行 except
# except 不指定类型时，处理所有异常
# --------------------------------------------------------
def div_v1(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except:
        print('try中发生异常')

div_v1(2, 1)
div_v1(2, 0)
div_v1('2', 2)


# --------------------------------------------------------
# 知识点：try ... except ...（指定具体异常类型）
# 可以写多个 except，分别处理不同类型的异常
# --------------------------------------------------------
def div_v2(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except ZeroDivisionError:
        print('try中发生了除数为0的异常')
    except TypeError:
        print('try中发生了类型异常')

div_v2(2, 1)
div_v2(2, 0)
div_v2('2', 2)


# --------------------------------------------------------
# 知识点：try ... except ...（用元组指定多个异常类型）
# 多个异常类型可以用元组合并在一个 except 中处理
# --------------------------------------------------------
def div_v3(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except (ZeroDivisionError, TypeError):
        print('try中发生了除数为0的异常或者类型异常')

div_v3(2, 1)
div_v3(2, 0)
div_v3('2', 2)


# --------------------------------------------------------
# 知识点：try ... except ... else ...
# else 必须在所有 except 之后
# else 子句在 try 没有发生任何异常时执行
# --------------------------------------------------------
def div_v4(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except ZeroDivisionError:
        print('try中发生了除数为0的异常')
    except:
        print('发生了除0以外的异常')
    else:
        print('try中没有异常')

div_v4(2, 1)
div_v4(2, 0)
div_v4('2', 2)


# --------------------------------------------------------
# 知识点：try ... except ... as ...
# as 后面为异常实例对象的名称，可以获取异常详细信息
# --------------------------------------------------------
def div_v5(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    # e：ZeroDivisionError('division by zero')
    except ZeroDivisionError as e:
        print(type(e) is ZeroDivisionError)
        print(e)
        print(ZeroDivisionError('division by zero'))
    except Exception as e:
        print(type(e) is TypeError)
        print(e)

div_v5(2, 0)
div_v5(2, '0')


# --------------------------------------------------------
# 知识点：try ... finally ...
# finally 子句作为 try 语句结束前的最后一项任务执行
# 无论 try 是否产生异常，finally 都会被执行
# --------------------------------------------------------
def div_v6(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    finally:
        print("执行finally子句")

try:
    div_v6(2, 0)
except ZeroDivisionError:
    print("（finally执行后，异常继续向上传播，此处捕获）")


# --------------------------------------------------------
# 知识点：try ... except ... else ... finally ...（完整结构）
# except：发生异常时执行
# else：没有异常时执行
# finally：任何情况下都执行
# --------------------------------------------------------
def div_v7(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except:
        print('except在发生异常时执行')
    else:
        print('else在没有异常时执行')
    finally:
        print('finally在任何情况下都会被执行')

div_v7(2, 1)
div_v7(2, 0)


# --------------------------------------------------------
# 知识点：raise 抛出异常（传入异常实例）
# raise 语句可以主动抛出异常
# raise 后面可以是 异常实例 / 异常类 / 没有内容
# --------------------------------------------------------
def div_v8(a, b):
    if b == 0:
        raise ZeroDivisionError('除数为0')
    c = a / b
    print(f"{a} / {b} = {c}")

div_v8(2, 1)
try:
    div_v8(2, 0)
except ZeroDivisionError as e:
    print(f"捕获到异常: {e}")


# --------------------------------------------------------
# 知识点：raise 抛出异常（传入异常类，不带参数）
# --------------------------------------------------------
def div_v9(a, b):
    if b == 0:
        raise ZeroDivisionError
    c = a / b
    print(f"{a} / {b} = {c}")

div_v9(2, 1)
try:
    div_v9(2, 0)
except ZeroDivisionError as e:
    print(f"捕获到异常: {e}")


# --------------------------------------------------------
# 知识点：raise 不带参数（重新抛出当前异常，必须在 except 中使用）
# --------------------------------------------------------
def div_v10(a, b):
    try:
        c = a / b
        print(f"{a} / {b} = {c}")
    except ZeroDivisionError:
        print("捕获到除零异常，重新抛出")
        raise  # 重新抛出当前异常

div_v10(2, 1)
try:
    div_v10(2, 0)
except ZeroDivisionError as e:
    print(f"外层捕获: {e}")


# --------------------------------------------------------
# 知识点：assert 断言（基本用法）
# assert 用于判断表达式，表达式为 False 时触发 AssertionError
# assert expression  等价于：if not expression: raise AssertionError
# --------------------------------------------------------
# 注意：以下示例原本使用 input()，这里用固定值演示可运行版本

num = 3  # 奇数，断言通过
assert num % 2
print(f'{num}为奇数')

# 等价写法
num = 3
if not num % 2:
    raise AssertionError
print(f'{num}为奇数')


# --------------------------------------------------------
# 知识点：assert 断言（带错误提示信息）
# assert expression [, arguments]
# 等价于：if not expression: raise AssertionError(arguments)
# --------------------------------------------------------
num = 3  # 奇数，断言通过
assert num % 2, f'断言失败, {num}是偶数'
print(f'{num}为奇数')

# 等价写法
num = 3
if not num % 2:
    raise AssertionError(f'断言失败, {num}是偶数')
print(f'{num}为奇数')

# 演示断言失败的情况
num_even = 4  # 偶数，断言失败
try:
    assert num_even % 2, f'断言失败, {num_even}是偶数'
except AssertionError as e:
    print(f"AssertionError: {e}")

# === END p09 ===
