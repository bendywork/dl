# Python 中的魔术方法（Dunder Methods）

## 核心概念

魔术方法（Magic Methods），又叫双下方法（Dunder Methods，dunder = **d**ouble **under**score），是 Python 预定义的特殊方法，方法名以 `__` 开头和结尾。

**两个关键事实：**
1. **方法名是固定的**——Python 解释器硬编码了对应关系，名字写错就不生效
2. **不需要注解或装饰器**——只要方法名对了，解释器就能识别并调用

```python
class MyData:
    # 没有任何注解，直接定义就能生效
    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)
```

**调用机制：**

```
你写 len(obj)
  ↓
解释器去找 obj.__len__()
  ↓
找到了？调用它，返回结果
找不到？抛 TypeError
```

这就是 **鸭子类型**——不关心你是什么类，只关心你有没有那个方法。

---

## 构造与销毁

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__init__` | `MyClass()` | 初始化实例 |
| `__new__` | `MyClass()` | 创建实例（在 `__init__` 之前） |
| `__del__` | 对象被回收 | 析构，不推荐依赖 |

```python
class Person:
    def __init__(self, name):
        self.name = name  # Person("Tom") → self.name = "Tom"
```

---

## 字符串表示

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__str__` | `str(obj)` / `print(obj)` | 面向用户的可读字符串 |
| `__repr__` | `repr(obj)` / 交互式输出 | 面向开发者的明确表示 |
| `__format__` | `f"{obj:spec}"` | 自定义格式化 |

```python
class Point:
    def __str__(self):
        return f"({self.x}, {self.y})"      # print(p) → (1, 2)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"  # repr(p) → Point(1, 2)
```

---

## 索引与容器访问

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__getitem__` | `obj[key]` | 获取元素 |
| `__setitem__` | `obj[key] = val` | 设置元素 |
| `__delitem__` | `del obj[key]` | 删除元素 |
| `__contains__` | `x in obj` | 成员判断 |
| `__len__` | `len(obj)` | 长度 |
| `__iter__` | `for x in obj` | 迭代 |
| `__next__` | `next(obj)` | 迭代下一个值 |

```python
class MyData:
    def __init__(self):
        self._data = {"a": 1, "b": 2, "c": 3}

    def __getitem__(self, key):
        return self._data[key]    # d["a"] → 1

    def __len__(self):
        return len(self._data)    # len(d) → 3
```

> HuggingFace 的 `DatasetDict` 和 `Dataset` 就是通过 `__getitem__` 实现了 `dataset["train"]` 和 `ds[0]` 的访问方式。

---

## 比较运算

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__eq__` | `a == b` | 等于 |
| `__ne__` | `a != b` | 不等于 |
| `__lt__` | `a < b` | 小于 |
| `__le__` | `a <= b` | 小于等于 |
| `__gt__` | `a > b` | 大于 |
| `__ge__` | `a >= b` | 大于等于 |

```python
class Score:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.value  # s1 < s2 → 比较数值
```

---

## 算术运算

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__add__` | `a + b` | 加法 |
| `__sub__` | `a - b` | 减法 |
| `__mul__` | `a * b` | 乘法 |
| `__truediv__` | `a / b` | 除法 |
| `__floordiv__` | `a // b` | 整除 |
| `__mod__` | `a % b` | 取模 |
| `__pow__` | `a ** b` | 幂运算 |

反向运算（右操作数）：`__radd__`、`__rsub__`、`__rmul__` 等（当左操作数不支持时调用）

原地运算：`__iadd__`（`+=`）、`__isub__`（`-=`）、`__imul__`（`*=`）等

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)  # v1 + v2
```

---

## 类型转换

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__int__` | `int(obj)` | 转整数 |
| `__float__` | `float(obj)` | 转浮点 |
| `__bool__` | `bool(obj)` | 转布尔 |
| `__str__` | `str(obj)` | 转字符串 |
| `__list__` | `list(obj)` | 转列表 |

```python
class Counter:
    def __init__(self, count):
        self.count = count

    def __bool__(self):
        return self.count > 0   # if counter: → 判断是否 > 0

    def __int__(self):
        return self.count       # int(counter) → 直接拿数值
```

---

## 可调用对象

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__call__` | `obj(args)` | 把实例当函数调用 |

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
double(5)  # → 10
```

> PyTorch 的 `nn.Module` 的前向传播就是 `__call__` → 调用 `forward()`。

---

## 上下文管理器

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__enter__` | `with obj as x:` | 进入 with 块 |
| `__exit__` | with 块结束/异常 | 退出 with 块 |

```python
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        print(f"耗时: {time.time() - self.start:.2f}s")

with Timer():
    do_something()  # 自动计时
```

---

## 属性访问

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__getattr__` | `obj.xxx` 且属性不存在 | 动态生成属性 |
| `__getattribute__` | `obj.xxx` 任何属性访问 | 拦截所有属性访问 |
| `__setattr__` | `obj.xxx = val` | 拦截属性设置 |
| `__delattr__` | `del obj.xxx` | 拦截属性删除 |

---

## 描述符

| 魔术方法 | 触发方式 | 说明 |
|---------|---------|------|
| `__get__` | 访问描述符属性 | 获取值 |
| `__set__` | 设置描述符属性 | 设置值 |
| `__delete__` | 删除描述符属性 | 删除值 |

> Python 的 `property`、`classmethod`、`staticmethod` 底层都是描述符实现的。

---

## 速查对照表

| 你的写法 | 实际调用 |
|---------|---------|
| `obj[key]` | `obj.__getitem__(key)` |
| `obj[key] = val` | `obj.__setitem__(key, val)` |
| `del obj[key]` | `obj.__delitem__(key)` |
| `len(obj)` | `obj.__len__()` |
| `str(obj)` | `obj.__str__()` |
| `repr(obj)` | `obj.__repr__()` |
| `int(obj)` | `obj.__int__()` |
| `bool(obj)` | `obj.__bool__()` |
| `for x in obj` | `obj.__iter__()` → `__next__()` |
| `x in obj` | `obj.__contains__(x)` |
| `a + b` | `a.__add__(b)` |
| `a < b` | `a.__lt__(b)` |
| `obj(args)` | `obj.__call__(args)` |
| `with obj:` | `obj.__enter__()` / `__exit__()` |

---

## 实战示例：模拟 HuggingFace Dataset 的访问方式

```python
class SimpleDataset:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, index):
        return self._data[index]  # ds[0] / ds["train"]

    def __len__(self):
        return len(self._data)    # len(ds)

    def __repr__(self):
        return f"SimpleDataset(num_rows={len(self._data)})"

    def __iter__(self):
        return iter(self._data)   # for row in ds

    def __contains__(self, item):
        return item in self._data # item in ds
```

只要实现了这些魔术方法，你的类就能像内置容器一样用 `[]`、`len()`、`for...in`、`in` 来访问——这就是 **Pythonic** 风格。
