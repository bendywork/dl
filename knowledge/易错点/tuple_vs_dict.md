# Python 元组（tuple）与字典（dict）完全区别

---

## 一、本质定位

| 维度 | tuple | dict |
|------|-------|------|
| 数据结构 | 有序序列 | 键值映射 |
| 可变性 | 不可变（immutable） | 可变（mutable） |
| 索引方式 | 整数下标 0, 1, 2... | 任意可哈希 key |
| 底层实现 | C 数组（PyTupleObject） | 哈希表（PyDictObject） |
| 内存布局 | 紧凑连续内存 | 散列桶 + 动态扩容 |

---

## 二、创建语法

```python
# tuple
t1 = (1, 2, 3)
t2 = 1, 2, 3          # 括号可省略
t3 = (42,)            # 单元素必须加逗号！
t4 = ()               # 空元组
t5 = tuple([1, 2, 3]) # 从可迭代对象转换

# dict
d1 = {"a": 1, "b": 2}
d2 = dict(a=1, b=2)
d3 = dict([("a", 1), ("b", 2)])
d4 = {k: v for k, v in zip("abc", [1,2,3])}
d5 = {}               # 空字典（不是空集合！）
```

### 易错点 1：单元素元组陷阱
```python
x = (42)    # ❌ 这是 int，不是 tuple
x = (42,)   # ✅ 这才是 tuple
print(type((42)))   # <class 'int'>
print(type((42,)))  # <class 'tuple'>
```

### 易错点 2：空花括号是 dict 不是 set
```python
x = {}        # dict，不是 set
x = set()     # 空 set 必须用构造函数
```

---

## 三、访问方式

```python
t = (10, 20, 30)
t[0]      # 10  — 正向索引
t[-1]     # 30  — 反向索引
t[1:3]    # (20, 30) — 切片，返回新 tuple

d = {"x": 10, "y": 20}
d["x"]          # 10  — KeyError if missing
d.get("z")      # None — 安全访问
d.get("z", 0)   # 0   — 带默认值
d["z"]          # ❌ KeyError
```


---

## 四、可变性与哈希性

### tuple 不可变但内部元素可以是可变对象
```python
t = ([1, 2], [3, 4])
t[0].append(99)   # ✅ 修改列表内容合法
t[0] = [9, 9]     # ❌ TypeError: 不能替换元素

# 哈希陷阱：含可变对象的 tuple 不可哈希
t_hashable = (1, 2, 3)
hash(t_hashable)   # ✅ 正常

t_unhashable = ([1, 2], 3)
hash(t_unhashable) # ❌ TypeError: unhashable type: 'list'
```

### dict 的 key 必须可哈希
```python
d = {}
d[(1, 2)] = "ok"    # ✅ tuple 作 key
d[[1, 2]] = "bad"   # ❌ list 不可哈希，TypeError
d[{"a": 1}] = "bad" # ❌ dict 不可哈希，TypeError
```

### 易错点 3：tuple 作字典 key 的隐藏坑
```python
# 只要 tuple 内含不可哈希元素，整体就不可哈希
d = {}
d[((1, 2), (3, 4))] = "nested ok"   # ✅
d[((1, 2), [3, 4])] = "nested bad"  # ❌ 因为内层有 list
```

---

## 五、内存与性能

```python
import sys
t = (1, 2, 3, 4, 5)
d = {"a":1, "b":2, "c":3, "d":4, "e":5}

sys.getsizeof(t)  # 80 bytes（Python 3.11）
sys.getsizeof(d)  # 232 bytes（哈希表有额外开销）
```

| 操作 | tuple | dict |
|------|-------|------|
| 索引访问 | O(1) | O(1) 均摊 |
| 遍历 | O(n) 最快（连续内存） | O(n) |
| 查找元素 | O(n) 线性搜索 | O(1) 均摊（key查找） |
| 内存占用 | 极小，无哈希桶开销 | 较大，预分配桶 |
| 创建速度 | 极快（字面量有缓存） | 较慢 |

### 易错点 4：tuple 的 in 操作是线性的
```python
t = tuple(range(10000))
10000 in t  # O(n) 遍历，慢！

d = {i: True for i in range(10000)}
10000 in d  # O(1) 哈希查找，快
```


---

## 六、解包操作

```python
# tuple 解包
a, b, c = (1, 2, 3)
first, *rest = (1, 2, 3, 4, 5)   # first=1, rest=[2,3,4,5]
*head, last = (1, 2, 3, 4, 5)    # head=[1,2,3,4], last=5

# 函数返回多值本质是返回 tuple
def minmax(lst):
    return min(lst), max(lst)   # 返回 tuple

lo, hi = minmax([3, 1, 4, 1, 5])

# dict 解包（Python 3.5+）
d1 = {"a": 1}
d2 = {"b": 2}
merged = {**d1, **d2}   # {"a":1, "b":2}

# 易错点 5：** 解包键冲突时后者覆盖
d3 = {**{"a": 1}, **{"a": 99}}  # {"a": 99}
```

---

## 七、遍历方式

```python
t = (10, 20, 30)
# tuple 遍历
for v in t: ...
for i, v in enumerate(t): ...

d = {"x": 1, "y": 2, "z": 3}
# dict 遍历（3种）
for k in d: ...            # 遍历 key
for v in d.values(): ...   # 遍历 value
for k, v in d.items(): ... # 遍历键值对

# 易错点 6：遍历时修改 dict 会报错
for k in d:
    del d[k]   # ❌ RuntimeError: dictionary changed size during iteration

# 正确做法：遍历副本
for k in list(d.keys()):
    del d[k]   # ✅
```

---

## 八、常用方法对比

```python
# tuple 方法（只有2个）
t = (1, 2, 2, 3)
t.count(2)   # 2  — 计数
t.index(3)   # 3  — 首次出现位置，找不到抛 ValueError

# dict 常用方法
d = {"a": 1, "b": 2}
d.keys()             # dict_keys 视图
d.values()           # dict_values 视图
d.items()            # dict_items 视图
d.get("c", 0)        # 安全访问
d.setdefault("c", 0) # 不存在则设置并返回默认值
d.update({"c": 3})   # 批量更新
d.pop("a")           # 删除并返回值
d.pop("z", None)     # 安全删除
d.popitem()          # 删除并返回最后插入的 (k,v)（Python 3.7+ LIFO）
d.copy()             # 浅拷贝
```


---

## 九、隐藏的坑汇总

### 坑1：tuple 的"不可变"是浅层的
```python
t = ([1, 2], [3, 4])
t[0].append(99)    # 不报错！内部列表被修改
print(t)           # ([1, 2, 99], [3, 4])
# tuple 不可变 = 不能重新绑定元素，不等于内容不变
```

### 坑2：dict 视图是动态的
```python
d = {"a": 1, "b": 2}
keys = d.keys()    # 不是快照，是实时视图
d["c"] = 3
print(keys)        # dict_keys(['a', 'b', 'c']) 自动更新
# 要快照就用 list(d.keys())
```

### 坑3：dict.setdefault vs get
```python
d = {}
d.get("x", [])         # 返回 []，但不写入 d
d.setdefault("x", [])  # 返回 []，且写入 d
d["x"].append(1)       # setdefault 后可直接操作
```

### 坑4：dict 合并的覆盖顺序
```python
# Python 3.9+ 的 | 运算符
a = {"x": 1, "y": 2}
b = {"y": 99, "z": 3}
c = a | b   # {"x":1, "y":99, "z":3} 右边覆盖左边
c = b | a   # {"y":2, "x":1, "z":3} 顺序不同结果不同
```


### 坑5：tuple 拼接创建新对象
```python
t = (1, 2)
t += (3,)       # 不是原地修改，而是创建新 tuple
id_before = id(t)
t += (4,)
id(t) == id_before  # False！每次都是新对象

# 对比 list：
lst = [1, 2]
lst += [3]      # 原地修改，id 不变
```

### 坑6：字典推导式与 zip 的陷阱
```python
keys = ["a", "b", "c"]
vals = [1, 2]             # 长度不等！
d = dict(zip(keys, vals)) # {"a":1, "b":2} 多余的 key 被丢弃，不报错
```

### 坑7：defaultdict vs 普通 dict
```python
from collections import defaultdict
d = defaultdict(list)
d["x"].append(1)   # key 不存在时自动创建 []，不报 KeyError

# 普通 dict 的等价写法（更显式）：
d = {}
d.setdefault("x", []).append(1)
```

### 坑8：OrderedDict 与 dict 的相等比较
```python
from collections import OrderedDict
od = OrderedDict([("a",1), ("b",2)])
d  = {"b":2, "a":1}
od == d   # True！dict 比较不考虑顺序
# 但 OrderedDict 之间比较会考虑顺序
```


---

## 十、拷贝陷阱

```python
import copy

# tuple：浅拷贝与原对象共享内部引用
t = ([1, 2], [3, 4])
t2 = t[:]           # 浅拷贝
t2[0].append(99)    # t[0] 也被修改！
t3 = copy.deepcopy(t)   # 深拷贝才真正独立

# dict：同理
d = {"a": [1, 2]}
d2 = d.copy()           # 浅拷贝，d2["a"] 与 d["a"] 是同一个列表
d2["a"].append(99)      # d["a"] 也变了！
d3 = copy.deepcopy(d)   # 深拷贝
```

### 判断是否需要深拷贝
```python
# 如果容器内只含不可变对象（int/str/tuple...），浅拷贝够用
t = (1, "hello", (2, 3))
t2 = t[:]   # 安全

# 如果含 list/dict/自定义对象，必须 deepcopy
d = {"x": {"nested": [1,2,3]}}
d2 = copy.deepcopy(d)   # 必须
```


---

## 十一、底层实现差异

### tuple 底层（CPython）
```
PyTupleObject {
    ob_refcnt   // 引用计数
    ob_type     // 类型指针
    ob_size     // 元素数量
    ob_item[1]  // 指针数组（连续内存）
}
```
- 固定大小，创建时一次性分配内存
- 元素是 `PyObject*` 指针的连续数组
- 小 tuple（len≤20）有对象池缓存，避免重复分配

```python
# tuple 字面量有编译期优化
import dis
dis.dis("x = (1, 2, 3)")
# LOAD_CONST (1, 2, 3)  ← 整个 tuple 是一个常量！
```

### dict 底层（CPython 3.6+）
```
PyDictObject {
    ma_used      // 已用条目数
    ma_version   // 版本号（每次修改+1）
    ma_keys      // 指向 PyDictKeysObject
    ma_values    // 指向值数组（分离式）或 NULL（组合式）
}

PyDictKeysObject {
    dk_size      // 哈希表大小（2的幂次）
    dk_lookup    // 查找函数指针
    dk_entries[] // 条目数组（key, hash, value）
}
```


### dict 哈希冲突处理
```python
# 开放寻址法（open addressing）
# 探测序列：i = (5*i + 1 + perturb) % size
# perturb 逐步右移，保证覆盖所有槽位

# 实际效果：
# - 负载因子 > 2/3 时自动扩容（×4）
# - 扩容后重新哈希所有 key
# - 删除用"墓碑标记"，不立即释放槽位
```

### dict 3.6+ 紧凑有序实现
```python
# Python 3.7+ 字典保证插入顺序（语言规范）
# 实现：indices 数组 + entries 数组分离
#
# indices: [_, 0, _, 2, 1, _, ...]  哈希槽 → entries 下标
# entries: [(k0,h0,v0), (k1,h1,v1), (k2,h2,v2)]  按插入顺序
#
# 优点：遍历按插入顺序；内存更紧凑

d = {}
d["c"] = 3; d["a"] = 1; d["b"] = 2
list(d.keys())  # ['c', 'a', 'b'] 保证插入顺序
```

### tuple 小对象缓存
```python
# CPython 对长度 0~20 的 tuple 有 free_list 缓存
a = (1, 2)
del a
b = (3, 4)  # 可能复用 a 的内存槽

# 验证（CPython 实现细节）：
a = ()
b = ()
a is b   # True！空 tuple 是单例
```


---

## 十二、使用场景选择

| 场景 | 选 tuple | 选 dict |
|------|---------|---------|
| 固定结构的多值返回 | ✅ | |
| 需要按名称访问字段 | | ✅ |
| 作为字典的 key | ✅（可哈希） | ❌ |
| 数据不应被修改 | ✅ | |
| 快速成员查找 | ❌ O(n) | ✅ O(1) |
| 存储同类型有序数据 | ✅ | |
| JSON-like 结构 | | ✅ |
| 函数关键字参数传递 | | ✅ **kwargs |
| 内存敏感场景 | ✅ 更小 | |

### 推荐替代方案
```python
# 有字段名的 tuple → namedtuple 或 dataclass
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(1, 2)
p.x    # 按名访问，且不可变，可哈希

# Python 3.7+ 更推荐 dataclass
from dataclasses import dataclass
@dataclass(frozen=True)   # frozen=True 使其不可变且可哈希
class Point:
    x: float
    y: float
```

---

## 十三、快速记忆口诀

```
tuple = 不可变有序序列，位置索引，可作 key，内存紧凑
dict  = 可变键值映射，名称索引，O(1)查找，保留插入顺序

坑点速记：
- 单元素 tuple 必须加逗号 (42,)
- {} 是 dict 不是 set
- tuple 不可变是浅层的，内部列表仍可变
- dict 遍历时不能增删 key
- dict 视图是动态的，要快照用 list()
- in 操作：tuple O(n)，dict O(1)
```

