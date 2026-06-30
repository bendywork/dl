# Python import：模块 vs 类

## 一句话总结

`import module` 绑定模块对象，`from module import Class` 绑定类对象 — 导入机制相同，绑定到当前命名空间的对象不同。

---

## 核心区别

### import 模块

```python
import collections        # collections 是一个 <class 'module'> 对象
d = collections.deque()  # 通过 模块.类名 访问 deque
```

- 拿到的是 **module 对象**
- 访问类/函数都要加 `模块.` 前缀
- 不污染当前命名空间

### from import 类

```python
from collections import deque  # deque 直接是 <class 'type'> 对象
d = deque()                    # 直接用类名
```

- 拿到的是 **类对象**（type 对象）
- 跳过了模块引用，直接用
- 把名字拉进了当前命名空间

---

## 关键差异对比

| 维度 | `import module` | `from module import Class` |
|------|----------------|---------------------------|
| 绑定的对象 | module 对象 | 类对象 (type) |
| 访问方式 | `module.Class` | 直接用 `Class` |
| 命名空间污染 | 不污染，一个名字对应一个模块 | 会污染，可能覆盖同名变量 |
| 模块重载 | `importlib.reload(m)` 后所有 `m.Class` 自动刷新 | reload 后旧的引用不变（快照隔离） |
| 同名冲突风险 | 低 | 高（容易覆盖当前作用域已有的同名变量） |

---

## 深入理解

### 1. 执行机制相同

```python
from collections import deque
```

**仍然会执行整个 `collections` 模块的代码**（首次导入时），不是只导入 `deque` 一个类。Python 的导入流程是：
1. 找到模块文件
2. 执行模块顶层代码（只执行一次，结果缓存到 `sys.modules`）
3. 从模块命名空间中取出指定的名字绑定到当前作用域

```python
import collections         # 步骤：1→2→把 collections 绑定到当前作用域
from collections import deque  # 步骤：1→2→把 deque 绑定到当前作用域
```

第 1、2 步完全一样。区别只在第 3 步：绑了什么名字。

### 2. 模块重载的行为差异

```python
import importlib
import collections

# hot reload 整个模块后，collections.deque 自动指向新版本
importlib.reload(collections)
```

```python
from collections import deque

# reload 后 deque 仍然指向旧对象，不自动刷新
import collections
importlib.reload(collections)
deque is collections.deque  # False，deque 是旧版本
```

**原因**：`from import` 拿到的是值引用（类对象的引用），reload 改变了 `collections.deque` 这个名字的指向，但不会回溯更新之前已经取走的名字。

### 3. 命名空间污染

```python
# 安全：一个名字一个模块
import os
import sys
```

```python
# 危险：deque 可能覆盖当前文件内已有的 deque
deque = "自己定义的 deque 变量"
from collections import deque  # 覆盖了上面的字符串
# deque 不再是变量，变成了类
```

### 4. __name__ 属性不同

```python
import collections
print(collections.__name__)  # 'collections'

from collections import deque
print(deque.__name__)        # 'deque'
print(deque.__module__)      # 'collections'（可通过这个知道它来自哪个模块）
```

---

## 选择建议

| 场景 | 推荐方式 |
|------|---------|
| 需要调用模块内很多类/函数 | `import module` 或 `import module as m` |
| 只用一个类且名字不冲突 | `from module import Class` |
| 需要 hot reload | `import module`（from import 拿到的引用不刷新） |
| 类名太长 | `from module import LongClassName as Short` |
| 大型项目、严肃工程 | 倾向 `import module`，命名空间清晰 |
| 脚本、快速原型 | `from module import Something`，少打字 |
