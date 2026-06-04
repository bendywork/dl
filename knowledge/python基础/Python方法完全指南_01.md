# Python 方法完全指南

## 📌 核心问题

> Python 中的"方法"并非只有一种。不同类型的方法决定了：谁能调用它？它能访问什么？它的第一个参数是什么？  
> 理解这四种方法，是写出结构清晰、可维护的 Python 代码的基础。

---

## 🌱 根源与动机

Python 的面向对象设计借鉴了 Smalltalk 和 C++，但加入了更灵活的函数模型。  
核心问题是：**一个函数，应该绑定到实例、类、还是什么都不绑定？**

| 问题 | 对应方法类型 |
|------|------------|
| 我需要访问/修改实例数据 | 实例方法 |
| 我需要访问/修改类本身 | 类方法 `@classmethod` |
| 我只是一个工具函数，放在类里只为逻辑归类 | 静态方法 `@staticmethod` |
| 我是模块级别的工具函数 | 模块级函数 |

---

## 📐 四种方法详解

### 1. 模块级函数（Module-level Function）

**定义方式：** 直接在模块顶层定义，不属于任何类。

```python
# split_tokens.py
def split_by_char(text: str) -> list[str]:
    """按字符切分文本"""
    return list(text)
```

**第一个参数：** 普通参数，没有 `self` 或 `cls`。

**调用方式：**
```python
# 直接调用（在同模块内）
result = split_by_char("你好世界")

# 导入后调用
from split_tokens import split_by_char
result = split_by_char("你好世界")

# 或导入整个模块
import split_tokens
result = split_tokens.split_by_char("你好世界")
```

**典型使用场景：**
- 通用工具函数，与任何类无关
- 数据预处理、格式转换
- 脚本入口 `main()` 函数

---

### 2. 实例方法（Instance Method）

**定义方式：** 类内部定义，第一个参数必须是 `self`（约定俗成，不是关键字）。

```python
class Tokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab          # 实例属性
        self.unk_token = "<UNK>"

    def encode(self, text: str) -> list[int]:
        """将文本转换为 token id 列表（实例方法）"""
        return [self.vocab.get(char, 0) for char in text]
```

**第一个参数：** `self`，指向调用该方法的实例对象本身。

**调用方式：**
```python
tokenizer = Tokenizer(vocab={"你": 1, "好": 2})
ids = tokenizer.encode("你好")   # 通过实例调用，self 自动传入
# 等价于：Tokenizer.encode(tokenizer, "你好")  ← 不常用但合法
```

**典型使用场景：**
- 需要读取或修改 `self.xxx` 实例属性
- 每个实例的行为取决于自身状态
- 绝大多数业务方法

---

### 3. 类方法（Class Method）

**定义方式：** 使用 `@classmethod` 装饰器，第一个参数必须是 `cls`。

```python
class Tokenizer:
    _default_vocab_path = "vocab.json"   # 类属性

    def __init__(self, vocab: dict):
        self.vocab = vocab

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        """从文件加载词表，创建 Tokenizer 实例（工厂方法）"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)   # cls 就是 Tokenizer 本身，等价于 Tokenizer(vocab)

    @classmethod
    def set_default_path(cls, path: str):
        """修改类属性"""
        cls._default_vocab_path = path
```

**第一个参数：** `cls`，指向类本身（不是实例）。

**调用方式：**
```python
# 通过类名调用（最常见）
tokenizer = Tokenizer.from_file("vocab.json")

# 通过实例调用也可以（但不推荐，容易混淆）
t = Tokenizer({})
tokenizer2 = t.from_file("vocab.json")  # cls 仍然是 Tokenizer，不是 t
```

**典型使用场景：**
- 工厂方法（`from_file`、`from_config`、`from_pretrained`）
- 修改或读取类属性
- 子类继承时需要返回正确类型（`cls(...)` 而非 `Tokenizer(...)`）

---

### 4. 静态方法（Static Method）

**定义方式：** 使用 `@staticmethod` 装饰器，没有 `self` 或 `cls` 参数。

```python
class SplitTokensTools:

    @staticmethod
    def is_chinese_char(char: str) -> bool:
        """判断是否为中文字符（纯工具函数，不需要访问实例或类）"""
        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF or   # CJK 统一汉字
            0x3400 <= code <= 0x4DBF       # CJK 扩展A
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本，去除多余空格"""
        return " ".join(text.split())
```

**第一个参数：** 无隐式参数，与普通函数完全相同。

**调用方式：**
```python
# 通过类名调用
result = SplitTokensTools.is_chinese_char("你")   # True

# 通过实例也可以调用（不推荐）
tools = SplitTokensTools()
result = tools.is_chinese_char("A")   # False
```

**典型使用场景：**
- 逻辑上属于这个类，但不需要访问类或实例数据
- 纯工具/辅助函数（验证、格式化、判断）
- 希望通过类名调用以体现归属关系

---

## 💡 四种方法对比表格

| 维度 | 模块级函数 | 实例方法 | 类方法 | 静态方法 |
|------|-----------|---------|--------|---------|
| 定义位置 | 模块顶层 | 类内部 | 类内部 | 类内部 |
| 装饰器 | 无 | 无 | `@classmethod` | `@staticmethod` |
| 第一个参数 | 普通参数 | `self`（实例） | `cls`（类） | 无隐式参数 |
| 能访问实例属性 | ❌ | ✅ | ❌ | ❌ |
| 能访问类属性 | ❌ | ✅（通过 `self.__class__`） | ✅ | ❌ |
| 能修改实例状态 | ❌ | ✅ | ❌ | ❌ |
| 能修改类状态 | ❌ | 间接可以 | ✅ | ❌ |
| 调用方式 | `func()` | `instance.method()` | `Class.method()` | `Class.method()` |
| 典型用途 | 通用工具 | 业务逻辑 | 工厂方法 | 辅助工具 |

---

## 🔧 代码示例

结合文本分类项目（Tokenizer + SplitTokensTools），完整演示四种方法协作：

```python
# ============================================================
# 文件结构：
#   text_specify/
#   ├── split_tokens.py      ← 模块级函数
#   ├── SplitTokensTools.py  ← 静态方法
#   ├── Tokenizer.py         ← 实例方法 + 类方法
#   └── TextClassifyDataset.py
# ============================================================

# --------- split_tokens.py（模块级函数）---------
def split_by_whitespace(text: str) -> list[str]:
    """按空格切分（模块级函数）"""
    return text.strip().split()


def split_by_char(text: str) -> list[str]:
    """按字符切分（模块级函数）"""
    return list(text.strip())


# --------- SplitTokensTools.py（静态方法）---------
class SplitTokensTools:
    """切词工具集合：纯工具函数，用类名命名空间来组织"""

    @staticmethod
    def is_chinese_char(char: str) -> bool:
        code = ord(char)
        return 0x4E00 <= code <= 0x9FFF

    @staticmethod
    def remove_punctuation(text: str) -> str:
        import re
        return re.sub(r"[^\w\s一-鿿]", "", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return " ".join(text.split())


# --------- Tokenizer.py（实例方法 + 类方法）---------
import json


class Tokenizer:
    PAD_TOKEN = "<PAD>"    # 类属性
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab                          # 实例属性
        self.id2token = {v: k for k, v in vocab.items()}

    # --- 实例方法 ---
    def encode(self, text: str) -> list[int]:
        """文本 → token id 列表"""
        return [self.vocab.get(char, 0) for char in text]

    def decode(self, ids: list[int]) -> str:
        """token id 列表 → 文本"""
        return "".join(self.id2token.get(i, self.UNK_TOKEN) for i in ids)

    def __len__(self) -> int:
        """返回词表大小（魔术方法）"""
        return len(self.vocab)

    # --- 类方法（工厂方法）---
    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        """从 JSON 文件加载词表，返回 Tokenizer 实例"""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)   # 用 cls 而不是 Tokenizer，子类也能正确工作

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Tokenizer":
        """从文本列表自动构建词表"""
        vocab = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 1}
        for text in texts:
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
        return cls(vocab)


# --------- 使用示例 ---------
if __name__ == "__main__":
    # 使用模块级函数
    from split_tokens import split_by_char
    chars = split_by_char("深度学习很有趣")
    print(chars)  # ['深', '度', '学', '习', '很', '有', '趣']

    # 使用静态方法（通过类名）
    is_cn = SplitTokensTools.is_chinese_char("深")
    print(is_cn)  # True

    # 使用类方法（工厂方法）
    texts = ["深度学习", "自然语言处理", "文本分类"]
    tokenizer = Tokenizer.from_texts(texts)
    print(f"词表大小: {len(tokenizer)}")  # 调用 __len__

    # 使用实例方法
    ids = tokenizer.encode("深度学习")
    print(f"编码结果: {ids}")
    decoded = tokenizer.decode(ids)
    print(f"解码结果: {decoded}")
```

---

## 导入方式详解

### from xxx import xxx 的几种形式

```python
# 1. 导入模块级函数
from split_tokens import split_by_char, split_by_whitespace
result = split_by_char("文本")   # 直接用函数名

# 2. 导入整个模块
import split_tokens
result = split_tokens.split_by_char("文本")  # 需要模块名前缀

# 3. 导入类
from Tokenizer import Tokenizer
t = Tokenizer(vocab={})          # 先实例化，再用实例方法

# 4. 导入类后调用类方法（不需要实例化）
from Tokenizer import Tokenizer
t = Tokenizer.from_file("vocab.json")   # 类方法，直接用类名调用

# 5. 导入类后调用静态方法（不需要实例化）
from SplitTokensTools import SplitTokensTools
result = SplitTokensTools.is_chinese_char("你")   # 静态方法，直接用类名
```

### 类导入 vs 实例化的区别

```python
from Tokenizer import Tokenizer

# ❌ 错误：直接用类名调用实例方法
# Tokenizer.encode("文本")   # TypeError: encode() missing 1 required positional argument: 'text'
#                               ↑ 因为 encode 是实例方法，需要 self

# ✅ 正确：先实例化，再调用实例方法
tokenizer = Tokenizer(vocab={"你": 1})   # 实例化
ids = tokenizer.encode("你好")            # 实例方法，self=tokenizer

# ✅ 正确：类方法不需要实例化，直接用类名
tokenizer2 = Tokenizer.from_file("vocab.json")   # 类方法

# ✅ 正确：静态方法不需要实例化
clean = SplitTokensTools.clean_text("  文本  ")  # 静态方法
```

**核心规则：**
- 实例方法 → 必须先 `obj = Class(...)` 再 `obj.method()`
- 类方法 / 静态方法 → 可以直接 `Class.method()` 不需要实例化

---

## ⚠️ 易错点与常见误解

### 1. 忘写 `self`，导致实例属性访问失败

```python
# ❌ 错误：忘写 self
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(text):          # 少了 self！
        return [self.vocab.get(c, 0) for c in text]   # NameError: name 'self' is not defined

# ✅ 正确
    def encode(self, text):    # 第一个参数必须是 self
        return [self.vocab.get(c, 0) for c in text]
```

---

### 2. 加了 `@staticmethod` 但还是写了 `self`

```python
# ❌ 错误：静态方法里写 self，但 Python 不会自动传入
class SplitTokensTools:
    @staticmethod
    def is_chinese(self, char):   # self 是普通参数，调用时必须手动传！
        return 0x4E00 <= ord(char) <= 0x9FFF

# 调用时：SplitTokensTools.is_chinese("你")  → TypeError：缺少 char 参数

# ✅ 正确：静态方法不要 self
    @staticmethod
    def is_chinese(char):
        return 0x4E00 <= ord(char) <= 0x9FFF
```

---

### 3. 位运算符 `|` 误用成 `or`（条件判断中）

```python
# ❌ 错误：| 是位运算，不是逻辑运算
if 0x4E00 <= code <= 0x9FFF | 0x3400 <= code <= 0x4DBF:  # 运算符优先级混乱！
    pass

# ✅ 正确：逻辑"或"用 or
if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
    pass

# 注意：| 用于集合合并或类型联合（Python 3.10+）
vocab1 | vocab2          # 集合合并（正确用途）
int | str                # 类型联合注解（正确用途）
```

---

### 4. 判空应用 `not text` vs `is None` vs `len(text) == 0`

```python
text = ""
tokens = []

# ❌ 容易混淆的判空
if not text:          # True，但包含了 None、""、0、[]，语义模糊
    pass

# ✅ 明确判断是否为 None
if text is None:      # 只判断是否为 None，空字符串不触发
    raise ValueError("text 不能为 None")

# ✅ 明确判断是否为空字符串
if len(text) == 0:    # 只判断长度
    return []

# ✅ 实际工程中的推荐写法（先 None 后空）
def encode(self, text: str) -> list[int]:
    if text is None:
        raise ValueError("text 不能为 None")
    if len(text) == 0:
        return []
    return [self.vocab.get(c, 0) for c in text]
```

**判空选择指南：**

| 判断方式 | 触发条件 | 适用场景 |
|---------|---------|---------|
| `not x` | `None`、`""`、`0`、`[]`、`{}` | 快速"有没有值"，语义宽泛 |
| `x is None` | 仅 `None` | 参数校验，明确区分 None 和空值 |
| `len(x) == 0` | 空列表/字符串等 | 明确要求非空集合/字符串 |
| `x == ""` | 仅空字符串 | 严格匹配空字符串 |

---

## 🔗 知识延伸

### 命名规范

| 命名风格 | 适用对象 | 示例 |
|---------|---------|------|
| `snake_case` | 函数名、方法名、变量名、模块名 | `split_by_char`, `vocab_size` |
| `PascalCase`（大驼峰） | 类名 | `Tokenizer`, `SplitTokensTools` |
| `UPPER_SNAKE_CASE` | 常量 | `MAX_SEQ_LEN = 512` |
| `_single_underscore` | 约定私有（外部不应访问） | `_vocab_cache` |
| `__double_underscore` | 真正私有（名称改写，防子类覆盖） | `__secret_key` |
| `__magic__` | 魔术方法（dunder） | `__init__`, `__len__`, `__repr__` |

**下划线前缀的本质区别：**

```python
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab              # 公开属性，外部可随意访问
        self._cache = {}                # 约定私有，外部"不应该"访问（但技术上可以）
        self.__compiled = None          # 真正私有，名称改写为 _Tokenizer__compiled

    def _build_cache(self):             # 约定内部方法
        pass

    def __repr__(self):                 # 魔术方法，控制 print(tokenizer) 的输出
        return f"Tokenizer(vocab_size={len(self.vocab)})"
```

### 方法解析顺序（MRO）

Python 用 C3 线性化算法确定多继承时的方法查找顺序：

```python
class CharTokenizer(Tokenizer):
    @classmethod
    def from_file(cls, path: str) -> "CharTokenizer":
        # cls 是 CharTokenizer，不是 Tokenizer
        # 这就是为什么要用 cls(...) 而不是 Tokenizer(...)
        return super().from_file(path)   # 调用父类的 from_file
```

### 与其他知识点的联系

- `@property` 装饰器：将方法伪装成属性访问，`tokenizer.vocab_size` 而非 `tokenizer.get_vocab_size()`
- `__slots__`：限制实例属性，减少内存占用（大量小对象时有用）
- 描述符协议：`__get__`、`__set__`、`__delete__`，是 `@property` 和方法绑定的底层实现
- 抽象方法 `@abstractmethod`：强制子类实现特定方法（配合 `ABC` 基类使用）

---

## 📚 参考资料

- Python 官方文档：[Descriptor HowTo Guide](https://docs.python.org/3/howto/descriptor.html)
- PEP 8：Python 代码风格指南（snake_case / PascalCase 规范）
- PEP 3135：`super()` 无参数调用
