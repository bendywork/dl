
# Python 代码规范

## 📌 核心问题

> Python 写法灵活，但"能跑"不等于"规范"。本文整合 PEP 8、Google Style Guide 与大厂实践，涵盖命名、分包、类型注解、类写法、函数、导入、Lint 工具链，帮你从一开始就养成工业级习惯。

---

## 命名规范

### 总览

| 场景 | 风格 | 示例 |
|------|------|------|
| 类名 | PascalCase | `TextClassifier` |
| 函数 / 方法 / 变量 | snake_case | `load_model`, `batch_size` |
| 常量 | UPPER_SNAKE_CASE | `MAX_SEQ_LEN = 512` |
| 私有成员（约定） | `_single_underscore` | `_hidden_dim` |
| 名称改写（强私有） | `__double_underscore` | `__weights` |
| 模块 / 包名 | 全小写，下划线分隔 | `data_loader`, `model_utils` |

### 代码示例

```python
# ✅ 正确命名示范
MAX_VOCAB_SIZE = 50000          # 常量：UPPER_SNAKE_CASE
LEARNING_RATE = 1e-4


class TextClassifier:           # 类：PascalCase
    """基于 RNN 的文本分类器。"""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        self.vocab_size = vocab_size    # 公共属性：snake_case
        self._hidden_dim = hidden_dim   # 约定私有（外部尽量不访问）
        self.__weights = None           # 强私有（名称改写保护）

    def forward(self, x):               # 方法：snake_case
        ...

    def _compute_loss(self, logits, labels):  # 内部辅助方法
        ...


def load_pretrained_model(model_path: str) -> TextClassifier:  # 函数：snake_case
    ...


# ❌ 错误示范
class text_classifier: ...      # 类名不用 snake_case
def LoadModel(): ...            # 函数名不用 PascalCase
maxVocabSize = 50000            # 不用 camelCase
```

### 私有成员说明

```python
class MyModel:
    def __init__(self):
        self.public_attr = 1        # 完全公开
        self._internal_attr = 2     # 约定私有，外部可访问但不推荐
        self.__mangled_attr = 3     # 名称改写：实际存储为 _MyModel__mangled_attr

m = MyModel()
print(m.public_attr)        # ✅ 正常访问
print(m._internal_attr)     # ⚠️ 可访问，但不推荐
# print(m.__mangled_attr)   # ❌ AttributeError
print(m._MyModel__mangled_attr)  # ✅ 强行访问（不推荐）
```

---

## 分包规范

### 典型 ML/DL 项目目录结构

```
my_nlp_project/
├── README.md
├── requirements.txt
├── setup.py                    # 可安装包配置
├── pyproject.toml              # 现代构建配置（推荐）
├── .env                        # 环境变量（不提交 git）
├── .gitignore
│
├── configs/                    # 配置文件（yaml/json）
│   ├── model_config.yaml
│   └── train_config.yaml
│
├── data/                       # 原始数据（通常不提交 git）
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/                  # Jupyter 探索性分析
│   └── 01_eda.ipynb
│
├── scripts/                    # 可执行脚本入口
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
│
├── src/                        # 核心源码包（可安装）
│   └── my_nlp_project/
│       ├── __init__.py
│       ├── data/               # 数据处理模块
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── tokenizer.py
│       ├── models/             # 模型定义
│       │   ├── __init__.py
│       │   ├── rnn.py
│       │   └── transformer.py
│       ├── training/           # 训练逻辑
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── loss.py
│       └── utils/              # 通用工具
│           ├── __init__.py
│           ├── logger.py
│           └── metrics.py
│
├── tests/                      # 测试（镜像 src 结构）
│   ├── __init__.py
│   ├── test_data/
│   └── test_models/
│
└── outputs/                    # 训练输出（不提交 git）
    ├── checkpoints/
    └── logs/
```

### 各层职责说明

| 目录 | 职责 |
|------|------|
| `configs/` | 超参数、路径等配置，与代码解耦 |
| `data/` | 原始数据，只读，通过脚本生成 processed |
| `notebooks/` | 探索性分析，不放业务逻辑 |
| `scripts/` | 命令行入口，只做参数解析和调用 |
| `src/` | 可安装的核心业务代码 |
| `tests/` | 测试代码，目录结构镜像 src |
| `outputs/` | 模型 checkpoint、日志，不提交 git |

### `__init__.py` 使用规范

```python
# src/my_nlp_project/__init__.py
# 顶层包：只暴露最核心的公共接口，避免循环导入

from .models.rnn import RNNClassifier
from .models.transformer import TransformerClassifier

__version__ = "0.1.0"
__all__ = ["RNNClassifier", "TransformerClassifier"]


# src/my_nlp_project/models/__init__.py
# 子包：按需暴露，方便外部 from models import xxx

from .rnn import RNNClassifier
from .transformer import TransformerClassifier

# 如果什么都不暴露，保持空文件也可以
# 空 __init__.py 只是标记这是一个包
```

---

## 类型注解规范

### 基本类型注解

```python
# Python 3.9+ 可直接用内置类型
def greet(name: str) -> str:
    return f"Hello, {name}"

def add(a: int, b: int) -> int:
    return a + b

def process(values: list[int]) -> dict[str, int]:
    return {"sum": sum(values), "count": len(values)}
```

### Optional、Union 等用法

```python
from typing import Optional, Union

# Optional[X] 等价于 Union[X, None]，表示"可以是 X 也可以是 None"
def find_token(token_id: int, vocab: dict[int, str]) -> Optional[str]:
    return vocab.get(token_id)  # 找不到返回 None


# Python 3.10+ 可用 X | None 简写
def find_token_v2(token_id: int, vocab: dict[int, str]) -> str | None:
    return vocab.get(token_id)


# Union：多种类型之一
def encode(text: Union[str, list[str]]) -> list[int]:
    if isinstance(text, str):
        text = [text]
    return [hash(t) for t in text]
```

### List、Dict、Tuple 精确写法

```python
from typing import List, Dict, Tuple  # Python 3.8 及以下需要这样导入

# Python 3.9+ 推荐直接用内置
def batch_encode(texts: list[str]) -> list[list[int]]:
    ...

def get_label_map() -> dict[str, int]:
    ...

# Tuple：固定长度，每个位置类型可不同
def split_dataset(data: list) -> tuple[list, list, list]:
    # 返回 train, val, test 三个列表
    ...

# ⚠️ 易错：不要写成 tuple[str:int]（冒号是切片语法，不是类型注解语法）
# ❌ 错误
def wrong() -> tuple[str:int]: ...

# ✅ 正确：逗号分隔
def correct() -> tuple[str, int]: ...

# 更复杂的例子：返回两个映射字典
def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    word2idx = {w: i for i, w in enumerate(tokens)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word
```

### dataclass 标准写法

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置类。"""
    vocab_size: int                         # 必填，无默认值
    hidden_dim: int = 256                   # 有默认值
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-4
    device: str = "cuda"

    # ⚠️ 可变默认值必须用 field(default_factory=...)
    # 不能写 layer_dims: list[int] = [256, 128]  ← 所有实例共享同一个列表！
    layer_dims: list[int] = field(default_factory=lambda: [256, 128])
    tags: dict[str, str] = field(default_factory=dict)

    # Optional 字段
    pretrained_path: Optional[str] = None


# 使用示例
cfg = ModelConfig(vocab_size=10000)
print(cfg)
# ModelConfig(vocab_size=10000, hidden_dim=256, num_layers=2, ...)
```

---

## 类的写法规范

### Python vs Java 类的主要区别

| 特性 | Python | Java |
|------|--------|------|
| 访问控制 | 约定（`_` / `__`），无强制 | `private` / `protected` / `public` |
| getter/setter | 用 `@property`，不写无意义的 get/set | 通常显式写 |
| 接口 | 用 ABC 或 Protocol | `interface` 关键字 |
| 重载 | 不支持，用默认参数 / `*args` | 支持方法重载 |
| 构造函数 | `__init__`，可只有一个 | 多构造函数重载 |

### dataclass vs 普通类的选择原则

```python
# ✅ 用 dataclass：纯数据容器，主要存属性，行为少
@dataclass
class TrainBatch:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


# ✅ 用普通类：有复杂初始化逻辑、大量方法、继承关系
class RNNClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._build_layers()

    def _build_layers(self):
        ...

    def forward(self, x):
        ...

    def predict(self, text: str) -> str:
        ...
```

### 魔术方法规范

```python
class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        self._word2idx: dict[str, int] = {w: i for i, w in enumerate(tokens)}
        self._idx2word: dict[int, str] = {i: w for w, i in self._word2idx.items()}

    def __repr__(self) -> str:
        """供开发者调试用，应能反映对象状态。"""
        return f"Vocabulary(size={len(self._word2idx)})"

    def __str__(self) -> str:
        """供用户友好展示用（print 时调用）。"""
        return f"词表（{len(self._word2idx)} 个词）"

    def __len__(self) -> int:
        """支持 len(vocab)。"""
        return len(self._word2idx)

    def __contains__(self, token: str) -> bool:
        """支持 token in vocab。"""
        return token in self._word2idx

    def __getitem__(self, token: str) -> int:
        """支持 vocab['hello'] 语法。"""
        return self._word2idx[token]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vocabulary):
            return NotImplemented
        return self._word2idx == other._word2idx


# 使用
vocab = Vocabulary(["hello", "world", "AI"])
print(len(vocab))           # 3
print("hello" in vocab)     # True
print(vocab["hello"])       # 0
print(repr(vocab))          # Vocabulary(size=3)
```

### @property 替代 getter/setter

```python
class Model:
    def __init__(self, lr: float) -> None:
        self._lr = lr

    # ✅ Python 风格：用 @property
    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"学习率必须为正数，得到 {value}")
        self._lr = value

    # ❌ Java 风格（不推荐）
    def get_lr(self) -> float:
        return self._lr

    def set_lr(self, value: float) -> None:
        self._lr = value


m = Model(lr=1e-3)
print(m.lr)         # ✅ 直接访问，像属性
m.lr = 1e-4         # ✅ 像赋值，自动触发验证
```

---

## 函数规范

### 参数命名与设计

```python
# ✅ 参数名具体、有意义
def compute_accuracy(predictions: list[int], labels: list[int]) -> float:
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


# ✅ 合理使用 * 强制关键字参数（避免位置混淆）
def train(
    model,
    dataloader,
    *,                          # 之后的参数必须用关键字传递
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    ...

# 调用时必须写关键字
train(model, loader, epochs=5, lr=1e-4)
# train(model, loader, 5, 1e-4)  ❌ TypeError
```

### 可变默认值陷阱

```python
# ❌ 经典陷阱：默认值是列表/字典，所有调用共享同一对象
def append_token(token: str, token_list: list = []) -> list:
    token_list.append(token)
    return token_list

print(append_token("hello"))   # ['hello']
print(append_token("world"))   # ['hello', 'world']  ← 不是 ['world']！！


# ✅ 正确：用 None 作默认值，函数内创建
def append_token_correct(token: str, token_list: list | None = None) -> list:
    if token_list is None:
        token_list = []
    token_list.append(token)
    return token_list

print(append_token_correct("hello"))   # ['hello']
print(append_token_correct("world"))   # ['world']  ✅
```

### Docstring 规范（Google Style）

```python
def build_vocab(
    texts: list[str],
    max_size: int = 10000,
    min_freq: int = 1,
) -> tuple[dict[str, int], dict[int, str]]:
    """从文本列表构建词汇表。

    Args:
        texts: 训练文本列表，每个元素是一个句子。
        max_size: 词表最大大小，超出部分按频率截断。
        min_freq: 词频阈值，低于此值的词不计入词表。

    Returns:
        包含两个元素的元组：
        - word2idx: 词到索引的映射字典。
        - idx2word: 索引到词的映射字典。

    Raises:
        ValueError: 当 texts 为空列表时抛出。

    Example:
        >>> texts = ["hello world", "world is great"]
        >>> w2i, i2w = build_vocab(texts, max_size=100)
        >>> w2i["world"]
        0
    """
    if not texts:
        raise ValueError("texts 不能为空")

    from collections import Counter
    counter = Counter(w for text in texts for w in text.split())
    vocab = [w for w, freq in counter.most_common(max_size) if freq >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word
```

---

## 导入规范

### 三段式导入（严格顺序）

```python
# 第一段：标准库
import os
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

# 第二段：第三方库（空一行隔开）
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 第三段：本地模块（空一行隔开）
from src.models.rnn import RNNClassifier
from src.data.tokenizer import Tokenizer
from src.utils.metrics import compute_f1
```

### 导入规则

```python
# ✅ 明确导入，避免命名空间污染
from collections import Counter, defaultdict

# ❌ 禁止 star import，调试噩梦
from numpy import *
from torch.nn import *


# ✅ 长导入用括号换行
from src.training.trainer import (
    Trainer,
    TrainerConfig,
    EarlyStopping,
)

# ✅ 别名：仅在约定俗成时使用
import numpy as np          # ✅ 业界标准别名
import pandas as pd         # ✅ 业界标准别名
import matplotlib.pyplot as plt  # ✅ 业界标准别名

# ❌ 随意起别名，可读性变差
import torch as t           # ❌ 没人这样写
```

---

## Lint 工具链

### 各工具职责

| 工具 | 职责 | 典型命令 |
|------|------|---------|
| `black` | 代码格式化（自动修复） | `black src/ tests/` |
| `isort` | import 排序（自动修复） | `isort src/ tests/` |
| `flake8` | 风格检查（不自动修复） | `flake8 src/` |
| `pylint` | 深度静态分析，检查逻辑问题 | `pylint src/` |
| `mypy` | 类型检查 | `mypy src/` |

### 推荐配置文件

**`pyproject.toml`（推荐，统一管理）**

```toml
[tool.black]
line-length = 88
target-version = ["py39", "py310"]

[tool.isort]
profile = "black"           # 与 black 兼容
line_length = 88
known_first_party = ["src"]

[tool.mypy]
python_version = "3.10"
strict = true               # 开启严格模式
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["numpy.*", "torch.*"]
ignore_missing_imports = true
```

**`.flake8`**

```ini
[flake8]
max-line-length = 88        # 与 black 保持一致
extend-ignore = E203, W503  # black 格式化会触发的误报
exclude =
    .git,
    __pycache__,
    .venv,
    build,
    dist
```

### 推荐工作流

```bash
# 开发时：先格式化，再检查
isort src/ tests/
black src/ tests/
flake8 src/ tests/
mypy src/

# 或者用 pre-commit hook 自动在 commit 前执行
# .pre-commit-config.yaml 示例：
# repos:
#   - repo: https://github.com/psf/black
#     hooks: [id: black]
#   - repo: https://github.com/pycqa/isort
#     hooks: [id: isort]
#   - repo: https://github.com/pycqa/flake8
#     hooks: [id: flake8]
```

---

## ⚠️ 易错点汇总

### 1. tuple 类型注解用逗号，不用冒号

```python
# ❌ 错误：冒号是切片语法，用在类型注解里是 bug
def wrong() -> tuple[str:int]: ...

# ✅ 正确：逗号分隔每个位置的类型
def correct() -> tuple[str, int]: ...
def build_maps() -> tuple[dict[str, int], dict[int, str]]: ...
```

### 2. 可变默认参数陷阱

```python
# ❌ 错误：列表/字典作默认值，所有调用共享
def bad(items: list = []) -> list: ...

# ✅ 正确：用 None，函数内初始化
def good(items: list | None = None) -> list:
    if items is None:
        items = []
    return items
```

### 3. dataclass 可变字段必须用 field()

```python
# ❌ 错误：所有实例共享同一个列表对象
@dataclass
class Bad:
    layers: list[int] = [256, 128]  # ValueError!

# ✅ 正确
@dataclass
class Good:
    layers: list[int] = field(default_factory=lambda: [256, 128])
```

### 4. `__repr__` vs `__str__` 用途不同

```python
# __repr__：给开发者看，应能重建对象（或包含调试信息）
def __repr__(self) -> str:
    return f"Model(hidden_dim={self.hidden_dim}, layers={self.num_layers})"

# __str__：给用户看，可读性优先
def __str__(self) -> str:
    return f"文本分类模型（词表大小：{self.vocab_size}）"
```

### 5. from xxx import * 污染命名空间

```python
# ❌ 永远不要用，尤其在大型项目中
from numpy import *     # 会覆盖内置函数，引入未知名称

# ✅ 明确导入
import numpy as np
from numpy import ndarray, zeros
```

### 6. 类名误用 snake_case

```python
# ❌ 错误
class text_classifier: ...
class rnn_model: ...

# ✅ 正确
class TextClassifier: ...
class RNNModel: ...
```

---

## 参考资料

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 — Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 557 — Data Classes](https://peps.python.org/pep-0557/)
- [black 文档](https://black.readthedocs.io/)
- [mypy 文档](https://mypy.readthedocs.io/)
