# 机器学习复习 Day01

## 1. Python 文档注释（Docstring）

### 读取 docstring
```python
def greet(name):
    """向用户打招呼。"""
    return f"Hello, {name}"

print(greet.__doc__)   # 直接访问 __doc__ 属性
help(greet)            # 格式化输出
```

### 常见格式

**Google 风格：**
```python
def func(x, y):
    """计算两数之和。

    Args:
        x (int): 第一个数
        y (int): 第二个数

    Returns:
        int: 两数之和
    """
```

**NumPy 风格：**
```python
def func(x, y):
    """
    Parameters
    ----------
    x : int
    y : int

    Returns
    -------
    int
    """
```

### 用 inspect 模块操作
```python
import inspect
print(inspect.getdoc(greet))          # 清理缩进后的 docstring
print(inspect.cleandoc(greet.__doc__)) # 手动清理缩进
```

---

## 2. train_test_split 返回值顺序

### 常见 Bug

`train_test_split` 的返回顺序是 `x_train, x_test, y_train, y_test`，顺序写错会导致数据错乱。

```python
# ❌ 错误写法
self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(x, y, ...)

# ✅ 正确写法
self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, ...)
```

---

## 3. NumPy Shape 理解

| Shape | 含义 | 例子 |
|-------|------|------|
| `(800,)` | 1维，800个元素 | `[0, 1, 1, 0, ...]` |
| `(800, 1)` | 2维，800行1列 | `[[0], [1], [1], ...]` |
| `(800, 2)` | 2维，800行2列 | `[[1.2, 0.5], [0.3, 1.1], ...]` |

### 为什么 y 是 `(n,)` 而 x 是 `(n, 2)`

- `x`：每个样本有多个特征（坐标值），必须是二维
- `y`：每个样本只有一个标签值，天然是一维

```python
x, y = make_circles(n_samples=1000, ...)
print(x.shape)  # (1000, 2) → 1000个点，每点2个坐标
print(y.shape)  # (1000,)   → 1000个标签，每点1个类别（0或1）
```

---

## 4. 分类类型与 y 的 Shape

| 分类类型 | 例子 | y的shape | 说明 |
|----------|------|----------|------|
| 二分类 | 是猫/不是猫 | `(n,)` | 每样本1个值 |
| 多分类 | 数字0-9 | `(n,)` | 每样本1个值，值域扩展 |
| 多标签分类 | 文章同时属于多个话题 | `(n, 类别数)` | 每样本多个值 |

### 多标签分类（Multi-label Classification）

一个样本同时属于多个类别，y 必须是二维：

```python
# 3个类别：0=科技  1=财经  2=体育
y_raw = [
    {0, 2},   # 科技 + 体育
    {1, 2},   # 财经 + 体育
    {0, 1},   # 科技 + 财经
]

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)
print(y.shape)  # (3, 3)
print(y)
# [[1 0 1]
#  [0 1 1]
#  [1 1 0]]
```

### 理解 `(3, 3)` 的含义

```
(样本数, 类别总数)
```

每一行是一个样本，每一列是一个类别，格子里填 0 或 1 表示"这个样本是否属于这个类别"。

对单个样本来说，y 是一个向量，每一位对应一个类别的是/否判断：

```
样本1的y = [1, 0, 1]
           ↑  ↑  ↑
          类0 类1 类2
          是  否  是
```

> 多标签分类本质是**同时做多个二分类**，每个类别独立问一次"属不属于"。

### One-Hot 与多标签的关系

| 场景 | y的shape | 说明 |
|------|----------|------|
| sklearn 分类器 | `(n,)` | 内部自动处理 |
| PyTorch CrossEntropyLoss | `(n,)` | 直接传整数类别 |
| Keras categorical_crossentropy | `(n, 类别数)` | 需要 One-Hot |
| Keras sparse_categorical_crossentropy | `(n,)` | 不需要 One-Hot |

---

## 5. 特征工程：多项式扩展

### 代码示例

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(x_train)  # 学习规则 + 转换
x_test = poly.transform(x_test)        # 只转换，不重新学习
```

### 转换效果

原始特征 `[x1, x2]` → 多项式扩展后 `[1, x1, x2, x1², x1·x2, x2²]`，从 2 列扩展到 6 列。

这样逻辑回归就能拟合圆形决策边界（线性模型本身无法分类圆形数据）。

---

## 6. fit_transform vs transform

### 核心原则

`fit` 的本质是**从数据中学习规则**，`transform` 是**用已学到的规则转换数据**。

```
训练集：fit_transform = 学习规则 + 按规则转换
测试集：transform     = 只按规则转换（规则已经学好了）
```

### 为什么测试集不能 fit

- 测试集模拟真实世界中未见过的新数据
- 现实预测时不可能拿到所有未来数据先 fit 一遍
- 如果重新 fit，训练集和测试集会用不同的标准，评估结果失去参考意义

**反例（StandardScaler）：**
```python
# 训练集：均值=10, 标准差=3  → 用这个标准化
# 测试集重新fit：均值=9.5, 标准差=2.8 → 规则变了！
# 模型在A规则数据上训练，却在B规则数据上测试 → 评估无效
```

### 为什么要保存 poly 为实例属性

```python
# ✅ 正确：保存为实例属性，后续预测新数据时复用
self.poly = PolynomialFeatures(degree=2)
self.x_train = self.poly.fit_transform(self.x_train)
self.x_test = self.poly.transform(self.x_test)
```

如果 `poly` 只是局部变量，方法结束后丢失，后续无法对新数据做一致的转换。

> **核心原则：** 规则只从训练集学一次，之后所有数据（测试集、生产数据）都必须服从这套规则，保证数据处于同一个"坐标系"下，这是防止**数据泄露**的关键。

---

## 7. 模型持久化：JSON vs joblib

训练好的模型需要保存下来，避免每次使用都重新训练。主要有两种方式。

### JSON 持久化（摘要参数存储）

```python
import json

json_dump_file = "./output/01/ml.json"
with open(json_dump_file, "w", encoding="utf-8") as writer:
    json.dump(
        {
            'poly': poly.get_feature_names_out(['x1', 'x2']).tolist(),  # 多项式组合规则
            'algo': {
                'intercept': algo.intercept_.tolist(),  # 截距项
                'coef': algo.coef_.tolist()             # 系数项
            }
        },
        writer,
        indent=2,           # 每级缩进2个空格，格式化输出
        ensure_ascii=False  # 中文直接输出，不转义为 \uXXXX
    )
```

**为什么要 `.tolist()`：** `numpy` 数组不是标准 Python 类型，`json.dump` 无法序列化，必须先转成 Python 原生 list。

**存储内容示例：**
```json
{
  "poly": ["1", "x1", "x2", "x1^2", "x1 x2", "x2^2"],
  "algo": {
    "intercept": [-0.123],
    "coef": [[1.23, -0.45, 2.11, 0.33, -1.02, 1.87]]
  }
}
```

### joblib 持久化（完整模型存储）

```python
import joblib

joblib.dump(model, "./output/01/ml.joblib")  # 保存
model = joblib.load("./output/01/ml.joblib") # 加载，直接推理
```

### 两种方式对比

| 对比项 | JSON（摘要参数） | joblib（完整模型） |
|--------|----------------|------------------|
| 存储内容 | 关键参数（系数、截距、规则） | 完整 Python 对象（含所有状态） |
| 跨语言 | ✅ Java/Go/C++ 等任何语言均可读取 | ❌ 仅限 Python |
| 恢复方式 | 需要手动重建模型（读参数→赋值） | 直接 `load` 即可推理 |
| 文件大小 | 小（只有数字） | 大（含完整对象结构） |
| 使用场景 | 跨语言部署、模型导出给其他系统 | Python 内部复用、快速原型 |
| 局限性 | 需要自己实现推理逻辑 | 强依赖 Python + sklearn 版本 |

### 核心结论

> - **JSON**：存的是"模型学到了什么"（参数摘要），任何语言都能读懂并复现推理逻辑，适合**生产部署、跨系统集成**
> - **joblib**：存的是"模型本身"（Python 对象序列化），拿来就用，适合**Python 内部快速复用**，但换个语言或 sklearn 版本可能就失效

你的理解完全正确。JSON 方式本质上是把模型的"知识"翻译成通用格式，而 joblib 是把 Python 对象直接冷冻保存。

---

## 8. 完整复习案例代码

```python
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class MachineLearning:
    def __init__(self, model=None):
        self.model = model
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.poly = None

    def init_data_set(self):
        x, y = make_circles(n_samples=1000, factor=0.1, noise=0.2, random_state=24)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=24
        )

    def feature_engineering(self):
        self.poly = PolynomialFeatures(degree=2)
        self.x_train = self.poly.fit_transform(self.x_train)
        self.x_test = self.poly.transform(self.x_test)

    def create_model(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        pred_train = self.model.predict(self.x_train)
        pred_test = self.model.predict(self.x_test)
        print(f"训练集准确率: {metrics.accuracy_score(self.y_train, pred_train)}")
        print(f"测试集准确率: {metrics.accuracy_score(self.y_test, pred_test)}")
        print(f"分类报告:\n{metrics.classification_report(self.y_test, pred_test)}")


if __name__ == "__main__":
    ml = MachineLearning()
    ml.init_data_set()
    ml.feature_engineering()
    ml.create_model()
    ml.train()
    ml.predict()
```

### 运行结果

```
训练集准确率: 0.9825
测试集准确率: 0.97
```

训练集与测试集准确率接近，说明模型没有过拟合，多项式特征对圆形决策边界拟合效果良好。
