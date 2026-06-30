# PyTorch 中置信度与 Softmax/交叉熵的关系

## 📌 核心问题
> 1. PyTorch 里什么叫置信度？和理论定义一样吗？
> 2. Softmax 本身不就是交叉熵损失函数吗？为什么说置信度在 softmax 之后、交叉熵之前？

## 🌱 根源：模型输出的完整数据流

```
输入 x → 模型 → logits → softmax → 概率(置信度) → 交叉熵 → 损失值
                      ^^^^^^^^^   ^^^^^^^^^^^^     ^^^^^^^^
                      归一化        模型的"看法"     打分
```

这三步是**独立操作**，不是一回事。

## 📐 问题一：置信度是什么

### 理论定义

```
置信度 = max(softmax(logits))
```

模型对**自己预测的类别**有多确定。就是 softmax 输出的最大概率值。

### PyTorch 中计算

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, 0.1]])
probs = F.softmax(logits, dim=1)

# 置信度 = max softmax 概率
confidence, predicted = probs.max(dim=1)
# confidence = 0.7285, predicted = 类别0
```

### PyTorch 和理论定义一致

| 概念 | 值 | 说明 |
|------|---|------|
| **置信度 (confidence)** | 0~1 | max softmax 概率 |
| **logit 最大值** | 任意实数 | softmax 之前的原始值，和置信度正相关但不等 |
| **softmax 概率分布** | 0~1 | 完整分布，所有类别加起来=1 |

## 📐 问题二：Softmax 和交叉熵是两回事

### 它们各自做什么

| 操作 | 做什么 | 输入 | 输出 |
|------|--------|------|------|
| **softmax** | 归一化成概率 | logits（任意实数） | 概率（0~1，和=1） |
| **交叉熵** | 衡量概率有多错 | 概率 + 真实标签 | 标量（损失值） |

- Softmax 只回答：**"模型认为各类别的概率是多少？"**
- 交叉熵只回答：**"这个概率分布和真实标签差多远？"**

### 拆开看是三步

```
第1步 softmax:  logits [2.0, 0.5, 0.1] → 概率 [0.73, 0.16, 0.11]
                只做归一化，跟损失无关

第2步 log:      概率 [0.73, 0.16, 0.11] → log概率 [-0.32, -1.82, -2.22]
                只做对数变换，还是跟损失无关

第3步 交叉熵:   取负 + 选正确类别 → Loss = 0.3168
                这一步才叫「算损失」
```

### PyTorch 打包的原因：数值稳定性

PyTorch 的 `CrossEntropyLoss` 把三步打包成一步：

```python
# 打包版（推荐）
ce = nn.CrossEntropyLoss()    # 传 logits，内部自动 softmax+log+取负
loss = ce(logits, target)

# 拆开版（等价，但不推荐）
probs = F.softmax(logits, dim=1)        # 第1步
log_probs = torch.log(probs)            # 第2步
loss = -log_probs[0, target]            # 第3步
```

**打包不是因为它俩是一回事，而是为了数值稳定性。** 先 softmax 再 log，对极小概率会算 `-log(0.0001)` 这种不稳定值。PyTorch 内部用 `log_softmax` 一步算，数学等价但数值更稳。

### PyTorch 提供的四种选择

| 损失函数 | 输入 | 内部打包了什么 |
|---------|------|-------------|
| `CrossEntropyLoss` | **logits** | softmax + log + NLLLoss |
| `NLLLoss` | log_softmax 结果 | 只做取负+选类别（第3步） |
| `BCELoss` | sigmoid 后的概率 | 只做交叉熵计算 |
| `BCEWithLogitsLoss` | **logits** | sigmoid + BCELoss |

**规律**：名字带 `WithLogits` 的打包了激活函数，不带的没打包。

## 💡 关键理解

1. **置信度 = softmax 输出 = 损失函数的输入**。视频说"置信度是损失之前的数据对象"完全正确——先有模型的看法（概率），再对这个看法打分（损失）

2. **置信度和损失是一体两面**：`Loss = -log(置信度_正确类别)`。置信度越高（模型越确定且对了），损失越低

3. **softmax 不是损失函数**，它只是概率归一化。交叉熵才是损失函数

4. **PyTorch 打包 softmax+交叉熵**是为了数值稳定，不是因为它俩是一回事。不要被 PyTorch 的 API 设计误导了概念理解

## ⚠️ 易错点

1. **"softmax 就是交叉熵"** → 错。Softmax 做归一化，交叉熵做评估，两个独立操作
2. **"CrossEntropyLoss 需要先 softmax"** → 错。传 logits 就行，内部自动做
3. **"置信度 = logit 最大值"** → 不精确。logit 最大值和置信度正相关但不等，必须经过 softmax 才是真正的概率
4. **"BCELoss 传 logits"** → 错。BCELoss 传概率，传 logits 会 NaN。用 BCEWithLogitsLoss 才能传 logits
5. **"打包版和拆开版结果不一样"** → 数学上完全一样，只是打包版数值更稳定

## 🔗 知识延伸

- [[01_交叉熵损失函数详解]]：交叉熵的完整理论推导
- [[14_PyTorch的Linear层W形状详解]]：PyTorch 内部计算的另一个"打包"案例
- [[13_反向传播中W转置的由来]]：交叉熵梯度如何回传

## 📚 参考资料

- PyTorch 源码：`torch/nn/modules/loss.py` → CrossEntropyLoss = LogSoftmax + NLLLoss
- PyTorch 源码：`torch/nn/functional.py` → `log_softmax` 的数值稳定实现
- 《Deep Learning》Goodfellow et al., Chapter 6.2 (softmax 数值稳定性)
