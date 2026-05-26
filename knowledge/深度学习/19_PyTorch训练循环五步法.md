# PyTorch 训练循环五步法

## 📌 核心问题
> 训练一个神经网络，代码应该如何组织？每一步的顺序为什么不能颠倒？

---

## 🌱 根源与动机

神经网络的训练本质是一个**迭代优化过程**：

1. 用当前参数跑一次前向传播，得到预测值
2. 计算预测值与真实值之间的误差（loss）
3. 对 loss 关于每个参数求偏导（反向传播）
4. 用梯度更新参数（optimizer step）

PyTorch 的自动求导机制（autograd）让第 3 步变得简单，但也带来了一个设计决策：**梯度默认是累积的**，而不是每次清零。这个设计是为了支持某些特殊场景（如梯度累积训练大 batch），但在标准训练中，我们必须手动清零梯度。

---

## 📐 理论推导

训练循环对应的参数更新公式：

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta)$$

其中：
- $\theta$：模型参数
- $\eta$：学习率
- $\mathcal{L}$：损失函数
- $\nabla_\theta \mathcal{L}$：loss 对参数的梯度

PyTorch 中每个参数 `p` 的梯度存储在 `p.grad` 字段里。`backward()` 是**累加**到 `p.grad`，而不是替换。因此如果不清零，第 N 步的梯度会叠加在第 N-1 步的梯度上，导致参数更新错误。

---

## 💡 关键理解

### 训练循环五步（顺序不可颠倒）

```
第一步：outputs = net(inputs)         # 前向传播
第二步：loss = criterion(outputs, labels)  # 计算 loss
第三步：optimizer.zero_grad()         # 清空梯度
第四步：loss.backward()               # 反向传播（计算梯度）
第五步：optimizer.step()              # 更新参数
```

### 为什么 zero_grad 在 backward 之前，而不是之后？

**直觉理解：** `zero_grad` 是"清空黑板"，`backward` 是"写新内容"。你必须先清空再写，否则新内容会叠加在旧内容上。

**具体原因：**
- `backward()` 执行时，会把当前 mini-batch 的梯度**累加**到 `param.grad` 上
- 如果上一轮的 `param.grad` 没有清零，本轮 backward 会叠加上去
- `step()` 用的是叠加后的错误梯度更新参数
- **结果：** 参数更新方向错误，loss 不收敛

为什么不在 `step()` 之后清零？技术上可行，但容易在多 loss 场景下出错，且放在 `backward` 之前是业界惯例，语义更清晰（"我要开始一次新的梯度计算了，先清空"）。

### forward 与 training 职责分离

| 职责 | 代码位置 | 内容 |
|------|---------|------|
| 网络结构描述 | `nn.Module.forward()` | 定义数据从输入到输出的流向，只管"长什么样" |
| 训练执行引擎 | 外部训练循环 | 管理 loss、backward、optimizer，只管"怎么训练" |

**原则：** `forward` 里不应该出现 `zero_grad`、`backward`、`optimizer.step()`。这些是训练逻辑，不属于网络定义。

### net.train() vs net.eval()

| 方法 | 作用 | 影响的层 |
|------|------|---------|
| `net.train()` | 开启训练模式 | Dropout 随机 drop 神经元；BatchNorm 用 mini-batch 统计 |
| `net.eval()` | 开启评估模式 | Dropout 关闭（全神经元参与）；BatchNorm 用全局统计量 |

**重要：** `net.train()` 会**递归地**将所有子模块设为训练模式。直接设置 `self.training = True` 只影响当前模块，是错误做法。

### torch.no_grad() 的本质

PyTorch 在每次前向传播时，会构建一张**计算图（computation graph）**，记录每个张量的运算历史，用于 backward 时反向追踪求梯度。

`torch.no_grad()` 的作用是：**在该上下文中，不构建计算图**。

好处：
- 节省显存（不存储中间激活值）
- 加快推理速度（无需维护梯度信息）
- 评估时不需要梯度，使用 `no_grad` 是正确且必要的

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 定义网络（只负责描述结构，不管训练逻辑）
# ============================================================
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # 只描述数据流向，返回 raw logits


# ============================================================
# 2. 数据加载（独立函数，不放进 nn.Module）
# ============================================================
def data_load(batch_size=32):
    # 模拟数据：100 个样本，10 维特征，3 分类
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# ============================================================
# 3. 训练循环（五步法）
# ============================================================
def train_one_epoch(net, loader, criterion, optimizer):
    net.train()  # 开启训练模式（影响 Dropout/BN）

    total_loss = 0.0
    for inputs, labels in loader:
        # 第一步：前向传播
        outputs = net(inputs)           # outputs: raw logits, shape=(B, num_classes)

        # 第二步：计算 loss
        loss = criterion(outputs, labels)

        # 第三步：清空梯度（必须在 backward 之前）
        optimizer.zero_grad()

        # 第四步：反向传播（计算梯度）
        loss.backward()

        # 第五步：更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# ============================================================
# 4. 评估循环（eval 模式 + no_grad）
# ============================================================
def evaluate(net, loader, criterion):
    net.eval()  # 关闭 Dropout，BN 使用全局统计量

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不构建计算图，节省显存和计算
        for inputs, labels in loader:
            outputs = net(inputs)           # 仍然传 raw logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  # 取最大 logit 对应的类别
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# 5. 主训练流程
# ============================================================
def main():
    # 超参数
    input_dim = 10
    hidden_dim = 64
    num_classes = 3
    lr = 0.01
    epochs = 20

    # 初始化
    net = SimpleClassifier(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()        # 内部已含 log_softmax，传 raw logits
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_loader = data_load(batch_size=32)
    val_loader = data_load(batch_size=32)    # 示例中复用，实际应用需独立验证集

    # 训练
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(net, val_loader, criterion)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")


if __name__ == "__main__":
    main()
```

---

## ⚠️ 易错点与常见误解

1. **`zero_grad` 放在 `step()` 之后**
   → 下一轮 `backward` 会叠加梯度，参数更新错误。必须放在 `backward` 之前。

2. **评估时忘记 `net.eval()`**
   → Dropout 仍在随机 drop，BatchNorm 仍用 mini-batch 统计，评估结果不稳定且不可复现。

3. **评估时忘记 `torch.no_grad()`**
   → 仍在构建计算图，浪费显存，可能导致 OOM；评估结果不受影响，但性能损失明显。

4. **在 `forward` 里写训练逻辑（zero_grad / backward / step）**
   → 严重违反职责分离原则，导致代码无法复用、无法测试、难以 debug。

5. **直接设置 `self.training = True` 代替 `net.train()`**
   → 只影响顶层模块，子模块（如内部的 Dropout、BN）不受影响，隐性 bug。

6. **`torch.no_grad()` 只包裹 loss 计算，不包裹 forward**
   → 计算图在 forward 时已经构建完毕，包裹 loss 计算无效。必须从 forward 开始就在 `no_grad` 上下文里。

7. **手动展开 `Sequential` 时每层都传原始 `x`**
   → 所有层变成并联而非串联。正确写法：每层输入是上一层输出。
   ```python
   # 错误
   out1 = layer1(x)
   out2 = layer2(x)   # 应该是 layer2(out1)
   
   # 正确
   out1 = layer1(x)
   out2 = layer2(out1)
   ```

---

## 🔗 知识延伸

- **梯度累积（Gradient Accumulation）：** 利用 PyTorch 梯度默认累积的特性，多个 mini-batch 累积梯度后再 `step()`，模拟大 batch 训练（显存不足时常用）
- **`nn.CrossEntropyLoss`：** 内部集成了 `log_softmax`，传入 raw logits；详见文档 `20_CrossEntropyLoss正确用法.md`
- **`PyTorch Lightning`：** 将五步法封装进 `training_step`，框架自动调用 `zero_grad` / `backward` / `step`，开发者只需实现前向和 loss 计算
- **`BatchNorm` / `Dropout`：** 这两种层在训练和推理时行为不同，是 `train()` / `eval()` 切换意义的核心

---

## 📚 参考资料

- PyTorch 官方文档：[torch.optim](https://pytorch.org/docs/stable/optim.html)
- PyTorch 官方文档：[torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
- PyTorch 官方文档：[nn.Module.train](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train)
