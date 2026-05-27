# CrossEntropyLoss 正确用法

## 📌 核心问题
> `nn.CrossEntropyLoss` 应该传入什么？为什么不能先做 softmax 再传入？

---

## 🌱 根源与动机

交叉熵损失（Cross-Entropy Loss）是分类任务最常用的损失函数。其数学定义为：

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{p}_c$$

其中 $y_c$ 是 one-hot 标签，$\hat{p}_c$ 是模型对类别 $c$ 的预测概率。

要得到概率 $\hat{p}_c$，需要先对网络输出的 raw logits（原始得分）做 **softmax**：

$$\hat{p}_c = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$$

再取 log：

$$\log \hat{p}_c = \log \left(\frac{e^{z_c}}{\sum_{j} e^{z_j}}\right) = z_c - \log \sum_j e^{z_j}$$

这就是 **log-softmax**。

---

## 📐 理论推导

**PyTorch 的 `nn.CrossEntropyLoss` 内部已经把以下两步合并在一起：**

```
nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss
```

即：
1. 对输入 `z`（raw logits）做 log-softmax，得到 log 概率
2. 用 NLLLoss（负对数似然损失）计算最终 loss

完整公式：

$$\mathcal{L}(z, y) = -z_y + \log \sum_j e^{z_j}$$

其中 $y$ 是正确类别的索引，$z_y$ 是该类别对应的 logit。

PyTorch 采用数值稳定的 **log-sum-exp** 技巧实现，防止 $e^{z_j}$ 数值溢出：

$$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \quad m = \max_j z_j$$

---

## 💡 关键理解

### 为什么必须传 raw logits？

`CrossEntropyLoss` **假定输入是 raw logits**，然后在内部做 log-softmax。

如果你在外部已经做了 softmax，再传进去，相当于：

```
输入：softmax(z) = p（已经是概率，范围 (0,1)，和为 1）
内部操作：log_softmax(p) = log(softmax(p))
         = log(softmax(softmax(z)))   ← softmax 嵌套！
```

**Softmax 嵌套的后果：**

- `softmax(z)` 输出的概率值范围是 $(0, 1)$，数值比 raw logits 更"集中"
- 对已经是概率的输入再做 softmax，会把分布压得更平坦（接近均匀分布）
- `log` 作用在非常小的数上，导致 loss 值异常偏小（接近 0）
- 反向传播时梯度极小，参数几乎不更新，**模型无法收敛**

### 数值对比

```python
import torch
import torch.nn.functional as F

z = torch.tensor([2.0, 1.0, 0.1])  # raw logits

# 正确路径：直接传 logits
p_correct = z  # 直接用
log_p = F.log_softmax(p_correct, dim=0)
print("正确 log_p:", log_p)  # tensor([-0.4170, -1.4170, -2.3170])

# 错误路径：先 softmax 再传入
p_wrong = F.softmax(z, dim=0)       # 先做 softmax → [0.6590, 0.2424, 0.0986]
log_p_wrong = F.log_softmax(p_wrong, dim=0)  # 再做 log_softmax
print("错误 log_p:", log_p_wrong)   # 数值被严重压缩，接近均匀分布

# CrossEntropyLoss 表现
criterion = torch.nn.CrossEntropyLoss()
label = torch.tensor([0])  # 正确类别是 0

loss_correct = criterion(z.unsqueeze(0), label)
loss_wrong = criterion(p_wrong.unsqueeze(0), label)
print(f"正确 loss: {loss_correct.item():.4f}")   # 约 0.4170
print(f"错误 loss: {loss_wrong.item():.4f}")     # 异常偏小，接近 0
```

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 错误示例 vs 正确示例对比
# ============================================================

class WrongClassifier(nn.Module):
    """错误：forward 里做了 softmax，传给 CrossEntropyLoss"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        # ❌ 错误：对 logits 做了 softmax
        return F.softmax(logits, dim=1)


class CorrectClassifier(nn.Module):
    """正确：forward 返回 raw logits，由 CrossEntropyLoss 内部处理"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        # ✅ 正确：直接返回 raw logits，不做任何激活
        return logits


# ============================================================
# 验证两种方式的 loss 差异
# ============================================================

def compare_loss():
    torch.manual_seed(42)

    input_dim = 10
    num_classes = 3
    batch_size = 4

    X = torch.randn(batch_size, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))

    criterion = nn.CrossEntropyLoss()

    # 错误模型
    wrong_net = WrongClassifier(input_dim, num_classes)
    wrong_out = wrong_net(X)
    wrong_loss = criterion(wrong_out, y)

    # 正确模型（相同参数）
    correct_net = CorrectClassifier(input_dim, num_classes)
    correct_net.fc.weight = wrong_net.fc.weight  # 使用完全相同的参数
    correct_net.fc.bias = wrong_net.fc.bias
    correct_out = correct_net(X)
    correct_loss = criterion(correct_out, y)

    print(f"错误 loss（softmax 嵌套）: {wrong_loss.item():.6f}")
    print(f"正确 loss（raw logits）:   {correct_loss.item():.6f}")
    print(f"Loss 差异比: {correct_loss.item() / wrong_loss.item():.2f}x")


# ============================================================
# 推理时获取概率：在外部手动 softmax
# ============================================================

def inference_example():
    """推理时如果需要概率值，在 net 外部做 softmax"""
    net = CorrectClassifier(10, 3)

    with torch.no_grad():
        x = torch.randn(1, 10)
        logits = net(x)                          # raw logits

        # 需要概率时，在外部做 softmax
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)  # argmax 对 logits 和 probs 结果相同

        print(f"Logits:  {logits.numpy()}")
        print(f"Probs:   {probs.numpy()}")
        print(f"Predicted class: {pred_class.item()}")


if __name__ == "__main__":
    print("=== Loss 对比实验 ===")
    compare_loss()
    print()
    print("=== 推理示例 ===")
    inference_example()
```

---

## ⚠️ 易错点与常见误解

1. **在 `forward` 里加了 `nn.Softmax()` 层**
   → 如果损失函数是 `CrossEntropyLoss`，删掉 softmax；如果需要用 `NLLLoss`，改成 `nn.LogSoftmax()`。

2. **认为 softmax 嵌套只会让 loss"略有偏差"**
   → 实际上影响极大：loss 可以从正常的 1.0 降到 0.01 量级，梯度几乎为 0，模型彻底无法训练。

3. **评估阶段的 `argmax` 没问题，就以为传概率也行**
   → `argmax(softmax(z)) == argmax(z)` 确实成立（softmax 单调保序），但一旦你用这个输出算 loss（哪怕只是 debug），就会出问题。保持 `forward` 返回 logits 是一致性最好的做法。

4. **把 `log_softmax + NLLLoss` 和 `CrossEntropyLoss` 混用**
   → 两者等价，但不能组合使用（如 `NLLLoss` 接 `softmax` 输出而非 `log_softmax` 输出），务必搞清楚每个函数的期望输入。

5. **多标签分类（multi-label）误用 CrossEntropyLoss**
   → `CrossEntropyLoss` 用于**多分类（multi-class，每个样本只属于一个类）**；多标签分类（每个样本可属于多个类）应使用 `nn.BCEWithLogitsLoss`（内置 sigmoid）。

---

## 🔗 知识延伸

- **`nn.BCEWithLogitsLoss`：** 二分类 / 多标签分类的对应版本，内置 sigmoid，同样要传 raw logits
- **`nn.NLLLoss`：** 需要手动先做 `log_softmax`，再传入；`CrossEntropyLoss = LogSoftmax + NLLLoss`
- **数值稳定性：** log-sum-exp trick 是防止 softmax 溢出的标准做法，PyTorch 内部已实现
- **温度缩放（Temperature Scaling）：** 模型校准技术，在 logits 除以温度 $T$ 后再做 softmax，不影响 argmax 但改变概率分布的峰度
- **训练循环五步法：** 详见 `19_PyTorch训练循环五步法.md`

---

## 📚 参考资料

- PyTorch 官方文档：[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- PyTorch 官方文档：[nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
- PyTorch 官方文档：[nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
