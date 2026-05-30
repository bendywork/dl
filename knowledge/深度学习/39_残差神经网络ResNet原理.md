# 残差神经网络（ResNet）原理

## 📌 核心问题
> 网络越深，效果反而越差？残差网络解决了深度神经网络训练中的"退化问题"——不是过拟合，而是更深网络反而比浅网络训练误差更高。

## 🌱 根源与动机

### 问题是怎么发现的？

2015年之前，大家有个朴素直觉：网络越深，表达能力越强，效果应该越好。但实际实验发现：

56层网络 vs 20层网络（在同一个数据集上）
→ 56层的训练误差和测试误差都比20层高！

这不是过拟合（过拟合是训练误差低、测试误差高），而是连训练集上都学不好。说明深层网络存在优化困难。

### 为什么深层难优化？

直觉上，一个56层网络"至少应该"不差于20层网络——因为可以把前20层学成和那个20层网络一模一样，后面36层做恒等映射（identity mapping）就行了。

但问题在于：常规网络结构很难学出恒等映射。每一层都在做非线性变换 F(x)，想让 F(x) = x 其实非常困难。

## 💡 核心洞察：残差学习

何恺明的天才想法：与其让网络直接学目标映射 H(x)，不如让网络学残差 F(x) = H(x) - x

这样：
- 如果最优解就是恒等映射，那只需要让 F(x) = 0 就行了——学全零比学恒等映射容易得多
- 如果最优解不是恒等映射，那网络学的是"微调"，比从零开始学完整映射也更容易

### 残差块结构

```
普通块:    y = F(x)          → 直接学完整映射
残差块:    y = F(x) + x      → 学残差，再加回输入（shortcut/skip connection）
```

```
输入 x ──→ [Conv → BN → ReLU → Conv → BN] ──→ (+) ──→ ReLU ──→ 输出 y
  │                                              ↑
  └────────── shortcut connection ───────────────┘
                    (恒等映射)
```

关键：那条"跳过去"的线就是残差连接，它让梯度可以直接回传。

## 📐 为什么残差有效？两个角度

### 角度1：前向传播——学习难度降低

- 学 F(x) = 0（权重初始化本来就接近0）远比学 F(x) = x 容易
- 网络只需要学"和恒等映射差多少"，而不是"完整映射是什么"

### 角度2：反向传播——梯度流通

这是更根本的原因。普通深层网络的梯度链：

```
∂L/∂x = ∂L/∂y · ∂y/∂x = ∂L/∂y · ∂F/∂x
```

每一层乘一个 ∂F/∂x，连乘多次后梯度消失。

残差网络的梯度链：

```
∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
```

注意那个 +1！即使 ∂F/∂x 很小接近0，梯度仍然可以通过 shortcut 以 ∂L/∂y · 1 的路径直接回传。这打破了梯度消失的魔咒。

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 普通块 vs 残差块对比
# ============================================================

class PlainBlock(nn.Module):
    """普通卷积块：y = F(x)"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x))  # 没有shortcut


class ResidualBlock(nn.Module):
    """残差块：y = F(x) + x"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)  # 核心：+ x 就是残差连接


# ============================================================
# 2. 梯度对比实验：证明残差连接解决梯度消失
# ============================================================

def gradient_experiment():
    """堆叠20层，对比普通网络和残差网络的梯度大小"""
    torch.manual_seed(42)
    x = torch.randn(1, 64, requires_grad=True)

    # 普通网络：20层，每层权重0.9（模拟梯度衰减）
    plain_x = x.clone().detach().requires_grad_(True)
    for i in range(20):
        plain_x = torch.sigmoid(0.9 * plain_x)

    loss = plain_x.sum()
    loss.backward()
    print(f"普通网络（20层sigmoid）梯度大小: {x.grad.abs().mean():.8f}")

    # 残差网络：20层，同样权重，但有shortcut
    res_x = x.clone().detach().requires_grad_(True)
    temp = res_x
    for i in range(20):
        temp = torch.sigmoid(0.9 * temp) + temp  # + temp 就是残差连接

    loss = temp.sum()
    loss.backward()
    print(f"残差网络（20层sigmoid+shortcut）梯度大小: {res_x.grad.abs().mean():.8f}")


# ============================================================
# 3. 完整 ResNet 在 MNIST 上的实战
# ============================================================

class ResNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res1 = self._make_layer(32, 2)
        self.res2 = self._make_layer(32, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_resnet():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNetMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()

        acc = correct / len(test_ds)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")


if __name__ == "__main__":
    print("=== 梯度对比实验 ===")
    gradient_experiment()
    print("\n=== ResNet MNIST 训练 ===")
    train_resnet()
```

## ⚠️ 易错点与常见误解

1. **"残差就是跳过一层"** → 不对。残差连接不是跳过，而是学一个增量。网络仍然在学习，只是学的目标从"完整映射"变成了"残差"
2. **"退化就是过拟合"** → 不对。退化是训练误差也变高，过拟合是训练误差低测试误差高。这是两个不同问题
3. **"shortcut必须是恒等映射"** → 不对。当输入输出维度不同时（如通道数变化），用 1x1 Conv 做投影：y = F(x) + W_proj · x，这叫 projection shortcut
4. **"残差只能用于CNN"** → 不对。Transformer、扩散模型 U-Net、甚至RNN都大量使用残差连接。残差是一种通用的网络设计范式
5. **"深层网络一定比浅层好"** → 没有残差连接时不一定；有残差连接后，至少不会比浅层差（因为多余的层可以学成恒等映射）

## 🔗 知识延伸

- **ResNet → DenseNet**：残差是加法融合，DenseNet 改为拼接融合（channel维度拼接）
- **ResNet → Pre-Activation ResNet**：把 BN+ReLU 放到卷积前面，效果更好
- **残差 → Transformer**：Transformer 每个 Block 都有残差连接（output = F(x) + x），是深层Transformer能训练的关键
- **残差 → 扩散模型 U-Net**：U-Net 的 skip connection 本质也是残差思想

## 📚 参考资料

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016 (ResNet原论文)
- He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016 (Pre-Activation ResNet)
- 论文中的核心图：Figure 2 (plain vs residual 网络的误差对比曲线) 和 Figure 3 (残差块结构图)
