"""
VGG 网络
=========
2014 年 ImageNet 亚军，核心思想：用更小的 3×3 卷积堆叠替代大卷积核

关键洞察：
- 2 个 3×3 卷积 = 1 个 5×5 感受野，但参数更少、非线性更强
- 3 个 3×3 卷积 = 1 个 7×7 感受野，同理

VGG 的设计哲学：简洁、统一、可扩展
所有卷积都是 3×3 + stride=1 + padding=1，所有池化都是 2×2 maxpool
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================== 超参数 ====================
batch_size = 64
learning_rate = 0.001
epochs = 10

# ==================== 1. 数据准备 ====================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 2. VGG 模型 ====================
class VGGBlock(nn.Module):
    """
    VGG 基本块：若干个 3×3 卷积 + 1 个 2×2 最大池化

    这是 VGG 的核心设计——重复使用相同的"卷积+池化"模式
    通道数逐块翻倍：64 → 128 → 256 → 512
    """
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3, padding=1  # 3×3 卷积, padding=1 保持尺寸
            ))
            layers.append(nn.BatchNorm2d(out_channels))  # VGG 原版没有 BN，现代版加上
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 尺寸减半
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG11Small(nn.Module):
    """
    简化版 VGG-11，适配 CIFAR-10 (32×32)

    VGG-11 配置: [1, 1, 2, 2, 2] 个卷积 per block
    通道数:       [64, 128, 256, 512, 512]

    VGG 的核心优势 vs AlexNet:
    - AlexNet: 大卷积核(11×11, 5×5)，参数多，非线性少
    - VGG:     全用 3×3 小卷积核堆叠，参数少，非线性多

    为什么 2 个 3×3 ≡ 1 个 5×5？
    - 1 个 3×3: 感受野 3×3
    - 2 个 3×3: 感受野 5×5（第一个看3×3，第二个在3×3上再看3×3 = 5×5）
    - 但参数: 2×(3×3×C×C) = 18C² < 1×(5×5×C×C) = 25C²
    - 且 2 个 ReLU > 1 个 ReLU → 非线性表达能力更强
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3→64, 1个卷积, 32×32 → 16×16
            VGGBlock(3, 64, num_convs=1),

            # Block 2: 64→128, 1个卷积, 16×16 → 8×8
            VGGBlock(64, 128, num_convs=1),

            # Block 3: 128→256, 2个卷积, 8×8 → 4×4
            VGGBlock(128, 256, num_convs=2),

            # Block 4: 256→512, 2个卷积, 4×4 → 2×2
            VGGBlock(256, 512, num_convs=2),

            # Block 5: 512→512, 2个卷积, 2×2 → 1×1
            VGGBlock(512, 512, num_convs=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = VGG11Small()
print(f"模型结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")


# ==================== 3. 训练 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("\n" + "=" * 50)
print("开始训练 VGG-11")
print("=" * 50)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | 准确率: {acc:.1f}%")


# ==================== 4. 测试 ====================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

print(f"\n测试准确率: {100 * correct / total:.1f}%")


# ==================== 5. VGG 关键知识点 ====================
print("""
VGG vs AlexNet 对比：

| 维度           | AlexNet              | VGG                   |
|---------------|---------------------|----------------------|
| 卷积核大小      | 11×11, 5×5, 3×3    | 全部 3×3             |
| 设计哲学        | 逐层手动设计        | 统一模块重复堆叠      |
| 非线性层数      | 少                  | 多（每个3×3后都有ReLU）|
| 参数效率        | 低（大卷积核参数多） | 高（小卷积核堆叠）    |
| 深度            | 8 层                | 11-19 层             |
| ImageNet 准确率 | 84.7%              | 92.3%（VGG-16）      |

VGG 的核心贡献：证明了"更深的网络 + 更小的卷积核"优于"浅网络 + 大卷积核"
""")
