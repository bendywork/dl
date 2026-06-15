"""
AlexNet 网络
=============
2012 年 ImageNet 冠军，深度卷积神经网络的里程碑
关键创新：ReLU 激活、Dropout、GPU 训练、数据增强

简化版：适配 CIFAR-10 (32×32 彩色图片, 10类)
原始 AlexNet 是为 ImageNet (224×224) 设计的，这里缩小了通道数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 批次大小
batch_size = 64
# 学习率
learning_rate = 0.001
# 训练轮次
epochs = 10

# ==================== 1. 数据准备 ====================
# CIFAR-10: 32×32 彩色图片, 10 个类别 (飞机、汽车、鸟、猫...)
# 数据增强：随机翻转 + 随机裁剪（AlexNet 的关键创新之一）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪（先补4像素边距）
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # 均值
                         (0.2470, 0.2435, 0.2616))    # 标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
print(f"CIFAR-10 类别: {train_dataset.classes}")


# ==================== 2. AlexNet 模型（简化版适配 CIFAR-10） ====================
class AlexNetSmall(nn.Module):
    """
    简化版 AlexNet，适配 32×32 输入

    AlexNet 的核心结构：
    5 个卷积层（提取特征）+ 3 个全连接层（分类）

    关键创新点：
    1. ReLU 激活（代替 sigmoid/tanh，训练更快）
    2. Dropout（防止过拟合）
    3. 最大池化（降维 + 平移不变性）
    4. 局部响应归一化（LRN，现在已被 BatchNorm 替代）
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # ===== 特征提取部分（卷积层） =====
        self.features = nn.Sequential(
            # Conv1: 3→48, 3×3卷积, ReLU
            # 原版: 11×11卷积 → 缩小为3×3适配32×32输入
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32→16

            # Conv2: 48→128, 3×3卷积, ReLU
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16→8

            # Conv3: 128→192, 3×3卷积, ReLU
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 192→192, 3×3卷积, ReLU
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 192→128, 3×3卷积, ReLU
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8→4
        )

        # ===== 分类部分（全连接层） =====
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                     # AlexNet 创新：Dropout 防过拟合
            nn.Linear(128 * 4 * 4, 1024),        # 展平 → 2048
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, num_classes),          # 输出 10 类
        )

    def forward(self, x):
        x = self.features(x)        # 卷积特征提取
        x = x.view(x.size(0), -1)   # 展平: (batch, 128, 4, 4) → (batch, 2048)
        x = self.classifier(x)       # 全连接分类
        return x

model = AlexNetSmall()
print(f"\n模型结构:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")


# ==================== 3. 训练 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5epoch学习率减半

print("\n" + "=" * 50)
print("开始训练 AlexNet")
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
    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | 准确率: {acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")


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


# ==================== 5. AlexNet 关键知识点总结 ====================
print("""
AlexNet 关键创新点（2012）：

1. ReLU 激活函数
   - 替代 sigmoid/tanh → 梯度不饱和 → 训练更快
   - sigmoid: 梯度 ∈ (0, 0.25)，连乘后消失
   - ReLU:    梯度 = 1（正区间），梯度不消失

2. Dropout 正则化
   - 训练时随机丢弃 50% 神经元 → 防止过拟合
   - 相当于训练很多子网络的集成

3. 数据增强
   - 随机翻转、随机裁剪 → 扩充训练数据
   - 模型看到更多变化 → 泛化更好

4. 最大池化
   - 降维 + 保留最显著特征
   - 比平均池化效果更好

5. GPU 训练
   - 首次在大规模数据集上用 GPU → 训练速度飞跃
""")
