"""
图像分类综合实践
================
用 PyTorch 官方 torchvision 预训练模型做迁移学习
快速在自定义数据上获得高准确率

场景：CIFAR-10 分类，使用预训练 ResNet-18 微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# ==================== 超参数 ====================
batch_size = 64
learning_rate = 0.001
epochs = 5
num_classes = 10

# ==================== 1. 数据准备 ====================
# 迁移学习需要匹配预训练模型的输入尺寸和归一化方式
# ImageNet 预训练模型期望: 224×224 输入, ImageNet 均值/标准差归一化
train_transform = transforms.Compose([
    transforms.Resize(224),                    # CIFAR-10 32×32 → 224×224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet 均值
                         std=[0.229, 0.224, 0.225])       # ImageNet 标准差
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


# ==================== 2. 加载预训练模型 ====================
# 方法一：直接加载 PyTorch 官方预训练 ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 方法二（备选）：加载预训练 VGG-16
# model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

# 方法三（备选）：加载预训练 AlexNet
# model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# ===== 迁移学习：替换最后全连接层 =====
# 预训练模型输出 1000 类（ImageNet），我们要改成 10 类（CIFAR-10）
# ResNet-18 的最后层: model.fc = nn.Linear(512, 1000)
# 替换为:
model.fc = nn.Linear(512, num_classes)

print(f"替换后的分类层: {model.fc}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")


# ==================== 3. 可选：冻结特征提取层 ====================
# 策略 A：全部微调（默认，效果最好但训练慢）
# 所有参数都可训练，不做任何修改

# 策略 B：冻结卷积层，只训练分类层（训练快，效果稍差）
# 取消注释下面两行即可启用：
# for param in model.parameters():
#     param.requires_grad = False
# model.fc.weight.requires_grad = True
# model.fc.bias.requires_grad = True

# 策略 C：分层学习率（高级技巧）
# 卷积层用小学习率，新分类层用大学习率
# optimizer = optim.Adam([
#     {'params': model.conv1.parameters(), 'lr': 1e-5},
#     {'params': model.layer1.parameters(), 'lr': 1e-5},
#     {'params': model.layer2.parameters(), 'lr': 1e-4},
#     {'params': model.layer3.parameters(), 'lr': 1e-4},
#     {'params': model.layer4.parameters(), 'lr': 1e-3},
#     {'params': model.fc.parameters(), 'lr': 1e-2},
# ])


# ==================== 4. 训练 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

device = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")

print("\n" + "=" * 60)
print(f"迁移学习训练 ResNet-18 (预训练 ImageNet → 微调 CIFAR-10)")
print("=" * 60)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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
    elapsed = time.time() - start_time
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
          f"准确率: {acc:.1f}% | 用时: {elapsed:.0f}s | LR: {scheduler.get_last_lr()[0]:.6f}")


# ==================== 5. 测试 ====================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

print(f"\n测试准确率: {100 * correct / total:.1f}%")


# ==================== 6. 迁移学习策略总结 ====================
print("""
迁移学习三种策略对比：

| 策略              | 做法                  | 训练速度 | 准确率 | 适用场景           |
|------------------|----------------------|---------|--------|-------------------|
| A. 全部微调       | 所有层都训练          | 慢      | 最高   | 数据多，计算资源足   |
| B. 冻结+只训分类层 | 卷积层冻结，只训FC    | 快      | 中等   | 数据少，快速验证     |
| C. 分层学习率      | 浅层小LR，深层大LR    | 中      | 高     | 数据中等，精细化调优  |

迁移学习为什么有效？
  预训练模型的卷积层已经学会了通用的视觉特征：
  - 浅层: 边缘、纹理（所有图像任务都通用）
  - 深层: 形状、物体部件（大部分任务可复用）
  - 只有最后的分类层需要针对新任务重新学习

  相当于：已经学会"看图"的学生，只需要学"认新类别"，不用从头学视觉
""")
