"""
图像分类基础概念
=================
用最简单的全连接网络做 MNIST 手写数字分类
理解：数据加载 → 模型定义 → 训练 → 评估 的完整流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================== 超参数 ====================
batch_size = 64
learning_rate = 0.001
epochs = 5

# ==================== 1. 数据准备 ====================
# 图像预处理：转为张量 + 归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image → Tensor, 自动归一化到 [0, 1]
])

# MNIST: 28×28 灰度手写数字, 10 个类别 (0-9)
train_dataset = datasets.MNIST('./base/datas/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./base/datas/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"单张图片 shape: {train_dataset[0][0].shape}")  # (1, 28, 28)


# ==================== 2. 模型定义（最简单的全连接网络） ====================
class SimpleClassifier(nn.Module):
    """
    全连接网络做图像分类
    输入: 28×28 = 784 维展平向量
    输出: 10 维 logits（对应 0-9 十个数字）
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # (1, 28, 28) → (784,)
        self.fc1 = nn.Linear(784, 128)        # 784 → 128
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)         # 128 → 64
        self.fc3 = nn.Linear(64, 10)          # 64 → 10（10个类别）

    def forward(self, x):
        x = self.flatten(x)      # 展平: (batch, 1, 28, 28) → (batch, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)          # 输出 logits, 不加 softmax（CrossEntropyLoss 内部有）
        return x

model = SimpleClassifier()
print(f"\n模型结构:\n{model}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")


# ==================== 3. 损失函数和优化器 ====================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（内部包含 softmax）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ==================== 4. 训练循环 ====================
print("\n" + "=" * 50)
print("开始训练")
print("=" * 50)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: (batch, 1, 28, 28)
        # labels: (batch,)

        # 前向传播
        logits = model(images)                    # (batch, 10)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)               # 选概率最大的类别
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | 训练准确率: {acc:.1f}%")


# ==================== 5. 测试评估 ====================
print("\n" + "=" * 50)
print("测试评估")
print("=" * 50)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"测试准确率: {test_acc:.1f}%")


# # ==================== 6. 可视化几个预测结果 ====================
# import matplotlib.pyplot as plt
#
# model.eval()
# images, labels = next(iter(test_loader))
# with torch.no_grad():
#     logits = model(images)
#     preds = logits.argmax(dim=1)
#
# fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(images[i][0], cmap='gray')
#     color = 'green' if preds[i] == labels[i] else 'red'
#     ax.set_title(f'预测: {preds[i]} / 真实: {labels[i]}', color=color)
#     ax.axis('off')
# plt.suptitle('图像分类预测结果（绿色=正确，红色=错误）')
# plt.tight_layout()
# plt.savefig('classification_results.png', dpi=100)
# plt.show()
# print("\n预测结果图已保存到 classification_results.png")
