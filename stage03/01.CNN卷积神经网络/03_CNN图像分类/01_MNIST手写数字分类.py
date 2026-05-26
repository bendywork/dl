# -*- coding: utf-8 -*-
"""
CNN图像分类完整案例——MNIST手写数字识别
- 完整的训练/评估/推理流程
- 每一步都有详细注释
- 包含训练曲线可视化和预测结果可视化
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. 数据准备
# ============================================================
def prepare_data(batch_size=64):
    """
    MNIST数据集：28×28灰度手写数字，0-9共10类
    torchvision自动下载
    """
    # 训练集做数据增强，测试集只做归一化
    train_transform = transforms.Compose([
        transforms.ToTensor(),                          # PIL → Tensor，自动归一化到[0,1]
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST的均值和标准差
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"每个batch: {batch_size}")
    print(f"训练batch数: {len(train_loader)}")
    print(f"测试batch数: {len(test_loader)}")

    return train_loader, test_loader


# ============================================================
# 2. 模型定义
# ============================================================
class MNISTNet(nn.Module):
    """
    MNIST分类CNN
    结构: 卷积块1 → 卷积块2 → 全连接分类头

    输入: [batch, 1, 28, 28]
    输出: [batch, 10] (10类数字的logits)
    """
    def __init__(self):
        super().__init__()

        # 卷积块1: 1通道 → 32通道
        # 卷积核3×3, padding=1保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 参数: 1×3×3×32 = 288
        # 池化2×2, 尺寸减半
        # → 28×28 → 14×14

        # 卷积块2: 32通道 → 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 参数: 32×3×3×64 = 18,432
        # 池化2×2
        # → 14×14 → 7×7

        # 全连接分类头
        # 64通道 × 7×7 = 3136 维特征
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: [batch, 1, 28, 28]

        # 卷积块1
        x = F.relu(self.conv1(x))    # [batch, 32, 28, 28]
        x = F.max_pool2d(x, 2)       # [batch, 32, 14, 14]

        # 卷积块2
        x = F.relu(self.conv2(x))    # [batch, 64, 14, 14]
        x = F.max_pool2d(x, 2)       # [batch, 64, 7, 7]

        # 展平 → 送入全连接
        x = x.view(x.size(0), -1)    # [batch, 3136]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))      # [batch, 128]
        x = self.dropout(x)
        x = self.fc2(x)              # [batch, 10] logits

        return x


# ============================================================
# 3. 训练函数
# ============================================================
def train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向
        logits = model(data)
        loss = loss_fn(logits, target)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * len(data)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += len(data)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch} 训练: loss={avg_loss:.4f}, acc={accuracy:.3f}")
    return avg_loss, accuracy


# ============================================================
# 4. 评估函数
# ============================================================
def evaluate(model, device, test_loader, loss_fn):
    """在测试集上评估"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            logits = model(data)
            loss = loss_fn(logits, target)

            total_loss += loss.item() * len(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += len(data)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"         评估: loss={avg_loss:.4f}, acc={accuracy:.3f}")
    return avg_loss, accuracy


# ============================================================
# 5. 训练曲线可视化
# ============================================================
def plot_training_curves(train_losses, train_accs, test_losses, test_accs, save_dir):
    """绘制训练/评估的loss和accuracy曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, test_accs, 'r-', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.show()


# ============================================================
# 6. 预测结果可视化
# ============================================================
def visualize_predictions(model, device, test_loader, save_dir, num_samples=16):
    """可视化模型的预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    probs = probs.cpu()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx >= num_samples:
            break
        img = images[idx].squeeze(0)
        # 反归一化显示
        img = img * 0.3081 + 0.1307
        ax.imshow(img, cmap='gray')
        true_label = labels[idx].item()
        pred_label = preds[idx].item()
        confidence = probs[idx, pred_label].item()
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True:{true_label} Pred:{pred_label} ({confidence:.2f})', color=color)
        ax.axis('off')

    plt.suptitle('CNN预测结果（绿色=正确，红色=错误）', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
    plt.show()


# ============================================================
# 7. 主流程
# ============================================================
def main():
    # 配置
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    SAVE_DIR = './output/cnn_mnist'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据
    train_loader, test_loader = prepare_data(BATCH_SIZE)

    # 模型
    model = MNISTNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型结构:\n{model}")
    print(f"\n总参数量: {total_params:,}")
    print(f"  卷积层参数: {sum(p.numel() for p in model.conv1.parameters()) + sum(p.numel() for p in model.conv2.parameters()):,}")
    print(f"  全连接层参数: {sum(p.numel() for p in model.fc1.parameters()) + sum(p.numel() for p in model.fc2.parameters()):,}")

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_test_acc = 0.0

    print(f"\n开始训练，共{EPOCHS}个epoch...")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch)
        test_loss, test_acc = evaluate(model, device, test_loader, loss_fn)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最优模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"  → 保存最优模型 (acc={test_acc:.3f})")

    print("-" * 50)
    print(f"训练完成! 最优测试准确率: {best_test_acc:.3f}")

    # 可视化
    plot_training_curves(train_losses, train_accs, test_losses, test_accs, SAVE_DIR)
    visualize_predictions(model, device, test_loader, SAVE_DIR)


if __name__ == '__main__':
    main()
