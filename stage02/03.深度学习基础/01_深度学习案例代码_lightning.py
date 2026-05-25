# -*- coding: utf-8 -*-
"""
用 PyTorch Lightning 实现与 01_深度学习案例代码_fixed.py 完全一致的分类效果
核心对比：手写训练循环 vs Lightning 封装
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


class ClassifyNetwork(pl.LightningModule):
    """
    对比原文件：
        原文件 ClassifyNetwork 继承 nn.Module，训练循环手写在 training() 函数里
        这里继承 pl.LightningModule，训练循环由 Lightning 框架驱动
        网络结构 features + classify 完全一致
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU()
        )
        self.classify = nn.Linear(8, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # 推理时直接返回概率（与原文件 eval 分支一致）
        return torch.softmax(self.classify(self.features(x)), dim=1)

    def _get_raw_score(self, x):
        # 取 raw logits 用于 loss（与原文件 fix1 一致）
        return self.classify(self.features(x))

    # ------------------------------------------------------------------ #
    # 以下三个方法替代了原文件手写的 for epoch / for batch 训练循环         #
    # Lightning 框架自动调用它们，顺序：training_step -> validation_step   #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        # 对应原文件：前向 + loss + zero_grad + backward + step 五步
        # Lightning 自动处理 zero_grad / backward / step，这里只需写前向和 loss
        x, y = batch
        loss = self.loss_fn(self._get_raw_score(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 对应原文件评估循环，自动在 torch.no_grad() 上下文中执行
        x, y = batch
        loss = self.loss_fn(self._get_raw_score(x), y)
        proba = self(x)
        pred_idx = torch.argmax(proba, dim=1)
        acc = metrics.accuracy_score(y.cpu().numpy(), pred_idx.cpu().numpy())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        # 对应原文件：optim.SGD(params=net.parameters(), lr=0.01)
        return optim.SGD(self.parameters(), lr=0.01)


def training():
    # 1. 数据准备（与原文件完全一致）
    X, Y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=24)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24)

    # 2. 封装成 DataLoader（原文件手动切 batch，这里交给 DataLoader）
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.int64)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 3. 模型创建（与原文件结构一致）
    net = ClassifyNetwork(in_features=2, num_classes=2)

    # 4. Trainer 替代原文件的 for epoch 循环
    #    max_epochs=100 对应原文件 total_epoch=100
    os.makedirs("../../base/out/dl/lightning", exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=100,
        default_root_dir="../../base/out/dl/lightning",
        enable_progress_bar=True,
    )
    trainer.fit(net, train_loader, test_loader)

    # 5. 模型保存（与原文件 torch.save 对应）
    torch.save({'net': net, 'net_param': net.state_dict()}, "../../base/out/dl/lightning/final.pkl")
    print("训练完成，模型已保存")


def interface():
    obj = torch.load("../../base/out/dl/lightning/final.pkl", map_location='cpu', weights_only=False)
    net = obj['net']
    net.eval()

    x = [
        [0.05, -0.01],
        [0.1, 0.3],
        [-0.4, 0.2],
        [1.0, 1.2],
        [0.0, 0.75],
        [0.0, -1.2]
    ]
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32)
        proba = net(x)
        pred_idx = torch.argmax(proba, dim=1).tolist()
        pred_proba = proba[range(len(pred_idx)), pred_idx].round(decimals=3).tolist()
        result = [{'id': i, 'proba': p} for i, p in zip(pred_idx, pred_proba)]
        print(result)


if __name__ == '__main__':
    training()
    # interface()
