# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/10 20:58
Create User : 19410
Fixed By : 2026/05/25
Fix List:
    1. 评估阶段 loss 计算时，net.eval() 返回的是 softmax 概率，传入 CrossEntropyLoss 导致 loss 失真
       -> 评估时单独取 raw score 用于 loss，概率值仅用于 argmax 预测
    2. named_parameters() 返回迭代器，不能直接解包为单个 (name, param)
       -> 改为 for 循环遍历
    3. test_hoot_pre_feature 普通函数多了 self 参数
       -> 删除 self
    4. interface01 中 torch.no_grad() 未包裹前向过程
       -> 将 net(x) 移入 with torch.no_grad() 块内
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


class ClassifyNetwork(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """
        分类模型
        :param in_features: 分类模型输入的原始特征向量数目
        :param num_classes: 分类模型对应的类别数目
        """
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        # 1. 特征提取
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU()
        )

        # 2. 决策输出
        self.classify = nn.Linear(8, self.num_classes)

    def forward(self, x):
        """
        前向过程
            NOTE:
                bs: 样本数目
                in_features: 每个样本的向量维度大小
                置信度：模型损失计算前的数据对象
        :param x: 输入的原始特征向量，FloatTensor格式，shape形状为: [bs, in_features]
        :return: 前向结果，训练时候一般为置信度值，推理的时候可以直接返回预测结果，FloatTensor格式，shape形状为: [bs, num_classes]
        """
        # 1. 样本特征向量提取 [bs,in_features] --> [bs,8]
        features = self.features(x)

        # 2. 基于提取的特征向量进行分类决策 [bs,8] --> [bs,num_classes]
        score = self.classify(features)

        # 3. 基于不同的结果返回不同要求的数据
        if self.training:
            return score
        return torch.softmax(score, dim=1)


class Predictor(object):
    def __init__(self):
        super().__init__()
        # 1. 模型恢复
        obj = torch.load("./output/02/models/000099.pkl", map_location='cpu')
        net = obj['net']
        net.eval()
        print(f"模型恢复完成: \n{net}\n\n")
        self.net = net

    @torch.no_grad()
    def predict(self, x):
        # 2. 数据转换
        x = torch.tensor(x, dtype=torch.float32)

        # 3. 模型预测
        y_pred_proba = self.net(x)
        print(f"获取预测概率对象为: {type(y_pred_proba)} - {y_pred_proba.shape}")

        # 4. 结果拼接
        y_pred_proba = y_pred_proba.numpy()
        y_pred_idx_per_sample = np.argmax(y_pred_proba, axis=1).tolist()
        y_pred_proba_per_sample = y_pred_proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample] \
            .round(3).tolist()
        result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
        return result


def _hook_fn(_m, _m_input, _m_output):
    print(f"模块_{type(_m)}")
    return None


# Fix3: 普通函数不应有 self 参数
def test_hoot_pre_feature(_m, _m_input, _m_output):
    print("test_hoot_pre_feature")
    return None


# noinspection DuplicatedCode
def training():
    # 1. 加载数据
    X, Y = (
        make_circles(
            n_samples=1000,
            noise=0.1,
            factor=0.2,
            random_state=24
        ))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
    print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
    print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
    print(f"类别取值: {np.unique(y_train)} - {np.bincount(y_train)} -- {np.unique(y_test)} - {np.bincount(y_test)}")

    # 2. 模型创建
    net = ClassifyNetwork(in_features=x_train.shape[1], num_classes=len(np.unique(y_train)))
    # net.features.register_forward_hook(test_hoot_pre_feature)  # 注册前向过程的 hook
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(params=net.parameters(), lr=0.01)

    # Fix2: named_parameters() 返回迭代器，必须用 for 循环遍历
    for name, param in net.named_parameters():
        print(f"参数名: {name}, shape: {param.shape}")

    # 3. 模型训练 + 模型评估 + 模型持久化
    total_epoch = 1
    batch_size = 8
    test_batch_size = batch_size * 2
    total_train_batch = len(x_train) // batch_size
    total_test_batch = len(x_test) // test_batch_size + (1 if len(x_test) % test_batch_size != 0 else 0)
    model_output_dir = "output/02/models"
    os.makedirs(model_output_dir, exist_ok=True)

    for epoch in range(total_epoch):
        # 训练
        net.train()
        train_rnd_indexes = np.random.permutation(len(x_train))
        for batch_idx in range(total_train_batch):
            si = batch_size * batch_idx
            ei = si + batch_size
            train_batch_indexes = train_rnd_indexes[si: ei]
            batch_x_train = torch.tensor(x_train[train_batch_indexes], dtype=torch.float32)
            batch_y_train = torch.tensor(y_train[train_batch_indexes], dtype=torch.int64)

            score = net(batch_x_train)  # training=True，返回 raw logits
            loss = loss_fn(score, batch_y_train)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"Train Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_train_batch} Loss:{loss.item():.3f}")

        # Fix1: 评估阶段 loss 必须用 raw logits，不能用 softmax 后的概率
        # 方案：临时切换回 training=True 只取 score，或直接调用内部层绕过 forward 的分支
        # 这里采用更清晰的方案：eval 模式下手动走一遍 features + classify 取 raw score
        net.eval()
        with torch.no_grad():
            test_indexes = list(range(len(x_test)))
            for batch_idx in range(total_test_batch):
                si = test_batch_size * batch_idx
                ei = si + test_batch_size
                test_batch_indexes = test_indexes[si: ei]
                batch_x_test = torch.tensor(x_test[test_batch_indexes], dtype=torch.float32)
                batch_y_test = torch.tensor(y_test[test_batch_indexes], dtype=torch.int64)

                # 取 raw logits 用于 loss（绕过 forward 的 softmax 分支）
                raw_score = net.classify(net.features(batch_x_test))  # [bs, num_classes]
                loss = loss_fn(raw_score, batch_y_test)

                # 取概率用于 argmax 预测（走完整 forward，含 softmax）
                proba = net(batch_x_test)  # [bs, num_classes]，softmax 概率
                pred_idx = torch.argmax(proba, dim=1)
                acc = metrics.accuracy_score(batch_y_test.cpu().numpy(), pred_idx.cpu().numpy())

                print(f"Test Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_test_batch} "
                      f"Batch-number:{batch_x_test.shape[0]} Loss:{loss.item():.3f} Accuracy:{acc:.3f}")

        # 模型持久化
        torch.save(
            {
                'net': net,
                'net_param': net.state_dict(),
                'epoch': epoch
            },
            os.path.join(model_output_dir, f"{epoch:06d}.pkl")
        )


# noinspection DuplicatedCode
def interface01():
    # 1. 模型恢复
    obj = torch.load("output/02/models/000099.pkl", map_location='cpu', weights_only=False)
    net = obj['net']
    net.eval()
    print(f"模型恢复完成: \n{net}\n\n")

    # 2. 数据转换
    x = [
        [0.05, -0.01],
        [0.1, 0.3],
        [-0.4, 0.2],
        [1.0, 1.2],
        [0.0, 0.75],
        [0.0, -1.2]
    ]
    x = torch.tensor(x, dtype=torch.float32)

    # Fix4: no_grad 必须包裹前向过程，否则梯度图已经在 net(x) 时构建完毕
    with torch.no_grad():
        y_pred_proba = net(x)
        print(f"获取预测概率对象为: {type(y_pred_proba)} - {y_pred_proba.shape}")

        y_pred_proba = y_pred_proba.numpy()
        y_pred_idx_per_sample = np.argmax(y_pred_proba, axis=1).tolist()
        y_pred_proba_per_sample = y_pred_proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample] \
            .round(3).tolist()
        result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
        print(result)


def interface02():
    p = Predictor()

    r0 = p.predict([[0, 0.3], [1.2, 1.5]])
    print(r0)

    while True:
        v = input("请输入样本特征，使用空格隔开:")
        if "q" == v:
            break

        x = list(map(lambda s: float(s.strip()), v.split(" ")))
        r = p.predict([x])
        print(r)


if __name__ == '__main__':
    training()
    # interface01()
    # interface02()
