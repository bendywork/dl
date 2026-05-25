import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from ClassifyNetwork import ClassifyNetwork


class Predictor(object):

    def __init__(self):
        super().__init__()
        self.model = None
        self.net = None

    def load_model(self):
        """
        模型加载
        :return: None
        """
        self.model = torch.load("../../base/out/dl/models/000099.pkl", map_location='cpu')
        net = self.model['net']
        net.eval()
        print(f"模型恢复完成: \n{net}\n\n")
        self.net = net

    @torch.no_grad()
    def predict(self, x):
        """
        模型预测
        :param x: 输入的样本特征向量，FloatTensor格式，shape形状为: [bs, in_features]
        :return: 预测结果，FloatTensor格式，shape形状为: [bs, num_classes]
        """
        x = torch.tensor(x, dtype=torch.float32)
        y_pred_proba = self.net(x)
        print(f"获取预测概率对象为: {type(y_pred_proba)} - {y_pred_proba.shape}")
        y_pred_proba = y_pred_proba.numpy()
        y_pred_idx_per_sample = np.argmax(y_pred_proba, axis=1).tolist()
        y_pred_proba_per_sample = y_pred_proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample] \
            .round(3).tolist()
        result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
        return result

    def _hook_fn(_m, _m_input, _m_output):
        # 一般情况下为日志持久化相关操作
        print(f"模块_{type(_m)}")
        return None

    def training(self):
        # 1. 加载数据
        X, Y = (
            make_circles(
                n_samples=1000,  # 样本数目
                noise=0.1,  # 噪声样本比例
                factor=0.2,  # 内圈直径是外圈直径的factor倍
                random_state=24  # 随机数种子
            ))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
        print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
        print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
        print(f"类别取值: {np.unique(y_train)} - {np.bincount(y_train)} -- {np.unique(y_test)} - {np.bincount(y_test)}")

        # 2. 模型创建
        # 模型结构创建
        net = ClassifyNetwork(in_features=x_train.shape[1], num_classes=len(np.unique(y_train)))
        # 损失函数创建
        loss_fn = nn.CrossEntropyLoss()
        # 优化器创建
        opt = optim.SGD(params=net.parameters(), lr=0.01)

        # 定义hook
        # handle = net.features.register_forward_hook(_hook_fn)
        # handle.remove()  # 删除，不进行hook操作

        # 3. 模型训练+模型评估+模型持久化
        total_epoch = 100
        batch_size = 8
        test_batch_size = batch_size * 2
        total_train_batch = len(x_train) // batch_size
        total_test_batch = len(x_test) // test_batch_size + (1 if len(x_test) % test_batch_size != 0 else 0)
        model_output_dir = "../../base/out/dl/models"
        os.makedirs(model_output_dir, exist_ok=True)
        for epoch in range(total_epoch):
            # 训练
            net.train()  # 将net模块内部的对应的属性进行标记 --> 因为某些模块在训练和推理的时候采用的是不同逻辑
            train_rnd_indexes = np.random.permutation(len(x_train))
            for batch_idx in range(total_train_batch):
                # 获取当前批次的数据x + y
                si = batch_size * batch_idx
                ei = si + batch_size
                train_batch_indexes = train_rnd_indexes[si: ei]
                batch_x_train = torch.tensor(x_train[train_batch_indexes], dtype=torch.float32)
                batch_y_train = torch.tensor(y_train[train_batch_indexes], dtype=torch.int64)

                # 前向过程
                score = net(batch_x_train)  # [bs,num_classes]
                loss = loss_fn(score, batch_y_train)

                # 反向过程
                opt.zero_grad()  # 重置当前优化器对应的所有参数的梯度为0
                loss.backward()  # 计算和当前损失相同的所有参数的梯度值
                opt.step()  # 参数更新

                print(f"Train Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_train_batch} Loss:{loss.item():.3f}")

            # 评估
            net.eval()
            with torch.no_grad():
                test_indexes = list(range(len(x_test)))
                for batch_idx in range(total_test_batch):
                    # 获取当前批次的数据x + y
                    si = test_batch_size * batch_idx
                    ei = si + test_batch_size
                    test_batch_indexes = test_indexes[si: ei]
                    batch_x_test = torch.tensor(x_test[test_batch_indexes], dtype=torch.float32)
                    batch_y_test = torch.tensor(y_test[test_batch_indexes], dtype=torch.int64)

                    # 前向过程
                    score = net(batch_x_test)  # [bs,num_classes]
                    loss = loss_fn(score, batch_y_test)

                    # 效果评估
                    pred_idx = torch.argmax(score, dim=1)  # 获取预测的类别id
                    acc = metrics.accuracy_score(batch_y_test.cpu().numpy(), pred_idx.cpu().numpy())

                    print(f"Test Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_test_batch} "
                          f"Batch-number:{batch_x_test.shape[0]} Loss:{loss.item():.3f} Accuracy:{acc:.3f}")

            # 模型持久化
            ## torch.save 和joblib.dump类似，底层采用pickle进行二进制的数据存储
            if epoch == 99:
                torch.save(
                    {
                        'net': net,  # 模型对象(参数 + 结构)
                        'net_param': net.state_dict(),  # 模型网络对应的所有参数
                        'epoch': epoch
                    },
                    os.path.join(model_output_dir, f"{epoch:02d}.pkl")
                )


def interface02():
    p = Predictor()

    # 加载模型
    p.load_model()

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
    interface02()