import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split



class ClassifyNetwork(nn.Module):

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        # 首先确定分类分多少类 输入多少维度的特征
        self.in_features = in_features
        self.num_classes = num_classes

        # 在确认网络提取的特征是什么样子的
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU()
        )

        # 最后定义好我们网络的输出结构 决策输出
        self.classify = nn.Linear(8, num_classes)


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
        features = self.features(x)
        classify_score = self.classify(features)
        if self.training:
            return classify_score
        return torch.softmax(classify_score, dim=1)


    def predict(self, x):
        """
        模型预测
        :param x: 输入的样本特征向量，FloatTensor格式，shape形状为: [bs, in_features]
        :return: 预测结果，FloatTensor格式，shape形状为: [bs, num_classes]
        """
        return self.forward(x)

    def evaluate(self, x, y):
        """
        模型评估
        :param x: 输入的样本特征向量，FloatTensor格式，shape形状为: [bs, in_features]
        :param y: 输入的样本标签向量，FloatTensor格式，shape形状为: [bs, num_classes]
        :return: 评估结果，FloatTensor格式，shape形状为: [num_classes]
        """
        y_pred = self.predict(x)
        return metrics.accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())

    def save(self, path):
        """
        模型保存
        :param path: 模型保存路径
        :return: None
        """
        torch.save(self.state_dict(), path)


    def load(self, path):
        """
        模型加载
        :param path: 模型加载路径
        :return: None
        """
        self.load_state_dict(torch.load(path))





