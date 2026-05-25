from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
from torch import optim
import torch

# 深度学习代码流程练习
# 分类网络搭建流程
# 自定义模块
class DeepLearning_Classify(nn.Module):
    
    # 自定义网络结构
    def __init__(self, in_features: int, num_classes: int):
        super(DeepLearning_Classify, self).__init__()
        print("深度学习代码流程练习")
        self.in_features = in_features
        self.num_classes = num_classes
        # 定义网络结构 输入的结构一定是【bs，in_features】bs个样本 in_features个特征输入 对应就是bs行数据，每条数据有in_features个特征
        # 内部全连接的层数和每层的神经元数目也是由自己定义的 这里我们定义两层全连接层 每层8个神经元 激活函数使用ReLU
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 8),
            nn.ReLU(), 
            nn.Linear(8, 8),
            nn.ReLU()
        )
        # 置信度输出
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
            # 训练时候返回置信度值
            return score
        # 推理时候返回预测结果 置信度值经过argmax操作得到预测结果 [bs,num_classes] --> [bs]
        return score.argmax(dim=1)

    def forward_with_base_api(self, x):
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
        # features = self.features[0](x)
        # features = self.features[1](features)
        # features = self.features[2](features)
        # features = self.features[3](features)
        # 换一种写法
        # features = self.features(x)
        # 第三种写法
        for layer in self.features:
            x =layer(x)
        features = x
         # 2. 基于提取的特征向量进行分类决策 [bs,8] --> [bs,num_classes]
        score = self.classify(features)
        # 3. 基于不同的结果返回不同要求的数据
        if self.training:
            # 训练时候返回置信度值
            return score
        # 推理时候返回预测结果 置信度值经过argmax操作得到预测结果 [bs,num_classes] --> [bs]
        return score.argmax(dim=1)

'''
第一步 数据加载（网络结构不会干涉数据加载流程，因此数据加载流程是独立于网络结构的）
'''
def data_load() -> tuple:
        print("数据加载")
        x, y = make_circles(n_samples=1000, factor=0.5, noise=0.06, random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
        print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
        print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
        return x_train, x_test, y_train, y_test

def training():
    # 1. 数据加载
    x_train, x_test, y_train, y_test = data_load()
    # 2. 模型创建
    net = DeepLearning_Classify(in_features=x_train.shape[1], num_classes=len(np.unique(y_train)))
    # 3. 定义好损失函数+优化器
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(params=net.parameters(), lr=0.01)
    # 开始训练前需要设置一些运行参数
    # 我这个模型一共需要训练多少轮次呢 整个训练集整体要循环训练多少次
    total_epoch = 100
    # 我这个模型每一轮需要训练的数据量是多少呢
    batch_size = 8
    # 我这个模型，要经过 total_epoch 次总体数据集循环，单看一轮循环中， 训练数据集 一次只能训练batch_size条数据
    # ，那么训练数据集又要被切成多少个 batch_size 大小的小批次进行训练呢
    total_train_batch = len(x_train) // batch_size
    # 开始训练
    for epoch in range(total_epoch):
        # 标记一下网络现在是训练模式
        net.train()
        # 每一轮训练都要把训练数据集切成一个个小批次进行训练
        for batch_idx in range(total_train_batch):
            # 先从numpy数据集中拿出数据
            x_batch = x_train[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
            y_batch = y_train[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
            # 再把数据转换成torch的Tensor格式
            x_train_batch = torch.tensor(x_batch, dtype=torch.float32)
            y_train_batch = torch.tensor(y_batch, dtype=torch.int64)
            # 训练步骤：前向 + loss + zero_grad + backward + step 五步 
            # 训练第一步 前向过程得到置信度值
            score = net(x_train_batch)
            # 训练第二步 基于置信度值和标签计算 loss
            loss = loss_fn(score, y_train_batch)
            # 训练第三步 优化器梯度清零
            opt.zero_grad()
            # 训练第四步 反向传播计算损失
            loss.backward()
            # 训练第五步 优化器更新参数
            opt.step()
        # 训练第六步 及时评估训练状态
        # 现在进入评估状态
        net.eval()
        # 评估数据集也要切成一个个小批次进行评估
        total_test_batch = len(x_test) // batch_size
        with torch.no_grad():
            for batch_idx in range(total_test_batch):
                x_test_batch = x_test[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
                y_test_batch = y_test[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
                x_test_batch = torch.tensor(x_test_batch, dtype=torch.float32)
                y_test_batch = torch.tensor(y_test_batch, dtype=torch.int64)
                pred = net(x_test_batch)
                acc = (pred == y_test_batch).float().mean().item()
                print(f"第{epoch}轮评估完成，评估准确率为: {acc:.3f}")

        print(f"第{epoch}轮训练完成，loss值为: {loss.item()}")



if __name__ == '__main__':
    training()