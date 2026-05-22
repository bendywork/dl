from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
import torch

# 深度学习代码流程练习
# 分类网络搭建流程
class DeepLearning_Classify(nn.Module):
    
    # 自定义网络结构
    def __init__(self, in_features: int, num_classes: int):
        super(DeepLearning_Classify, self).__init__()
        print("深度学习代码流程练习")
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.in_features = in_features
        self.num_classes = num_classes
        # 定义网络结构 
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 8),
            nn.ReLU(), 
            nn.Linear(8, 24),
            nn.ReLU()
        )
        self.classify = nn.Linear(24, self.num_classes)


    def data_load(self):
        print("数据加载")
        x, y = make_circles(n_samples=1000, factor=0.5, noise=0.06, random_state=0)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=24)
        print(f"训练数据shape形状为: {type(self.x_train)} - {self.x_train.shape} -- {type(self.y_train)} - {self.y_train.shape}")
        print(f"评估数据shape形状为: {type(self.x_test)} - {self.x_test.shape} -- {type(self.y_test)} - {self.y_test.shape}")


    def model_building(self):
        print("模型构建")
        
        


    def model_training(self):
        print("模型训练")

    def model_evaluation(self):
        print("模型评估")

    def model_deployment(self):
        print("模型部署")