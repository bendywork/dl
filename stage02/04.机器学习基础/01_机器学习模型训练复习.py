import json
import os
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import joblib


# 机器学习模型训练复习
class MachineLearning:
    def __init__(self, model=None):
        self.model = model
        self.poly = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.poly = None
        self.pred_train = None
        self.pred_test = None

    ## 生成数据集
    def init_data_set(self):
        print("Generating data set for training...")
        '''
        n_samples: 样本数量
        factor: 内圈和外圈的比例
        noise: 数据的噪声水平
        '''
        x, y = make_circles(n_samples=1000, factor=0.1, noise=0.2, random_state=24)
        # 将数据集分为训练集和测试集
        '''
        test_size: 测试集占总数据的比例
        random_state: 随机种子，确保每次运行结果一致
        '''
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=24)
        '''
        打印训练集与测试集的shape与大小
        '''
        print(f"Training set: x_train shape: {self.x_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"Test set: x_test shape: {self.x_test.shape}, y_test shape: {self.y_test.shape}")
        '''
        打印类别取值的分布情况
        '''
        # print(f"Training set class distribution: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")    
        # print(f"Test set class distribution: {dict(zip(*np.unique(self.y_test, return_counts=True)))}")


    # 特征工程
    def feature_engineering(self):
        print("Performing feature engineering on the data...")
        poly = PolynomialFeatures(degree=2)
        self.x_train = poly.fit_transform(self.x_train)
        self.x_test = poly.transform(self.x_test) 
        self.poly = poly
        print(f"多项式扩展转换规则: {poly.get_feature_names_out(['x1', 'x2'])}")
        print(f"转换后训练数据shape形状为: {type(self.x_train)} - {self.x_train.shape}")
        print(f"转换后评估数据shape形状为: {type(self.x_test)} - {self.x_test.shape}")
        

    # 创建模型
    def create_model(self):
        print(f"Creating model: {self.model}")
        # 这里以逻辑回归为例，实际可以根据需要选择不同的模型
        ''' LogisticRegression的参数说明：
            max_iter: 最大迭代次数，默认值为100。对于某些数据集，可能需要增加这个值以确保模型收敛。
        '''
        self.model = LogisticRegression(max_iter=1000)  

    ## 训练模型
    def train(self, data=None):
        print(f"Training {self.model} with data: {data}")
        self.model.fit(self.x_train, self.y_train)

    # 预测与模型评估
    def predict(self, input_data=None):
        print(f"Predicting with {self.model} for input: {input_data}")
        self.pred_train = self.model.predict(self.x_train)
        self.pred_test = self.model.predict(self.x_test)
        print(f"预测结果类型: {type(self.pred_train)} - {self.pred_train.shape}")
        print(f"训练数据上的准确率: {metrics.accuracy_score(self.y_train, self.pred_train)}")
        print(f"评估数据上的准确率: {metrics.accuracy_score(self.y_test, self.pred_test)}")
        print(f"训练数据上的分类报告: \n{metrics.classification_report(self.y_train, self.pred_train)}\n")
        print(f"评估数据上的分类报告: \n{metrics.classification_report(self.y_test, self.pred_test)}\n")

    # 模型持久化与恢复
    def save_model_with_file(self, file_path):
        print(f"Saving model to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_obj = {'model': self.model, 'poly': self.poly}
        joblib.dump(model_obj, file_path)
    
    # 从文件中加载模型
    def load_model_with_file(self, file_path):
        print(f"Loading model from {file_path}...")
        model_obj = joblib.load(file_path)
        poly = model_obj['poly']
        algo = model_obj['model']
        return poly, algo
    
    # 将模型保存为JSON格式
    def save_model_with_json(self, file_path):
        print(f"Saving model to {file_path} in JSON format...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(self.model, LogisticRegression):
            json_dump_file = file_path
            with open(json_dump_file, "w", encoding="utf-8") as writer:
                json.dump(
                    {
                        'poly': self.poly.get_feature_names_out(['x1', 'x2']).tolist(),  # 获取多项式的组合规则，并转换为list输出
                        'algo': {
                            'intercept': self.model.intercept_.tolist(),  # 获取LR的截距项，并转换为list输出
                            'coef': self.model.coef_.tolist()  # 提取LR的参数项，并转换为list输出(json默认仅支持普通python类型)
                        }
                    },  # 持久化的对象
                    writer,  # 输出文件对象
                    indent=2,  # json格式化空格 -- 每个级别前面空2个空格
                    ensure_ascii=False  # 中文不进行编码输出，直接输出中文
                )
    
    # 从JSON文件中加载模型
    def load_model_with_json(self, file_path):
        print(f"Loading model from {file_path} in JSON format...")
        with open(file_path, "r", encoding="utf-8") as reader:
            model_data = json.load(reader)
            poly_features = model_data['poly']
            algo_params = model_data['algo']
            print(f"模型恢复完成: {poly_features} -- {algo_params}")
            return LogisticRegression(intercept=np.array(algo_params['intercept']), coef=np.array(algo_params['coef'])), poly_features

if __name__ == "__main__":
    ml_model = MachineLearning()
    ml_model.init_data_set()
    ml_model.feature_engineering()
    ml_model.create_model()
    ml_model.train()
    ml_model.predict()
