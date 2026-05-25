#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/3 13:26
# @Author  : fntp
# @File    : 机器学习-demo.py.py
# @Software: PyCharm

import os
import warnings
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
import json


class MachineModel:

    def __init__(self, poly=None):
        super().__init__()
        # 做自己的额外init操作
        self.model = None
        self.poly = poly

    # 实例方法
    def transformer_x(self,x1, x2):
        features = []
        if not self.poly:
            return None
        for p in self.poly:
            if p == "1":
                features.append(1)
            elif p == "x1":
                features.append(x1)
            elif p == "x2":
                features.append(x2)
            elif p == "x1^2":
                features.append(x1 ** 2)
            elif p == "x1 x2":
                features.append(x1 * x2)
            elif p == "x2^2":
                features.append(x2 ** 2)
        return np.array(features)

    def recover_model(self, model_json_file_path):
        with open(model_json_file_path, "r") as f:
            data = json.load(f)
        self.poly=data["poly"]
        model = LogisticRegression()
        model.coef_ = np.array(data["algo"]["coef"])
        model.intercept_ = np.array(data["algo"]["intercept"])
        model.classes_ = np.array([0, 1])  # 必须加
        self.model = model
        return model


    def train(self):
        X,Y = make_circles(
            n_samples= 10000,
            noise= 0.1,
            factor=0.2,
            random_state=24
        )
        x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
        # print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
        # print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
        # print(f"类别取值: {np.unique(y_train)} - {np.bincount(y_train)} -- {np.unique(y_test)} - {np.bincount(y_test)}")

        # 特征工程
        # 2. 特征工程
        poly = PolynomialFeatures(degree=2)
        x_train = poly.fit_transform(x_train)
        x_test = poly.transform(x_test)
        print(f"多项式扩展转换规则: {poly.get_feature_names_out(['x1', 'x2'])}")
        print(f"转换后训练数据shape形状为: {type(x_train)} - {x_train.shape}")
        print(f"转换后评估数据shape形状为: {type(x_test)} - {x_test.shape}")

        # 模型创建
        logistic = LogisticRegression()
        logistic.fit(x_train,y_train)

        # 模型验证
        y_predict_test = logistic.predict(x_test)
        print(f"评估数据上的准确率: {metrics.accuracy_score(y_test, y_predict_test)}")
        print(f"评估数据上的分类报告: \n{metrics.classification_report(y_test, y_predict_test)}\n")

        # 6. 模型持久化
        joblib_dump_file = "../../base/out/ml/scx-logistic.pkl"
        os.makedirs(os.path.dirname(joblib_dump_file), exist_ok=True)
        joblib.dump({
            'poly': poly,
            'algo': logistic
        }, joblib_dump_file)

        if isinstance(logistic, LogisticRegression):
            json_dump_file = "../../base/out/ml/scx-logistic.json"
            with open(json_dump_file, "w", encoding="utf-8") as writer:
                json.dump(
                    {
                        'poly': poly.get_feature_names_out(['x1', 'x2']).tolist(),  # 获取多项式的组合规则，并转换为list输出
                        'algo': {
                            'intercept': logistic.intercept_.tolist(),  # 获取LR的截距项，并转换为list输出
                            'coef': logistic.coef_.tolist()  # 提取LR的参数项，并转换为list输出(json默认仅支持普通python类型)
                        }
                    },  # 持久化的对象
                    writer,  # 输出文件对象
                    indent=2,  # json格式化空格 -- 每个级别前面空2个空格
                    ensure_ascii=False  # 中文不进行编码输出，直接输出中文
                )



if __name__ == '__main__':
    m = MachineModel()
    current_model = m.recover_model("../../base/out/ml/scx-logistic.json")
    predict_result = current_model.predict(np.array(m.transformer_x(0.3, 0.8)).reshape(1, -1))
    proba = current_model.predict_proba(np.array(m.transformer_x(0.3, 0.8)).reshape(1, -1))
    print(proba)
    y_pred_idx_per_sample = np.argmax(proba, axis=1).tolist()
    y_pred_proba_per_sample = proba[range(len(y_pred_idx_per_sample)), y_pred_idx_per_sample].round(3).tolist()
    result = list(map(lambda t: {'id': t[0], 'proba': t[1]}, zip(y_pred_idx_per_sample, y_pred_proba_per_sample)))
    print(result)
