# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/12 20:42
Create User : 19410
Desc : xxx
"""

import torch
import torch.nn as nn
import numpy as np


def m1_create_tensor():
    # 创建一个张量 默认浮点型是 float32 默认整型是 int64
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y = torch.tensor(np.array([[5, 6], [7, 8]]), dtype=torch.float32)  # 从 NumPy 数组创建张量
    p = torch.rand(2,3)
    # 区别于numpy
    q = np.random.rand(2, 3)
    z = torch.zeros((2, 2))  # 创建一个全零张量
    w = torch.ones((2, 2))  # 创建一个全一张量
    print("张量 x:")
    print(x)
    print("张量 y:")
    print(y)
    print("张量 z:")
    print(z)
    print("张量 w:")
    print(w)
    print("张量 p:")
    print(p, type(p))
    print("ndarry q:")
    print(q, type(q))

    # 张量的形状
    print("张量 x 的形状:", x.shape)

    # 张量的类型
    print("张量 x 的数据类型:", x.dtype)

    # 张量的设备
    print("张量 x 的设备:", x.device)

    # 是否需要梯度更新
    print("张量 x 是否需要梯度更新:", x.requires_grad)


def m2_tensor_operations():
    return 


if __name__ == "__main__":
    m1_create_tensor()
