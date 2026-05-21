# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/12 20:42
Create User : 19410
Desc : xxx
"""

import torch
import torch.nn as nn
import numpy as np
import inspect


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


def m2_tensor_create__operations():
    # tensor object 2 python object
    tensor_demo = torch.tensor([1, 2, 3])
    print("张量对象:", tensor_demo)
    py_list = tensor_demo.tolist()  # 将张量转换为 Python 列表 不支持GPU张量转为python对象？为什么
    print("Python 列表:", py_list)
    print("张量对象的方法和属性:")
    # print(inspect.getsource(type(tensor_demo)))  # 打印张量对象的方法和属性
    # 基于numpy对象创建tensor对象，不共享内存
    np_array = np.array([4, 5, 6])
    tensor_base_np = torch.tensor(np_array)  # 从 NumPy 数组创建张量
    print("NumPy 数组:", np_array)
    print("基于 NumPy 数组创建的张量:", tensor_base_np)
    np_array[0] = 10  # 修改 NumPy 数组的第一个元素
    print("修改后的 NumPy 数组:", np_array)
    print("修改后的张量:", tensor_base_np)  # 张量不会受到影响，因为它不共享内存
    # 如果是想要共享内存，可以使用 torch.from_numpy() 方法
    tensor_from_np = torch.from_numpy(np_array)  # 从 NumPy 数组创建张量并共享内存
    print("从 NumPy 数组创建的张量并共享内存:", tensor_from_np)
    np_array[1] = 20  # 修改 NumPy 数组的第二个元素
    print("修改后的 NumPy 数组:", np_array)
    print("修改后的张量:", tensor_from_np)  # 张量会受到影响
    # numpy only support CPU tensor, if you want to use GPU tensor, you need to use torch.tensor() method
    return 


def m3_tensor_reshape_operations():
    print("张量重塑操作:")
    x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])  # 创建一个 4x2 的张量
    print("原始张量 x:")
    x = x.reshape(1,2,4)  # 将张量重塑 为 1x2x4 的张量
    print(x)
    # 使用 view() 方法重塑张量
    y = x.view(4)  # 将张量重塑为一维张量
    print("使用 view() 方法重塑后的张量 y:") 
    # view() 方法只能用于连续的张量，如果张量不是连续的，则需要使用 reshape() 方法
    print(y)

def m4_type_convert():
    # 张量类型转换
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    x = x.to(dtype=torch.int64)  # 将张量转换为 int64 类型
    print("张量 x 的数据类型:", x.dtype)

def m5_device_convert():
    # 张量设备转换
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    x = x.to(device)  # 将张量移动到 GPU 上
    print("张量 x 的设备: {},x values is {}", x.device, x)

def m6_tensor_add():
    # 张量加法 两个张量的形状必须相同或者满足广播机制 并且都必须在1个设备上CPU或者GPU
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.tensor([4, 5, 6], dtype=torch.float32)
    z = x + y  # 使用 + 运算符进行张量加法
    print("张量 x:", x)
    print("张量 y:", y)
    print("张量 z (x + y):", z)

def m7_tensor_mul():
    # 张量乘法 两个张量的形状必须相同或者满足广播机制 并且都必须在1个设备上CPU或者GPU
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.tensor([4, 5, 6], dtype=torch.float32)
    z = x * y  # 使用 * 运算符进行张量乘法
    print("张量 x:", x)
    print("张量 y:", y)
    print("张量 z (x * y):", z)
    # 与数字想乘
    a = 2
    z = x * a  # 张量与数字相乘
    print("张量 z (x * a):", z)

def m8_tensor_matmul():
    # 张量矩阵乘法
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # 2x2 的张量
    z = torch.matmul(x, y)  # 使用 torch.matmul() 方法进行矩阵乘法
    print("张量 x:")
    print(x)
    print("张量 y:")
    print(y)
    print("张量 z (x @ y):")
    print(z)


def m9_tensor_dot():
    # 张量点积
    x = torch.tensor([1, 2, 3], dtype=torch.float32)  # 1D 张量
    y = torch.tensor([4, 5, 6], dtype=torch.float32)  # 1D 张量
    z = torch.dot(x, y)  # 使用 torch.dot() 方法进行点积
    print("张量 x:", x)
    print("张量 y:", y)
    # 点积是两个向量的乘积，结果是一个标量 所以 z 的形状是 () 而不是 (1,) 或者 (3,)
    print("张量 z (x . y):", z)


def m10_tensor_argmax():
    # 张量求最大值的索引
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    max_index = torch.argmax(x, dim=1)  # 使用 torch.argmax() 方法求最大值的索引
    print("张量 x:")
    print(x)
    print("张量 x 中最大值的索引: {}, 值：{}".format(max_index, x[torch.arange(x.size(0)), max_index]))  # 打印最大值的索引和对应的值

def m11_top_k_values():
    # 张量求前 k 个最大值及其索引
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    top_k_values, top_k_indices = torch.topk(x, k=1, dim=1)  # 使用 torch.topk() 方法求前 k 个最大值及其索引
    print("张量 x:")
    print(x)
    print("张量 x 中前 k 个最大值: {}, 索引: {}".format(top_k_values, top_k_indices))  # 打印前 k 个最大值和对应的索引


def m12_tensor_mean():
    # 张量求平均值
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    mean_value = torch.mean(x)  # 使用 torch.mean() 方法求平均值
    print("张量 x:")
    print(x)
    print("张量 x 的平均值:", mean_value)

def m13_tensor_sum():
    # 张量求和
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    sum_value = torch.sum(x)  # 使用 torch.sum() 方法求和
    print("张量 x:")
    print(x)
    print("张量 x 的总和:", sum_value)

def m14_tensor_mean_dim():
    # 张量按维度求平均值
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    mean_dim0 = torch.mean(x, dim=0)  # 按维度 0 求平均值
    mean_dim1 = torch.mean(x, dim=1)  # 按维度 1 求平均值
    print("张量 x:")
    print(x)
    print("张量 x 按维度 0 的平均值:", mean_dim0)
    print("张量 x 按维度 1 的平均值:", mean_dim1)

def m15_tensor_sum_dim():
    # 张量按维度求和
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    sum_dim0 = torch.sum(x, dim=0)  # 按维度 0 求和
    sum_dim1 = torch.sum(x, dim=1)  # 按维度 1 求和
    print("张量 x:")
    print(x)
    print("张量 x 按维度 0 的总和:", sum_dim0)
    print("张量 x 按维度 1 的总和:", sum_dim1)

def m16_tensor_transpose():
    # 张量转置
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    x_transpose = torch.transpose(x, dim0=0, dim1=1)  # 使用 torch.transpose() 方法进行转置
    print("张量 x:")
    print(x)
    print("张量 x 的转置:")
    print(x_transpose)

def m17_tensor_permute():
    # 张量维度交换
    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)  # 2x2x2 的张量
    x_permute = torch.permute(x, (1, 0, 2))  # 使用 torch.permute() 方法进行维度交换
    print("张量 x:")
    print(x)
    print("张量 x 的维度交换:")
    print(x_permute)

def m18_tensor_squeeze():
    # 张量去除维度
    x = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)  # 1x2x2 的张量
    x_squeeze = torch.squeeze(x)  # 使用 torch.squeeze() 方法去除维度
    print("张量 x:")
    print(x)
    print("张量 x 去除维度后的结果:")
    print(x_squeeze)


def m19_tensor_unsqueeze():
    # 张量增加维度
    x = torch.tensor([1, 2, 3], dtype=torch.float32)  # 1D 的张量
    x_unsqueeze = torch.unsqueeze(x, dim=0)  # 使用 torch.unsqueeze() 方法增加维度
    print("张量 x:")
    print(x)
    print("张量 x 增加维度后的结果:")
    print(x_unsqueeze)


def m20_tensor_stack():
    # 张量堆叠
    x = torch.tensor([1, 2, 3], dtype=torch.float32)  # 1D 的张量
    y = torch.tensor([4, 5, 6], dtype=torch.float32)  # 1D 的张量
    z = torch.stack((x, y), dim=0)  # 使用 torch.stack() 方法进行堆叠
    print("张量 x:", x)
    print("张量 y:", y)
    print("张量 z (x 和 y 堆叠):")
    print(z) 


def m21_tensor_cat():
    # 张量连接
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # 2x2 的张量
    z = torch.cat((x, y), dim=0)  # 使用 torch.cat() 方法进行连接
    print("张量 x:")
    print(x)
    print("张量 y:")
    print(y)
    print("张量 z (x 和 y 连接):")
    print(z)


def m22_tensor_split():
    # 张量分割
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)  # 3x2 的张量
    split_x = torch.split(x, split_size_or_sections=1, dim=0)  # 使用 torch.split() 方法进行分割
    print("张量 x:")
    print(x)
    print("张量 x 分割后的结果:")
    for i, split in enumerate(split_x):
        print("分割 {}:".format(i))
        print(split)


def m23_tensor_chunk():
    # 张量分块
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)  # 3x2 的张量
    chunk_x = torch.chunk(x, chunks=3, dim=0)  # 使用 torch.chunk() 方法进行分块
    print("张量 x:")
    print(x)
    print("张量 x 分块后的结果:")
    for i, chunk in enumerate(chunk_x):
        print("分块 {}:".format(i))
        print(chunk)


def m24_tensor_gather():
    # 张量收集
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    indices = torch.tensor([[0, 1], [1, 0]])  # 索引张量
    gathered_x = torch.gather(x, dim=1, index=indices)  # 使用 torch.gather() 方法进行收集
    print("张量 x:")
    print(x)
    print("索引张量:")
    print(indices)
    print("收集后的张量:")
    print(gathered_x)


if __name__ == "__main__":
    # m1_create_tensor()
    # m2_tensor_create__operations()
    # m3_tensor_reshape_operations()
    # m4_type_convert()
    # m5_device_convert()
    # m6_tensor_add()
    # m7_tensor_mul()
    # m8_tensor_matmul()
    # m9_tensor_dot()
    # m10_tensor_argmax()
    m11_top_k_values()