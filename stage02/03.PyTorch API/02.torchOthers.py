import torch
from torch import nn

def m1_torch_parameters():
    # 张量参数
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2x2 的张量
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # 2x2 的张量
    z = torch.add(x, y)  # 使用 torch.add() 方法进行加法运算
    print("张量 x:")
    print(x)
    print("张量 y:")
    print(y)
    print("张量 z (x 和 y 的和):")
    print(z)
    # 使用Parameter来定义一个可训练的参数
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))  # 定义一个可训练的参数
    print("可训练的参数 param:")
    print(param)
    # 对比使用 torch.tensor 定义的张量和 torch.nn.Parameter 定义的参数
    tensor = torch.tensor([1.0, 2.0, 3.0])  # 定义一个普通的张量
    print("普通的张量 tensor:")
    print(tensor)
    print("参数 param 和张量 tensor 是否相等:", torch.equal(param.data, tensor ))
    print("参数 param 是否需要梯度:", param.requires_grad)
    print("张量 tensor 是否需要梯度:", tensor.requires_grad)



def m2_torch_simple_linear():
    w = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # 定义权重参数 2x2 的矩阵
    print("权重参数 w: {}".format(w))
    x = torch.rand(2,9) # 定义输入张量 2x9 的矩阵
    print("输入张量 x: {}".format(x))
    b = nn.Parameter(torch.tensor([1.0, 2.0]))  # 定义偏置参数 1x2 的向量
    print("偏置参数 b: {}".format(b))
    y = w @ x + b  # 进行线性变换
    print("线性变换结果 y: {}".format(y))




if __name__ == "__main__":
    # m1_torch_parameters()
    m2_torch_simple_linear()