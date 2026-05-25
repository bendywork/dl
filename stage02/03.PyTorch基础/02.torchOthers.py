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

def m3_torch_data_split_and_combine():
    # 创建一个张量
    x = torch.tensor([[1, 2, 1], [3, 4, 2], [5, 6, 3]], dtype=torch.float32)  # 3x3 的张量
    print("原始张量 x:")
    print(x)
    # 使用 torch.split() 方法进行数据分割 dim=1 表示按列分割， [2, 1] 表示第一部分包含前两列，第二部分包含最后一列
    split1, split2 = torch.split(x, [2, 1],dim = 1)  # 将张量分割成两个部分，第一部分包含前两列，第二部分包含最后一列
    print("分割后的张量 split1:")
    print(split1)
    print("分割后的张量 split2:")
    print(split2)
    # 使用 torch.cat() 方法进行数据组合 dim=0 表示沿着第0维（行）进行组合
    combined = torch.cat((split1, split2), dim=0)  # 将分割后的张量沿着第0维（行）进行组合
    print("组合后的张量 combined:")
    print(combined)


def m4_test_linear():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)  # 定义一个线性层，输入特征数为3，输出特征数为2，包含偏置项
    print("线性层 linear: {}".format(linear))
    x = torch.rand(1, 3)  # 定义输入张量，形状为1x3
    print("输入张量 x: {}".format(x))
    y = linear(x)  # 进行线性变换
    print("线性变换结果 y: {}".format(y))
    print("Linear shape:{}".format(y.shape))  # 输出线性变换结果的形状
    linear.register_parameter("大鸡巴_weight", nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]])))  # 注册一个新的参数 custom_weight
    # 拿到Linear的属性信息
    for name, param in linear.named_parameters():
        print("参数名称: {}, 参数值: {}".format(name, param))
 
if __name__ == "__main__":
    # m1_torch_parameters()
    # m2_torch_simple_linear()
    # m3_torch_data_split_and_combine()
    m4_test_linear()