# ==============================================================
# 文件：10_nn.Module基础.py
# 主题：PyTorch nn.Module 核心 API
# API：Module定义 / forward / parameters() / named_parameters()
#       state_dict() / load_state_dict() / children() / modules() / apply()
# ==============================================================
import torch
import torch.nn as nn
import copy

SEP = "=" * 60

# ==============================================================
# 1. 定义自定义 Module
# ==============================================================
print(SEP)
print("1. 定义自定义 Module")
print(SEP)

class MLP(nn.Module):
    """一个简单的多层感知机，演示 Module 基本用法"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()         # 必须调用父类 __init__
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """定义前向传播逻辑"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLP(input_dim=8, hidden_dim=16, output_dim=3)
print("模型结构:")
print(model)

# 调用 forward：通过 model(x) 触发，而非 model.forward(x)
x = torch.randn(4, 8)   # batch=4, features=8
out = model(x)
print("输入 shape:", x.shape)
print("输出 shape:", out.shape)

# ==============================================================
# 2. parameters() / named_parameters()
# ==============================================================
print()
print(SEP)
print("2. parameters() / named_parameters()")
print(SEP)

# parameters() 返回所有可学习参数的迭代器
total = sum(p.numel() for p in model.parameters())
print("总参数量:", total)

# named_parameters() 同时返回名称和参数张量
print("各层参数名称和 shape:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")

# 冻结某层参数（迁移学习常用）
for param in model.fc1.parameters():
    param.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"冻结 fc1 后，可训练参数量: {trainable}/{total}")
# 恢复
for param in model.fc1.parameters():
    param.requires_grad = True

# ==============================================================
# 3. state_dict() / load_state_dict()
# ==============================================================
print()
print(SEP)
print("3. state_dict() / load_state_dict()")
print(SEP)

# state_dict() 返回所有参数和 buffer 的字典
sd = model.state_dict()
print("state_dict 键:")
for k, v in sd.items():
    print(f"  {k}: {tuple(v.shape)}")

# 保存和加载（此处用内存演示，实际用 torch.save 到文件）
import copy
saved_sd = copy.deepcopy(sd)

# 修改参数（模拟训练）
with torch.no_grad():
    for p in model.parameters():
        p.fill_(999.0)

# 恢复
model.load_state_dict(saved_sd)
print("load_state_dict 后 fc1.weight 前3值:", model.fc1.weight.flatten()[:3].tolist())

# strict=False 允许忽略不匹配的键（微调时常用）
partial_sd = {"fc1.weight": saved_sd["fc1.weight"]}
model.load_state_dict(partial_sd, strict=False)
print("strict=False 允许部分加载")

# ==============================================================
# 4. children() / modules()
# ==============================================================
print()
print(SEP)
print("4. children() / modules()")
print(SEP)

# children() 仅返回直接子模块（不递归）
print("children() 直接子模块:")
for child in model.children():
    print(" ", type(child).__name__)

# modules() 递归返回所有模块（包括自身）
print("modules() 所有模块（递归）:")
for mod in model.modules():
    print(" ", type(mod).__name__)

# named_children() / named_modules() 同时返回名称
print("named_children():")
for name, child in model.named_children():
    print(f"  {name}: {type(child).__name__}")

# 嵌套 Module 演示
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(4,8), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(8,4), nn.Sigmoid())
    def forward(self, x):
        return self.block2(self.block1(x))

deep = DeepNet()
print("DeepNet named_modules():")
for name, mod in deep.named_modules():
    print(f"  [{name or "root"}]: {type(mod).__name__}")

# ==============================================================
# 5. apply()  对所有子模块递归应用一个函数
# ==============================================================
print()
print(SEP)
print("5. apply()")
print(SEP)

# 常用于权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        print(f"  初始化 Linear: {m.weight.shape}")

print("apply xavier 初始化:")
model.apply(init_weights)

# ==============================================================
# 6. train() / eval() 切换训练/推理模式
# ==============================================================
print()
print(SEP)
print("6. train() / eval() 模式")
print(SEP)

# Dropout 和 BatchNorm 在 train/eval 下行为不同
model.train()   # 开启 Dropout（按概率丢弃）
print("train 模式: model.training =", model.training)
out_train = model(x)

model.eval()    # 关闭 Dropout（直接透传）
print("eval 模式: model.training =", model.training)
with torch.no_grad():
    out_eval = model(x)

print("train/eval 输出相同?", torch.allclose(out_train, out_eval))  # False（dropout 随机）
print()
print("10_nn.Module基础.py 运行完毕!")
