# ==============================================================
# 文件：12_激活函数.py
# 主题：PyTorch 常用激活函数
# API：ReLU / Sigmoid / Tanh / Softmax / GELU / LeakyReLU / ELU / Swish(SiLU)
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

SEP = "=" * 60

# 测试输入：包含负数、零、正数
x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])

# ==============================================================
# 1. ReLU  max(0, x)
# 优点：简单高效，不存在梯度消失（正区间）
# 缺点："Dead ReLU" — 负区间梯度永远为0，神经元可能永久不激活
# ==============================================================
print(SEP)
print("1. ReLU  f(x) = max(0, x)")
print(SEP)
relu = nn.ReLU()
print("输入:", x.tolist())
print("ReLU:", relu(x).tolist())
print("使用场景：CNN隐藏层的默认选择，计算快")

# ==============================================================
# 2. LeakyReLU  max(negative_slope*x, x)
# 负区间有微小斜率，解决 Dead ReLU 问题
# ==============================================================
print()
print(SEP)
print("2. LeakyReLU  f(x) = max(0.01x, x)")
print(SEP)
leaky = nn.LeakyReLU(negative_slope=0.01)
print("LeakyReLU:", [round(v,3) for v in leaky(x).tolist()])
print("使用场景：GAN discriminator，避免梯度死亡")

# ==============================================================
# 3. ELU  x if x>0 else alpha*(e^x - 1)
# 负区间平滑，均值接近0（更快收敛），但计算较慢
# ==============================================================
print()
print(SEP)
print("3. ELU")
print(SEP)
elu = nn.ELU(alpha=1.0)
print("ELU:", [round(v,3) for v in elu(x).tolist()])
print("使用场景：替代 ReLU，负区间连续可微，适合深层网络")

# ==============================================================
# 4. Sigmoid  1 / (1 + e^(-x))  输出 (0,1)
# 缺点：两端梯度消失，输出非零均值（可用 tanh 替代）
# ==============================================================
print()
print(SEP)
print("4. Sigmoid  f(x) = 1/(1+e^-x)")
print(SEP)
sigmoid = nn.Sigmoid()
print("Sigmoid:", [round(v,3) for v in sigmoid(x).tolist()])
print("使用场景：二分类输出层、门控机制（LSTM门）")

# ==============================================================
# 5. Tanh  (e^x - e^(-x)) / (e^x + e^(-x))  输出 (-1,1)
# 零均值输出，比 Sigmoid 更好；但仍有梯度消失问题
# ==============================================================
print()
print(SEP)
print("5. Tanh")
print(SEP)
tanh = nn.Tanh()
print("Tanh:", [round(v,3) for v in tanh(x).tolist()])
print("使用场景：RNN 内部激活，输出范围(-1,1)，比Sigmoid收敛快")

# ==============================================================
# 6. GELU  x * Phi(x)  其中 Phi 是标准正态 CDF
# 平滑版 ReLU，BERT/GPT 等 Transformer 模型的默认选择
# ==============================================================
print()
print(SEP)
print("6. GELU")
print(SEP)
gelu = nn.GELU()
print("GELU:", [round(v,3) for v in gelu(x).tolist()])
# 手动验证 GELU 近似公式
gelu_approx = 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
print("GELU 近似公式:", [round(v,3) for v in gelu_approx.tolist()])
print("使用场景：BERT、GPT、ViT等Transformer架构")

# ==============================================================
# 7. SiLU (Swish)  x * Sigmoid(x)
# 平滑、非单调，EfficientNet 中使用
# ==============================================================
print()
print(SEP)
print("7. SiLU (Swish)  f(x) = x * sigmoid(x)")
print(SEP)
silu = nn.SiLU()
print("SiLU:", [round(v,3) for v in silu(x).tolist()])
# 手动验证
manual_silu = x * torch.sigmoid(x)
print("手动 Swish:", [round(v,3) for v in manual_silu.tolist()])
print("一致:", torch.allclose(silu(x), manual_silu))
print("使用场景：EfficientNet、MobileNetV3、部分LLM（如LLaMA使用SwiGLU）")

# ==============================================================
# 8. Softmax  将 logits 转为概率分布（总和=1）
# 注意：只用于输出层，中间层不要用（梯度消失严重）
# ==============================================================
print()
print(SEP)
print("8. Softmax")
print(SEP)
softmax = nn.Softmax(dim=-1)
logits = torch.tensor([2.0, 1.0, 0.5])
probs = softmax(logits)
print("logits:", logits.tolist())
print("Softmax:", [round(v,4) for v in probs.tolist()])
print("概率总和:", probs.sum().item())
print("使用场景：多分类输出层（与 CrossEntropyLoss 配合时不需要手动调用）")

# ==============================================================
# 激活函数对比汇总
# ==============================================================
print()
print(SEP)
print("激活函数对比汇总")
print(SEP)
print(f"{"名称":12} {"值域":15} {"梯度消失":10} {"主要使用场景"}")
print("-" * 70)
summary = [
    ("ReLU",     "[0, +inf)",   "负区间有",  "CNN标准选择"),
    ("LeakyReLU","(-inf,+inf)", "基本无",    "GAN"),
    ("ELU",      "(-1,+inf)",   "较少",      "深层网络"),
    ("Sigmoid",  "(0, 1)",      "两端严重",  "二分类输出/LSTM门"),
    ("Tanh",     "(-1, 1)",     "两端有",    "RNN内部"),
    ("GELU",     "(-0.17,+inf)","极少",     "Transformer"),
    ("SiLU",     "(-0.28,+inf)","极少",     "EfficientNet"),
    ("Softmax",  "(0,1) 和=1",  "较严重",   "多分类输出层"),
]
for name, rng, grad, usage in summary:
    print(f"{name:12} {rng:15} {grad:10} {usage}")
print()
print("12_激活函数.py 运行完毕!")
