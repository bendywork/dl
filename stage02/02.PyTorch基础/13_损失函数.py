# ==============================================================
# 文件：13_损失函数.py
# 主题：PyTorch 常用损失函数及使用场景
# API：CrossEntropyLoss / MSELoss / BCELoss / BCEWithLogitsLoss
#       NLLLoss / L1Loss / HuberLoss
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

SEP = "=" * 60

# ==============================================================
# 1. CrossEntropyLoss
# 公式：-sum(y * log(softmax(logit)))
# 内部自动做 softmax + log + nll，输入不需要手动 softmax
# 使用场景：多分类（互斥类别），如图像分类、语言模型
# ==============================================================
print(SEP)
print("1. CrossEntropyLoss")
print(SEP)
criterion_ce = nn.CrossEntropyLoss()
# logits: (batch, num_classes)，targets: (batch,) 整数类别
logits = torch.tensor([[2.0, 1.0, 0.5],
                       [0.5, 2.5, 0.1],
                       [0.1, 0.3, 3.0]])
targets = torch.tensor([0, 1, 2])
loss = criterion_ce(logits, targets)
print("logits shape:", logits.shape)
print("targets:", targets.tolist())
print("CrossEntropyLoss:", round(loss.item(), 4))

# 等价于 NLLLoss(log_softmax(logits))
manual = F.nll_loss(F.log_softmax(logits, dim=1), targets)
print("手动等价结果:", round(manual.item(), 4))
print("使用场景：多分类任务的标准损失")

# ==============================================================
# 2. NLLLoss  Negative Log Likelihood Loss
# 输入必须是 log_softmax 后的概率，targets 是整数类别
# 使用场景：手动控制 softmax 的场合；TextCNN / LSTM 分类
# ==============================================================
print()
print(SEP)
print("2. NLLLoss")
print(SEP)
nll = nn.NLLLoss()
log_probs = F.log_softmax(logits, dim=1)
loss_nll = nll(log_probs, targets)
print("NLLLoss（需先 log_softmax）:", round(loss_nll.item(), 4))
print("等于 CrossEntropyLoss:", torch.isclose(loss, loss_nll).item())
print("使用场景：与 log_softmax 组合，等价于 CrossEntropyLoss")

# ==============================================================
# 3. MSELoss  Mean Squared Error
# 公式：mean((pred - target)^2)
# 使用场景：回归任务，要求输出连续值
# ==============================================================
print()
print(SEP)
print("3. MSELoss")
print(SEP)
mse = nn.MSELoss()
pred = torch.tensor([1.5, 2.3, 0.8, 3.1])
target = torch.tensor([1.0, 2.0, 1.0, 3.0])
loss_mse = mse(pred, target)
print("pred:", pred.tolist())
print("target:", target.tolist())
print("MSELoss:", round(loss_mse.item(), 4))
manual_mse = ((pred - target) ** 2).mean()
print("手动计算:", round(manual_mse.item(), 4))
print("使用场景：房价预测、图像超分辨率、回归")

# ==============================================================
# 4. BCELoss  Binary Cross Entropy
# 输入必须是 sigmoid 后的概率 [0,1]
# 使用场景：二分类、多标签分类（各标签独立）
# ==============================================================
print()
print(SEP)
print("4. BCELoss")
print(SEP)
bce = nn.BCELoss()
probs = torch.sigmoid(torch.tensor([2.0, -1.0, 0.5, -2.0]))
labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
loss_bce = bce(probs, labels)
print("sigmoid 概率:", [round(v,3) for v in probs.tolist()])
print("标签:", labels.tolist())
print("BCELoss:", round(loss_bce.item(), 4))
print("使用场景：需要先 sigmoid，多标签分类")

# ==============================================================
# 5. BCEWithLogitsLoss  = sigmoid + BCELoss（数值更稳定）
# 直接输入 logits，内部用 log-sum-exp 避免数值溢出
# 使用场景：二分类/多标签，推荐替代 BCELoss
# ==============================================================
print()
print(SEP)
print("5. BCEWithLogitsLoss（推荐替代 BCELoss）")
print(SEP)
bce_logits = nn.BCEWithLogitsLoss()
raw_logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
loss_bcel = bce_logits(raw_logits, labels)
print("BCEWithLogitsLoss（直接输入logits）:", round(loss_bcel.item(), 4))
print("与 BCELoss 结果一致:", torch.isclose(loss_bce, loss_bcel).item())
print("使用场景：直接输入 logits，推荐始终使用此版本")

# ==============================================================
# 6. L1Loss  Mean Absolute Error
# 公式：mean(|pred - target|)
# 特点：对异常值更鲁棒；梯度不连续（x=0处不可微）
# 使用场景：图像重建、目标检测边界框回归
# ==============================================================
print()
print(SEP)
print("6. L1Loss (MAE)")
print(SEP)
l1 = nn.L1Loss()
loss_l1 = l1(pred, target)
print("L1Loss:", round(loss_l1.item(), 4))
print("MSE vs L1 对比（有离群点时）:")
p2 = torch.tensor([1.0, 1.0, 1.0, 10.0])  # 10.0 是异常值
t2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
print("  MSELoss:", round(mse(p2, t2).item(), 2), "（离群点放大误差）")
print("  L1Loss:", round(l1(p2, t2).item(), 2), "（离群点影响较小）")

# ==============================================================
# 7. HuberLoss  MSE 和 MAE 的折中
# 误差小时用 MSE（平滑），误差大时用 MAE（鲁棒）
# 使用场景：强化学习（DQN），回归任务中有异常值时
# ==============================================================
print()
print(SEP)
print("7. HuberLoss (SmoothL1)")
print(SEP)
huber = nn.HuberLoss(delta=1.0)
loss_huber = huber(pred, target)
print("HuberLoss (delta=1.0):", round(loss_huber.item(), 4))
print("有离群点时:")
print("  MSE:", round(mse(p2, t2).item(), 2))
print("  L1:", round(l1(p2, t2).item(), 2))
print("  Huber:", round(huber(p2, t2).item(), 2), "（折中）")
print("使用场景：DQN等强化学习，坐标回归（目标检测）")

# ==============================================================
# 损失函数选择指南汇总
# ==============================================================
print()
print(SEP)
print("损失函数选择指南")
print(SEP)
print(f"{"任务类型":20} {"推荐损失函数":30} {"注意事项"}")
print("-" * 70)
guide = [
    ("多分类",            "CrossEntropyLoss",       "logits直接输入，无需softmax"),
    ("二分类",            "BCEWithLogitsLoss",      "logits直接输入，数值稳定"),
    ("多标签分类",        "BCEWithLogitsLoss",      "每个标签独立sigmoid"),
    ("回归（无异常值）",  "MSELoss",                "梯度连续，收敛快"),
    ("回归（有异常值）",  "HuberLoss / L1Loss",    "对离群点鲁棒"),
    ("图像重建",          "MSELoss / L1Loss",      "SSIM也常用"),
    ("语言模型",          "CrossEntropyLoss",       "targets=下一个token的id"),
    ("强化学习DQN",       "HuberLoss",             "SmoothL1Loss旧名称"),
]
for task, loss, note in guide:
    print(f"{task:20} {loss:30} {note}")
print()
print("13_损失函数.py 运行完毕!")
