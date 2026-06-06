# ==============================================================
# 文件：14_优化器与调度器.py
# 主题：PyTorch 优化器与学习率调度器
# 优化器：SGD / Adam / AdamW / RMSprop
# 调度器：StepLR / CosineAnnealingLR / ReduceLROnPlateau / OneCycleLR
# ==============================================================
import torch
import torch.nn as nn

SEP = "=" * 60

# 共用模型
def make_model():
    return nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# ==============================================================
# 1. SGD  随机梯度下降
# ==============================================================
print(SEP)
print("1. SGD")
print(SEP)
model = make_model()
# momentum：动量项，加速收敛方向、抑制震荡
# weight_decay：L2正则化（等价于 AdamW 中的解耦正则）
# nesterov：Nesterov 加速梯度，通常比普通 momentum 更好
sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
print("SGD param groups lr:", sgd.param_groups[0]["lr"])
print("使用场景：CV任务（ResNet等）配合 momentum=0.9 表现优秀")

# ==============================================================
# 2. Adam  自适应矩估计
# 为每个参数维护一阶（均值）和二阶（方差）动量
# ==============================================================
print()
print(SEP)
print("2. Adam")
print(SEP)
model2 = make_model()
adam = torch.optim.Adam(
    model2.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),   # (一阶动量衰减, 二阶动量衰减)
    eps=1e-8,             # 数值稳定项
    weight_decay=0
)
print("Adam: lr={}, betas={}".format(
    adam.param_groups[0]["lr"],
    adam.param_groups[0]["betas"]))
print("使用场景：NLP/LLM/默认选择，收敛快但可能过拟合")

# ==============================================================
# 3. AdamW  Adam + 解耦权重衰减
# Adam 的 weight_decay 有缺陷（作用于自适应梯度），AdamW 修复了这个问题
# ==============================================================
print()
print(SEP)
print("3. AdamW (推荐替代 Adam)")
print(SEP)
model3 = make_model()
adamw = torch.optim.AdamW(
    model3.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01   # 解耦正则，直接作用于参数
)
print("AdamW weight_decay:", adamw.param_groups[0]["weight_decay"])
print("使用场景：Transformer/BERT/GPT微调，几乎所有现代LLM训练")

# ==============================================================
# 4. RMSprop
# ==============================================================
print()
print(SEP)
print("4. RMSprop")
print(SEP)
model4 = make_model()
rmsprop = torch.optim.RMSprop(
    model4.parameters(),
    lr=1e-3,
    alpha=0.99,    # 梯度平方的滑动平均系数
    eps=1e-8
)
print("RMSprop: lr={}".format(rmsprop.param_groups[0]["lr"]))
print("使用场景：RNN训练、强化学习（DQN标配）")

# ==============================================================
# 学习率调度器
# ==============================================================

# ==============================================================
# 5. StepLR  每隔 step_size 个 epoch，lr *= gamma
# ==============================================================
print()
print(SEP)
print("5. StepLR")
print(SEP)
model5 = make_model()
opt5 = torch.optim.SGD(model5.parameters(), lr=0.1)
scheduler_step = torch.optim.lr_scheduler.StepLR(opt5, step_size=3, gamma=0.1)
lrs = []
for epoch in range(10):
    lrs.append(opt5.param_groups[0]["lr"])
    scheduler_step.step()
print("StepLR(step=3, gamma=0.1) 各epoch lr:", [round(lr, 5) for lr in lrs])

# ==============================================================
# 6. CosineAnnealingLR  余弦退火，lr 从初始值平滑降到 eta_min
# ==============================================================
print()
print(SEP)
print("6. CosineAnnealingLR")
print(SEP)
model6 = make_model()
opt6 = torch.optim.AdamW(model6.parameters(), lr=1e-3)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt6, T_max=10, eta_min=1e-6)
lrs = []
for epoch in range(10):
    lrs.append(round(opt6.param_groups[0]["lr"], 7))
    scheduler_cos.step()
print("CosineAnnealingLR 各epoch lr:", lrs)
print("使用场景：Transformer训练，lr从高到低平滑衰减")

# ==============================================================
# 7. ReduceLROnPlateau  监控指标无改善时降低 lr
# ==============================================================
print()
print(SEP)
print("7. ReduceLROnPlateau")
print(SEP)
model7 = make_model()
opt7 = torch.optim.Adam(model7.parameters(), lr=1e-2)
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt7,
    mode="min",       # 监控 loss，最小化
    factor=0.5,       # lr *= 0.5
    patience=2,       # 容忍 2 个 epoch 不改善
    verbose=False
)
# 模拟损失曲线
fake_losses = [1.0, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5]   # epoch 2-4 停滞
for i, loss_val in enumerate(fake_losses):
    scheduler_plateau.step(loss_val)
    lr_now = opt7.param_groups[0]['lr']
    print(f'  epoch {i}: loss={loss_val}, lr={lr_now:.6f}')
print("使用场景：不确定训练总epoch数，根据val_loss自动降lr")

# ==============================================================
# 8. OneCycleLR  一次循环策略（warmup + cosine decay）
# 使用场景：超快速收敛（fastai默认），小数据集上经验很好
# ==============================================================
print()
print(SEP)
print("8. OneCycleLR")
print(SEP)
model8 = make_model()
opt8 = torch.optim.Adam(model8.parameters(), lr=1e-3)
total_steps = 50
scheduler_1cycle = torch.optim.lr_scheduler.OneCycleLR(
    opt8,
    max_lr=1e-2,
    total_steps=total_steps,
    pct_start=0.3,    # 前30% warmup 到 max_lr
    anneal_strategy="cos"
)
lrs = []
for step in range(total_steps):
    lrs.append(opt8.param_groups[0]["lr"])
    scheduler_1cycle.step()
print("OneCycleLR: 前5步 lr:", [round(lr, 5) for lr in lrs[:5]])
print("第15步（接近peak）lr:", round(lrs[14], 5))
print("最后5步 lr:", [round(lr, 7) for lr in lrs[-5:]])

# ==============================================================
# 9. 完整训练循环示例（AdamW + CosineAnnealingLR）
# ==============================================================
print()
print(SEP)
print("9. 完整训练循环示例")
print(SEP)
torch.manual_seed(42)
X = torch.randn(100, 16)
y = torch.randn(100, 1)

train_model = make_model()
optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.MSELoss()

for epoch in range(20):
    train_model.train()
    optimizer.zero_grad()          # 1. 清空梯度
    pred = train_model(X)          # 2. 前向传播
    loss = criterion(pred, y)      # 3. 计算损失
    loss.backward()                # 4. 反向传播
    # 梯度裁剪（防止梯度爆炸）
    nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
    optimizer.step()               # 5. 更新参数
    scheduler.step()               # 6. 更新学习率（epoch结束后）
    if epoch % 5 == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:2d}: loss={loss.item():.4f}, lr={lr_now:.6f}")

print()
print("14_优化器与调度器.py 运行完毕!")
