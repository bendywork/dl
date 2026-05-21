# 11_PyTorch梯度计算开关详解.md

## 📌 核心问题
> 什么情况下需要手动关闭梯度计算（torch.no_grad / requires_grad=False）？什么时候又必须保留？

## 🌱 根源与动机

梯度 = 参数更新的方向。只有需要更新参数时才需要梯度。不需要更新时开着梯度，白白占显存、拖慢速度。

---

## 一、必须关闭梯度的场景

### 1. 模型评估/推理（最常见）

```python
model.eval()
with torch.no_grad():
    output = model(x_test)
    loss = criterion(output, y_test)
```

原因：评估时只看结果，不更新参数。开着梯度白白占内存、拖慢速度。

### 2. 预测/部署

```python
with torch.no_grad():
    prediction = model(input_data)
```

原因：线上推理，不需要训练，梯度完全没用。

### 3. 特征提取（冻结预训练模型）

```python
# 冻结整个backbone，只训练新加的层
for param in pretrained_model.parameters():
    param.requires_grad = False

# 只训练分类头
classifier = nn.Linear(512, 10)  # 这个默认requires_grad=True
```

原因：预训练参数不需要更新，关掉梯度节省显存，计算更快。

### 4. 生成对抗样本/对抗训练中的特殊步骤

```python
with torch.no_grad():
    perturbation = generate_perturbation(x)
```

---

## 二、必须保留梯度的场景

### 1. 训练

```python
model.train()
output = model(x)          # 前向，保留梯度
loss = criterion(output, y)
loss.backward()             # 反向，计算梯度
optimizer.step()            # 更新参数
```

### 2. 对输入求梯度（特殊需求：可视化/对抗攻击）

```python
x = input_data.clone().requires_grad_(True)  # 对输入求梯度，不是对参数
output = model(x)
loss = output.sum()
loss.backward()
grad = x.grad  # 拿到损失对输入的梯度
```

### 3. 梯度惩罚（WGAN-GP等）

```python
alpha = torch.rand(...)
interpolated = alpha * real + (1 - alpha) * fake
interpolated.requires_grad_(True)
d_interpolated = discriminator(interpolated)
gradients = torch.autograd.grad(d_interpolated, interpolated)
```

---

## 三、关掉梯度省了什么

| 指标 | 梯度开 | 梯度关（no_grad） |
|------|--------|-----------------|
| 显存 | 需要存每层的中间值（反向要用） | 不存，省约50%显存 |
| 速度 | 每步要记录计算图 | 跳过计算图构建，快约30% |
| 安全 | 梯度可能意外泄露 | 不会误更新参数 |

---

## 四、两种写法的区别

```python
# torch.no_grad() — 临时关闭，with块结束自动恢复
with torch.no_grad():
    output = model(x)  # 这里面不计算梯度
output2 = model(x)     # 这里又恢复了梯度计算

# requires_grad=False — 永久关闭，直到手动改回来
for param in model.parameters():
    param.requires_grad = False  # 这个参数永远不求梯度了
```

**怎么选：**
- 推理/评估 → `with torch.no_grad()`，临时性的
- 冻结某些层 → `requires_grad=False`，长期性的

---

## 💡 关键理解

**不需要更新参数的时候，就关掉梯度。需要更新参数（或需要对输入求梯度做分析），就保留。**

---

## ⚠️ 易错点与常见误解

1. **"model.eval()等于no_grad"** — 不是。eval()只改变BN和Dropout的行为，不关梯度。两者要配合使用。

2. **"no_grad里不能算loss"** — 可以算，loss值正常返回，只是不构建计算图，不能backward()。

3. **"冻结层后optimizer还要包含这些参数"** — 不需要，过滤掉requires_grad=False的参数：
```python
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

4. **"推理时忘记no_grad"** — 结果不会错，但显存翻倍、速度变慢，大模型可能OOM。

## 🔗 知识延伸

- torch.inference_mode()：PyTorch 2.0新增，比no_grad更快，完全禁用autograd
- torch.compile()：PyTorch 2.0编译优化，自动推断哪些需要梯度
- 梯度检查点（gradient checkpointing）：用时间换显存，训练时也不存所有中间值
