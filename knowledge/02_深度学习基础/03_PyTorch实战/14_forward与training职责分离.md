# forward 与 training 职责分离

## 📌 核心问题
> `forward` 里要不要写训练逻辑？训练循环应该放在哪里？

## 🌱 根源与动机

PyTorch 的设计哲学是**结构与优化解耦**：

- 网络"长什么样、数据怎么流"→ `forward` 负责
- 网络"怎么学、参数怎么更新"→ 训练循环负责

两者职责不同，混在一起会导致网络无法复用（换个 loss 函数或优化器就得改网络结构）。

## 📐 职责划分

| 职责 | 位置 | 包含内容 |
|------|------|---------|
| 网络结构描述 | `__init__` | 定义层：Linear、ReLU、Sequential 等 |
| 数据流向描述 | `forward` | wx+b → 激活 → 下一层，最终输出 score |
| 训练驱动流程 | 类外的 `training()` 函数 | epoch 循环、batch 切分、loss、backward、step |

## 💡 关键理解

**`forward` 里的 `if self.training` 不是训练过程**，只是一个状态标志位，用来切换输出格式：

```python
if self.training:
    return score          # 训练模式：返回 raw logits，给 CrossEntropyLoss 用
return score.argmax(dim=1)  # 推理模式：直接返回预测类别 id
```

`self.training` 由外部的 `net.train()` / `net.eval()` 设置，`forward` 本身感知不到"梯度、loss、参数更新"任何一件事。

## 🔧 标准执行顺序

```
1. data_load()               准备数据
         ↓
2. net = MyNetwork(...)      创建网络实例
         ↓
3. loss_fn / optimizer       创建损失函数和优化器
         ↓
4. for epoch in range(N):
       net.train()
       for batch:
           score = net(x)    ← PyTorch 自动路由到 forward
           loss = loss_fn(score, y)
           opt.zero_grad()
           loss.backward()
           opt.step()

       net.eval()
       with torch.no_grad():
           for batch:
               net(x)        ← 再次调用 forward，self.training=False
```

## ⚠️ 易错点

1. **`net(x)` 不是直接调用 `forward`**，是调用 `nn.Module.__call__`，它内部才路由到 `forward`，同时触发注册的 hook。永远写 `net(x)`，不要写 `net.forward(x)`。

2. **`data_load` 不应该写进 `nn.Module` 类里**。网络不应该知道数据怎么来，数据加载是训练流程的职责，放在外部函数中。

3. **`forward` 里不要写 `loss.backward()`**。backward 是训练循环的事，放进 forward 会导致推理时也触发梯度计算。

## 🔗 知识延伸

- [[PyTorch梯度计算开关详解]] — `torch.no_grad()` 为什么要包裹前向过程
- [[PyTorch_state_dict与named_parameters对比]] — 训练结束后如何保存参数
