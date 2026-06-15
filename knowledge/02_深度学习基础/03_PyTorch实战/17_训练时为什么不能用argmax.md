# 训练时为什么不能用 argmax

## 📌 核心问题
> forward 训练时为什么必须返回 raw logits，而不是 argmax 结果？

## 🌱 根源：argmax 不可导

训练的本质是反向传播，反向传播通过链式法则从 loss 往回求梯度：

```
loss → softmax → raw score(logits) → 每一层的 W
```

`CrossEntropyLoss` 接收 raw logits，内部执行 `softmax + log + NLLLoss`，整个过程**连续可导**，梯度能沿链路传回每一层。

`argmax` 是"取最大值的下标"，这个操作**没有导数**，梯度在这里断掉，参数无法更新，网络完全学不了东西。

## 💡 正确写法

```python
def forward(self, x):
    score = self.features(x)       # 只算一次前向
    if self.training:
        return score               # 训练：返回 raw logits → CrossEntropyLoss
    return score.argmax(dim=1)     # 推理：返回预测类别（不需要反向传播）
```

## ⚠️ 易错点

1. **训练时返回 argmax** → 梯度断裂，loss 能算出来但参数不会更新，网络不会学习
2. **训练时做了两次前向** → 如下错误写法：
```python
# ❌ 错误：training 分支没有 return，score 被丢弃，又重新算了一遍
def forward(self, x):
    if self.training:
        x = self.features(x)              # 没有 return，结果丢掉了
    return self.features(x).argmax(dim=1) # 训练时也走到这里
```
3. **CrossEntropyLoss 接收 argmax 结果** → argmax 输出是类别 id（整数），不是 logits，传入 loss 会报维度错误

## 🔗 知识延伸
- [[forward与training职责分离]] — forward 职责边界
- [[CrossEntropyLoss内部含softmax]] — 为什么不能在 forward 里再加 softmax
