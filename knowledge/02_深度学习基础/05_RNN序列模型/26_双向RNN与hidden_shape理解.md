# 双向RNN与hidden shape理解

## 📌 核心问题
> 双向RNN的 hidden 输出第一维为什么是2，和样本数有什么关系？

## 🌱 根源与动机
双向RNN让每个token同时感知左侧上文和右侧下文，需要正向和反向各跑一遍序列，因此会产生两个最终隐状态。

## 📐 理论推导

hidden 第一维计算公式：
```
num_directions = 2 if bidirectional else 1
hidden 第一维 = num_layers × num_directions
```

示例：num_layers=1，bidirectional=True → 1 × 2 = 2

各参数组合对应的 hidden shape：

| num_layers | bidirectional | hidden 第一维 |
|-----------|--------------|-------------|
| 1         | False        | 1           |
| 1         | True         | 2           |
| 3         | False        | 3           |
| 3         | True         | 6           |

## 💡 关键理解

hidden shape 为 `[num_layers × num_directions, bs, hidden_size]`：
- 第一维：层数 × 方向数，与样本数无关
- 第二维：bs（样本数），永远不变
- 第三维：每个方向的隐状态维度

双向处理过程（同一批样本，被两个方向各处理一次）：
```
输入序列：[x0, x1, x2, x3, x4]

正向：x0 → x1 → x2 → x3 → x4 → 得到最终隐状态 h_forward
反向：x4 → x3 → x2 → x1 → x0 → 得到最终隐状态 h_backward

hidden = [h_forward, h_backward]  # shape: [2, bs, hidden_size]
```

- `hidden[0]`：正向跑完整个序列后的隐状态
- `hidden[1]`：反向跑完整个序列后的隐状态

output 的变化：
- 单向：output shape = `[bs, t, hidden_size]`
- 双向：output shape = `[bs, t, hidden_size × 2]`，每个时间步的正向+反向隐状态在特征维拼接

## 🔧 代码实现

```python
import torch
from torch import nn

bs, t, e = 1, 5, 128

# 双向RNN
birnn = nn.RNN(
    input_size=e,
    hidden_size=256,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)

token_embs = torch.randn(bs, t, e)
output, hidden = birnn(token_embs)

print(output.shape)   # [1, 5, 512]  → hidden_size × 2
print(hidden.shape)   # [2, 1, 256]  → num_layers × num_directions, bs, hidden_size
```

## ⚠️ 易错点与常见误解
1. hidden 第一维是 2，不是样本数变成了 2，样本数永远在第二维（batch_first=True 时在第一维）
2. 双向RNN参数量是单向的 2 倍，正向和反向用的是完全独立的两套权重，不共享
3. 反向RNN append 的顺序是从 t-1 到 0，拼接前需要翻转对齐时间步

## 🔗 知识延伸
- 双向RNN手动实现见：`stage03/02.RNN循环神经网络/02_RNN理解/03_双向RNN练习.py`
- API使用见：`stage03/02.RNN循环神经网络/02_RNN理解/04_rnn_api练习.py`
- 权重共享的模型：孪生网络（Siamese Network）、语言模型的Embedding与输出层共享

## 📚 参考资料
- PyTorch 官方文档：nn.RNN
