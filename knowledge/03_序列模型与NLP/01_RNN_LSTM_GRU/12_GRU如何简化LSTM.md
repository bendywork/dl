# GRU 如何简化 LSTM

## 📌 核心问题
> GRU 是 LSTM 的简化版，如何用更少的门实现同样的加法传递效果？

## 🌱 根源与动机

LSTM 用 3 个门 + 2 条状态线（C_t 和 h_t）解决了长时依赖，但结构较重。GRU 的思路：**保留加法传递的核心，精简门控结构**。

## 📐 GRU 结构

### 两个门

**重置门 r_t**（计算候选时 h_{t-1} 参与多少）：
```
r_t = σ(W_r · [h_{t-1}, x_t])
```

**更新门 z_t**（最终合并时新旧比例 = LSTM 遗忘门+输入门合体）：
```
z_t = σ(W_z · [h_{t-1}, x_t])
```

### 候选内容

```
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])
```

r_t=0 时 h_{t-1} 完全不参与，相当于"重来"；r_t=1 时完全参与。

### 加法传递（核心）

```
h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h̃_t
      ↑ 保留旧信息     ↑ 写入新信息
```

## 💡 关键理解

### 为什么 z_t 能替代遗忘门+输入门

LSTM 的 f_t 和 i_t 独立，没有约束。GRU 的 z_t 天然互补：
- z 和 (1-z) 加起来恒等于 1
- z=1 → 全保留旧信息，z=0 → 全写入新内容
- 不需要额外约束就自动平衡新旧比例

### 重置门 vs 更新门的区别

- **r_t（重置门）**：管"怎么算"——计算 h̃_t 时 h_{t-1} 参与多少
- **z_t（更新门）**：管"怎么合"——最终 h_t 中新旧信息的比例

## 🔧 LSTM vs GRU 对比代码

```python
import torch
import torch.nn as nn

e, h = 128, 256
x = torch.randn(1, 5, e)

lstm = nn.LSTM(input_size=e, hidden_size=h, batch_first=True)
gru = nn.GRU(input_size=e, hidden_size=h, batch_first=True)

print(f"LSTM 参数量: {sum(p.numel() for p in lstm.parameters()):,}")
print(f"GRU  参数量: {sum(p.numel() for p in gru.parameters()):,}")
# GRU ≈ LSTM 的 75%
```

| | LSTM | GRU |
|---|---|---|
| 门数量 | 3（遗忘f、输入i、输出o） | 2（重置r、更新z） |
| 状态线 | 2条（C_t + h_t） | 1条（h_t） |
| 参数量 | 4组 W | 3组 W（≈75%） |
| 核心机制 | C_t = f⊙C_{t-1} + i⊙C̃ | h_t = z⊙h_{t-1} + (1-z)⊙h̃ |

## ⚠️ 易错点与常见误解

1. **GRU 效果一定比 LSTM 差？** — 不一定，很多任务表现相当，数据量小时 GRU 可能更好（过拟合风险低）。

2. **重置门 r_t 和更新门 z_t 搞混？** — r 管的是"怎么算"（候选内容计算时），z 管的是"怎么合"（最终合并时）。

## 🔗 知识延伸

- [[24_LSTM如何解决长时依赖]] — LSTM 加法传递原理
- [[01_RNN原理与梯度消失]] — RNN 梯度消失的数学推导

## 📚 参考资料

- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", 2014
