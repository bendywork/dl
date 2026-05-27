# RNN 基础知识点整理

## 📌 核心问题
> 本文整理 RNN 手动实现过程中涉及的关键知识点，包括：Conv1d 维度顺序、RNN 两个 Linear 的作用与设计逻辑、PyTorch API 等价写法，以及循环实现中的易错陷阱。

---

## 🌱 根源与动机

RNN 是处理序列数据的基础架构，能够在时间步之间传递隐状态，从而捕获上下文依赖关系。在手动实现 RNN 时，需要深刻理解以下几个层面的细节：

1. NLP 与 PyTorch 卷积 API 的维度约定差异
2. RNN 公式中两个线性变换矩阵的对应关系
3. PyTorch 常用操作的等价写法，避免混淆
4. 循环体中变量更新顺序的正确性

---

## 📐 理论推导

### 知识点1：Conv1d 需要 permute 的原因

Conv1d API 设计要求输入格式为 `[bs, channels, length]`，而 NLP 中 token embedding 的习惯格式是 `[bs, length, channels]`，两者维度顺序相反。

使用 Conv1d 处理形状为 `[bs, t, e]` 的 token embedding 时，完整流程为：

1. `permute(0, 2, 1)`：`[bs, t, e]` → `[bs, e, t]`，满足 Conv1d 对 `channels` 在前的要求
2. Conv1d 沿 `t`（序列长度）方向滑动，提取局部上下文特征
3. `permute(0, 2, 1)`：`[bs, e, t]` → `[bs, t, e]`，恢复 NLP 习惯格式，方便后续 mean pooling

---

### 知识点2：RNN 两个 Linear 的作用

RNN 核心公式：

```
H_t = act(U · X_t + W · H_{t-1})
```

对应到代码实现：

- `input_linear = nn.Linear(e, e*2)`：对应 **U 矩阵**，负责对当前输入 X_t 做线性变换，维度 e → 2e
- `hidden_linear = nn.Linear(2*e, e*2)`：对应 **W 矩阵**，负责对上一时刻隐状态 H_{t-1} 做线性变换，维度 2e → 2e

> 注意：`hidden_linear` 的输入维度是 `2e` 而非 `e`，因为隐状态的维度等于输出维度（即 `2e`），与输入嵌入维度 `e` 不同。

---

### 知识点3：torch.zeros 两种写法等价

```python
torch.zeros(bs, 2*e)       # 展开写法：多个整数参数
torch.zeros((bs, 2*e))     # tuple 写法：传入一个元组
```

两者完全等价。`torch.zeros` 的 `*size` 参数既可以接收展开的多个整数，也可以接收一个 tuple，PyTorch 内部统一处理。

---

### 知识点4：torch.cat 与 torch.concat 等价

`torch.cat` 和 `torch.concat` 是同一函数的两个名字，`torch.concat` 是 `torch.cat` 的别名，行为完全相同。

---

### 知识点5：range 倒序遍历

`range(t-1, -1, -1)` 参数解析：

| 参数 | 值 | 含义 |
|------|-----|------|
| start | `t-1` | 从最后一个索引开始 |
| stop | `-1` | 到 -1 停止（不含），即最小值为 0 |
| step | `-1` | 每次递减 1 |

示例（t=5）：`range(4, -1, -1)` → `4, 3, 2, 1, 0`

用途：双向 RNN 的反向链，从最后一个 token 倒序遍历到第一个 token，使反向隐状态能够捕获"未来"上下文信息。

---

## 💡 关键理解

- Conv1d 与 NLP 维度约定的差异是工程实践中的常见陷阱，`permute` 是桥梁，不是"多余操作"
- RNN 的两个 Linear 分别对应公式中的 U 和 W，`hidden_linear` 输入维度由隐状态维度决定，而非输入嵌入维度
- `torch.zeros(a, b)` 和 `torch.zeros((a, b))` 在语义上完全等价，选一种风格保持一致即可
- `range` 倒序是实现双向 RNN 反向链的标准写法，务必理解三个参数的含义

---

## 🔧 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# 知识点1：Conv1d 的 permute 用法
# ==============================
bs, t, e = 2, 10, 64
x = torch.randn(bs, t, e)  # NLP 习惯：[bs, t, e]

conv = nn.Conv1d(in_channels=e, out_channels=e, kernel_size=3, padding=1)

x_permuted = x.permute(0, 2, 1)        # [bs, t, e] → [bs, e, t]
x_conv = conv(x_permuted)              # Conv1d 沿 t 方向滑动
x_out = x_conv.permute(0, 2, 1)        # [bs, e, t] → [bs, t, e]
x_pooled = x_out.mean(dim=1)           # mean pooling → [bs, e]
print("Conv1d output:", x_pooled.shape)  # [2, 64]


# ==============================
# 知识点2：RNN 两个 Linear
# ==============================
input_linear = nn.Linear(e, e * 2)      # U 矩阵：e → 2e
hidden_linear = nn.Linear(e * 2, e * 2) # W 矩阵：2e → 2e


# ==============================
# 知识点5：range 倒序遍历（双向 RNN 反向链示意）
# ==============================
t_len = 5
forward_indices = list(range(t_len))           # 0, 1, 2, 3, 4
backward_indices = list(range(t_len - 1, -1, -1))  # 4, 3, 2, 1, 0
print("正向:", forward_indices)
print("反向:", backward_indices)


# ==============================
# 知识点6：RNN 手动实现（正确写法）
# ==============================
class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        x: [bs, t, input_size]
        """
        bs, t, _ = x.shape
        h_prev = torch.zeros(bs, self.hidden_linear.out_features)

        outputs = []
        for i in range(t):
            current_input = x[:, i, :]                              # [bs, input_size]
            input_transform = self.input_linear(current_input)      # U · X_t
            hidden_transform = self.hidden_linear(h_prev)           # W · H_{t-1}
            h_cur = F.relu(input_transform + hidden_transform)      # 激活
            h_prev = h_cur                                           # 更新 h_prev（最后才更新）
            outputs.append(h_cur)

        # [t, bs, hidden_size] → [bs, t, hidden_size]
        return torch.stack(outputs, dim=1)


# 验证
rnn = ManualRNN(input_size=e, hidden_size=e * 2)
x_input = torch.randn(bs, t, e)
rnn_output = rnn(x_input)
print("RNN output:", rnn_output.shape)  # [2, 10, 128]
```

---

## ⚠️ 易错点与常见误解

### 1. Conv1d permute 顺序写错

错误：忘记 permute，直接将 `[bs, t, e]` 传入 Conv1d  
正确：必须先 `permute(0, 2, 1)` 转为 `[bs, e, t]`，卷积结束后再 permute 回来

---

### 2. hidden_linear 输入维度与输出维度混淆

错误：`nn.Linear(e, e*2)`（隐状态的输入维度写成 `e`）  
正确：隐状态维度 = 输出维度 = `2e`，所以应为 `nn.Linear(2*e, 2*e)`

---

### 3. 循环内 h_prev 被提前覆盖

```python
# ❌ 错误写法：h_prev（即 ht_1）在中途被覆盖
ht = matmul(xi, uw.T) + ub
ht_1 = matmul(ht_1, ww.T) + wb   # h_prev 被覆盖！下一轮拿不到正确历史状态
h = relu(ht + ht_1)

# ✅ 正确写法：用临时变量分开计算，最后才更新 h_prev
input_transform = matmul(current_input, input_weight.T) + input_bias
hidden_transform = matmul(h_prev, hidden_weight.T) + hidden_bias
h_cur = relu(input_transform + hidden_transform)
h_prev = h_cur   # 最后统一更新
```

---

### 4. range 倒序的 stop 参数

`range(t-1, -1, -1)` 的 stop 是 `-1`（不含），最后一个值是 `0`，覆盖完整序列。  
常见错误：写成 `range(t-1, 0, -1)`，导致索引 0 的 token 被跳过。

---

## 🔗 知识延伸

- **双向 RNN（BiRNN）**：正向链使用 `range(t)`，反向链使用 `range(t-1, -1, -1)`，最终将两个方向的隐状态拼接
- **LSTM / GRU**：在 RNN 基础上引入门控机制，解决长序列梯度消失问题，两个 Linear 扩展为多个 Linear 对应多个门
- **Conv1d 与 RNN 的对比**：Conv1d 捕获局部固定窗口内的上下文，RNN 捕获全局历史依赖，两者各有适用场景
- **torch.cat vs torch.stack**：`cat` 在已有维度上拼接（不增加新维度），`stack` 沿新维度堆叠（会增加一个维度）

---

## 📚 参考资料

- PyTorch 官方文档：[torch.nn.Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- PyTorch 官方文档：[torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- 课程代码：`stage03/02.RNN循环神经网络/02_RNN理解/01_RNN理解.py`
