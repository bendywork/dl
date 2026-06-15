# PyTorch dim 维度规则详解

## 📌 核心问题
> `dim=0` 到底是从左数还是从右数？和 broadcasting 的对齐方式为什么不同？

---

## 🌱 两个规则，两个场景

### 规则一：cat / split / squeeze 等操作 — 从左数

```python
shape: (3, 4, 5)
        ↑  ↑  ↑
      dim0 dim1 dim2
```

`dim=0` 就是 shape 的**第一个数字**，即最外层维度。

```python
a = torch.zeros(3, 2)
b = torch.zeros(3, 1)

torch.cat([a, b], dim=0)  # ❌ 报错：dim=0以外的维度必须相同，2 ≠ 1
torch.cat([a, b], dim=1)  # ✅ 正确：(3, 2+1) = (3, 3)
```

**cat 规则：沿 dim=N 拼接时，第 N 维可以不同，其余维度必须完全相同。**

---

### 规则二：Broadcasting（广播）— 从右对齐

```
(3, 4, 5)
      (5)   ← 从右边对齐，左边不足的维度自动扩展为 1
```

Broadcasting 不需要指定 dim，是自动的形状匹配机制。

---

## 💡 关键理解

| 场景 | 对齐方向 | 说明 |
|------|----------|------|
| `cat` / `split` / `squeeze` / `unsqueeze` | **从左数** | dim=0 是第一个维度 |
| Broadcasting | **从右对齐** | 自动补齐左边缺失的维度 |

两个规则完全独立，不会混用。

---

## 🔧 代码验证

```python
import torch

a = torch.zeros(3, 2)
b = torch.zeros(3, 1)

# cat dim=1：沿列方向拼，行数必须相同
combined = torch.cat([a, b], dim=1)
print(combined.shape)  # torch.Size([3, 3])

# broadcasting：从右对齐
x = torch.ones(3, 4, 5)
y = torch.ones(5)       # 自动扩展为 (1, 1, 5) → (3, 4, 5)
z = x + y
print(z.shape)          # torch.Size([3, 4, 5])
```

---

## ⚠️ 易错点

1. **cat 和 broadcasting 混淆**：cat 从左数 dim，broadcasting 从右对齐，两者无关
2. **cat 方向选错**：split 沿哪个 dim 切，cat 就应该沿同一个 dim 拼回去
3. **broadcasting 不是操作**：它是自动触发的，不需要指定 dim

---

## 🔗 知识延伸

- `torch.stack` 与 `torch.cat` 的区别：stack 会新增一个维度，cat 不会
- `tensor.unsqueeze(dim)` / `tensor.squeeze(dim)`：同样从左数 dim
