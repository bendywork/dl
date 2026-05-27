# PyTorch topk 与 argmax 详解

## 📌 核心问题
> 如何从张量中取出最大值的索引，或前 k 个最大值及其索引？

---

## 🔧 argmax：取最大值的索引

```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
max_index = torch.argmax(x, dim=1)
# tensor([1, 1])  ← 每行最大值所在的列索引
```

**注意**：print 占位符必须用 f-string 或 .format()，不能直接用 `{}`：

```python
# 错误写法（{} 不会被替换）
print("索引: {}, 值: {}", max_index, values)

# 正确写法
print(f"索引: {max_index}, 值: {values}")
print("索引: {}, 值: {}".format(max_index, values))
```

---

## 🔧 topk：取前 k 个最大值及其索引

### 函数签名

```python
torch.topk(input, k, dim=None, largest=True, sorted=True)
```

| 参数 | 含义 |
|------|------|
| `input` | 输入张量 |
| `k` | 取前 k 个 |
| `dim` | 沿哪个维度操作 |
| `largest` | True=最大值，False=最小值 |
| `sorted` | 结果是否按降序排列 |

### 返回值

返回具名元组，包含 `values` 和 `indices`：

```python
top_k_values, top_k_indices = torch.topk(x, k=2, dim=1)
# 等价于
result = torch.topk(x, k=2, dim=1)
result.values
result.indices
```

---

## 📐 3x3 示例详解

```python
x = [[1, 5, 3],
     [9, 2, 7],
     [4, 8, 6]]
```

### dim=1，k=2（对每行取前2大）

每行**横向**扫描，挑出最大的 k 个：

```
第0行 [1, 5, 3] → 5(列1), 3(列2)
第1行 [9, 2, 7] → 9(列0), 7(列2)
第2行 [4, 8, 6] → 8(列1), 6(列2)
```

```
values  = [[5, 3],
           [9, 7],
           [8, 6]]

indices = [[1, 2],
           [0, 2],
           [1, 2]]

形状：(3,3) → (3,2)   # 行数不变，列变成 k
```

### dim=0，k=2（对每列取前2大）

每列**纵向**扫描，挑出最大的 k 个：

```
第0列 [1, 9, 4] → 9(行1), 4(行2)
第1列 [5, 2, 8] → 8(行2), 5(行0)
第2列 [3, 7, 6] → 7(行1), 6(行2)
```

```
values  = [[9, 8, 7],   ← 三列各自的第1大
           [4, 5, 6]]   ← 三列各自的第2大

indices = [[1, 2, 1],
           [2, 0, 2]]

形状：(3,3) → (2,3)   # 列数不变，行变成 k
```

---

## 💡 核心规律

```
哪个 dim → 那个维度的大小从 n 变成 k，其余维度不变
```

**topk 不是降维，而是"替换"那个维度的大小：**

```
原始：(3, 3)
topk(k=2, dim=0) → (2, 3)   # dim=0 的大小3 → 变成k=2
topk(k=2, dim=1) → (3, 2)   # dim=1 的大小3 → 变成k=2
```

**dim 的直觉：**

```
dim=1 → 沿列比较 → 每行出结果 → 输出行数不变
dim=0 → 沿行比较 → 每列出结果 → 输出列数不变
```

---

## ⚠️ 易错点

1. **dim=0 结果看起来像转置**：不是真的转置，是因为每列独立排名后，"第几大"变成了行的编号，视角换了
2. **print 中用 {} 不加 .format()**：`{}` 不会自动替换，必须配合 `.format()` 或使用 f-string
3. **topk 维度数不变**：(3,3) → (2,3) 仍是二维，只是某个轴的大小变了

---

## 🔗 知识延伸

- `torch.argmax()`：只返回最大值的索引，等价于 `topk(k=1).indices.squeeze()`
- `torch.sort()`：返回全部排序结果，topk 是其子集（只取前k个，更高效）
- `torch.argsort()`：返回排序后的索引
