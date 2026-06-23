# Scaled Dot-Product Attention 三个疑问详解

## 疑问1：为什么 transpose(-2, -1) 比 transpose(1, 2) 更通用？

### 问题背景
```python
K.transpose(1, 2)    # 示例写法
K.transpose(-2, -1)  # 工业写法，为什么更好？
```

### 正数索引 vs 负数索引

```python
# 3维 tensor [bs, t, d_k]
# 正数索引：0=bs,  1=t,  2=d_k
# 负数索引：-3=bs, -2=t, -1=d_k

# 4维 tensor [bs, heads, t, d_k]
# 正数索引：0=bs, 1=heads, 2=t,  3=d_k
# 负数索引：-4=bs,-3=heads,-2=t, -1=d_k
```

### 对比

```python
# 3维时，两种写法都正确
K.transpose(1, 2)    # 交换 t 和 d_k  ✓
K.transpose(-2, -1)  # 交换 t 和 d_k  ✓

# 4维时（Multi-Head Attention）
K.transpose(1, 2)    # 交换 heads 和 t  ✗ 不是你要的
K.transpose(-2, -1)  # 交换 t 和 d_k   ✓ 永远正确
```

### 结论
负数索引从最后一维开始数，**永远指向倒数第1和倒数第2维**，
不管 tensor 有几个维度都适用，Multi-Head Attention 扩展到4维时不需要改代码。

---

## 疑问2：为什么用 ** 0.5 而不是 np.sqrt？

### np.sqrt 的问题

```python
# np.sqrt 是 numpy 函数，numpy 运行在 CPU
# tensor 可能在 GPU 上

similarity / np.sqrt(self.d_k)
# 触发 CPU↔GPU 同步，性能损耗
```

### 推荐写法

```python
similarity / (self.d_k ** 0.5)   # 纯 Python float，无额外开销
similarity / math.sqrt(self.d_k) # math 库，同样是纯 Python
```

### 结论
tensor 运算尽量不引入 numpy，避免设备切换开销，大模型训练时积累明显。

---

## 疑问3：为什么 x^0.5 等于开根号？分数次幂怎么理解？

### 指数规律

```
x^a × x^b = x^(a+b)   核心规律：指数相加
```

### 推导 0.5 次幂

```
x^0.5 × x^0.5 = x^(0.5+0.5) = x^1 = x

自己乘自己等于 x，满足这个条件的就是 √x
所以 x^0.5 = √x
```

### 分数次幂通用规律

```
x^(1/n) = x 的 n 次方根

x^(1/2) = √x    平方根
x^(1/3) = ∛x    立方根
x^(m/n) = (x^m) 的 n 次方根
```

### 例子：x^(2/3)

```
8^(2/3) = (8^2) 的立方根 = 64 的立方根 = ∛64 = 4
反向验证：4^3 = 64 = 8^2  ✓
```

### 代码验证

```python
8 ** (2/3)    # = 4.0
256 ** 0.5    # = 16.0
256 ** (1/2)  # = 16.0
import math
math.sqrt(256) # = 16.0  三种写法完全等价
```

---

## 总结

| 疑问 | 结论 |
|------|------|
| transpose(-2,-1) | 负数索引从末尾数，任意维度通用 |
| ** 0.5 vs np.sqrt | 避免 numpy 引入 CPU↔GPU 同步开销 |
| 分数次幂 | x^(m/n) = (x^m)的n次方根，0.5次幂就是开平方 |
