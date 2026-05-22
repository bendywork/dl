# PyTorch state_dict 与 named_parameters 对比

## 📌 核心问题
> `state_dict()` 和 `named_parameters()` 都能拿到模型的权重，它们到底有什么区别？什么场景用哪个？

## 🌱 一句话区分

```
named_parameters() = 可学习参数（给优化器用）
state_dict()       = 可学习参数 + 不可学习状态（给保存/加载用）
```

## 📐 从简单到底层，逐层拆解

### 第一层：包含的内容不同

| 内容 | named_parameters() | state_dict() |
|------|:------------------:|:------------:|
| Linear 的 weight/bias | ✓ | ✓ |
| BatchNorm 的 weight/bias | ✓ | ✓ |
| BatchNorm 的 running_mean | ✗ | ✓ |
| BatchNorm 的 running_var | ✗ | ✓ |
| BatchNorm 的 num_batches_tracked | ✗ | ✓ |

**Buffer（running_mean 等）不是参数，不会参与梯度计算，但是模型状态的一部分。**

实际对比（Linear+BatchNorm+ReLU+Linear 模型）：

```
named_parameters():  6 个 — weight×2 + bias×2 + BN.weight + BN.bias
state_dict():        9 个 — 上面6个 + running_mean + running_var + num_batches_tracked
```

### 第二层：返回类型不同

| | named_parameters() | state_dict() |
|---|---|---|
| 返回类型 | `generator`（迭代器） | `OrderedDict`（有序字典） |
| 加载方式 | `for name, p in model.named_parameters()` | `sd = model.state_dict()` |
| 内存 | 懒加载，不占额外内存 | 全量拷贝，双倍内存 |

### 第三层：内存关系不同（关键差异）

```python
# named_parameters 返回的是模型内部的【同一个对象】
for name, param in model.named_parameters():
    if '0.weight' in name:
        param.data[0, 0] = 999.0    # 修改了 param
        break
print(model[0].weight.data[0, 0])   # 输出 999.0 ← 模型内部也变了！

# state_dict 返回的是【拷贝】，不是同一个对象
sd = model.state_dict()
sd['0.weight'][0, 0] = -1.0         # 修改了拷贝
print(model[0].weight.data[0, 0])   # 输出 999.0 ← 模型内部没变
```

| | named_parameters() | state_dict() |
|---|---|---|
| 和模型内部的关系 | **同一对象**（引用） | **独立拷贝**（复制） |
| 修改它影响模型吗 | ✓ 直接影响 | ✗ 不影响 |
| data_ptr() 相同吗 | ✓ 相同 | ✗ 不同 |

### 第四层：冻结参数时的行为

```python
model[0].weight.requires_grad = False  # 冻结第一层
```

| | named_parameters() | state_dict() |
|---|---|---|
| 冻结的参数还在吗 | ✓ 在（requires_grad=False） | ✓ 在 |
| 区别 | 可以通过 requires_grad 判断是否冻结 | 无法区分（全是普通 Tensor） |

**注意**：`named_parameters()` 包含冻结参数（requires_grad=False），但 `parameters()` 传给优化器时可以过滤：
```python
optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
```

### 第五层：底层实现原理

```
nn.Module 内部维护两个字典：
  _parameters:  存 Parameter（通过 self.weight = nn.Parameter(...) 注册）
  _buffers:     存 Tensor（通过 self.register_buffer(...) 注册）

named_parameters():
  → 遍历 _parameters，返回 (name, Parameter) 对
  → 递归遍历所有子模块（recurse=True）

state_dict():
  → 遍历 _parameters + _buffers，返回 OrderedDict
  → 值通过 .detach() 拷贝，与模型内部断开
  → 递归遍历所有子模块
```

### 第六层：完整关系图

```
nn.Module
├── _parameters (Parameter, requires_grad=True/False)
│   ├── Linear.weight          ← named_parameters() ✓  state_dict() ✓
│   ├── Linear.bias            ← named_parameters() ✓  state_dict() ✓
│   ├── BatchNorm.weight       ← named_parameters() ✓  state_dict() ✓
│   └── BatchNorm.bias         ← named_parameters() ✓  state_dict() ✓
│
├── _buffers (Tensor, requires_grad=False)
│   ├── BatchNorm.running_mean ← named_parameters() ✗  state_dict() ✓
│   ├── BatchNorm.running_var  ← named_parameters() ✗  state_dict() ✓
│   └── BatchNorm.num_batches  ← named_parameters() ✗  state_dict() ✓
│
└── _modules (子模块，递归遍历)
    └── ...
```

## 💡 关键理解：用途决定选择

| 场景 | 用哪个 | 原因 |
|------|--------|------|
| 传给优化器 | `parameters()` | 优化器只需要可学习参数 |
| 保存模型到磁盘 | `state_dict()` | 必须保存 buffer（如 BN 统计量） |
| 加载模型 | `load_state_dict()` | 恢复全部状态（含 buffer） |
| 冻结/解冻层 | `named_parameters()` | 需要 name 定位 + requires_grad 控制 |
| 统计参数量 | `parameters()` | 只关心可训练参数 |
| 迁移学习 | `state_dict()` | 预训练权重含 BN 统计量 |
| 梯度裁剪 | `parameters()` | 只裁剪有梯度的参数 |

**核心逻辑**：优化器只管"学习"→ 用 parameters；保存/加载要管"状态"→ 用 state_dict。

## 🔧 代码速查

```python
# 保存 & 加载
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))

# 优化器
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 冻结某层
for name, param in model.named_parameters():
    if 'layer1' in name:
        param.requires_grad = False

# 参数量统计
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## ⚠️ 易错点

1. **"state_dict 不含 buffer"** → 错。state_dict 包含 buffer，这正是它存在的意义
2. **"named_parameters 不含冻结参数"** → 错。冻结参数还在，只是 requires_grad=False
3. **"修改 state_dict 的值能改变模型"** → 错。state_dict 是拷贝，不影响模型；要改模型需 `load_state_dict()`
4. **"保存模型用 torch.save(model)"** → 不推荐。应保存 state_dict，否则 pickle 绑定了类结构
5. **"optimizer 需要 state_dict 里的 buffer"** → 不需要。优化器只优化参数，不优化 running_mean

## 🔗 知识延伸

- [[11_PyTorch梯度计算开关详解]]：requires_grad 冻结机制
- [[14_PyTorch的Linear层W形状详解]]：Parameter 的存储细节

## 📚 参考资料

- PyTorch 源码：`torch/nn/modules/module.py` → `state_dict()` 和 `named_parameters()` 实现
- [PyTorch Saving/Loading](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
