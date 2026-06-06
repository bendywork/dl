# ==============================================================
# 文件：08_索引与切片.py
# 主题：PyTorch 张量的高级索引与切片操作
# API：gather / scatter_ / index_select / masked_select / where / nonzero / take
# ==============================================================
import torch
import torch.nn.functional as F

SEP = "=" * 60

# ==============================================================
# 1. index_select  按指定轴、指定索引选取行/列
# ==============================================================
print(SEP)
print("1. index_select")
print(SEP)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
idx = torch.tensor([0, 2])
print("原始张量 x:")
print(x)
print("index_select(dim=0, [0,2]) -> 第0行和第2行:")
print(torch.index_select(x, dim=0, index=idx))
print("index_select(dim=1, [0,2]) -> 第0列和第2列:")
print(torch.index_select(x, dim=1, index=idx))

# ==============================================================
# 2. gather  按 index 张量聚集值
#    output[i][j] = input[i][index[i][j]]  (dim=1)
#    核心用途：NLL Loss 取目标类别概率
# ==============================================================
print()
print(SEP)
print("2. gather")
print(SEP)
probs = torch.tensor([[0.1,0.3,0.4,0.2],[0.5,0.1,0.2,0.2],[0.1,0.6,0.2,0.1]])
labels = torch.tensor([[2],[0],[1]])
gathered = torch.gather(probs, dim=1, index=labels)
print("概率矩阵 probs:")
print(probs)
print("真实标签 labels（列向量）:", labels.T)
print("对应类别概率 gathered:")
print(gathered)

# ==============================================================
# 3. scatter_  gather 的逆操作，把值分散写入目标位置
#    核心用途：生成 one-hot 编码
# ==============================================================
print()
print(SEP)
print("3. scatter_ (in-place)")
print(SEP)
# 生成 one-hot
num_classes = 5
labels_1d = torch.tensor([0, 2, 4, 1])
one_hot = torch.zeros(4, num_classes)
one_hot.scatter_(1, labels_1d.unsqueeze(1), 1.0)
print("标签:", labels_1d.tolist())
print("one-hot 编码:")
print(one_hot)

# scatter_ 用 src 张量赋值
src = torch.full((2, 3), 0.5)
out = torch.zeros(2, 5)
idx = torch.tensor([[0, 2, 4],[1, 3, 0]])
out.scatter_(1, idx, src)
print("scatter_ 用 src 张量赋值:")
print(out)

# ==============================================================
# 4. masked_select  布尔掩码选取，返回 1D 张量
# ==============================================================
print()
print(SEP)
print("4. masked_select")
print(SEP)
x = torch.randn(3, 4)
torch.manual_seed(0)
x = torch.randn(3, 4)
mask = x > 0
selected = torch.masked_select(x, mask)
print("原始张量:")
print(x.round(decimals=2))
print("掩码 (>0):")
print(mask)
print("所有正数值 (展平为1D):", selected.round(decimals=2))

# ==============================================================
# 5. where  按条件从两个张量中选值
# ==============================================================
print()
print(SEP)
print("5. where")
print(SEP)
a = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
result = torch.where(a > 0, a, torch.zeros_like(a))
print("a:", a.tolist())
print("where(a>0, a, 0):", result.tolist())
# 广播支持
x2 = torch.tensor([[1, 2],[3, 4]])
y2 = torch.tensor([[10,20],[30,40]])
cond = torch.tensor([[True,False],[False,True]])
print("广播 where 示例:")
print(torch.where(cond, x2, y2))

# ==============================================================
# 6. nonzero  返回非零元素的索引
# ==============================================================
print()
print(SEP)
print("6. nonzero")
print(SEP)
x = torch.tensor([[0,1,0],[2,0,3],[0,0,4]])
nz = torch.nonzero(x)
print("张量 x:")
print(x)
print("nonzero (N,2) 坐标矩阵:")
print(nz)
rows, cols = torch.nonzero(x, as_tuple=True)
print("as_tuple=True -> rows:", rows.tolist(), "cols:", cols.tolist())
print("非零值:", x[rows, cols].tolist())

# ==============================================================
# 7. take  把张量视为 1D，按线性索引取值
# ==============================================================
print()
print(SEP)
print("7. take")
print(SEP)
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
idx = torch.tensor([0, 4, 8])
print("张量 x:")
print(x)
print("take([0,4,8]) 主对角线:", torch.take(x, idx).tolist())
print("take([1,3,5,7]):", torch.take(x, torch.tensor([1,3,5,7])).tolist())

# ==============================================================
# 综合示例：用 gather 手动实现 NLL Loss
# ==============================================================
print()
print(SEP)
print("综合示例：手动实现 NLL Loss")
print(SEP)
logits = torch.tensor([[2.0,1.0,0.5],[0.5,2.5,0.1],[0.1,0.3,3.0],[1.0,2.0,0.5]])
targets = torch.tensor([0, 1, 2, 1])
log_probs = F.log_softmax(logits, dim=1)
target_lp = log_probs.gather(1, targets.unsqueeze(1))
manual_loss = -target_lp.mean()
official_loss = F.nll_loss(log_probs, targets)
print("手动 NLL Loss:", round(manual_loss.item(), 4))
print("官方 NLL Loss:", round(official_loss.item(), 4))
print("结果一致:", torch.isclose(manual_loss, official_loss).item())
print()
print("08_索引与切片.py 运行完毕!")
