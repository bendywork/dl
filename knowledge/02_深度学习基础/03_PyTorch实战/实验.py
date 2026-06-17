import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
max_index = torch.argmax(x, dim=1)
# tensor([1, 1])  ← 每行最大值所在的列索引
print(max_index)
# print(x[max_index])
# 每行取出其最大值所在的元素，应该这样写：
print(torch.amax(x, dim=1))
print(x.gather(1, max_index.unsqueeze(1)))