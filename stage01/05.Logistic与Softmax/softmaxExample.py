import torch
import torch.nn as nn


softmax = torch.tensor([[2.0, 2.2, 2.9, 11, 20, 30, 12]])
torch.device("cpu")
result = torch.softmax(softmax, dim=1)

print("=== 多标签分类（Softmax）===")
print(f"原始logits: {softmax.tolist()}")
print(f"result-tolist-结果: {result.tolist()}")
print(f"各类概率: {result.tolist()[0]}")
print(f"预测类别: {result.argmax(dim=1).item()}")
print(f"概率和: {result.sum().item():.4f}")
print()
