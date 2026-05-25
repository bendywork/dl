import torch
import torch.nn.functional as F

# 场景：多标签分类 - 图片可以同时是"猫"和"狗"
# 模型最后一层输出两个logits（原始分数）
logits = torch.tensor([2.0, 1.5])  # 猫的得分: 2.0, 狗的得分: 1.5
torch.device("cpu")
# 使用两个独立的Sigmoid函数
prob_cat = torch.sigmoid(logits[0])  # 猫的概率
prob_dog = torch.sigmoid(logits[1])  # 狗的概率

print("=== 多标签二分类（两个独立Sigmoid）===")
print(f"原始logits: {logits.tolist()}")
print(f"是猫的概率: {prob_cat.item():.4f}")
print(f"是狗的概率: {prob_dog.item():.4f}")
print(f"概率和: {prob_cat.item() + prob_dog.item():.4f}")
print(f"结论: 概率和 ≠ 1 ({prob_cat.item() + prob_dog.item():.4f} ≠ 1)")