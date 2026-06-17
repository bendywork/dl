import torch


x = torch.tensor(2.0, requires_grad=True)
y = x ** 3            # y = f(x)
z = torch.sin(y)      # z = g(y) = g(f(x))
z.backward()          # 自动用链式法则：dz/dx = cos(x³) · 3x²
print(x.grad)         # tensor(-11.7873)  ≈ cos(8) * 12