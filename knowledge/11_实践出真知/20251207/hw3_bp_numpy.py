# -*- coding: utf-8 -*-
"""
作业 20251207 - 第三题
参考 stage02/03.深度学习基础/02_BP过程理解.py，
将逐元素标量写法改为 NumPy 矩阵运算结构。

网络结构：输入2 → 隐层3(sigmoid) → 输出2(sigmoid)，MSE损失
原始参数与脚本保持一致，可对照验证数值。
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """已知 sigmoid 输出 a，直接求导：a*(1-a)"""
    return a * (1.0 - a)


class TwoLayerNet:
    """
    两层全连接网络（矩阵版 BP）
    Layer1: [input_size → hidden_size] + sigmoid
    Layer2: [hidden_size → output_size] + sigmoid
    Loss:   MSE = 0.5 * sum((pred - y)^2)
    """

    def __init__(self, W1: np.ndarray, W2: np.ndarray,
                 b1: float, b2: float, lr: float = 0.5):
        # W1: [input_size, hidden_size]
        # W2: [hidden_size, output_size]
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.b1 = b1    # 标量 bias（与原版保持一致：整层共用同一个 bias 值）
        self.b2 = b2
        self.lr = lr

        # 中间变量（前向时填充，反向时使用）
        self.x = self.h = self.o = None

    # ── 前向 ──────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [bs, input_size]
        返回 o: [bs, output_size]
        """
        self.x = x
        # 第一层：net_h = x @ W1 + b1  →  h = sigmoid(net_h)
        self.net_h = x @ self.W1 + self.b1          # [bs, hidden]
        self.h = sigmoid(self.net_h)                  # [bs, hidden]
        # 第二层：net_o = h @ W2 + b2  →  o = sigmoid(net_o)
        self.net_o = self.h @ self.W2 + self.b2      # [bs, output]
        self.o = sigmoid(self.net_o)                  # [bs, output]
        return self.o

    # ── 损失 ──────────────────────────────────────────────
    def loss(self, y: np.ndarray) -> float:
        return float(0.5 * np.sum((self.o - y) ** 2))

    # ── 反向 ──────────────────────────────────────────────
    def backward(self, y: np.ndarray):
        """
        y: [bs, output_size]
        推导（链式法则，矩阵形式）：
          dL/dnet_o = (o - y) * sigmoid'(net_o)         [bs, output]
          dL/dW2   = h.T @ delta_o                       [hidden, output]
          delta_h  = delta_o @ W2.T * sigmoid'(net_h)   [bs, hidden]
          dL/dW1   = x.T @ delta_h                       [input, hidden]
        """
        bs = self.x.shape[0]

        # 输出层误差项
        delta_o = (self.o - y) * sigmoid_deriv(self.o)   # [bs, output]

        # W2 梯度
        dW2 = self.h.T @ delta_o / bs                     # [hidden, output]

        # 隐层误差项（反向传播穿过 W2）
        delta_h = (delta_o @ self.W2.T) * sigmoid_deriv(self.h)  # [bs, hidden]

        # W1 梯度
        dW1 = self.x.T @ delta_h / bs                    # [input, hidden]

        # 参数更新
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2

    def step(self, x: np.ndarray, y: np.ndarray):
        """前向 + 计算损失 + 反向，返回损失值"""
        self.forward(x)
        l = self.loss(y)
        self.backward(y)
        return l


# ─────────────────────────────────────────────
# 对照原标量版：使用完全相同的初始参数
# ─────────────────────────────────────────────
def build_net_from_original():
    _w = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                     0.4, 0.45, 0.5, 0.55, 0.6, 0.65], dtype=np.float64)
    _b = np.asarray([0.35, 0.65], dtype=np.float64)

    # 原始参数排列：
    # W1[input, hidden] — 按原标量编号拼回矩阵
    # w1~w6: h1←x1,x2 | h2←x1,x2 | h3←x1,x2
    W1 = np.array([
        [_w[0], _w[2], _w[4]],   # x1 → h1, h2, h3
        [_w[1], _w[3], _w[5]],   # x2 → h1, h2, h3
    ])  # shape [2, 3]

    # W2[hidden, output] — w7~w12
    W2 = np.array([
        [_w[6],  _w[7]],   # h1 → o1, o2
        [_w[8],  _w[9]],   # h2 → o1, o2
        [_w[10], _w[11]],  # h3 → o1, o2
    ])  # shape [3, 2]

    return TwoLayerNet(W1, W2, b1=float(_b[0]), b2=float(_b[1]), lr=0.5)


def main():
    net = build_net_from_original()

    x = np.array([[5.0, 10.0]])   # [1, 2]
    y = np.array([[0.01, 0.99]])  # [1, 2]

    # 第一次迭代，打印对照信息
    l0 = net.step(x, y)
    print(f"第1次迭代后  loss={l0:.6f}  pred={net.o}")

    losses = [l0]
    for _ in range(10000):
        losses.append(net.step(x, y))

    print(f"\n10000次迭代后  loss={losses[-1]:.8f}  pred={net.o}")

    # 可视化损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("BP (NumPy Matrix Version) — Loss Curve")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
