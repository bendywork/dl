# 05 · 稀疏 & 滑动窗口 Attention（长序列降复杂度）

标准 Self-Attention 的 O(n²) 复杂度在长序列场景下是灾难——文本生成 100K tokens 时每个 token 要 attend 100K 个位置。稀疏 Attention 和滑动窗口通过**限制每个 token 能 attend 的范围**来降复杂度。

## 三种主流稀疏策略

### 1. Sliding Window Attention（滑窗，Mistral）

```
每个 token 只 attend 前后 w 个 token

"我 今天 吃 了 一 个 红 色 的 苹 果"
              ↑
          "色" 只看到 前面 4 个 + 后面 4 个 = 8 个 token
          而不是整句 11 个 token

复杂度: O(n × w)  ←  w=4096 时远小于 n=128000
```

| 使用 | 窗口大小 | 说明 |
|------|---------|------|
| Mistral 7B | w=4096 | 配合 Sliding Window，32K 上下文 |
| Gemma 2 | w=4096 | |

**如何看到远距离信息**：每层都滑动，信息靠层间传递从远距离"渗透"过来（类似 CNN 感受野堆叠扩大的原理）。

### 2. Sparse Attention（Longformer / BigBird）

```
不只看窗口，还选择性看一些全局 token

标准:          每一个 token ← 所有 token        O(n²)
Longformer:    每一个 token ← 窗口 token + 全句共用的 [CLS] token  O(n·w + n·g)
BigBird:       每一个 token ← 窗口 + 随机 + 全局 token  O(n·w + n·g + n·r)
```

| 模型 | 策略 | 复杂度 |
|------|------|--------|
| Longformer | 滑窗 + 全局 [CLS] | O(n·w) |
| BigBird | 滑窗 + 随机 + 全局 | O(n·w) |

### 3. Dilated Attention（扩张窗口，LongNet）

标准滑窗有"近视眼"问题——相邻层只能用同一个窗口感受附近的 token。Dilated 在不同层用不同 stride（间距 1, 2, 4, 8...），**浅层看细，深层看广**。

```
Layer 1: window [i-4, i+4], stride=1   → 局部密度高
Layer 4: window [i-4, i+4], stride=8   → 跨越 64 个 token
Layer 8: window [i-4, i+4], stride=64  → 跨越 512 个 token
```

| 使用 | 上下文 | 说明 |
|------|-------|------|
| LongNet | 10 亿 token | Microsoft 论文 |

## 稀疏 vs 全 Attention：什么时候用谁

| 场景 | 推荐 |
|------|------|
| 短文本（< 4K） | 全 Attention（复杂度 O(n²) 不大） |
| 中等长度（4K-32K） | Sliding Window + FlashAttention |
| 长文本（> 32K） | 稀疏 Attention / Dilated |
| 文长且要理解全文（法律、论文） | BigBird（随机 + 全局） |

## 一句话总结

> 滑窗是最简单有效的优化——Mistral 靠它把 32K 上下文跑在 7B 模型上。稀疏是更复杂的优化——Longformer/BigBird 证明了"不用每个 token 看所有 token 也能理解全文"。
