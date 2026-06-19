# Attention 扩展与进阶理解

> 软/硬 Attention、图像 Attention、层次化 Attention、状态传递演进

---

## 1. 软 Attention vs 硬 Attention

### 一句话区分

| | 软 Attention | 硬 Attention |
|---|---|---|
| 选择方式 | **全看，分权重** | **只看一个，挑一个** |
| 数学 | 连续可导（softmax） | 离散选择（argmax / 采样） |
| 训练 | 反向传播直接训 | 需要强化学习 / 梯度估计 |
| 权重分布 | 所有位置都有非零权重 | 只有选中的位置权重=1，其余=0 |

### 软 Attention（Soft Attention）— 你学的就是这个

```
权重 = softmax(score)     → 所有位置都分到权重，和为1
输出 = Σ(权重ᵢ × Vᵢ)      → 连续加权求和，处处可导
```

特征：
- 处处可导 → 端到端反向传播 → 训练简单
- 每个位置都参与 → 计算 O(n²)，n 大时慢
- 你文件里的 `qkv_attention_value` 就是软 Attention（softmax 后加权求和）

### 硬 Attention（Hard Attention）

```
选中位置 = argmax(score)  → 只挑权重最大的一个位置
输出 = V[选中位置]         → 独热选择，不可导
```

特征：
- 不可导（argmax 梯度为 0） → 不能直接用反向传播 → 用 REINFORCE 等强化学习方法训练
- 只取一个 → 计算快 → n 很大时优势明显
- 问题是容易选错（训练早期很难选中正确的），训得慢

### 为什么不直接用硬 Attention？

| 问题 | 解释 |
|------|------|
| argmax 不可导 | 神经网络靠梯度下降训，argmax 没梯度 |
| 训不动 | 需要用策略梯度（REINFORCE），方差大，收敛慢 |
| 软硬权衡 | 软 Attention 更好训，所以成为主流 |

### 实际普遍使用

**99% 的场景用软 Attention**。硬 Attention 只在极少数场景出现（如强化学习 agent 选择"看画面哪个区域"）。你只需知道概念区别，实战全部是软 Attention。

---

## 2. 图像 Attention：CNN 特征充当 K/V

### 图像怎么"编码"

| 文本 | 图像 |
|------|------|
| 分词 → token id → Embedding 查表 | Patch 切块 / CNN 下采样 |
| 得到 `[bs, seq_len, hidden]` | 得到 `[bs, H×W, hidden]` |

两条路线：
- **ViT 路线**：原图 → 切 16×16 patch → Linear 投影 → 图像 token 序列
- **CNN 混合路线**：原图 → CNN backbone → 特征图 `[C, H, W]` → flatten → 每个空间位置 = 一个"图像 token"

### QKV 分配（按任务）

| 任务 | Q | K, V | 说明 |
|------|----|------|------|
| 图像描述生成 | decoder 状态 | CNN 特征图 | 跟文本 Seq2Seq+Attention 完全一样 |
| 目标检测（DETR） | 学到的 object query | CNN 特征图 | 每个 query 负责找一种物体 |
| 分类（ViT） | `[CLS]` token | 图像 patch | 一个 token 查所有 patch |
| 图像生成 | 噪声 token / query | 条件图像特征 | diffusion 里的 cross-attention |

### 文本 → 图像的 Attention 本质映射

```
文本：分词 → embedding → [N个token, e] → Q查K → 看源句不同位置
图像：CNN/Patch → H×W个特征向量 → Q查K → 看图像不同区域
```

**本质完全一样：把"信息源"变成一组向量 → 用 Q 去查 → 按权重取。**

---

## 3. 长文本高效 Attention：层次化 + 稀疏

### 核心思路（先粗后细）

```
文本 → 分段（句子/段落）→ 每段压成摘要向量 → 摘要间算 Attention
                                              ↓
                               找出相关段 → 段内细粒度 Attention
```

### 已有方案

| 方案 | 做法 | 对应直觉 |
|------|------|---------|
| HAN (层次化 Attention) | 词级 → 句级，两级 Attention | "先找相关段，再看段内词" |
| Longformer | 局部滑窗 + 少量全局 token | "上下文窗口 + 关键位置全局看" |
| BigBird | 滑窗 + 随机 + 全局，三种 mask | "多视角混合关注" |
| Linformer | K/V 序列线性投影压缩 | "先建索引，压缩后查" |
| Perceiver | 学少量"归纳点" attend 全文 | "只让少数 delegate 去读全文" |

### 复杂度压缩

```
标准 Self-Attention: O(n²)    n=10000 → 1亿次
层次化 Attention:   O(m² + n) m=50   → 2500 + 200/段
差距三个数量级
```

---

## 4. K 和 V 的长度约束

### 数学硬约束：K 和 V 序列长度必须相等

```
Q:  [bs, qt,  e]     qt 可以是任意长度
K:  [bs, kvt, e]     kvt ← 序列维
V:  [bs, kvt, e]     kvt ← 必须跟 K 一样

alpha: softmax(Q·Kᵀ)  → [bs, qt, kvt]
output: alpha · V       → [bs, qt, kvt] × [bs, kvt, e]
                           kvt 对不上就报错
```

### 可以不一样的是 Q 长度

```
Cross-Attention 的灵活性：
  Q: [bs, 1, e]    ← decoder 每步 1 个 query
  K: [bs, 50, e]   ← encoder 50 个 token
  V: [bs, 50, e]   ← 序列长度跟 K 对其
  qt=1 ≠ kvt=50    ← 可以不一样，没问题
```

| | 可以不一样？ | 为什么 |
|---|-----------|--------|
| Q 长度 vs K 长度 | ✅ 可以 | Cross-Attention 核心能力 |
| K 长度 vs V 长度 | ❌ 不能 | 矩阵乘法约束 |

---

## 5. 状态传递：从 RNN 到 Attention

### 你文件里的做法（仍依赖 RNN）

```python
hcn = (h0, c0)                              # LSTM 初始状态
for _t in range(dt):
    atte = attention_value(enc_out, hcn)      # attention 给 LSTM "加料"
    output, hcn = self.rnn_layer(input, hcn)  # ← LSTM 递推状态！
```

状态传递链 `hcnₜ₋₁ → hcnₜ` 走 LSTM 门控。Attention 只是辅助。

### Transformer 的做法（扔掉 RNN）

```
t-1 时刻输出 → Self-Attention（masked）→ 当前时刻输出
                                            ↓
                              追加到序列 → 下一步 Self-Attention 能看到它
```

| | Seq2Seq+Attention | Transformer |
|---|---|---|
| 状态传递者 | LSTM 的 h/c | Attention 输出向量 |
| 看历史 | LSTM 隐状态"记住" | Masked Self-Attention 直接看前面 |
| 训练 | 串行（循环逐 token） | 并行（mask 一下全部一次算完） |

> 如果 attention 值已包含"这一步该知道的信息"，下一步直接拿它继续 attend —— RNN 的循环状态传递就多余了。

---

## 6. 核心脉络回顾

```
分词 → Embedding → RNN/LSTM（token 融合）→ Seq2Seq（编码-解码生成）
                                              ↓
                                    Attention（加权查阅源序列）
                                              ↓
                                Self-Attention（序列内部互查）
                                              ↓
                     Multi-Head Attention（多个不同关注模式并行）
                                              ↓
                      Transformer（扔掉 RNN，纯 Attention 堆叠）
                                              ↓
                    BERT（Encoder） / GPT（Decoder）→ 大模型微调
```
