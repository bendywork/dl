# NLP 文本生成演进：从 RNN 到 Transformer

> **一条主线**：如何让机器一个词一个词地"写"出通顺的文本？
> **递进逻辑**：每个方法都在解决上一个方法解决不了的问题。

---

## 📌 核心问题：文本生成的本质

文本生成 = **给定已有上下文，预测下一个词，重复此过程直到结束**。

```
输入: "今天天气"
Step 1: "今天天气" → 预测 "很"    → "今天天气很"
Step 2: "今天天气很" → 预测 "好"  → "今天天气很好"
Step 3: "今天天气很好" → 预测 "，" → "今天天气很好，"
Step 4: "今天天气很好，" → 预测 "<EOS>" → 结束
```

数学表达（自回归）：

$$P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t | w_1, w_2, ..., w_{t-1})$$

**每一步都是：把已知的上下文"编码"成某种表示，再用这个表示在词表上选词。**

不同的生成模型，区别就在于：
1. **怎么编码上下文**（RNN 用隐藏状态 / Transformer 用自注意力）
2. **编码能看多远**（RNN 受限 / Transformer 全局）
3. **计算效率如何**（RNN 串行 / Transformer 并行）

---


## 1️⃣ RNN：序列建模的开端

### 1.1 要解决什么问题

在 RNN 之前，处理文本的方法（词袋、N-gram）都有一个致命缺陷：**无法处理任意长度的序列**。

- 词袋模型：丢掉了词序信息（"狗咬人" vs "人咬狗"无法区分）
- N-gram：只能看前 N 个词（N=3 时，"今天天气很好所以我们决定" 只看 "决定" 前两个词）

**RNN 的核心创新**：引入一个**隐藏状态 h**，像"记忆"一样，把之前所有词的信息压缩进去。

### 1.2 核心原理

```
时间步 t=1:  h₁ = tanh(W_hh · h₀ + W_xh · x₁ + b)
时间步 t=2:  h₂ = tanh(W_hh · h₁ + W_xh · x₂ + b)
时间步 t=3:  h₃ = tanh(W_hh · h₂ + W_xh · x₃ + b)
             ...
```

展开来看：

```
x₁ ──→ [RNN] ──→ h₁ ──→ [RNN] ──→ h₂ ──→ [RNN] ──→ h₃ ──→ ...
         ↑                 ↑                 ↑
        h₀=0             从h₁传来的"记忆"   从h₂传来的"记忆"
```

**关键**：所有时间步**共享同一组参数**（W_hh, W_xh, b），但隐藏状态 h 在时间步之间传递，形成"记忆"。

通用公式：

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)$$

### 1.3 RNN 如何做文本生成

```
训练阶段（Teacher Forcing）：
  输入: [BOS] 我 爱 中 国
  目标: 我 爱 中 国 [EOS]
  每步: h_t → 预测下一个词 → 和真实词算交叉熵损失

生成阶段（自回归）：
  Step 0: 输入 [BOS] → h₁ → 在词表上选概率最高的词 → "我"
  Step 1: 输入 "我"   → h₂ → 选词 → "爱"
  Step 2: 输入 "爱"   → h₃ → 选词 → "中"
  Step 3: 输入 "中"   → h₄ → 选词 → "国"
  Step 4: 输入 "国"   → h₅ → 选词 → [EOS] → 结束
```

### 1.4 RNN 的致命局限：梯度消失

**问题**：当序列很长时，h_t 对 h₁ 的梯度会指数级衰减。

数学原因：

$$\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{t} W_{hh} \cdot \text{diag}(\tanh'(h_{k-1}))$$

- $\tanh'$ 的最大值是 1，实际通常 < 1
- $W_{hh}$ 的特征值如果 < 1，连乘后趋近于 0
- 20 步之后，梯度几乎为 0 → **远处的信息传不过来**

直觉理解：

```
想象一个传话游戏：
第1个人: "明天下午3点在人民广场开会"
第5个人: "明天下午...在...开会"     ← 部分信息丢失
第20个人: "好像...有什么事..."       ← 几乎全忘了
```

**后果**：RNN 无法生成长文本——生成到后面就"忘记"了前面的内容，导致文本不连贯。

---


## 2️⃣ LSTM：长程记忆的突破

### 2.1 要解决什么问题

RNN 的核心缺陷是**梯度消失**——长序列中，远处的信息传不到当前步。

**LSTM 的思路**：如果 RNN 的记忆像"口头传话"（每传一次丢一点），那 LSTM 就引入"笔记本"——把重要信息直接**写下来**，需要时**读出来**，不需要时**擦掉**。

### 2.2 核心原理：三门一细胞

LSTM 在 RNN 的 h_t 基础上，引入了**细胞状态 C_t**（长期记忆）和三个**门**：

```
RNN 只有:  h_t = tanh(W·[h_{t-1}, x_t] + b)         ← 一条记忆线

LSTM 有:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t        ← 长期记忆（细胞状态）
           h_t = o_t ⊙ tanh(C_t)                    ← 短期输出（隐藏状态）
```

#### 遗忘门（Forget Gate）：决定从长期记忆中**忘掉什么**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- 输出 0~1 之间的值，0 = 完全忘记，1 = 完全保留
- 类比：翻笔记本，划掉不再需要的信息

#### 输入门（Input Gate）：决定往长期记忆中**写入什么新信息**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$：写不写（0=不写，1=写）
- $\tilde{C}_t$：写什么（候选新信息）

#### 输出门（Output Gate）：决定从长期记忆中**读出什么作为当前输出**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

- $o_t$：读不读（0=不输出，1=输出）
- $\tanh(C_t)$：把细胞状态压缩到 [-1,1] 便于输出

### 2.3 为什么 LSTM 能解决梯度消失

**关键**：细胞状态的更新是**加法**，不是乘法：

$$C_t = \underbrace{f_t \odot C_{t-1}}_{\text{保留旧记忆}} + \underbrace{i_t \odot \tilde{C}_t}_{\text{加入新记忆}}$$

梯度沿 $C_t$ 传播时：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

**没有连乘 W_hh！** 只要 $f_t$ 接近 1，梯度就可以几乎无损地传播很远。这就是"高速公路"——梯度直接"驶过"多个时间步，不需要反复乘以权重矩阵。

```
RNN 的梯度路径:  ∂L/∂h₁ = ... × W × tanh' × W × tanh' × W × tanh'  ← 连乘衰减
LSTM 的梯度路径: ∂L/∂C₁ = ... × f₃ × f₂ × f₁                     ← 只乘遗忘门
                  (fₜ≈1 时梯度几乎不衰减)
```

### 2.4 LSTM 如何做文本生成

与 RNN 结构相同，只是内部用 LSTM 单元替代简单 RNN 单元：

```python
# 生成循环
h, c = (zeros, zeros)  # 初始化隐藏状态和细胞状态
input_token = [BOS]

for step in range(max_len):
    output, (h, c) = lstm(input_token, (h, c))
    next_token = softmax(lm_head(output))  # 在词表上选词
    if next_token == [EOS]:
        break
    input_token = next_token
    generated.append(next_token)
```

**LSTM 生成的优势**：可以记住 100+ 步之前的信息，生成更连贯的长文本。

**LSTM 生成的局限**：仍然是**串行**的——每步必须等上一步算完，无法并行训练。

---


## 3️⃣ GRU：LSTM 的轻量替代

### 3.1 要解决什么问题

LSTM 效果好，但**参数太多**（每个时间步 4 组权重矩阵），计算开销大。

**GRU 的思路**：LSTM 的遗忘门和输入门功能有重叠——"忘掉旧的"和"写入新的"往往是联动的。能不能合并？

### 3.2 核心原理：两门合一状态

GRU 把 LSTM 的 3 个门简化为 2 个门，把细胞状态 C 和隐藏状态 h 合并为一个 h：

```
LSTM: C_t（长期） + h_t（短期） + 3门（遗忘/输入/输出）
GRU:  h_t（统一） + 2门（重置/更新）
```

#### 重置门（Reset Gate）：决定**忽略多少过去信息**（用于计算候选状态）

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

- $r_t ≈ 0$：忽略过去，"从零开始"思考 → 适合处理全新的话题
- $r_t ≈ 1$：保留过去，在旧记忆基础上更新 → 适合延续当前话题

#### 更新门（Update Gate）：决定**保留多少旧状态 vs 采用多少新状态**

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

**这是 LSTM 遗忘门 + 输入门的融合**：

| LSTM | GRU | 对应关系 |
|------|-----|---------|
| $f_t$（遗忘门） | $z_t$（更新门） | $z_t$ 扮演 $f_t$ 的角色：控制保留旧信息 |
| $i_t$（输入门） | $1 - z_t$ | LSTM 中 $f_t + i_t$ 不一定为1，GRU 中强制 $z + (1-z) = 1$ |
| $o_t$（输出门） | 无 | GRU 没有单独的输出门，h_t 直接输出 |

### 3.3 GRU vs LSTM 对比

| 维度 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3（遗忘/输入/输出） | 2（重置/更新） |
| 状态数量 | 2（C_t + h_t） | 1（h_t） |
| 参数量 | 4 × (d² + d×emb) | 3 × (d² + d×emb) |
| 训练速度 | 较慢 | 快约 15-20% |
| 长序列效果 | 略优 | 接近（差距很小） |
| 短序列效果 | 相当 | 相当 |
| 工程选择 | 需要极致长记忆时 | 追求效率时 |

**经验法则**：数据量大 → LSTM（更强的表达能力）；数据量小/需要速度 → GRU（更少参数，不易过拟合）。

### 3.4 GRU 文本生成

与 LSTM 完全相同的自回归流程，只是内部计算更简洁：

```python
h = zeros
input_token = [BOS]

for step in range(max_len):
    output, h = gru(input_token, h)  # 注意：只有一个状态 h
    next_token = softmax(lm_head(output))
    if next_token == [EOS]:
        break
    input_token = next_token
    generated.append(next_token)
```

---


## 4️⃣ Attention：打破序列瓶颈

### 4.1 要解决什么问题

RNN/LSTM/GRU 都有一个根本性的结构缺陷：**编码瓶颈**。

以翻译任务为例（Seq2Seq 架构）：

```
编码器: "我" → "爱" → "中" → "国" → 最终 h₄（一个固定长度的向量）
                                                         ↓
解码器: h₄ → "I" → "love" → "China" → <EOS>
```

**问题**：不管输入多长，全部信息都压缩到最后一个 h₄ 里。10 个词还能勉强，50 个词呢？100 个词呢？——这就是**编码瓶颈**。

### 4.2 Attention 的核心思想

**不用把所有信息塞进一个向量，而是让解码器每一步都能"回头看"输入序列的所有位置，自动决定该关注哪里。**

```
不用 Attention:
  解码器只能看 → [h₄] → 一个向量承载所有信息

用 Attention:
  解码器每步都能看 → [h₁, h₂, h₃, h₄] → 动态选择关注哪些
                     ↑    ↑    ↑    ↑
                   "我" "爱" "中" "国" 的编码
```

### 4.3 Attention 的计算过程（逐步详解）

假设解码器当前隐藏状态为 $s_t$，编码器输出为 $h_1, h_2, ..., h_n$。

**Step 1：计算注意力分数**——解码器状态 s_t 和每个编码器状态 h_i 的"匹配度"

$$e_{ti} = \text{score}(s_t, h_i)$$

常见打分方式：

| 类型 | 公式 | 来源 |
|------|------|------|
| 加性（Bahdanau） | $v^T \tanh(W_1 s_t + W_2 h_i)$ | 最早，可学习 |
| 乘性（Luong） | $s_t^T W h_i$ | 更高效 |
| 点积 | $s_t^T h_i$ | 最简单，要求维度一致 |

**Step 2：Softmax 归一化**——把分数变成概率分布

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{n} \exp(e_{tk})}$$

- $\sum_{i} \alpha_{ti} = 1$，所有注意力权重加起来等于 1
- $\alpha_{ti}$ 越大，表示第 t 步越关注第 i 个输入

**Step 3：加权求和**——按注意力权重混合编码器输出

$$c_t = \sum_{i=1}^{n} \alpha_{ti} \cdot h_i$$

**Step 4：用于生成**——把上下文向量和解码器状态拼接，预测下一个词

$$\hat{s}_t = \tanh(W_c [c_t; s_t])$$
$$P(w_{t+1}) = \text{softmax}(W_{out} \hat{s}_t)$$

### 4.4 Attention 解决了什么

| 问题 | RNN/LSTM/GRU | + Attention |
|------|-------------|------------|
| 编码瓶颈 | 所有信息压缩到一个向量 | 每步动态检索所有编码器状态 |
| 长序列丢失 | 远处信息传不到 | 直接"跳"到任意位置，无需传递 |
| 对齐问题 | 翻译时无法对应源词 | 注意力权重天然形成源-目标对齐 |
| 可解释性 | 黑盒 | 注意力权重可视化 = 模型在看哪里 |

### 4.5 Attention 的局限

Attention 是在 **RNN 之上**添加的机制，底层仍然是 RNN，所以：

1. **训练仍无法并行**：RNN 的隐藏状态必须串行计算
2. **计算距离仍受限**：Attention 解决了编码器的信息瓶颈，但解码器本身仍是串行 RNN
3. **注意力是"附加"而非"基础"**：Attention 只是 RNN 的辅助，RNN 才是骨架

→ 这引出了 Transformer：**让 Attention 成为主体，彻底抛弃 RNN。**

---


## 5️⃣ Transformer：并行化革命

### 5.1 要解决什么问题

RNN + Attention 的组合虽然有效，但 **RNN 的串行性是根本瓶颈**：

```
RNN: x₁ → h₁ → h₂ → h₃ → ... → h_T    必须一步一步算，无法并行
     时间复杂度: O(T)  串行步骤

     假设 T=1000，GPU 并行能力再强也帮不了——
     因为第 1000 步必须等第 999 步算完。
```

**Transformer 的核心颠覆**：**完全不用 RNN，只用 Attention。**

### 5.2 核心原理：自注意力（Self-Attention）

RNN 的 Attention 是"解码器看编码器"（Cross-Attention）。
Transformer 的 Self-Attention 是"序列内部每个位置看其他所有位置"。

#### QKV 框架

每个词的嵌入向量 x 被投影为三个角色：

| 角色 | 含义 | 类比 |
|------|------|------|
| **Q**（Query）| "我在找什么" | 搜索关键词 |
| **K**（Key）  | "我能提供什么" | 文档标签 |
| **V**（Value）| "我的实际内容" | 文档正文 |

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

#### 注意力计算

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

逐步拆解：

```
1. QKᵀ：每个位置和所有位置计算"匹配度"     → [T×T] 注意力分数矩阵
2. /√dₖ：缩放，防止点积过大导致 softmax 饱和  → 数值稳定性
3. softmax：分数变概率（每行和为1）           → [T×T] 注意力权重矩阵
4. × V：按权重加权求和                       → [T×d_v] 输出矩阵
```

**关键优势**：**O(1) 步就能看到任意远的位置**！

```
RNN: 第 1 个词的信息 → 传 99 步 → 到达第 100 个词（可能已衰减为零）
Self-Attention: 第 1 个词和第 100 个词直接计算 QKᵀ，一步到位
```

### 5.3 多头注意力（Multi-Head Attention）

为什么要多头？一个注意力头只能学一种"关注模式"。

```
头1 可能学: 语法关系（主语→动词）
头2 可能学: 共指关系（代词→实体）
头3 可能学: 语义相似（同义词互相关注）
头4 可能学: 位置邻近（相邻词关注）
...
```

数学：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

$$\text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)$$

- 每个头的维度：$d_k = d_{model} / h$
- 8 头 × 64 维 = 512 维（和单头一样参数量，但信息更丰富）

### 5.4 位置编码（Positional Encoding）

**Self-Attention 的问题**：它是"位置无关"的——"我爱你"和"你爱我"对 Self-Attention 来说没有本质区别，因为它是集合操作而非序列操作。

**解决方案**：给每个位置加一个"位置标签"：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- pos：位置编号（0, 1, 2, ...）
- 2i / 2i+1：嵌入向量的维度编号

**为什么用正弦/余弦**：
- 每个维度对应不同频率，低维高频（区分邻近位置），高维低频（区分远距离位置）
- 对于任意固定偏移 k，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数 → 模型可以学习相对位置

### 5.5 Transformer 完整架构（文本生成用 Decoder-Only）

现代文本生成模型（GPT 系列）使用 **Decoder-Only Transformer**：

```
输入: w₁ w₂ w₃ ... wₜ
  ↓ 词嵌入 + 位置编码
  ↓
┌─────────────────────────┐
│  Block 1                │
│  ├─ Masked Multi-Head  │ ← 因果掩码：第 t 步只能看 ≤t 的位置
│  ├─ Add & Norm         │
│  ├─ FFN                │ ← 两层全连接：d → 4d → d
│  └─ Add & Norm         │
└─────────────────────────┘
  ↓ （重复 N 个 Block，GPT-2 有 12-48 层）
  ↓
  lm_head (线性层) → logits → softmax → 预测下一个词
```

#### 因果掩码（Causal Mask）——生成的关键约束

```
Self-Attention 完整矩阵（4个词）:
      w₁  w₂  w₃  w₄
  w₁ [ ✓   ✗   ✗   ✗ ]   w₁ 只能看自己
  w₂ [ ✓   ✓   ✗   ✗ ]   w₂ 能看 w₁, w₂
  w₃ [ ✓   ✓   ✓   ✗ ]   w₃ 能看 w₁, w₂, w₃
  w₄ [ ✓   ✓   ✓   ✓ ]   w₄ 能看所有

  ✗ 位置填 -inf，softmax 后变 0 → 完全屏蔽未来信息
```

**为什么需要掩码**：生成时，模型不能"偷看"下一个词——这是自回归的基本约束。

### 5.6 Transformer 如何做文本生成

```python
# 伪代码：GPT 式自回归生成
def generate(prompt, max_len=100):
    tokens = tokenize(prompt)  # [w₁, w₂, ..., wₖ]

    for _ in range(max_len):
        # 1. 所有 token 并行过 Transformer
        logits = transformer(tokens)  # [k, vocab_size]

        # 2. 只取最后一个位置的预测
        next_logits = logits[-1]      # [vocab_size]

        # 3. 采样下一个词
        next_token = sample(next_logits, temperature=0.8)
        tokens.append(next_token)

        if next_token == EOS:
            break

    return detokenize(tokens)
```

**与 RNN 生成流程的关键区别**：

| | RNN | Transformer |
|---|---|---|
| 每步输入 | 只有上一个词 | 整个已生成序列 |
| 上下文编码 | 串行，从左到右 | 并行，一次性全部编码 |
| 训练并行性 | ❌ 不可并行 | ✅ 可并行（掩码保证自回归） |
| 生成并行性 | ❌ 仍然串行 | ❌ 仍然串行（自回归本质） |
| 训练速度 | 慢 | 快（GPU 并行优势大） |

> **注意**：生成阶段 Transformer 也是串行的（每步生成一个词），但**每步内部的 Self-Attention 是并行计算所有已有位置的**，而且训练时可以并行处理整个序列——这是 Transformer 的核心优势。

---


## 🔄 递进关系总结：一图看懂演进

```
RNN (1986)
 │  问题: 梯度消失，无法记住长距离依赖
 │  生成: 每步看一个词 + 上一步隐藏状态 → 预测下一个词
 │
 ├─→ LSTM (1997)
 │    │  解决: 引入细胞状态 + 三门，梯度可长程传播
 │    │  生成: 同 RNN，但能记住 100+ 步前的信息
 │    │  遗留: 仍然串行，编码瓶颈（所有信息压缩到一个 h）
 │    │
 │    ├─→ GRU (2014)
 │    │    │  解决: 简化 LSTM，参数更少，速度更快
 │    │    │  生成: 同 LSTM，几乎等价效果
 │    │    │  遗留: 串行性、编码瓶颈仍未解决
 │    │    │
 │    │    └─→ (GRU 并非 LSTM 的替代，而是轻量平替)
 │    │
 │    └─→ + Attention (2014, Bahdanau)
 │         │  解决: 编码瓶颈——解码器每步可动态关注编码器任意位置
 │         │  生成: 翻译质量大幅提升，注意力权重可视化
 │         │  遗留: 底层仍是 RNN → 串行训练，速度慢
 │         │
 │         └─→ Transformer (2017, Vaswani)
 │              │  解决: 抛弃 RNN，纯 Attention → 训练可并行
 │              │  生成: 自注意力一步看到所有位置，训练速度 ×10+
 │              │  进化: GPT 系列、BERT、所有现代大语言模型
 │              │
 │              └─→ GPT (2018→) / LLaMA / Claude / ...
```

### 每一代解决了什么、留下了什么

| 模型 | 核心突破 | 解决的问题 | 遗留的问题 |
|------|---------|-----------|-----------|
| RNN | 递归记忆 | 如何处理变长序列 | 梯度消失、长程遗忘 |
| LSTM | 门控+细胞状态 | 梯度消失、长程依赖 | 编码瓶颈、串行训练 |
| GRU | 两门简化 | LSTM 参数过多 | 同 LSTM 的遗留问题 |
| Attention | 动态加权检索 | 编码瓶颈、信息丢失 | 仍基于 RNN，串行训练 |
| Transformer | 纯自注意力 | 串行训练、并行化 | 生成仍串行、O(n²) 复杂度 |

### 五代模型的生成能力对比

以生成长度为 50 的连贯文本为例：

```
RNN:       前10个词连贯，之后逐渐"失忆"，输出混乱     ★★☆☆☆
LSTM:      前30个词连贯，超长文本仍有不一致            ★★★☆☆
GRU:       接近LSTM，略快但长文本略弱                  ★★★☆☆
+Attention: 翻译质量高，能关注到对应源词位置           ★★★★☆
Transformer: 全局一致性，可生成数千词的连贯文本        ★★★★★
```

---


## ⚠️ 易错点与常见误解

### 1. ❌ "LSTM 完全解决了梯度消失"
**纠正**：LSTM 大幅缓解了梯度消失，但没有完全消除。当遗忘门 $f_t$ 持续 < 1 时，梯度仍会指数衰减。只是 $f_t$ **可以学到接近 1**，不像 RNN 被固定权重连乘。所以叫"缓解"更准确。

### 2. ❌ "GRU 比 LSTM 好 / LSTM 比 GRU 好"
**纠正**：没有绝对优劣。GRU 参数少、速度快，小数据集上不易过拟合；LSTM 参数多、表达力强，大数据集长序列上略优。实际选择取决于任务、数据量、速度需求。

### 3. ❌ "Attention 就是 Transformer"
**纠正**：Attention 是一种机制（2014 年 Bahdanau 提出），Transformer 是一种架构（2017 年 Vaswani 提出）。Transformer 用了 Self-Attention，但还有位置编码、残差连接、层归一化、FFN 等组件。Attention ≠ Transformer。

### 4. ❌ "Transformer 生成也是并行的"
**纠正**：Transformer 的**训练**可以并行（一次处理整个序列），但**生成**仍然是串行的（自回归：每步生成一个词，添加到序列，再生成下一个）。生成阶段的加速主要来自 KV Cache（缓存已计算的 Key/Value，避免重复计算）。

### 5. ❌ "Self-Attention 每步都能看到所有信息，所以没有信息丢失"
**纠正**：Self-Attention 没有信息传递丢失的问题（不像 RNN 的梯度消失），但有**信息容量**的限制——所有位置的信息要压缩到一个固定维度的向量中，当序列极长时（>8K tokens），中间位置的信息可能被"稀释"。这也是长上下文窗口的研究热点。

---


## 📚 参考资料

### 经典论文

1. **Elman (1990)** — *"Finding Structure in Time"*
   简单 RNN（SRN/Elman Network）的奠基论文。

2. **Hochreiter & Schmidhuber (1997)** — *"Long Short-Term Memory"*
   LSTM 的原始论文，解决了 RNN 的梯度消失问题。

3. **Cho et al. (2014)** — *"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"*
   GRU 的提出论文。

4. **Bahdanau et al. (2014)** — *"Neural Machine Translation by Jointly Learning to Align and Translate"*
   Attention 机制首次引入 Seq2Seq，打破编码瓶颈。

5. **Vaswani et al. (2017)** — *"Attention Is All You Need"*
   Transformer 的原始论文，彻底抛弃 RNN，纯 Attention 架构。

6. **Radford et al. (2018)** — *"Improving Language Understanding by Generative Pre-Training"*
   GPT-1，Decoder-Only Transformer 用于文本生成的开端。

### 教程与资源

- Karpathy: *"The Unreasonable Effectiveness of Recurrent Neural Networks"* (2015)
- Lilian Weng 博客: *"Attention? Attention!"*
- Jay Alammar: *"The Illustrated Transformer"*
- Harvard NLP: *"The Annotated Transformer"*

### 动画演示

本文档配套 5 个 HTML 动画，与本文同目录：

| 文件 | 演示内容 |
|------|---------|
| `26_RNN文本生成动画.html` | RNN 隐藏状态传递 + 梯度消失可视化 |
| `26_LSTM文本生成动画.html` | LSTM 三门一细胞的信息流 |
| `26_GRU文本生成动画.html` | GRU 两门机制对比 LSTM |
| `26_Attention机制动画.html` | QKV 计算过程 + 注意力权重热力图 |
| `26_Transformer生成动画.html` | Self-Attention + 因果掩码 + 多头注意力 |

---

> 📝 **创建日期：2026-06-09**
> 📂 **所属领域：NLP / 文本生成 / 序列模型**
> 🔗 **递进链路：[[RNN]] → [[LSTM]] → [[GRU]] → [[Attention]] → [[Transformer]]**
