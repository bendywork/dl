# Seq2Seq + Attention 逐行理解

> 基于 `04_Seq2Seq+Attention理解.py` 逐行拆解，从 embedding 到 QKV Attention，包含学习者的口语化理解。

---

## 1. nn.Embedding 本质：建表 vs 查表

### 建表（`__init__` 里）

```python
self.embed_layer = nn.Embedding(
    num_embeddings=vocab_size,   # token→id 映射表大小
    embedding_dim=hidden_size    # 每个 token 的特征向量维度
)
```

**学习者的理解纠错过程**：
- 最初以为"这是在做 torch.embedding 的工作，只不过把 embedding 的结果存到 self.embed_layer 里方便调用"
- 纠正后：`nn.Embedding` 就是 `torch.nn.Embedding` 本身（文件第 9 行 `import torch.nn as nn`），不存在单独的"torch.embedding 函数"
- **`self.embed_layer` 存的不是 embedding 结果，而是 embedding 层本身**——即那张 `[vocab_size, hidden_size]` 的可训练查表（lookup table）

### 查表（`forward` 里）

```python
token_embed = self.embed_layer(token_ids)   # [bs, et] → [bs, et, hidden_size]
```

这才是真正做 embedding 操作的地方——token id 当索引去查表，拿到对应的向量。**这也就是为什么要挂成 `self.xxx`：让子模块被 nn.Module 注册，参数才能被优化器收集、被 state_dict 保存、被 model.to(device) 迁移。**

---

## 2. EncoderModule 结构拆解

### 2.1 LSTM 层

```python
self.rnn_layer = nn.LSTM(
    input_size=hidden_size,    # = embedding 维度，两个层的"插座"必须对上
    hidden_size=hidden_size,   # 每个时刻输出维度
    num_layers=num_layers,
    batch_first=True,          # 输入输出第一维是 batch，后续所有 shape 推理建立在此之上
    bidirectional=False
)
```

### 2.2 上下文变换层

```python
self.ctx_feature_layer = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU()
)
```

**学习者的困惑与解答**："这里我没看懂到底是做什么的。"

- 输入是 RNN 吐出来的汇总状态向量 `[bs, hidden_size]`
- 作用：过一道 `Linear + ReLU` 的可学习非线性精加工
  - `Linear`：可训练的线性变换，学如何重新调配成分比例（有用放大、噪声压低）
  - `ReLU`：截断负值，引入非线性
- **类比**：RNN 状态 = 原矿石 → Linear = 重新调配成分 → ReLU = 丢弃不合格成分 → 产出 ctx_embed 成品
- 不是硬性必需步骤，可省，但加了通常更好（给网络一个机会把"RNN 内部状态"翻译成"更适配 decoder 的语义表示"）

### 2.3 forward 三步走

```python
def forward(self, token_ids):
    token_embed = self.embed_layer(token_ids)        # ① 查表
    output, state = self.rnn_layer(token_embed)       # ② LSTM 前向
    if isinstance(state, tuple):
        state = state[0] + state[1]                  # ③ LSTM二元组 → 融合
    state = torch.mean(state, dim=0)                  # ④ 多层压单层
    ctx_embed = self.ctx_feature_layer(state)         # ⑤ 精加工
    return output, ctx_embed                          # [bs,t,e]  [bs,e]
```

**两个返回值是 Seq2Seq+Attention 的关键分工**：

| 返回值 | shape | 含义 | 去向 |
|--------|-------|------|------|
| `output` | `[bs, et, e]` | 逐 token 特征序列 | Attention 的 **K 和 V** |
| `ctx_embed` | `[bs, e]` | 全局语义压缩向量 | Decoder LSTM 的**初始状态 h0/c0** |

---

## 3. LSTM 返回结构 & 如何追源码

### 问题：为什么知道 `self.rnn_layer(token_embed)` 返回 `(output, state)`？

**学习者的困惑**："我点进 `nn.LSTM` 的 `__init__` 只看到 `super().__init__('LSTM', ...)`，没看到 forward，怎么知道返回结构？"

### 继承链

```
nn.LSTM  →  nn.RNNBase  →  nn.Module
  (壳)        (forward)       (基类)
```

- `nn.LSTM` 没有自己的 forward → 向上找父类 → 命中 `RNNBase.forward`
- `RNNBase.forward` 内部按 `self.mode` 分发：LSTM 调 `torch._C._nn.lstm`，GRU 调 gru，RNN 调 rnn_tanh
- 返回结构由 `RNNBase.forward` 固定：**LSTM → `(output, (h_n, c_n))`**；GRU/RNN → `(output, h_n)`

### Python vs Java：为什么 IDE 推不准

**学习者的洞察**："所以 py 不像 Java 那样将返回直接在底层表示出来，所以无法直接使用 PyCharm 的 introduce 接受返回值，因为不确定你到底要啥，看文档是最准确的方式对吧？"

- 根本原因：Python 动态类型，函数签名不强制声明返回类型，PyTorch 底层库又没补精确的类型注解 → IDE 静态推断退化
- 不是"Python 做不到"，而是"库作者没写"；Python 3.5+ 支持返回类型注解
- **最可靠的方式：看官方文档 Outputs/Returns 段**（如 `torch.nn.LSTM` 文档明确写 `output, (h_n, c_n)`）
- 兜底：运行时 `print(type(state), state[0].shape)` 直接看真实返回

### `state = torch.mean(state, dim=0)` 在做什么

```python
state.shape   # [num_layers, bs, hidden_size]
              # dim=0 ← 消除 num_layers 维度（对所有层取均值）
```

**多层 LSTM 每层学到的特征粒度不同（浅层偏局部、深层偏全局），取均值 = 让所有层都说句话，合成一个声音**。

---

## 4. Attention 机制核心理解（学习者的视角）

### 4.1 为什么需要 Attention

**学习者的理解**（已纠正精确化）：

> "传统的 Seq2Seq 中，编码器压缩语义信息然后解码器根据最后的完整的语义信息做生成预测得到每一个词表的单独置信度，每一次生成都是依赖于原始文本 token 的完整向量，这就好比如使用同样一批流程对数据做加工处理。"

> "你怎么知道数据在生成过程中是不是都需要经过一个完整的流程？换句话说你怎么知道所有数据生成都需要依赖同样的压缩语义信息？语言表达是灵活的，是分场景的。"

> **"比如你用喜庆的文本范式去生成葬礼的致辞，这明显不对，因为场景不符合。"**

→ 类比：生成的目标文本应该根据全局语义做**不同权重参考**，而不是盲目参考同一个压缩向量。

### 4.2 精确化：Attention 跳过了 C（压缩语义瓶颈）

**关键纠正**：注意力权重**不是跟压缩语义 C 算的，而是跟编码器每个 token 的输出向量 `output` 算的**。

| 方式 | 用谁计算 | 瓶颈 |
|------|---------|------|
| 传统 Seq2Seq | 一个 C → decoder 所有步共用 | 长序列信息被挤没 |
| Seq2Seq + Attention | **output 序列每个位置** → 每步动态查 | 缓解瓶颈 |

结论：注意力不是"参考 C 的哪些部分"，注意力是"直接看源序列的每个位置"，跳过了 C 这个瓶颈。

### 4.3 QKV 映射

```
Q = decoder 当前状态     → "我现在要找什么信息"（查询向量）
K = encoder 每个 token 输出 → "源序列每个位置有什么信息"（用来跟 Q 匹配）
V = encoder 每个 token 输出 → "源序列每个位置的实际内容"（被加权的内容）
```

> K 和 V 都是 encoder 输出序列，但语义分工不同：K 负责"匹配"，V 负责"取值"。

### 4.4 Attention = "加权查阅"三步走

**学习者的总结（完全正确）**：

> "我在生成 token 的时候，先拿到当前输入 [decoder 状态当作 Q]，然后跟之前编码器阶段的所有的 token 逐个看 output 做相关性计算，然后得到权重之后再继续跟每一个 token 逐个输出 output 做权重相乘加权求和（权重越大的拿的数据越多，权重越小的拿的数据越少）。"

流程：

| 步骤 | 操作 | 含义 |
|------|------|------|
| ① 算相关性 | `Q · Kᵀ` | "当前要生成的内容和源序列每个位置有多相关？" |
| ② 转权重 | `÷ √e + softmax` | 相关度 → 概率分布（和为 1） |
| ③ 加权取 | `权重 · V` | 按权重拿源信息的加权和 → 汇总向量 |

每次都要做一次，这就是为什么 Attention 能"每一步都重新关注源序列的不同位置"。

### 4.5 Attention 为什么叫"注意力"

> 解码器每生成一个 token 时，直接把注意力放到源序列的不同位置，给每个位置不同的权重——这跟"人做翻译时每写一个目标词就回头看源句的对应位置"一模一样。

---

## 5. 关键代码文件索引

- 源文件：`/Users/scx/data/codeData/python/dl/stage03/06.Attention网络/02_Attention结构详解/04_Seq2Seq+Attention理解.py`
- `EncoderModule`（第 15–49 行）：encoder，产出 `output` 和 `ctx_embed`
- `qkv_attention_value`（第 51–73 行）：Attention 数学本体
- `attention_value`（第 75–111 行）：封装 Attention，兼容 LSTM/RNN/GRU
- `DecoderModule`（第 114–211 行）：decoder，含训练/推理两套 forward
- `Seq2SeqModule`（第 214–234 行）：组装 encoder + decoder
