# RNN 结构衍生：四种输入-输出模式

## 📌 核心问题

> RNN 的核心优势是处理**序列**，但并非所有任务都是"序列进、序列出"。根据输入和输出的长度关系，RNN 衍生出四种经典结构，分别对应不同类型的深度学习任务。

---

## 🌱 根源：为什么需要不同结构？

RNN 的基础公式是时序展开：

$$h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b)$$

这个公式描述了**一个时间步**的计算。但实际任务中：
- 有时输入只有一个，需要扩展成序列输出（如看图写话）
- 有时输入很长，但只需要一个结论（如情感分析）
- 有时输入输出都要对齐（如词性标注）

**本质**：四种结构是对 RNN 时序维度的不同"接线方式"。

---

## 一、1 对 1（One-to-One）

### 结构图

```
x → [隐藏层] → y
```

### 特征
- **没有时序展开**，RNN 退化为标准前馈网络
- 输入是固定向量，输出也是固定向量

### 本质
这就是一个普通的全连接网络。把 RNN 的 $h_{t-1}$ 去掉就是它：

$$y = \phi(W x + b)$$

### 典型任务

| 任务 | 说明 |
|------|------|
| 图像分类 | 固定大小图片 → 类别标签 |
| 回归预测 | 固定特征 → 连续值 |

### 为什么放在 RNN 结构里讨论

因为它是序列长度 = 1 的退化情况，是理解其他模式的基准线。

---

## 二、1 对多（One-to-Many）

### 结构图

```
        y₁    y₂    y₃    y₄
        ↑     ↑     ↑     ↑
       [h₁]→[h₂]→[h₃]→[h₄]
        ↑
x → [编码]
```

或者等价地：

```
x → [h₀] → y₁
      ↓
     [h₁] → y₂
      ↓
     [h₂] → y₃
      ↓
     [h₃] → y₄
```

### 特征
- **单个输入**（可以是一个向量或一个编码后的上下文）
- **序列输出**，长度可变
- 解码器在 $t=0$ 时从输入 $x$ 获取初始状态，然后自回归生成

### 工作方式

1. 输入 $x$ 被编码为初始隐状态 $h_0$（或作为每个时间步的额外条件）
2. 从 $t=1$ 开始，$y_t$ 由 $h_{t-1}$ 生成
3. 上一时刻的输出 $y_{t-1}$ 作为下一时刻的输入（自回归）

$$h_t = f(h_{t-1}, y_{t-1}, x)$$
$$y_t = g(h_t)$$

### 典型任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **图像描述（Image Captioning）** | 一张图片 | "一只狗在草地上奔跑" |
| **音乐生成** | 风格/种子音符 | 完整旋律序列 |
| **文本续写（给定开头）** | 第一句话 | 续写段落 |
| **条件图像生成** | 文本 prompt | 图像 token 序列 |

### 关键技术点

- **Teacher Forcing**：训练时用真实 $y_{t-1}$ 而非生成的 $\hat{y}_{t-1}$
- **Exposure Bias**：训练和推理不一致 → 用 Scheduled Sampling 缓解
- **起始 token** `<SOS>` 和终止 token `<EOS>` 控制生成长度

---

## 三、多对 1（Many-to-One）

### 结构图

```
x₁ → [h₁]→[h₂]→[h₃]→[h₄]
                         ↓
                         y
```

### 特征
- **序列输入**，长度可变
- **单个输出**（向量/标量）
- 只取最后一个时间步的隐状态（或做全局池化）做预测

### 工作方式

1. 序列逐时间步输入
2. 只保留最后时刻的 $h_T$（或对所有 $h_t$ 做 pooling/attention）
3. $y = \text{Softmax}(W h_T + b)$

### 信息汇聚策略

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| 末位取 | $h_T$ | 序列有序，关键信息在后面 |
| 平均池化 | $\frac{1}{T}\sum h_t$ | 各时间步贡献均等 |
| 最大池化 | $\max_t h_t$ | 关键信息集中在少数位置 |
| 注意力汇聚 | $\sum \alpha_t h_t$ | 学习自动加权 |

### 典型任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **情感分析** | "这部电影太烂了" | 负面 |
| **文本分类** | 新闻正文 | 类别：体育/政治/科技 |
| **垃圾邮件检测** | 邮件全文 | spam / ham |
| **异常检测** | 时间序列 | 正常/异常 |
| **DNA 序列分类** | ATGC 序列 | 基因功能类别 |

### 代码示例

```python
import torch
import torch.nn as nn

class ManyToOneRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.rnn(embedded)
        # 取最后时刻隐状态（或 output[:, -1, :]）
        last_hidden = h_n.squeeze(0)           # (batch, hidden_dim)
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        return logits
```

---

## 四、多对多（Many-to-Many）

这一类型又分为两种子类：**等长对齐**和**不等长（Seq2Seq）**。

### 4.1 等长多对多（同步序列标注）

#### 结构图

```
x₁ → [h₁]→[h₂]→[h₃]→[h₄]
       ↓     ↓     ↓     ↓
      y₁    y₂    y₃    y₄
```

#### 特征
- 输入长度 = 输出长度，**一一对齐**
- 每个输入 token 对应一个输出标签
- 每个时间步都有预测

#### 工作方式

$$y_t = \text{Softmax}(W h_t + b)$$

每个 $h_t$ 独立预测对应位置的 $y_t$。但 $h_t$ 内部是双向循环的，已经看到过完整上下文。

#### 典型任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **词性标注（POS）** | "我/爱/北京" | 代词/动词/名词 |
| **命名实体识别（NER）** | "乔布斯/在/苹果/工作" | B-PER/O/B-ORG/O |
| **中文分词** | 字序列 | B/M/E/S 标签 |
| **帧级语音识别** | 声学特征序列 | 音素标签序列 |

#### 代码示例

```python
class ManyToManySync(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
    
    def forward(self, x):
        embedded = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, _ = self.rnn(embedded)             # (batch, seq_len, hidden_dim*2)
        logits = self.classifier(output)           # (batch, seq_len, num_tags)
        return logits
```

### 4.2 不等长多对多（Seq2Seq / Encoder-Decoder）

#### 结构图

```
x₁ → [h₁]→[h₂]→[h₃]→[h₄]     (编码器)
                         ↓
                      [c]      (上下文向量)
                         ↓
[ŷ₀]→[s₁]→[s₂]→[s₃]→[s₄]→[s₅] (解码器)
  ↓     ↓     ↓     ↓     ↓
 y₁    y₂    y₃    y₄    y₅
```

#### 特征
- 输入长度 $T$ ≠ 输出长度 $T'$
- 先编码 → 整个输入压缩为上下文向量 $c$ → 再解码输出
- 两个独立的 RNN（或 Transformer）

#### 工作方式

**编码阶段（多 → 1）** ：
$$h_t = \text{EncoderRNN}(x_t, h_{t-1})$$
$$c = h_T \quad \text{或} \quad c = \text{Attention}(\{h_t\}, s)$$

**解码阶段（1 → 多）** ：
$$s_{t'} = \text{DecoderRNN}(y_{t'-1}, s_{t'-1}, c)$$
$$P(y_{t'} | y_{<t'}, c) = \text{Softmax}(W s_{t'} + b)$$

#### 典型任务

| 任务 | 输入 | 输出 | 长度关系 |
|------|------|------|---------|
| **机器翻译** | "Hello world" | "你好世界" | 可能不等长 |
| **文本摘要** | 长文章 | 短摘要 | T >> T' |
| **问答生成** | 问题+上下文 | 答案文本 | 不等长 |
| **对话生成** | 对话历史 | 回复 | 不等长 |
| **语音识别** | 音频特征序列 | 文字序列 | T ≠ T' |

---

## 📊 四种结构的统一视角

```
                    输出维度
                 单点      序列
              ┌────────┬────────┐
输入    单点  │ 1 → 1  │ 1 → N  │
维度          │ 分类器  │ 生成器  │
              ├────────┼────────┤
       序列   │ N → 1  │ N → N  │
              │ 编码器  │ Seq2Seq │
              └────────┴────────┘
```

或者从信息流的角度：

| 结构 | 信息流 | 核心操作 | 代表架构 |
|------|--------|---------|---------|
| **1→1** | 点 → 点 | 映射 | MLP, CNN (图像分类) |
| **1→N** | 点 → 序列 | 展开/生成 | Decoder-only (GPT), VAE |
| **N→1** | 序列 → 点 | 汇聚/编码 | Encoder + Pooling, BERT [CLS] |
| **N→N** | 序列 → 序列 | 编码-解码 | Transformer, LSTM Seq2Seq |

---

## 🧠 Transformer 时代的对应关系

四种结构在 2017 年后的 Transformer 体系中同样存在：

| RNN 结构 | Transformer 等价 | 典型模型 |
|----------|-----------------|---------|
| 1→1 | 不适用（Transformer 专为序列设计） | ResNet, ViT [CLS] |
| 1→N | **Decoder-only**（自回归生成，起始于 prompt） | GPT 系列, LLaMA, Claude |
| N→1 | **Encoder-only**（全序列编码 + 汇聚） | BERT, RoBERTa |
| N→N（等长） | **Encoder-only**（每 token 输出） | BERT (token 分类), ViT |
| N→N（不等长） | **Encoder-Decoder** | T5, BART, 原始 Transformer |

### Decoder-only 的兴起（2023-2026）

2023 年后，Decoder-only 架构（GPT 系列）通过 **prompt 工程**将几乎所有任务都伪装成 1→N：

- 翻译：`"将以下英文翻译成中文: Hello world\n中文:"` → 自回归生成 `"你好世界"`
- 分类：`"以下是电影评论: 太烂了\n情感是:"` → 生成 `"负面"`
- 序列标注：`"提取人名: 乔布斯在苹果工作\n结果:"` → 生成 `"乔布斯"`

**本质上**，1→N 的生成模型通过 prompt 包装，可以模拟其他三种结构。

---

## 🔑 选择指南

| 你的任务 | 选哪种结构 | 为什么 |
|---------|-----------|--------|
| 输入输出都是定长向量 | 1→1（MLP） | 不需要时序建模 |
| 一个输入要生成序列 | 1→N（Decoder） | 输入作为初始条件，自回归展开 |
| 一个序列要输出一个结论 | N→1（Encoder） | 读完全文，做一次判断 |
| 序列中每个位置都要打标签 | N→N 同步 | 输入输出一一对齐 |
| 序列转成另一种序列（长度不同） | N→N Seq2Seq | 先理解再表达 |

---

## ⚠️ 易错点

1. **N→N 同步和 Seq2Seq 常被混淆**。关键区别：输入输出是否一一对齐？对齐就用同步 N→N（每个 $h_t$ 预测 $y_t$），不对齐就用 Encoder-Decoder。

2. **BERT 的 `[CLS]` token 就是 N→1 的体现**——整句编码后，只取 `[CLS]` 位置的输出做分类。

3. **GPT 的 prompt 不是 N→N，是 1→N**——prompt 虽然本身是一串 token，但作为"条件"，它一次性被编码（KV Cache），然后模型自回归生成。从信息流角度，这是条件化的 1→N。

4. **现代实践中结构界限模糊**——一个 GPT 模型可以同时完成分类（N→1 效果）、翻译（N→N 效果）、生成（1→N 效果），但这只是通过 prompt 实现的"伪装"，底层架构仍是 Decoder-only 的 1→N 自回归。

---

## 🔗 知识延伸

- [[05_Seq2Seq网络]]：不等长多对多的经典架构
- [[06_Attention网络]]：解决 Seq2Seq 中上下文瓶颈的关键机制
- [[14_BERT预训练与微调]]：N→1 和 N→N 同步在 Transformer 中的实现
- [[18_自编码器与机器翻译]]：N→N Seq2Seq 的完整实验
- [[LSTM为什么拆分输出与状态]]：理解 LSTM 隐状态设计对 N→1 最后时刻表示的影响

---

*创建日期：2026-06-09*
