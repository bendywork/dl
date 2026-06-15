# ELMo 上下文词向量

## 📌 核心问题

> Word2Vec / GloVe 给每个词分配**一个固定向量**，无法区分"苹果"在"我吃苹果"和"苹果发布了新 iPhone"中的不同含义。ELMo（Embeddings from Language Models）的核心目标是：**同一个词在不同上下文中得到不同的向量表示**。

---

## 🌱 根源与动机

### 静态词向量的致命缺陷

| 方法 | 类型 | 问题 |
|------|------|------|
| Word2Vec | 静态 | "bank"（银行/河岸）只有一个向量 |
| GloVe | 静态 | 同上，无法处理多义词 |
| ELMo | 动态 | 根据上下文动态生成词向量 |

**多义词示例：**
- "I went to the **bank** to deposit money."（金融机构）
- "I sat on the river **bank**."（河岸）

静态向量：`bank → [0.2, -0.5, 0.8, ...]`（始终相同）

ELMo 向量：`bank_金融 → [0.9, -0.1, ...]`，`bank_河岸 → [-0.3, 0.7, ...]`（根据上下文变化）

### ELMo 的提出背景

- **2018年，Allen Institute for AI**（AllenNLP 团队）发表论文《Deep contextualized word representations》
- 核心创新：用**深层双向语言模型**的所有隐藏层输出来表示词向量，而不只是最后一层

---

## 📐 理论推导

### 1. 双向语言模型目标函数

ELMo 基于**双向 LSTM 语言模型**：

**前向语言模型（Forward LM）：**

$$\mathcal{L}_{forward} = \sum_{k=1}^{N} \log p(t_k \mid t_1, t_2, \ldots, t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s)$$

**后向语言模型（Backward LM）：**

$$\mathcal{L}_{backward} = \sum_{k=1}^{N} \log p(t_k \mid t_{k+1}, t_{k+2}, \ldots, t_N; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s)$$

**联合训练目标（最大化）：**

$$\mathcal{L} = \sum_{k=1}^{N} \left( \log p(t_k \mid t_1, \ldots, t_{k-1}) + \log p(t_k \mid t_{k+1}, \ldots, t_N) \right)$$

其中：
- $t_k$：第 $k$ 个 token
- $\Theta_x$：词嵌入参数（前向后向共享）
- $\Theta_s$：Softmax 层参数（前向后向共享）
- $\overrightarrow{\Theta}_{LSTM}$、$\overleftarrow{\Theta}_{LSTM}$：各自独立的 LSTM 参数

### 2. ELMo 向量的三层结构

对于一个 L 层的双向语言模型，第 $k$ 个 token 有 $2L+1$ 个表示：

$$R_k = \{x_k^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} \mid j=1,\ldots,L\}$$

具体地，对于 L=2 的 ELMo：

| 层级 | 表示 | 含义 |
|------|------|------|
| 第0层 | $h_{k,0}^{LM}$（词嵌入层） | 字符卷积或词嵌入，表达**词法信息** |
| 第1层 | $h_{k,1}^{LM}$（LSTM第1层） | 低级特征，表达**语法信息** |
| 第2层 | $h_{k,2}^{LM}$（LSTM第2层） | 高级特征，表达**语义信息** |

**ELMo 向量计算公式：**

$$\text{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \cdot h_{k,j}^{LM}$$

其中：
- $\gamma^{task}$：全局缩放标量，允许整体缩放 ELMo 向量的幅度
- $s_j^{task}$：第 $j$ 层的权重（softmax 归一化，$\sum_j s_j = 1$），**针对不同下游任务学习**
- $h_{k,j}^{LM}$：第 $j$ 层对第 $k$ 个 token 的双向表示（前向+后向拼接）：$h_{k,j}^{LM} = [\overrightarrow{h}_{k,j}^{LM}; \overleftarrow{h}_{k,j}^{LM}]$

**关键洞察**：不同任务需要不同层的信息：
- **词性标注、句法分析** → 第1层权重更大（语法信息）
- **词义消歧、语义角色标注** → 第2层权重更大（语义信息）

### 3. 架构图示

```
输入句子：The bank can guarantee deposits will eventually cover future tuition costs.

字符 CNN/词嵌入
       ↓
[h_0: 词嵌入层]     "bank" → embedding vector

前向 LSTM 层1 →→→→→→→→→→→→→
后向 LSTM 层1 ←←←←←←←←←←←←←
[h_1: 拼接前向+后向]  "bank" → [→h1; ←h1]

前向 LSTM 层2 →→→→→→→→→→→→→
后向 LSTM 层2 ←←←←←←←←←←←←←
[h_2: 拼接前向+后向]  "bank" → [→h2; ←h2]

ELMo = γ * (s0*h0 + s1*h1 + s2*h2)
```

---

## 💡 关键理解

### 类比：专家团队投票

想象有3位专家（词法专家、语法专家、语义专家），每人给出对这个词的理解。最终结论是三位专家的**加权投票**，而权重取决于当前任务更看重哪位专家的意见。

- 情感分析任务 → 语义专家（第2层）权重最高
- 命名实体识别 → 词法+语法专家（第0+1层）权重更高

### Feature-based vs Fine-tuning

**ELMo 是 Feature-based 方法（特征提取）：**

```
预训练模型（冻结参数）
       ↓
提取 ELMo 特征（固定的上下文向量）
       ↓
下游任务模型（用提取的特征作为输入）
       ↓
只训练下游任务模型的参数
```

**BERT 是 Fine-tuning 方法（微调）：**

```
预训练模型（参数可更新）
       ↓
在下游任务数据上继续训练，更新全部参数
       ↓
端到端梯度回传
```

本质区别：
- ELMo 预训练模型参数**冻结**，只学层权重 $s_j$ 和 $\gamma$
- BERT 预训练模型参数**全部参与微调**

---

## 🔧 代码实现

### 方案1：使用 allennlp 官方库

```python
# pip install allennlp allennlp-models
# 注意：以下为 allennlp 早期版本 API 示意

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

# 加载预训练 ELMo 模型（需要下载权重文件）
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# num_output_representations: 需要几种加权组合（不同任务可以要不同组合）
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

# 演示：同一个词 "bank" 在不同上下文中的向量差异
sentences = [
    ["I", "went", "to", "the", "bank", "to", "deposit", "money"],   # 金融银行
    ["She", "sat", "on", "the", "bank", "of", "the", "river"],       # 河岸
]

# 将句子转为字符级 ID（ELMo 输入是字符，不是词）
character_ids = batch_to_ids(sentences)  # shape: [batch=2, max_len, max_char=50]

# 前向传播
with torch.no_grad():
    embeddings = elmo(character_ids)

# 获取 ELMo 向量
elmo_vecs = embeddings['elmo_representations'][0]  # shape: [2, max_len, 1024]

# "bank" 在两个句子中的位置都是索引4
bank_finance = elmo_vecs[0, 4, :]   # 金融语境下的 bank
bank_river   = elmo_vecs[1, 4, :]   # 河岸语境下的 bank

# 计算余弦相似度
cos_sim = torch.nn.functional.cosine_similarity(
    bank_finance.unsqueeze(0),
    bank_river.unsqueeze(0)
)
print(f"同一词'bank'在不同上下文的余弦相似度: {cos_sim.item():.4f}")
# 预期输出：远小于1（如 0.3~0.6），说明上下文确实影响了向量
```

### 方案2：手动实现双向 LSTM ELMo（简化版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleBiLSTMELMo(nn.Module):
    """
    简化版 ELMo：展示核心思想
    - 双向 LSTM 2层
    - 三层向量（词嵌入 + LSTM层1 + LSTM层2）加权求和
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层（第0层）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 双向 LSTM（每层独立存储，方便提取中间层）
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=embed_dim if i == 0 else hidden_dim * 2,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            for i in range(num_layers)
        ])
        
        # 投影层：统一各层维度（词嵌入层投影到 hidden_dim*2）
        self.embed_proj = nn.Linear(embed_dim, hidden_dim * 2)
    
    def forward(self, x):
        """
        x: [batch, seq_len] token ids
        返回: [batch, seq_len, hidden_dim*2] ELMo 向量（三层加权后）
             以及三层各自的输出（供任务层学习权重）
        """
        batch, seq_len = x.shape
        
        # 第0层：词嵌入
        h0 = self.embedding(x)                   # [B, T, embed_dim]
        h0_proj = self.embed_proj(h0)             # [B, T, hidden_dim*2]
        
        layer_outputs = [h0_proj]
        
        # 第1、2层：双向 LSTM
        current_input = h0
        for lstm in self.lstm_layers:
            output, _ = lstm(current_input)       # [B, T, hidden_dim*2]
            layer_outputs.append(output)
            current_input = output
        
        return layer_outputs  # [h0_proj, h1, h2]，每个 [B, T, hidden_dim*2]


class ELMoTaskHead(nn.Module):
    """
    下游任务使用 ELMo 特征的头部
    - ELMo 的 BiLSTM 参数冻结
    - 只学层权重 s_j 和缩放 gamma
    """
    def __init__(self, elmo_model, output_dim, num_classes):
        super().__init__()
        self.elmo = elmo_model
        
        # 冻结 ELMo 参数（feature-based 的核心）
        for param in self.elmo.parameters():
            param.requires_grad = False
        
        # 可学习的层权重（softmax归一化）
        num_elmo_layers = elmo_model.num_layers + 1  # +1 for embedding layer
        self.layer_weights = nn.Parameter(torch.ones(num_elmo_layers))
        self.gamma = nn.Parameter(torch.ones(1))
        
        # 下游任务分类头
        hidden_dim = elmo_model.hidden_dim * 2
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # 提取 ELMo 三层特征（ELMo 参数不更新）
        with torch.no_grad():
            layer_outputs = self.elmo(x)  # list of [B, T, D]
        
        # 层权重归一化（softmax 保证和为1）
        weights = F.softmax(self.layer_weights, dim=0)  # [3]
        
        # 加权求和
        elmo_vec = torch.zeros_like(layer_outputs[0])
        for i, h in enumerate(layer_outputs):
            elmo_vec += weights[i] * h  # [B, T, D]
        
        elmo_vec = self.gamma * elmo_vec  # 全局缩放
        
        # 取序列平均 → 分类
        pooled = elmo_vec.mean(dim=1)    # [B, D]
        logits = self.classifier(pooled) # [B, num_classes]
        return logits


# ===================== 测试 =====================
def demo_context_sensitive():
    """演示上下文敏感性"""
    # 简单词表
    vocab = {'<PAD>': 0, 'I': 1, 'went': 2, 'to': 3, 'the': 4,
             'bank': 5, 'deposit': 6, 'money': 7, 'sat': 8,
             'on': 9, 'river': 10, 'of': 11, 'she': 12}
    
    elmo = SimpleBiLSTMELMo(vocab_size=len(vocab), embed_dim=32, hidden_dim=64)
    elmo.eval()
    
    # 两个句子，bank 分别在第4个位置（0-indexed）
    sent1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])  # "I went to the bank deposit money <PAD>"
    sent2 = torch.tensor([[12, 8, 9, 4, 5, 9, 4, 10]])  # "she sat on the bank on the river"
    
    with torch.no_grad():
        layers1 = elmo(sent1)
        layers2 = elmo(sent2)
    
    # 取最后一层的 bank 向量（位置4）
    bank_vec1 = layers1[-1][0, 4, :]  # 金融语境
    bank_vec2 = layers2[-1][0, 4, :]  # 河岸语境
    
    cos_sim = F.cosine_similarity(bank_vec1.unsqueeze(0), bank_vec2.unsqueeze(0))
    print(f"[未训练模型] 同词不同上下文余弦相似度: {cos_sim.item():.4f}")
    print(f"bank (金融) 向量 norm: {bank_vec1.norm():.4f}")
    print(f"bank (河岸) 向量 norm: {bank_vec2.norm():.4f}")
    print(f"向量是否完全相同: {torch.allclose(bank_vec1, bank_vec2)}")
    # 即使未训练，因为上下文不同，BiLSTM 输出也不同！


def demo_feature_based_vs_static():
    """对比静态向量 vs ELMo 动态向量"""
    print("=" * 50)
    print("静态词向量（Word2Vec风格）：")
    # 静态：每个词 ID 对应固定向量
    static_embed = nn.Embedding(20, 32)
    bank_id = torch.tensor([5])
    vec_static = static_embed(bank_id)
    print(f"  任意上下文中 'bank' 向量相同: {vec_static.shape}")
    
    print("\nELMo 动态词向量：")
    elmo = SimpleBiLSTMELMo(vocab_size=20, embed_dim=32, hidden_dim=64)
    elmo.eval()
    
    ctx1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])   # bank 在金融语境
    ctx2 = torch.tensor([[8, 9, 10, 11, 5, 9, 10, 0]])  # bank 在河岸语境
    
    with torch.no_grad():
        out1 = elmo(ctx1)[-1][0, 4, :]
        out2 = elmo(ctx2)[-1][0, 4, :]
    
    same = torch.allclose(out1, out2, atol=1e-6)
    sim = F.cosine_similarity(out1.unsqueeze(0), out2.unsqueeze(0)).item()
    print(f"  不同上下文 bank 向量完全相同: {same}")
    print(f"  余弦相似度: {sim:.4f}（<1 说明上下文影响了向量）")


if __name__ == '__main__':
    demo_context_sensitive()
    demo_feature_based_vs_static()
```

**运行预期输出：**
```
[未训练模型] 同词不同上下文余弦相似度: 0.xxxx  （不等于1）
向量是否完全相同: False
静态词向量（Word2Vec风格）：
  任意上下文中 'bank' 向量相同: torch.Size([1, 32])
ELMo 动态词向量：
  不同上下文 bank 向量完全相同: False
  余弦相似度: 0.xxxx（<1 说明上下文影响了向量）
```

---

## ⚠️ 易错点与常见误解

### 1. ELMo 是 Feature-based，不是 Fine-tuning
**错误理解**：ELMo 和 BERT 一样，都是对预训练模型进行微调。

**正确理解**：
- ELMo 预训练完后，BiLSTM 的参数**被冻结**，不再更新
- 下游任务只学习层权重 $s_j$ 和 $\gamma$（参数量极少）
- BERT 则是把预训练模型的全部参数都放开，在下游任务上继续训练

### 2. 三层权重 $s_j$ 是下游任务学习的，不是预训练时固定的
**错误理解**：ELMo 在预训练时就确定了三层的权重。

**正确理解**：$s_j$ 是**任务相关**的参数，针对不同任务（情感分析、NER、问答）会学到不同的层权重，这也是 ELMo 的一大亮点。

### 3. ELMo 输入是字符级，不是词级
**错误理解**：ELMo 像 Word2Vec 一样，输入是词 ID。

**正确理解**：ELMo 使用**字符级 CNN** 来生成初始词表示，这使得它可以处理未登录词（OOV），因为任何词都可以用字符来表示。

### 4. 双向性与 BERT 的双向性不同
- **ELMo**：前向 LSTM 只看左边，后向 LSTM 只看右边，然后拼接。两个方向是**独立训练**的。
- **BERT**：通过 MLM（掩码语言模型），Transformer 可以同时看到**左右两边**的上下文（真正的双向）。
- ELMo 的"双向"是**浅层拼接**，BERT 才是**深度双向**。

### 5. ELMo 中 $\gamma$ 的作用
$\gamma$ 不是可有可无的，它解决了**不同层输出的数值范围不一致**问题，允许模型整体缩放 ELMo 特征的幅度以适应下游任务。

### 6. ELMo 不能直接用于文本生成
由于双向语言模型的前向和后向是分别训练的，ELMo **无法**像 GPT 那样做自回归生成（因为后向模型需要看未来的词）。

---

## 🔗 知识延伸

| 技术 | 与 ELMo 的关系 |
|------|---------------|
| **Word2Vec / GloVe** | ELMo 的前身，静态向量 ELMo 用上下文动态向量取代 |
| **BERT** | 用 Transformer 替代 BiLSTM，真正的深度双向，微调而非特征提取 |
| **GPT** | Transformer Decoder，单向语言模型，用于生成任务 |
| **ULMFiT** | 同期工作，也是迁移学习思路，但用 AWD-LSTM |
| **CoVe** | 更早的上下文向量方法，用机器翻译的 Encoder 提取特征 |

**ELMo 在 NLP 发展史中的地位：**
```
Word2Vec (2013) → GloVe (2014) → ELMo (2018) → BERT (2018) → GPT-2/3 → ...
静态词向量        静态词向量      动态上下文向量   深度双向预训练   大规模生成
```

ELMo 是从**静态**迈向**动态上下文**的关键一步，直接催生了 BERT 的诞生思路。

---

## 📚 参考资料

1. Peters et al. (2018). **Deep contextualized word representations**. NAACL 2018. [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)
2. AllenNLP ELMo 官方实现：[https://github.com/allenai/allennlp](https://github.com/allenai/allennlp)
3. 官方预训练权重下载：[https://allennlp.org/elmo](https://allennlp.org/elmo)
4. 知乎解读：[ELMo 论文笔记](https://zhuanlan.zhihu.com/p/37684922)
