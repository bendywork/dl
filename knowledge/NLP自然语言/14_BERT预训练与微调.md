# BERT 预训练与微调（面试核心考点）

## 📌 核心问题

> 如何让一个通用的语言模型理解**双向上下文**，并高效迁移到各种下游任务？BERT（Bidirectional Encoder Representations from Transformers，Devlin et al., 2018）通过**遮蔽语言模型（MLM）**和**下一句预测（NSP）**两个预训练任务，使 Transformer Encoder 学到深度双向语言表示，再通过**微调（Fine-tuning）**适配各种 NLP 任务。

---

## 🌱 根源与动机

### 为什么需要 BERT？

| 方法 | 问题 |
|------|------|
| Word2Vec / GloVe | 静态词向量，无上下文感知 |
| ELMo | 动态上下文，但 BiLSTM 是浅层双向（前向和后向独立训练再拼接） |
| GPT-1 | Transformer 架构，但单向（只看左边上下文），不适合理解任务 |
| **BERT** | Transformer Encoder，**真正的深度双向**，预训练 + 微调范式 |

### 核心创新

1. **MLM（Masked Language Model）**：随机遮蔽部分词，让模型预测被遮蔽的词，迫使模型同时利用左右双向上下文
2. **Fine-tuning 范式**：预训练一次，只需加一个简单的任务头，微调少量步骤即可适配下游任务

---

## 📐 理论推导

### 1. BERT 输入表示

BERT 的输入是三种 Embedding 的**逐元素加和**：

$$\text{Input} = \text{TokenEmbedding} + \text{SegmentEmbedding} + \text{PositionEmbedding}$$

**示例（句对任务）：**
```
输入序列:  [CLS] I love NLP [SEP] It is amazing [SEP]
Token Emb: E_[CLS] E_I E_love E_NLP E_[SEP] E_It E_is E_amazing E_[SEP]
Segment:    E_A    E_A  E_A    E_A   E_A     E_B  E_B  E_B       E_B
Position:   E_0    E_1  E_2    E_3   E_4     E_5  E_6  E_7       E_8
```

三种 Embedding 的作用：
- **Token Embedding**：词汇表中的词向量（WordPiece 分词后）
- **Segment Embedding**：区分句子A（$E_A$）和句子B（$E_B$），用于句对任务
- **Position Embedding**：可学习的位置向量（注意：BERT 是可学习的，不是 sin/cos 固定编码）

### 2. WordPiece Tokenization

BERT 不直接用词，而是用 **WordPiece** 子词分词（vocabulary size = 30,000）：

**原理（贪心/BPE变体）：**

从字符开始，不断合并频率最高的字符对，直到词表达到目标大小：
```
初始:  u n a f f o r d a b l e
合并高频对:
  → un af fo rd ab le
  → un afford able
  → unaffordable  （如果这个词足够常见）
  
对于未见词: "unaffordable" → "un", "##afford", "##able"
                         （## 表示是词的延续部分，不是词首）
```

**WordPiece 优势：**
- 处理 OOV（未登录词）：任何词都可以拆成子词/字符
- 词表规模可控（30K vs 原始词典可能百万级）
- 语素级别的语义共享（如 "play", "playing", "played" 共享 "play"）

### 3. MLM（Masked Language Model）预训练任务

**目标函数：**

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \hat{x}; \theta)$$

其中：
- $\mathcal{M}$：被选中 mask 的位置集合
- $\hat{x}$：被 mask 后的输入序列
- $P(x_i \mid \hat{x}; \theta)$：模型预测第 $i$ 位置原始词的概率

**15% 的 Mask 策略（80/10/10 规则）：**

随机选择 **15%** 的 token 进行处理，其中：
- **80%** → 替换为 `[MASK]`（主要学习任务）
- **10%** → 替换为随机词（增加鲁棒性）
- **10%** → 保持原词不变（让模型不依赖位置信息）

**为什么不是全部 mask？（面试必问）**
- 如果全 mask：测试/推理阶段从没见过 `[MASK]` token → **预训练/微调分布不匹配**
- 80/10/10 策略让模型在处理每个 token 时都不确定"这个词是否是原词"，迫使它学习真实的上下文表示
- 保持 10% 原词：使模型的 Representation 对实际词有 bias（而不全靠上下文猜）

**数学直觉：** 模型如果遇到一个位置，它不知道这个位置是真实词、被mask的、还是被随机替换的，所以对每个位置都要认真建模上下文。

### 4. NSP（Next Sentence Prediction）预训练任务

**任务定义：** 给定句子对 (A, B)，预测 B 是否是 A 的下一句：
- **IsNext（正样本）**：从语料中连续抽取两句话，50% 概率
- **NotNext（负样本）**：B 从其他文档随机抽取，50% 概率

**目标函数：**

$$\mathcal{L}_{NSP} = -[\mathbb{1}_{IsNext} \log P(IsNext) + \mathbb{1}_{NotNext} \log P(NotNext)]$$

NSP 的标签通过 `[CLS]` token 的输出接二分类头来预测。

**注意：后来的研究（RoBERTa, 2019）证明 NSP 对大多数任务帮助有限甚至有害**，RoBERTa 去掉了 NSP，效果更好。

### 5. BERT 总预训练目标

$$\mathcal{L}_{total} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

两个损失联合训练，参数共享（同一个 Transformer Encoder）。

### 6. BERT 规模参数

| 模型 | 层数 L | 隐藏维度 H | 注意力头数 A | 参数量 |
|------|--------|-----------|------------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

其中 $d_k = H / A = 768 / 12 = 64$（每头维度）

---

## 💡 关键理解

### [CLS] Token 的作用（面试必考）

`[CLS]`（Classification）是加在句子最开头的特殊 token。

**作用：** 经过所有 Transformer 层后，`[CLS]` 的输出向量 $C \in \mathbb{R}^H$ 被视为**整个序列的聚合表示**，用于：
- NSP 预训练任务的二分类
- 下游的句子分类任务

**为什么 [CLS] 能代表整个句子？**
- Transformer 的 Self-Attention 让 `[CLS]` 可以关注序列中所有其他 token
- 经过多层注意力，`[CLS]` 实际上聚合了全序列的信息
- 预训练时 NSP 任务强制 `[CLS]` 学习句子级别的语义

**注意：** `[CLS]` 的表示能力不如用全序列做平均池化（mean pooling），这在后来的 Sentence-BERT 中得到了验证和改进。

### Fine-tuning vs Feature-based

**Fine-tuning（BERT 的方式）：**
```
预训练 BERT（12层 Transformer）
       ↓
加入任务专用头（如分类层）
       ↓
在下游任务数据上更新 BERT 全部参数 + 任务头参数
       ↓
端到端梯度回传
```

**Feature-based（ELMo 的方式）：**
```
预训练 BiLSTM（冻结）
       ↓
提取固定的上下文特征向量
       ↓
只训练下游任务模型
```

BERT fine-tuning 的优势：预训练模型的参数可以根据任务调整，效果通常更好。

---

## 🔧 代码实现

### 完整文本分类微调示例（使用 transformers 库）

```python
"""
BERT 文本分类微调完整示例
任务：情感分析（二分类：正面/负面）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# ============================================================
# 1. 数据集
# ============================================================
class SentimentDataset(Dataset):
    """
    简单情感分析数据集
    实际使用时替换为真实数据（SST-2、IMDB等）
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # tokenizer 自动处理：
        # 1. WordPiece 分词
        # 2. 添加 [CLS] 和 [SEP]
        # 3. Padding / Truncation 到 max_length
        # 4. 生成 attention_mask（1=真实词，0=padding）
        # 5. token_type_ids（0=句子A，用于单句任务全为0）
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),        # [max_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [max_len]
            'token_type_ids': encoding['token_type_ids'].squeeze(0),  # [max_len]
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 2. 方案A：直接使用 BertForSequenceClassification
# ============================================================
class BERTClassifierV1:
    """使用 HuggingFace 内置分类模型，最简单"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        # BertForSequenceClassification 内部结构：
        # BERT Encoder → [CLS] 输出 → Dropout → Linear(768, num_labels)
    
    def predict(self, texts, device='cpu'):
        self.model.to(device).eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)


# ============================================================
# 3. 方案B：手动实现分类头（更灵活，面试展示用）
# ============================================================
class BERTClassifierV2(nn.Module):
    """
    自定义分类头：
    BERT Encoder → [CLS] → Dropout → LayerNorm → Linear → 分类
    """
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        input_ids:      [batch, seq_len]
        attention_mask: [batch, seq_len]，1=真实词，0=padding
        token_type_ids: [batch, seq_len]，0=句子A，1=句子B
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # outputs.last_hidden_state: [batch, seq_len, 768]（每个token的表示）
        # outputs.pooler_output:     [batch, 768]（[CLS]经过Linear+Tanh后的表示）
        
        # 取 [CLS] token 的表示（位置0）
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        # 注意：也可以用 outputs.pooler_output，但研究表明直接用 last_hidden_state[:, 0] 有时更好
        
        logits = self.classifier(cls_output)  # [batch, num_classes]
        return logits


# ============================================================
# 4. 训练函数
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸，BERT 微调的标准操作）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # 学习率调度（warmup + 线性衰减）
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# 5. 完整训练流程
# ============================================================
def main():
    # 超参数（论文推荐值）
    MODEL_NAME = 'bert-base-uncased'
    NUM_CLASSES = 2
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5  # BERT 微调关键：小学习率（1e-5 ~ 5e-5）
    WARMUP_RATIO = 0.1    # 前 10% 步数做 warmup
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {DEVICE}")
    
    # 模拟数据（实际替换为真实数据集）
    train_texts = [
        "This movie is absolutely fantastic!",
        "I hate this terrible film.",
        "Great performance by all actors.",
        "Worst movie I've ever seen.",
        "The plot was interesting and engaging.",
        "Boring and predictable storyline.",
        "Outstanding cinematography and direction.",
        "The script was poorly written.",
    ] * 100  # 扩展数据量用于演示
    
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0] * 100
    
    val_texts = [
        "Amazing film with great acting.",
        "Completely disappointed with this movie.",
        "A masterpiece of modern cinema.",
        "Waste of time and money.",
    ]
    val_labels = [1, 0, 1, 0]
    
    # 初始化 tokenizer
    print("加载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 演示 tokenizer 输出
    sample_text = "I love [MASK] learning!"
    tokens = tokenizer.tokenize(sample_text)
    print(f"\n示例 tokenization:")
    print(f"  原文: {sample_text}")
    print(f"  tokens: {tokens}")
    print(f"  [CLS] + tokens + [SEP]: ['[CLS]'] + {tokens} + ['[SEP]']")
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    print("\n加载 BERT 模型...")
    model = BERTClassifierV2(MODEL_NAME, NUM_CLASSES, DROPOUT).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器：分层学习率（BERT层用小lr，分类头用大lr）
    bert_params = list(model.bert.named_parameters())
    classifier_params = list(model.classifier.named_parameters())
    
    optimizer_grouped_parameters = [
        # BERT 底层：更小的学习率
        {'params': [p for n, p in bert_params], 'lr': LEARNING_RATE},
        # 分类头：更大的学习率
        {'params': [p for n, p in classifier_params], 'lr': LEARNING_RATE * 10}
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        weight_decay=0.01,   # L2 正则
        eps=1e-8
    )
    
    # 学习率调度：warmup + 线性衰减
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\n开始训练...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, criterion
        )
        val_loss, val_acc, val_preds, val_labels_list = evaluate(
            model, val_loader, DEVICE, criterion
        )
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  训练 loss: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"  验证 loss: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_classifier.pt')
            print(f"  ✓ 保存最优模型 (val_acc={val_acc:.4f})")
    
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    
    # 推理示例
    print("\n=== 推理示例 ===")
    model.eval()
    test_texts = [
        "This is an incredible masterpiece!",
        "I completely hated this movie."
    ]
    
    inputs = tokenizer(
        test_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs['token_type_ids']
        )
        probs = torch.softmax(logits, dim=1)
    
    label_map = {0: '负面', 1: '正面'}
    for text, prob in zip(test_texts, probs):
        pred = torch.argmax(prob).item()
        print(f"文本: {text}")
        print(f"预测: {label_map[pred]} (正面概率: {prob[1].item():.4f})")


# ============================================================
# 6. 不同下游任务的微调方式
# ============================================================

class BERTForNER(nn.Module):
    """命名实体识别（序列标注）：每个 token 都需要预测标签"""
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        # NER：对每个 token 预测标签（不只是 [CLS]）
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 用每个 token 的表示（不是 [CLS]）
        sequence_output = outputs.last_hidden_state  # [batch, seq, 768]
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)  # [batch, seq, num_labels]
        return logits


class BERTForQA(nn.Module):
    """
    抽取式问答（如 SQuAD）：
    给定 [CLS] question [SEP] context [SEP]，
    预测答案在 context 中的 start 和 end 位置
    """
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 同时预测 start 和 end 的 logits（或用两个独立的线性层）
        self.qa_outputs = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # 0=question, 1=context
        )
        
        sequence_output = outputs.last_hidden_state  # [batch, seq, 768]
        logits = self.qa_outputs(sequence_output)    # [batch, seq, 2]
        
        start_logits = logits[:, :, 0]  # [batch, seq]
        end_logits = logits[:, :, 1]    # [batch, seq]
        
        return start_logits, end_logits


# ============================================================
# 7. MLM 预测示例（展示 BERT 的 mask 预测能力）
# ============================================================
def demo_mlm_prediction():
    """演示 BERT 预测被 mask 词的能力"""
    from transformers import pipeline
    
    print("\n=== MLM 预测示例 ===")
    # 使用 pipeline 快速演示（内部用 BertForMaskedLM）
    fill_mask = pipeline('fill-mask', model='bert-base-uncased')
    
    examples = [
        "The [MASK] sat on the mat.",     # 预期：cat
        "I went to the [MASK] to deposit money.",  # 预期：bank
        "She [MASK] to the store yesterday.",      # 预期：went/walked/drove
    ]
    
    for text in examples:
        results = fill_mask(text)
        print(f"\n输入: {text}")
        for r in results[:3]:
            print(f"  '{r['token_str']}' 概率={r['score']:.4f}")


if __name__ == '__main__':
    main()
    # demo_mlm_prediction()  # 需要网络下载模型
```

---

## ⚠️ 易错点与常见误解

### 1. [CLS] 的作用是什么？它天生就有语义吗？
**错误理解**：`[CLS]` 是特殊 token，天然代表句子语义。

**正确理解**：
- `[CLS]` 的意义完全来自**预训练**（NSP 任务强制它聚合句子级信息）和**微调**（分类任务的梯度让它学到分类相关表示）
- 对于**句子语义相似度**任务，直接用 `[CLS]` 效果往往不如均值池化（Sentence-BERT 证明了这一点）
- 在推理时，`[CLS]` 的 Representation 质量取决于是否经过了**任务相关的 fine-tuning**

### 2. 为什么 MLM 用 80/10/10，不全部 mask？
**错误理解**：为了减少计算量（15% < 100%）。

**正确理解（预训练-微调分布不匹配问题）**：
- 如果所有被选的 15% 都替换成 `[MASK]`：微调和推理时输入中没有 `[MASK]`，模型会不适应
- 80% mask：主要学习任务，让模型学会预测词
- 10% 随机词：让模型学会纠错，增强表示的鲁棒性
- 10% 保持原词：让模型对真实 token 也保持良好的表示（不能全靠 `[MASK]` 标记来触发预测）

### 3. BERT 不能做生成任务的根本原因
**错误理解**：BERT 参数量不够大，所以不能生成。

**正确理解（架构决定的）**：
- BERT 是 Encoder-only 架构，每个 token 可以看到整个序列（双向注意力）
- 生成任务需要**自回归（autoregressive）**：预测第 $t$ 个词时只能看到 $t-1$ 个词（因果性）
- BERT 的双向注意力天然违反因果性，无法用于自回归生成
- 如果硬要用 BERT 生成，需要用 `[MASK]` 逐步替换（非自回归，速度慢，质量差）

### 4. NSP 被后续研究证明作用有限
**错误理解**：NSP 是 BERT 预训练的核心，必须保留。

**正确理解**：
- RoBERTa（2019）去掉 NSP + 更大 batch + 更多数据 → 效果超过原版 BERT
- NSP 的负样本是从**不同文档**随机抽取的，太容易（模型可能只是学到"话题一致性"而非真正的句子关系）
- ALBERT 将 NSP 换成 **SOP（Sentence Order Prediction）**，效果更好（正负样本都来自同一文档，只是顺序不同）

### 5. Fine-tune 时的学习率选择
**错误理解**：随便用 1e-3 的学习率微调就好。

**正确理解**：
- BERT 微调标准学习率：**1e-5 ~ 5e-5**（极小）
- 原因：BERT 预训练权重已经很好，大学习率会"遗忘"预训练知识（灾难性遗忘）
- 最好使用 **warmup + 线性衰减**：前 N 步学习率线性增大，后续线性减小

### 6. Position Embedding 的差异（BERT vs 原始 Transformer）
**错误理解**：BERT 用 sin/cos 固定位置编码。

**正确理解**：
- 原始 Transformer：**固定 sin/cos 编码**（可外推到更长序列）
- **BERT：可学习的位置嵌入**（Position Embedding，通过 `nn.Embedding` 实现）
- BERT 的最大序列长度固定为 **512**（因为可学习嵌入只训练了0-511）

### 7. Segment Embedding 的用途
**错误理解**：Segment Embedding 只在 NSP 任务中用，单句任务可以不用。

**正确理解**：
- 单句任务时，所有 token 的 Segment ID 都是 0（全 $E_A$），仍然需要输入
- Segment Embedding 帮助模型区分两个句子的边界，即使在单句任务中，对模型的理解也有辅助作用

### 8. BERT 的 [SEP] token 作用
**错误理解**：`[SEP]` 只是分隔符，没有实质意义。

**正确理解**：
- `[SEP]` 让模型知道句子的**边界**在哪里（尤其在句对任务中）
- 配合 Segment Embedding，模型可以识别哪些 token 属于句子A，哪些属于句子B
- 单句任务末尾也必须加 `[SEP]`（tokenizer 自动处理）

---

## BERT 变体对比

| 模型 | 主要改进 | 关键点 |
|------|---------|--------|
| **RoBERTa** (2019) | 去掉 NSP，动态 mask，更大 batch，更多数据 | 证明 NSP 无用，更充分的预训练 |
| **ALBERT** (2019) | 参数共享（跨层），Factorized Embedding，SOP | 极大减少参数量（~10x），但推理不快 |
| **DistilBERT** (2019) | 知识蒸馏，6层（BERT-base 12层的一半） | 66% 大小，60% 速度提升，保留 97% 性能 |
| **SpanBERT** (2020) | 遮蔽连续词段而非单个词，SBO 任务 | 抽取式问答任务表现更好 |
| **DeBERTa** (2020) | 分离位置和内容的注意力计算 | 诸多 NLU 任务 SOTA |
| **MacBERT** (2020) | 中文 BERT，使用 MLM 但以同义词替换 | 中文 NLP 的强基线 |

---

## 🔗 知识延伸

```
ELMo (2018) → BERT (2018) → RoBERTa (2019) → ALBERT (2019)
 BiLSTM       Transformer    去NSP+更多数据    参数压缩
  特征提取      预训练+微调
  
BERT → GPT-2 (2019) → GPT-3 (2020) → ChatGPT → GPT-4 → ...
Encoder-only  Decoder-only  超大规模     RLHF       多模态
理解任务       生成任务       少样本学习
```

**BERT 的局限性与突破方向：**
1. **序列长度限制（512）** → Longformer, BigBird（稀疏注意力）
2. **无法生成** → UniLM, BART（统一预训练框架）
3. **计算开销大** → DistilBERT, TinyBERT（知识蒸馏）
4. **单语** → mBERT, XLM-R（多语言）

---

## 📚 参考资料

1. Devlin et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2. Liu et al. (2019). **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
3. Lan et al. (2020). **ALBERT: A Lite BERT for Self-supervised Learning**. [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
4. Illustrated BERT（强烈推荐）：[https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)
5. HuggingFace transformers 文档：[https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
6. Sentence-BERT（改进 [CLS] 表示）：[https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
