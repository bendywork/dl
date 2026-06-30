# NLP / RAG / LLM 知识体系总结

---

# 一、整体技术栈

```text
算法/机制
    ↓
网络架构
    ↓
任务框架
    ↓
具体模型
    ↓
LLM
    ↓
RAG
    ↓
Agent
```

---

# 二、算法、架构、模型、系统的区别

## 算法（Algorithm）

解决具体问题的方法。

例如：

- Word2Vec
- GloVe
- FastText

作用：

```text
文本
 ↓
向量
```

---

## 机制（Mechanism）

解决某个局部问题。

例如：

- Attention
- Self-Attention
- Multi-Head Attention

作用：

```text
哪些词更重要
```

---

## 架构（Architecture）

神经网络骨架。

例如：

### CNN

```text
Embedding
 ↓
CNN
 ↓
Output
```

特点：

- 局部特征提取

---

### RNN

```text
Embedding
 ↓
RNN
 ↓
Output
```

特点：

- 序列建模

---

### LSTM

RNN改进版

解决：

```text
长期依赖问题
```

---

### GRU

LSTM简化版

---

### Transformer

核心组成：

```text
Attention
+
Feed Forward
+
Residual
+
LayerNorm
```

注意：

```text
Attention ≠ Transformer

Attention是零件
Transformer是整车
```

---

## 模型（Model）

利用架构训练出来的模型。

例如：

### BERT

```text
Transformer Encoder
```

用途：

- 分类
- NER
- 文本匹配

---

### GPT

```text
Transformer Decoder
```

用途：

- 文本生成

---

### T5

```text
Encoder
+
Decoder
```

用途：

- Seq2Seq

---

## 系统（System）

多个组件组合而成。

例如：

### RAG

```text
Embedding
+
Vector DB
+
Retriever
+
LLM
```

---

### Agent

```text
LLM
+
Tool
+
Memory
+
Planning
```

---

# 三、NLP发展主线

```text
统计NLP
    ↓
Word2Vec
    ↓
RNN
    ↓
LSTM / GRU
    ↓
Seq2Seq
    ↓
Attention
    ↓
Transformer
    ↓
BERT / GPT / T5
    ↓
LLM
    ↓
RAG
    ↓
Agent
```

---

# 四、Seq2Seq是什么

Seq2Seq：

```text
Sequence
 ↓
Sequence
```

即：

```text
输入序列
 ↓
输出序列
```

例如：

```text
中文
 ↓
英文
```

机器翻译。

---

## Seq2Seq不是架构

它是一种任务建模思想。

可以用：

- LSTM
- GRU
- Transformer

实现。

例如：

```text
LSTM Seq2Seq

GRU Seq2Seq

Transformer Seq2Seq
```

---

# 五、Embedding是什么

作用：

```text
文本
 ↓
向量
```

例如：

```text
贵州茅台
 ↓
[0.12,0.88,...]
```

用途：

- 语义表示
- 相似度计算
- 检索

---

# 六、文本分类

例如：

```text
帖子：
贵州茅台业绩超预期
```

输出：

```text
正面
```

流程：

```text
文本
 ↓
BERT
 ↓
类别
```

属于：

```text
Classification
```

---

# 七、NER

命名实体识别。

例如：

```text
贵州茅台董事长张德芹表示...
```

输出：

```text
贵州茅台 -> ORG

张德芹 -> PERSON
```

属于：

```text
序列标注任务
```

---

# 八、LLM是什么

LLM不是架构。

LLM是：

```text
Transformer
+
海量数据
+
大规模训练
```

得到的结果。

例如：

- GPT
- Qwen
- GLM
- DeepSeek

---

# 九、训练与推理

## Training

训练阶段：

```text
Forward
 ↓
Loss
 ↓
BackPropagation
 ↓
更新参数
```

目标：

```text
学习知识
```

---

## Inference

推理阶段：

```text
Forward
```

目标：

```text
使用知识
```

不更新参数。

---

# 十、RAG是什么

全称：

```text
Retrieval Augmented Generation
```

即：

```text
检索增强生成
```

---

## 工作流程

```text
用户问题
      ↓

Embedding
      ↓

Vector DB
      ↓

TopK召回
      ↓

相关文本
      ↓

LLM
      ↓

答案
```

---

# 十一、RAG的本质

RAG负责：

```text
Recall
Retrieve
Context
```

即：

```text
查资料
```

---

LLM负责：

```text
Reasoning
Generation
```

即：

```text
理解资料
+
回答问题
```

---

一句话：

```text
RAG = 图书管理员

LLM = 研究员
```

---

# 十二、向量数据库是什么

存储：

```text
Embedding向量
```

例如：

```json
{
  "text": "贵州茅台Q1利润增长15%",
  "vector": [...]
}
```

常见：

- Milvus
- Chroma
- FAISS

---

作用：

```text
语义检索
```

不是训练模型。

---

# 十三、Embedding模型与LLM关系

例如：

```text
Embedding:
BGE-M3

LLM:
Qwen
```

二者可能：

```text
Tokenizer不同
词表不同
模型不同
```

---

因此：

```text
Embedding负责检索

LLM负责推理
```

---

# 十四、LangChain是什么

不是模型。

是工作流框架。

负责连接：

```text
Embedding
Vector DB
Prompt
Retriever
LLM
```

形成完整链路。

---

# 十五、东方财富股吧项目

## 路线A：情感分析

```text
爬虫
 ↓
数据清洗
 ↓
BERT训练
 ↓
情绪预测
```

目标：

```text
正面
负面
中性
```

---

## 路线B：RAG知识库

```text
股吧帖子
 ↓
Chunk
 ↓
Embedding
 ↓
Milvus
 ↓
RAG
 ↓
Qwen
```

目标：

```text
智能问答
观点总结
舆情分析
```

---

# 十六、最重要的认知

```text
模型能力
来自训练

知识来源
来自RAG

最终答案
来自推理
```

---

牢记：

```text
Attention 是机制

Transformer 是架构

BERT/GPT 是模型

LLM 是训练后的大模型

RAG 是工程系统
```

---

最终链路：

```text
Word2Vec
    ↓
RNN
    ↓
LSTM/GRU
    ↓
Seq2Seq
    ↓
Attention
    ↓
Transformer
    ↓
BERT/GPT/T5
    ↓
LLM
    ↓
RAG
    ↓
Agent
```