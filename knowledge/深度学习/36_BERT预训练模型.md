# BERT 预训练模型

## 📌 核心问题
> 如何用大量无标注文本训练出能迁移到各种 NLP 任务的通用特征表示？

## 🌱 根源与动机

Word2Vec 给每个词一个固定向量，但"苹果"在"吃苹果"和"苹果手机"语义不同——它是**静态嵌入**。

BERT 的思路：**用双向 Transformer 对整个句子建模**，同一个词在不同上下文中输出不同向量——**动态上下文表示**。

预训练策略：
1. **MLM（Masked Language Model）**：随机 mask 15% token，让模型预测
2. **NSP（Next Sentence Prediction）**：判断两句话是否连续

## 📐 理论推导

### 输入表示

```
token = [CLS] + sentence_A + [SEP] + sentence_B + [SEP]
embedding = Token Embedding + Segment Embedding + Position Embedding
```

### 输出使用

| 任务 | 用哪个输出 |
|------|----------|
| 分类 | `[CLS]` token 的向量 → FC → softmax |
| NER/序列标注 | 每个 token 的向量 → FC → softmax |
| 问答 | 输出 span 的起止位置 |

### Fine-tuning 流程

```
BERT 冻结/微调参数 → 下游任务 head → 小学习率训练
```

## 💡 关键理解

- `[CLS]` 经过全层 Self-Attention，**聚合了整句话的信息**，适合做句子级分类
- BERT 是 **Encoder-only**，不能直接用于生成任务
- **微调比从零训练快得多**：预训练已经学会语言规律，微调只需调整任务特定部分
- 中文 BERT：以字为粒度（不分词），英文以 WordPiece 子词为粒度

## 🔧 代码实现

对应代码：`knowledge/深度学习/代码实践/BERT/`、`knowledge/深度学习/代码实践/Transformer/`
- `Transformer/02_Bert_API_初识.py` — 使用 HuggingFace Transformers 库
- `BERT/01_Bert_API使用.py` — BERT 特征提取与微调
- `BERT/02_GPT_API使用.py` — GPT 对比

## ⚠️ 易错点与常见误解

1. **BERT 不能做文本生成**（Encoder-only），生成任务用 GPT（Decoder-only）或 T5（Encoder-Decoder）
2. **微调时学习率要小**（1e-5~5e-5），否则破坏预训练的权重
3. **[CLS] 不是"句子开头"**，是专门设计的聚合 token，位置在最前
4. **NSP 后来被证明作用有限**，RoBERTa 去掉了 NSP 反而更好
5. **中文 BERT 输入是字**，不要以为要先分词

## 🔗 知识延伸

- [[Transformer原理与实现]] — BERT 的底层结构
- [[命名实体识别项目]] — BERT fine-tune 做 NER
- [[文本分类项目]] — BERT fine-tune 做意图识别/情感分析
- [[LLM大模型]] — GPT → GPT-2 → GPT-3 → ChatGPT 的演进

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/11_Bert.pdf`
- HuggingFace：https://huggingface.co/models
- ModelScope：https://modelscope.cn/models
