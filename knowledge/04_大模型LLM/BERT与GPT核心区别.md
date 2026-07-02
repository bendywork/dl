# BERT vs GPT 核心区别

> 两句大白话区分：BERT 是做完形填空的，GPT 是续写作文的。

---

## 一、架构层对比

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 注意力方向 | 双向 (看左右全部 token) | 单向/causal (只看左边) |
| Mask 类型 | 无 causal mask，只有 padding mask | 下三角 causal mask |
| 位置编码 | Learned Absolute | Learned (GPT-1/2) / RoPE (GPT-3+) |
| 典型模型 | BERT, RoBERTa, DeBERTa, ELECTRA | GPT-3/4, LLaMA, Qwen, DeepSeek |

### 注意力方向是根本区别

```
GPT (causal/单向):          BERT (双向):

"我 爱 北京 天安门"         "我 爱 北京 天安门"

我 → 我                    我 → 我 爱 北京 天安门
爱 → 我 爱                 爱 → 我 爱 北京 天安门
北京 → 我 爱 北京           北京 → 我 爱 北京 天安门
天安门 → 我 爱 北京 天安门   天安门 → 我 爱 北京 天安门
```

Bidirectional means each token can attend to ALL other tokens — left and right — in a single pass.

---

## 二、训练目标

| 维度 | BERT | GPT |
|------|------|-----|
| 训练任务 | MLM (Masked Language Model) + NSP | CLM (Causal Language Model) = Next Token Prediction |
| 训练方式 | 随机 mask 15% token，用上下文预测 | 给定前 n 个 token，预测第 n+1 个 |
| 类比 | CBOW (上下文→中心词) | Skip-gram 但自回归 |
| 能看到答案吗 | 不能 (被 mask 了) | 不能 (被 causal mask 遮住了) |
| 每个 batch 预测 | 15% 的位置 | 100% 的位置 (每个 token 都是预测目标) |

### MLM 的三个 mask 策略细节

```
原始:    今天 天气 很 好 适合 出去 玩
Mask:    今天 天气 很 [MASK] 适合 出去 玩
         → 模型根据"今天天气很 + 适合出去玩"预测"好"

实际 BERT mask 策略 (15% 选中词中):
  80% → [MASK]    (今天 天气 很 [MASK] 适合 出去 玩)
  10% → 随机词     (今天 天气 很 坏 适合 出去 玩)
  10% → 保持原词   (今天 天气 很 好 适合 出去 玩)
```

为什么要混入 10% 随机词 + 10% 原词？因为推理时没有 [MASK]，模型需要对非 mask token 也保持敏感。

---

## 三、能力边界

| 任务大类 | BERT | GPT |
|---------|------|-----|
| 文本分类 / 情感分析 | 适合 (拿 [CLS] 过分类头) | 勉强能做 (prompt 方式) |
| 命名实体识别 (NER) | 适合 (逐 token 标注) | 不太适合 |
| 序列标注 | 适合 | 不太适合 |
| 语义匹配 (两句话相似度) | 适合 | 勉强能做 |
| 抽取式问答 (SQuAD) | 适合 (预测 answer span 起止) | 勉强能做 |
| 文本生成 / 对话 | 不能 | 核心能力 |
| 代码生成 | 不能 | 核心能力 |
| 翻译 | 不能 | 可以做 (但不如 Enc-Dec) |
| 摘要 | 不能 | 适合 |
| 推理 (CoT) | 不能 | 核心能力 |

### BERT 不能生成，但能做"另一种生成"

BERT 的填空 (fill-mask) 不是自回归生成，一次只能填一个空：
```
输入:  巴黎是 [MASK] 的首都
输出:  巴黎是 法国 的首都  ← 一次搞定，不是逐个 token 输出
```

让它逐个 token 写一段话？做不到——它训练时从未见过"只根据左边预测右边"的模式。

---

## 四、输出方式

| 维度 | BERT | GPT |
|------|------|-----|
| 输出对象 | 每个 token 的上下文表示 (hidden states) | 词表上的概率分布 |
| 如何使用输出 | 接一个任务专用 head (分类/标注) | 直接 argmax/sample 得到下一个 token |
| 多任务 | 需要不同 head | prompt 统一所有任务 |

---

## 五、为什么只用了 Encoder / Decoder 的一半

```
                  BERT                         GPT
Transformer:  [Encoder][Encoder]...    [Decoder][Decoder]...

BERT 只用 Encoder:            GPT 只用 Decoder:
  Self-Attention (双向)         Masked Self-Attention (单向)
  FFN                           Cross-Attention (去掉, 因为没有 Encoder)
  省略 Cross-Attention           FFN
  省略 Causal Mask
```

- BERT 不需要 Decoder 因为不生成，Encoder 的双向表示就够做理解
- GPT 不需要 Encoder 因为没有外部输入要编码，自回归生成只需要 Decoder

---

## 六、关键洞察

1. **BERT 并未过时**：分类、NER、向量检索、特征提取，BERT 系模型更快更省，没必要用 GPT 做 embedding
2. **BERT 是理解型，GPT 是生成型**：选模型先判断任务是 NLU 还是 NLG
3. **两者可以互补**：BERT 做检索 (dense retrieval)，GPT 根据检索结果做生成 (RAG)
4. **Encoder-Decoder (T5/BART) 是第三种选择**：需要"理解输入→生成输出"的翻译/摘要场景，比 GPT 更直接
5. **BERT 的 transformer 层可以榨出知识**：Frozen BERT + 小分类头 在工业界仍然大量使用

---

## 七、一图总结

```
任务类型
│
├── 需要理解文本含义？
│   ├── 分类 / 标注 / 匹配 / 抽取 → BERT / RoBERTa
│   └── 同时还要生成新文本？
│       ├── 翻译 / 摘要 / 改写 → T5 / BART (Encoder-Decoder)
│       └── 对话 / 创作 / 推理 / 代码 → GPT / LLaMA (Decoder-only)
│
└── 只需要生成？
    └── 对话 / 创作 / 代码 → GPT / LLaMA
```
