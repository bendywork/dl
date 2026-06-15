# NLP 主要方向：原理、背景与发展历程

> 本文梳理自然语言处理的四大核心方向：文本标注、序列标注、文本分类、文本生成，涵盖技术原理、发展脉络与2026年6月前的关键里程碑。

---

## 一、文本标注（Text Annotation / Labeling）

### 1.1 核心问题

> 如何为原始文本赋予结构化语义标签，使机器能够理解语言中的实体、关系、情感、意图等信息？

文本标注是 NLP 的**数据基础**——没有高质量的标注数据，监督学习模型无从训练。它既指人工标注流程，也涵盖自动标注与弱监督标注技术。

### 1.2 技术原理

#### 1.2.1 标注体系设计

| 标注层级 | 标注对象 | 典型标签 |
|----------|---------|---------|
| 词级 | 单个 token | 词性（POS）、命名实体类型（NER）、语义角色 |
| 短语级 | 连续 token 序列 | 名词短语（NP）、动词短语（VP）、实体提及 |
| 句子级 | 完整句子 | 情感极性、意图分类、文本蕴含标签 |
| 篇章级 | 多句/全文 | 主题类别、篇章结构、话语关系（RST） |

#### 1.2.2 标注范式

- **人工标注**：众包平台（Amazon MTurk、Appen），专家标注（CoNLL、OntoNotes）
- **远程监督**：利用已有知识库（Freebase、Wikidata）自动对齐回标，如 Distant Supervision for Relation Extraction
- **弱监督**：Snorkel 框架——用户编写 labeling function，通过概率图模型融合噪声标注
- **主动学习**：模型选择不确定性最高的样本提交人工标注，最大化标注效率
- **Prompt-based 标注**：2022 年 GPT-3.5/4 出现后，利用 LLM 进行 few-shot / zero-shot 自动标注，逐步替代部分人工标注

#### 1.2.3 标注质量保障

- **IAA（Inter-Annotator Agreement）**：Cohen's κ、Fleiss' κ 衡量多标注员一致性
- **标注指南 + 黄金标准集**：定期校准
- **多轮迭代标注**：标注 → 训练 → 错误分析 → 修正指南 → 重新标注

### 1.3 发展历程

| 时期 | 关键进展 |
|------|---------|
| 1990s | MUC（Message Understanding Conference）推动命名实体标注规范化，CoNLL 建立标准数据集 |
| 2000s | ACE、OntoNotes 等大规模多层级标注语料建设；FrameNet（框架语义标注）、PropBank（谓词论元标注）、Penn Treebank（句法树标注） |
| 2010-2015 | 众包标注兴起；远程监督（Mintz et al., 2009）用于关系抽取；弱监督框架 Snorkel（2016） |
| 2016-2018 | 主动学习与深度学习结合；CT（Cross-Validation）标注策略 |
| 2019-2021 | 预训练模型（BERT）推动 few-shot 标注；数据增强（回译、同义词替换）降低标注需求 |
| 2022-2023 | LLM（GPT-3.5, GPT-4, Claude）引发标注范式变革——zero/few-shot 自动标注在多数任务上达到甚至超过众包质量；RLHF 标注（人类偏好标注）成为对齐训练核心 |
| 2024-2026 | **LLM-as-Judge** 大规模自动标注；多模态标注（图文、视频文字）；合成数据标注闭环（LLM 生成 + LLM 校验）；主动式持续标注流水线 |

---

## 二、序列标注（Sequence Labeling）

### 2.1 核心问题

> 输入一个 token 序列 $x = (x_1, x_2, ..., x_n)$，输出一个等长的标签序列 $y = (y_1, y_2, ..., y_n)$。每个 $y_i$ 依赖上下文 $x$ 以及相邻标签 $y_{i-1}, y_{i+1}$ 的约束。

序列标注是 NLP 中最基础也最广泛的任务范式之一。

### 2.2 典型任务

| 任务 | 输入 | 输出标签 |
|------|------|---------|
| 中文分词 | 字符序列 | `{B, M, E, S}` |
| 词性标注（POS Tagging） | 词序列 | Penn Treebank POS 标签集（45 种） |
| 命名实体识别（NER） | 词序列 | BIO/BIOES（PER, LOC, ORG, MISC...） |
| 浅层句法分析（Chunking） | 词序列 | BIO（NP, VP, PP...） |
| 语义角色标注（SRL） | 词序列 + 谓词位置 | BIO + 语义角色（ARG0, ARG1...） |

### 2.3 技术原理

#### 2.3.1 马尔可夫模型（HMM）

最早的概率序列建模方法。假设：
- **齐次马尔可夫假设**：$P(y_t | y_1...y_{t-1}) = P(y_t | y_{t-1})$
- **观测独立性假设**：$P(x_t | y_t)$，当前观测仅依赖当前标签

联合概率：
$$P(x, y) = P(y_1) \prod_{t=2}^{n} P(y_t | y_{t-1}) \prod_{t=1}^{n} P(x_t | y_t)$$

解码用维特比算法，学习用 Baum-Welch（EM）。

#### 2.3.2 条件随机场（CRF）

HMM 是**生成式**模型（建模 $P(x, y)$），CRF 是**判别式**模型（直接建模 $P(y|x)$），克服了 HMM 的强独立性假设。

线性链 CRF 的条件概率：

$$P(y|x) = \frac{1}{Z(x)} \exp\left(\sum_{t} \lambda_k f_k(y_t, y_{t-1}, x, t) + \sum_{t} \mu_j g_j(y_t, x, t)\right)$$

- $f_k$：转移特征函数（建模标签间依赖）
- $g_j$：状态特征函数（建模标签与输入的关联）
- 全局归一化，可融合任意重叠特征，无 HMM 的"label bias"问题

#### 2.3.3 BiLSTM-CRF

2015-2018 年的 SOTA 范式：

$$h_t^{\rightarrow} = \text{LSTM}(x_t, h_{t-1}^{\rightarrow})$$
$$h_t^{\leftarrow} = \text{LSTM}(x_t, h_{t+1}^{\leftarrow})$$
$$h_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$$

BiLSTM 自动提取上下文特征，CRF 层建模标签间转移约束。

#### 2.3.4 BERT + CRF / BERT + Linear

2019 年后预训练模型统一天下：
- BERT 的 Transformer 双向编码天然捕获长程上下文
- 最后一层输出 $h_t$ 送 CRF 层（保留标签转移约束）或直接 Softmax

#### 2.3.5 全局指针与多头标注

- **GlobalPointer**（苏剑林，2021）：将 NER 从 BIO 标注转化为头尾指针预测，避免 CRF 解码
- **多头标注**：W2NER 等统一 NER 框架，将实体识别建模为 token-pair 关系分类

#### 2.3.6 生成式统一框架

2023-2024 年，LLM 时代的新范式——将所有序列标注统一为文本生成，例如：
- 输入句子，输出结构化 JSON `{entities: [{text, type, start, end}, ...]}`
- GLiNER（2024）使用 Bi-encoder 直接对 span 进行分类，无需 LLM 即可高效零样本 NER

### 2.4 发展历程

| 时期 | 关键进展 |
|------|---------|
| 1980s-1990s | HMM 主导 POS Tagging 与浅层分词；维特比算法标准化 |
| 2001 | Lafferty et al. 提出 CRF，成为序列标注标准工具（CRF++, CRFSuite） |
| 2000s | MEMM（最大熵马尔可夫模型）尝试，但受 label bias 问题限制 |
| 2014-2015 | Collobert（2011）用 CNN 做序列标注开创神经网络方法；Huang et al.（2015）提出 BiLSTM-CRF 架构成为主流 |
| 2017-2018 | IDCNN-CRF、BiLSTM-CNN-CRF（字符级 CNN 特征 + 词级 LSTM）；ELMo 上下文词嵌入提升 |
| 2019 | BERT 出现，BERT+CRF 成为新基线；Ma & Hovy（2016）的 BiLSTM-CNN-CRF 被大规模取代 |
| 2020-2021 | 信息抽取统一框架（UIE）；GlobalPointer；Nested NER 结构（分层 BiLSTM，Biaffine） |
| 2022-2023 | GPT-3.5 展示出色的 few-shot NER；LLM+Instruct 范式；Flat-Lattice Transformer（中文 Lattice 结构） |
| 2024-2025 | GLiNER 推动 zero-shot NER 实用化；多模态序列标注（视觉-语言实体识别）；生成式通用 IE 统一 |
| 2025-2026 | 轻量级高效标注（小模型 + Distillation from LLM）；领域自适应序列标注无需目标域标注；扁平实体、嵌套实体、非连续实体统一建模 |

---

## 三、文本分类（Text Classification）

### 3.1 核心问题

> 给定文本 $x$，预测其所属类别 $y \in \mathcal{Y}$。是信息检索、舆情分析、内容风控等系统的核心组件。

### 3.2 典型任务

| 任务 | 输入 | 类别标签 | 类别层级 |
|------|------|---------|---------|
| 情感分析 | 评论/推文 | {正面, 负面, 中性} | 平面 |
| 主题分类 | 新闻全文 | {体育, 政治, 科技...} | 平面/层级 |
| 垃圾邮件检测 | 邮件正文 | {spam, ham} | 平面 |
| 意图识别 | 对话文本 | {订票, 查询, 投诉...} | 平面 |
| 仇恨言论检测 | 社交媒体帖子 | {toxic, severe_toxic, ...} | 多标签 |
| 关系分类 | 实体对+句子 | {出生地, 创始人...} | 平面/多类 |

### 3.3 技术原理

#### 3.3.1 词袋模型（BoW）+ 机器学习

将文本转为固定维度向量，然后送入分类器：

**特征表示**：
- **One-hot**：词表大小 $V$，每个词一个独热向量
- **TF-IDF**：$TF\text{-}IDF(w, d) = TF(w,d) \times \log\frac{N}{DF(w)}$
- **N-gram**：保留局部词序信息（如 bigram、trigram）

**分类器**：
- **朴素贝叶斯**：$P(y|x) \propto P(y) \prod_i P(x_i|y)$，特征条件独立假设
- **逻辑回归**：$P(y|x) = \sigma(W^T x + b)$，最大熵模型
- **SVM**：寻找最大间隔超平面，文本分类中线性 SVM 表现优异
- **XGBoost / LightGBM**：集成树模型，适合表格化后的文本特征

#### 3.3.2 神经网络方法

**TextCNN（Kim, 2014）**：
用多尺寸卷积核（如 2,3,4-gram）并行扫描句子矩阵，提取局部 n-gram 特征后池化拼接：
$$c_i = \text{ReLU}(W \cdot x_{i:i+h-1} + b)$$
$$\hat{c} = \max_i c_i \quad \text{(Max-over-time pooling)}$$

**TextRNN / BiLSTM**：
序列末位隐状态或时序池化结果作为句子表示，送入全连接层。

**TextRCNN**：
BiRNN 编码上下文 + Max Pooling 选重要特征。

**HAN（层级注意力网络）**：
词级注意力 → 句级注意力 → 篇章表示，适合长文档。

#### 3.3.3 预训练模型微调

**BERT fine-tuning（2019-）**：
$$h_{[CLS]} = \text{BERT}(x)$$
$$P(y|x) = \text{Softmax}(W h_{[CLS]} + b)$$

将文本分类统一为 `[CLS]` 表示 + 线性分类。

**Prompt-tuning / PET（2021-2022）**：
将分类转化为掩码语言模型填空：
- 输入："这部电影太棒了。情感是 [MASK]。" → 预测 "正面"
- 在小样本场景显著优于传统 fine-tuning

**SetFit（2022）**：
用对比学习在小样本（每个类别 8 个样本）达到接近全量 fine-tune 的效果。

#### 3.3.4 LLM Zero/Few-shot 分类

**In-Context Learning（2023-）**：
不需要训练，在 prompt 中提供少量示例（或多标签定义），让 LLM 直接分类。

优势：无需标注数据、无需训练、可灵活调整标签体系。
劣势：成本高、延迟大、一致性差。

### 3.4 发展历程

| 时期 | 关键进展 |
|------|---------|
| 1990s-2000s | TF-IDF + NB/SVM；Reuters-21578、20 Newsgroups 等数据集 |
| 2014 | Kim 提出 TextCNN，开启 NLP 的 CNN 时代；词向量（Word2Vec, GloVe）普及 |
| 2015-2017 | RNN/Attention 系列（HAN, BiLSTM-Attention）；FastText（高效工业级文本分类） |
| 2018 | ULMFiT 提出迁移学习三阶段（LM pre-train → Target LM fine-tune → Classifier fine-tune） |
| 2019 | BERT 预训练+微调一统天下，几乎所有文本分类任务 SOTA 被刷新 |
| 2020-2021 | PET/ADAPET prompt-tuning 解决小样本问题；SetFit 对比学习小样本分类；长文本（Longformer, BigBird）分类 |
| 2022 | GPT-3.5/InstructGPT 引发 LLM few-shot 分类；层级文本分类（HiAGM, HTCInfoMax） |
| 2023 | GPT-4/Claude 实现高质量 zero-shot 分类；多语言零样本分类成熟；极低成本分类（DistilBERT, TinyBERT 知识蒸馏） |
| 2024-2025 | 大模型蒸馏小模型分类流水线（LLM 标注 → 小模型学习）；极长文档（百万 token）分类；多模态文本分类（图文视频联合） |
| 2025-2026 | Text Classification as Agents（分类过程调用外部工具/知识图谱）；动态标签体系下的持续分类学习；解耦推理+分类的统一框架 |

---

## 四、文本生成（Text Generation）

### 4.1 核心问题

> 给定上下文/条件 $c$，生成连贯、流畅、符合目标的自然语言文本 $y = (y_1, y_2, ..., y_m)$。

文本生成是 NLP 的**终极挑战**——它要求模型同时掌握语法、语义、世界知识和语用学。

### 4.2 典型任务

| 任务 | 条件 $c$ | 输出 $y$ | 典型指标 |
|------|---------|---------|---------|
| 机器翻译 | 源语言句子 | 目标语言句子 | BLEU, COMET |
| 文本摘要 | 长文档 | 简短摘要 | ROUGE, BERTScore |
| 对话生成 | 对话历史+角色 | 回复文本 | BLEU, 人工评估 |
| 故事/创意写作 | 主题/开头 | 完整故事 | 人工评估 |
| 代码生成 | 自然语言描述 | 代码 | Pass@k, 功能正确性 |
| 数据到文本 | 结构化数据 | 自然语言描述 | BLEU, 事实准确度 |
| 可控文本生成 | 属性约束 | 符合属性的文本 | 属性准确率 + PPL |

### 4.3 技术原理

#### 4.3.1 自回归生成

文本生成的核心数学框架——语言模型逐 token 预测：

$$P(y) = \prod_{t=1}^{m} P(y_t | y_{<t})$$

每个步骤做 $|V|$ 类分类（词表大小，通常 3 万~25 万），训练时用**教师强制**（Teacher Forcing），推理时自回归采样。

#### 4.3.2 解码策略

| 策略 | 原理 | 特点 |
|------|------|------|
| **贪心解码** | 每步选最高概率 token | 确定性，但容易陷入重复 |
| **Beam Search** | 维护 k 条最优候选路径 | 适合翻译/摘要，但可能失去多样性 |
| **Top-k 采样** | 从概率最高的 k 个 token 中采样 | 增加多样性 |
| **Top-p（Nucleus）采样** | 从累积概率 ≥ p 的最小集合中采样 | 自适应候选大小 |
| **温度调节** | $P(y_t) = \text{softmax}(z_t / T)$，$T<1$ 更确定，$T>1$ 更随机 | 控制创造性与稳定性的平衡 |
| **对比搜索** | 惩罚与上下文相似的重复 token | 减少退化/重复 |
| **典型采样** | 按信息量（熵）过滤候选 token | 避免极端高频/低频 token |

#### 4.3.3 Seq2Seq + Attention

2014-2017 年的主流架构：

**编码器**：将源序列 $x$ 编码为隐状态序列 $h = (h_1, ..., h_n)$  
**注意力机制**：
$$\alpha_{t,i} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_j \exp(\text{score}(s_{t-1}, h_j))}$$
$$c_t = \sum_i \alpha_{t,i} h_i$$

**解码器**：$s_t = \text{RNN}(s_{t-1}, y_{t-1}, c_t)$，然后预测 $P(y_t | y_{<t}, c)$

#### 4.3.4 Transformer（自注意力 = 无RNN）

2017 年 "Attention is All You Need" 颠覆了整个领域：

- **Self-Attention**：每个 token 直接与序列中所有 token 交互
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Multi-Head**：多个注意力头并行，捕获不同子空间的依赖关系
- **Position Encoding**：正弦位置编码或可学习位置嵌入
- **Encoder-Decoder** 结构（翻译）vs **Decoder-only** 结构（GPT）

#### 4.3.5 预训练生成模型

**GPT 系列（Decoder-only）**：
- GPT-1（2018）：12 层单向 Transformer，BookCorpus 预训练
- GPT-2（2019）：1.5B 参数，展示 zero-shot 能力
- GPT-3（2020）：175B 参数，in-context learning 现象
- GPT-3.5/InstructGPT（2022）：RLHF 对齐训练
- GPT-4（2023）：多模态，推理能力质变
- GPT-4o / GPT-4.5（2024-2025）：实时多模态，增强推理、降本增效

**T5 / BART（Encoder-Decoder）**：
- T5（2019）：Text-to-Text 统一框架，所有 NLP = 文本到文本
- BART（2019）：去噪自编码器，encoder-decoder 结构

**其他架构**：
- PaLM/Gemini（2023-2025）：Google 系列，Gemini 2.5 Pro 达到极长上下文
- Claude（2023-2025）：Anthropic 系列，安全对齐领先
- LLaMA/Mistral/DeepSeek/Qwen（2023-2025）：开源模型推动生态民主化
- DeepSeek-R1（2025）：推理增强，GRPO 强化学习
- Kimi k2（2025）：MoE 架构，超长上下文

#### 4.3.6 RLHF 与对齐

预训练 → 监督微调（SFT）→ 奖励建模 → PPO/DPO 优化：

- **RLHF（PPO）**：用人类偏好训练奖励模型，再用 PPO 优化策略
- **DPO**（2023）：直接偏好优化，无需显式训练奖励模型，数学等价于 RLHF 但更稳定
- **KTO / SimPO**（2024）：进一步简化，去除成对比较需求

#### 4.3.7 推理时增强技术

- **Chain-of-Thought（CoT）**：引导模型逐步推理
- **Self-Consistency**：多次采样取多数
- **Tree-of-Thought / Graph-of-Thought**：多路径探索
- **ReAct / Tool-use**：思考-行动-观察循环
- **Inference-time Scaling**（2025）：增加推理计算量提升性能

#### 4.3.8 可控生成

- **CTRL / PPLM**：通过控制码或梯度引导属性
- **Classifier-free guidance**：扩散语言模型中的条件控制
- **Constrained Decoding**：通过有限状态机或文法约束输出

### 4.4 发展历程

| 时期 | 关键进展 |
|------|---------|
| 1950s-1960s | 规则机器翻译（ALPAC → MT 寒冬）；Eliza 对话系统；Shannon 奠基信息论 |
| 1970s-1980s | 基于模板的生成系统；SHRDLU 对话系统；统计语言模型（n-gram） |
| 1990s | 统计机器翻译（IBM Models 1-5）；短语 SMT |
| 2014 | Seq2Seq + Attention（Bahdanau, Luong）；神经机器翻译元年 |
| 2015-2016 | 复制机制（Pointer-Generator）；Coverage 机制；GAN 文本生成（SeqGAN） |
| 2017 | Transformer 诞生——整个 NLP 的分水岭 |
| 2018 | GPT-1, BERT；ULMFiT 迁移学习；预训练+微调范式确立 |
| 2019 | GPT-2（zero-shot）；T5（Text-to-Text）；BART；CTRL |
| 2020 | GPT-3（175B，few-shot）；扩散语言模型初步探索 |
| 2021 | Codex/GitHub Copilot；LaMDA；FLAN（指令微调）；Gopher, Chinchilla 缩放律 |
| 2022 | ChatGPT（RLHF 出圈）；GPT-4 rumor；Stable Diffusion 扩散模型；LLaMA 开源 |
| 2023 | GPT-4 发布；Claude 2；Open-source 爆发（Mistral, Llama 2, Qwen）; DPO; RAG 成熟；MoE 架构（Mixtral）；GPT-4V 多模态 |
| 2024 | GPT-4o 多模态实时交互；Claude 3.5（Artifacts）；Gemini 2.0；Llama 3（开源 405B）；Agent 元年（SWE-bench, WebArena）；扩散语言模型（LLaDA, D3LLM）；推理时 Scaling Law（OpenAI o1/o3）；合成数据大规模应用 |
| 2025 | DeepSeek R1 & V3 重塑开源格局；GPT-4.5；Claude 4（Extended Thinking）；Gemini 2.5 Pro（200万+上下文）；AI Agent 爆发（Manus, CUA, MCP 协议）；LLM 蒸馏与量化使端侧普及；小模型高性能化（Phi-4, Qwen2.5）；MoE 成为主流架构；视频生成模型（Sora, Veo 2）推动多模态文本生成 |
| 2025-2026 H1 | 推理模型军备竞赛（o4, R1-variants）；异构 Agent 系统（A2A 协议）；超大规模 RL 训练；GUI Agent 落地；Text-to-SQL/Code 高度成熟；多 Agent 协作框架标准化；Claude 推出更先进工具使用能力；开源 MoE 模型参数突破万亿级别；极低成本推理取代部分训练 |

---

## 五、四大方向的关系与交叉

```
         文本标注
         (数据基础)
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
序列标注   文本分类   文本生成
(理解)    (理解)    (生成)
    │        │        │
    └────────┼────────┘
             ▼
        统一的 LLM 范式
    (生成 = 理解 = 标注)
```

- **序列标注 → 文本分类**：序列标注的全局标签版本即文本分类（如情感分析可视为对整个句子标注情感）
- **文本标注 → 所有方向**：高质量标注数据是序列标注和文本分类的基石；偏好标注是 RLHF 的基石
- **文本生成 → 所有方向**：2023 年后，LLM 将分类、标注、序列标注都统一为"文本生成"任务——输入文本，输出结构化标签文本
- **生成式统一趋势**：2025-2026 年，NLP 整体趋向"一切皆生成"——所有任务通过 prompt 调度同一个生成模型完成

---

## 六、2026 年关键趋势总结

1. **推理时扩展**：从训练 scaling 转向推理 scaling——分配更多推理计算量提升效果
2. **MoE 与轻量化**：万亿参数 MoE + 蒸馏/量化 → 端侧部署强大模型
3. **Agent 化**：NLP 模型从"被动回答"走向"主动执行"，具备工具调用、规划、反思能力
4. **多模态融合**：文本生成 → 图文视频混合生成
5. **合成数据闭环**：LLM 生成标注 → 训练专用模型 → 部署 → 反馈再标注
6. **对齐与安全**：RLHF → DPO → Constitutional AI → 可解释对齐
7. **重新定义分类/标注**：传统判别式分类被 LLM 生成式分类逐步替代，标注流水线完全重构

---

*文档创建日期：2026-06-09*
*知识截止日期：2026年6月*
