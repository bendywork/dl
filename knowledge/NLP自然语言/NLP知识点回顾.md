# NLP知识点回顾

## 一、词向量

### 1. 分词——中文NLP的第一步

中文不像英文有天然空格分隔，分词是所有中文NLP任务的基础。分词质量直接决定下游任务效果。

---

### 1.1 Jieba分词

**定位**：最流行的中文分词工具，轻量、快速、易用。

**分词算法**：基于前缀词典 + 动态规划 + HMM

| 算法层 | 原理 | 作用 |
|--------|------|------|
| 前缀词典 | 构建词的前缀Trie树，快速判断是否成词 | 精确模式主路径 |
| 动态规划 | 基于词频的DAG（有向无环图）最大概率路径 | 从所有可能切分中选概率最大的 |
| HMM | 对未登录词（OOV）用隐马尔可夫模型预测 | 识别词典中没有的新词 |

**三种分词模式**：

```python
import jieba

# 精确模式：最精确切分，适合文本分析
jieba.cut("我来到北京清华大学", cut_all=False)
# → 我 / 来到 / 北京 / 清华大学

# 全模式：扫描所有可能成词，速度快但有冗余
jieba.cut("我来到北京清华大学", cut_all=True)
# → 我 / 来到 / 北京 / 清华 / 清华大学 / 华大 / 大学

# 搜索引擎模式：在精确模式基础上，对长词再切分
jieba.cut_for_search("我来到北京清华大学")
# → 我 / 来到 / 北京 / 清华 / 华大 / 大学 / 清华大学
```

**优缺点**：

| 优点 | 缺点 |
|------|------|
| 速度快（秒级处理MB级文本） | 对未登录词识别能力一般 |
| 安装简单、依赖少 | 细粒度领域分词需手动加词典 |
| 支持自定义词典 | 无词性标注（需搭配jieba.posseg） |
| 社区活跃、资料多 | 无法处理深层语义歧义 |

**自定义词典**：

```python
jieba.add_word("深度学习")        # 动态添加词
jieba.load_userdict("my_dict.txt") # 加载自定义词典文件
# 格式：词语 词频 词性
# 深度学习 5 n
```

---

### 1.2 HanLP分词

**定位**：面向生产环境的中文NLP工具包，学术级精度。

**分词算法演进**：

| 版本 | 算法 | 特点 |
|------|------|------|
| HanLP v1 | 感知机（Perceptron）+ CRF | 传统机器学习，精度高 |
| HanLP v2 | BERT + CRF / Transformer | 深度学习，SOTA级精度 |

**核心能力**（不只是分词，是一站式NLP）：

```
分词 → 词性标注 → 命名实体识别 → 依存句法分析 → 语义依存分析
```

```python
from hanlp_restful import HanLPClient

HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth='你的key')
result = HanLP.parse("我来到北京清华大学")
# 返回完整的分词+词性+NER+句法分析结果
```

**与Jieba的核心区别**：

| 对比 | Jieba | HanLP |
|------|-------|-------|
| 算法 | 词典+DP+HMM | 深度学习（BERT/Transformer） |
| 精度 | 中等（日常够用） | 高（学术/生产级） |
| 速度 | 快 | v1快，v2需GPU才快 |
| 功能 | 分词为主 | 分词+词性+NER+句法+语义 |
| 未登录词 | HMM（一般） | BERT上下文（强） |
| 部署 | 本地轻量 | v2可云端API |
| 适用场景 | 快速原型、中小项目 | 生产环境、高精度需求 |

**歧义消解能力对比**：

```
输入："结婚的和尚未结婚的"

Jieba：结婚 / 的 / 和 / 尚未 / 结婚 / 的  ← 错误切分（和尚未→和/尚未）
HanLP：结婚 / 的 / 和 / 尚未 / 结婚 / 的  ← 正确切分

输入："南京市长江大桥"

Jieba：南京市 / 长江大桥  ← 正确
       或：南京 / 市长 / 江大桥  ← 歧义切分（取决于词典）
HanLP：南京市 / 长江大桥  ← 基于上下文消歧，更稳定
```

---

### 1.3 Jiagu分词

**定位**：基于深度学习的中文NLP工具，主打易用和准确。

**分词算法**：BiLSTM + CRF

```python
import jiagu

jiagu.seg("我来到北京清华大学")
# → 我 / 来到 / 北京 / 清华大学
```

**特色功能**：

| 功能 | 说明 |
|------|------|
| 分词 | BiLSTM+CRF，精度高于Jieba |
| 词性标注 | 配套词性标注 |
| 命名实体识别 | 人名、地名、机构名 |
| 情感分析 | 正面/负面情感判断 |
| 文本纠错 | 错别字检测与纠正（特色） |
| 知识图谱关系抽取 | 实体关系抽取（特色） |

**优缺点**：

| 优点 | 缺点 |
|------|------|
| 深度学习模型，精度比Jieba高 | 社区不如Jieba活跃 |
| 自带情感分析、文本纠错 | 模型文件较大 |
| 接口简洁 | 更新维护频率较低 |
| 无需外部依赖 | 对长文本速度偏慢 |

---

### 1.4 GSim（gensim相关分词）

**定位**：GSim严格来说不是分词工具，而是一个**词向量/主题模型**库。但分词是它的前置步骤。

**gensim 核心能力**：

| 功能 | 说明 |
|------|------|
| Word2Vec | 训练词向量（CBOW / Skip-gram） |
| Doc2Vec | 文档向量 |
| FastText | 子词级词向量（解决OOV） |
| LDA | 主题模型 |
| TF-IDF | 关键词提取 |

**典型使用流程**：

```python
from gensim.models import Word2Vec

# 1. 先用Jieba分词（gensim本身不分词）
sentences = [jieba.lcut(text) for text in corpus]

# 2. 训练Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# 3. 使用词向量
model.wv["清华大学"]  # 获取词向量
model.wv.most_similar("清华大学")  # 相似词
```

**gensim在分词生态中的角色**：

```
分词工具（Jieba/HanLP/Jiagu）→ 切分好的语料 → gensim → 训练词向量/主题模型
```

**gensim 不是分词器，是分词的下游消费者**。它依赖外部分词结果作为输入。

---

### 1.5 四大工具对比总结

| 工具 | 算法 | 分词精度 | 速度 | 功能范围 | 适用场景 |
|------|------|---------|------|---------|---------|
| **Jieba** | 词典+DP+HMM | 中 | 快 | 分词+词性 | 快速原型、通用场景 |
| **HanLP** | BERT+CRF | 高 | v1快/v2需GPU | 全栈NLP | 生产环境、高精度 |
| **Jiagu** | BiLSTM+CRF | 中高 | 中 | 分词+情感+纠错 | 中小项目、需要情感分析 |
| **GSim(gensim)** | — | — | — | 词向量+主题模型 | 分词后的向量化 |

**选择决策**：

```
需要分词？
  ├── 快速够用 → Jieba
  ├── 高精度/生产级 → HanLP
  ├── 需要情感分析+纠错 → Jiagu
  └── 需要词向量训练 → Jieba/HanLP分词 + gensim训练
```

### 一句话总结
> Jieba是分词的"瑞士军刀"（快而通用），HanLP是"精密仪器"（深度学习高精度全栈），Jiagu是"多面手"（分词+情感+纠错），gensim不是分词器而是分词后的"向量工厂"（Word2Vec/LDA）。选谁取决于精度需求、速度要求和功能范围。

---

## 二、NLP三大任务类型

> NLP的所有应用，归根结底都可以归入三大任务类型：**文本分类**、**序列标注**、**生成**。理解这三大类型，就理解了NLP的整个版图。

### 2.1 文本分类——给整段文本贴标签

#### 核心思想

输入一段文本，输出一个（或多个）**离散的类别标签**。本质是：从文本空间映射到类别空间。

```
输入：一整段文本 → 模型 → 输出：类别标签
```

**关键特征**：模型的输出粒度是**文档级**，不关心文本内部每个词的标签，只关心整段文本属于哪个类别。

#### 典型子任务与举例

| 子任务 | 输入 | 输出 | 举例 |
|--------|------|------|------|
| 情感分析 | "这家餐厅味道不错但服务太差" | 正面/负面 | 电商评论分析、舆情监控 |
| 主题分类 | "美联储宣布加息25个基点" | 财经/体育/娱乐… | 新闻分类、RSS聚合 |
| 意图识别 | "帮我订明天去上海的机票" | 订票/查询/退改 | 对话系统、智能客服 |
| 垃圾检测 | "恭喜您中奖100万，点击领取" | 垃圾/正常 | 邮件过滤、短信拦截 |
| 关系分类 | "乔布斯创建了苹果公司" | 创立/收购/投资… | 知识图谱构建 |
| STS语义相似度 | "我很开心" vs "我十分快乐" | 相似度分数 | 搜索排序、问答匹配 |

#### 模型架构演变

```
传统：TF-IDF/SVM → 词向量平均+全连接 → TextCNN → BiLSTM+Attention → BERT微调
```

**最简实现（词向量平均 + 全连接）**：

```python
# 思路：把文本中每个词的向量求平均，得到文档向量，再用全连接分类
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embedding(x)        # [batch, seq_len, embed_dim]
        avg_embed = embeds.mean(dim=1)     # [batch, embed_dim] ← 文档向量
        logits = self.fc(avg_embed)        # [batch, num_classes]
        return logits
```

**TextCNN 的直觉**：用多种尺寸的一维卷积核，分别捕获"2词短语""3词短语""4词短语"的特征，再池化拼接送分类器——本质是**多粒度 n-gram 特征提取**。

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, filter_sizes=[2,3,4]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 多种尺寸的卷积核，每种尺寸捕获不同长度的短语模式
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embeds = self.embedding(x)            # [batch, seq_len, embed_dim]
        embeds = embeds.permute(0, 2, 1)      # [batch, embed_dim, seq_len] ← Conv1d需要
        conv_outs = [F.relu(conv(embeds)) for conv in self.convs]
        # 每个卷积输出: [batch, num_filters, seq_len-fs+1]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        # 每个池化输出: [batch, num_filters]
        cat = torch.cat(pooled, dim=1)        # [batch, num_filters*3]
        logits = self.fc(cat)
        return logits
```

#### 文本分类的局限

文本分类只能给**整段文本**一个标签，无法知道文本内部每个词的角色。比如"苹果公司发布了iPhone"——分类能判断这是科技新闻，但无法指出"苹果公司"是实体、"发布"是关系、"iPhone"是产品。这就是序列标注的用武之地。

---

### 2.2 序列标注——给文本中每个位置贴标签

#### 核心思想

输入一段文本，输出与输入**等长**的标签序列。本质是：对文本的每个token做逐位置的分类。

```
输入：我   来到  北京  清华大学
输出：O    O     B-LOC I-LOC  I-LOC
```

**关键特征**：输入多长，输出就多长。每个位置都有一个标签，且相邻标签之间通常有**依赖关系**（比如B-LOC后面大概率跟I-LOC，不会跟I-PER）。

#### 典型子任务与举例

| 子任务 | 标签体系 | 举例 |
|--------|---------|------|
| **命名实体识别（NER）** | BIO/BIOES | "张三在北京大学读书" → 张三/B-PER 北京大学/B-ORG |
| **词性标注（POS）** | 词性标签集 | "我/PN 喜欢/V 苹果/NN" |
| **分词** | B/M/E/S | "我/S 喜/B 欢/E 苹/B 果/E" |
| **槽位填充（Slot Filling）** | 领域槽位 | "订明天去上海的机票" → 明天/B-date 上海/B-dest |
| **关系抽取（序列式）** | 关系标签 | "乔布斯创立苹果" → 创立/REL |

#### BIO 标签体系——序列标注的核心编码方式

```
B = Begin（实体开始）
I = Inside（实体内部）
O = Outside（非实体）

例子：马云在杭州创建了阿里巴巴

B-PER I-PER O  B-LOC I-LOC O    O     B-ORG  I-ORG  I-ORG
 马云      在  杭州      创建  了   阿里巴巴

BIOES扩展：
E = End（实体结束）  S = Single（单字实体）
```

为什么用BIO而不用简单的"实体/非实体"？因为需要**区分实体的边界**。"阿里巴巴"是三个字组成的一个实体，不是三个独立的实体。

#### 模型架构演变

```
传统：HMM → CRF → BiLSTM → BiLSTM+CRF → BERT+CRF → BERT+Linear
```

**为什么需要CRF？** 因为序列标注中相邻标签有依赖关系，单独对每个位置做分类（如纯BiLSTM）会忽略这种依赖。CRF能学习标签之间的**转移概率**，比如约束"B-PER后面不能跟I-LOC"。

**BiLSTM+CRF**（序列标注经典架构）：

```python
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        # CRF转移矩阵: transitions[i][j] = 从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def _get_emission(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, num_tags]
        return emissions

    def forward(self, x, tags):
        emissions = self._get_emission(x)
        # 训练: 计算CRF的负对数似然损失 (gold_score - all_path_score)
        gold_score = self._score_sentence(emissions, tags)
        all_path_score = self._forward_algorithm(emissions)
        loss = all_path_score - gold_score
        return loss

    def decode(self, x):
        # 推理: Viterbi算法找最优标签路径
        emissions = self._get_emission(x)
        best_path = self._viterbi_decode(emissions)
        return best_path
```

**直觉理解BiLSTM+CRF**：
- **BiLSTM**负责"看文本"：根据上下文理解每个位置应该是什么标签（发射分数）
- **CRF**负责"看规则"：根据标签之间的合法转移关系修正预测（转移分数）
- 最终预测 = 发射分数 + 转移分数，取全局最优路径

#### 序列标注的局限

序列标注只能从**已有文本中抽取信息**（每个位置分配一个标签），无法创造新内容。比如你无法用序列标注让模型"写一段总结"或"回答问题"。这就是生成任务的领域。

---

### 2.3 生成——让模型创造新文本

#### 核心思想

输入一段文本（或提示），输出一段**新的、连贯的文本**。本质是：逐token预测下一个token的概率分布，自回归地生成完整序列。

```
输入：今天天气 → 模型 → 输出：很好，适合出去散步
```

**关键特征**：输出长度不固定，由模型自己决定何时停止（遇到`<EOS>`或达到最大长度）。这是与分类和序列标注最本质的区别——前两者的输出空间是**有限且预定义的**，生成的输出空间是**开放且无限的**。

#### 生成模型的核心原理：自回归

```
P(y1, y2, ..., yn | x) = P(y1|x) × P(y2|x,y1) × P(y3|x,y1,y2) × ... × P(yn|x,y1,...,yn-1)

每次预测下一个token，然后把预测结果拼回输入，继续预测下一个
```

```python
# 自回归生成的伪代码
def generate(model, prompt, max_len=50):
    tokens = tokenize(prompt)
    for _ in range(max_len):
        logits = model(tokens)              # [1, seq_len, vocab_size]
        next_token = sample(logits[:, -1, :])  # 只看最后一个位置的预测
        if next_token == EOS:
            break
        tokens.append(next_token)           # 拼回去，继续预测
    return detokenize(tokens)
```

#### 典型子任务与举例

| 子任务 | 输入 | 输出 | 举例 |
|--------|------|------|------|
| **机器翻译** | 英文原文 | 中文译文 | "Hello world" → "你好世界" |
| **文本摘要** | 长文档 | 短摘要 | 10页论文 → 200字摘要 |
| **对话生成** | 用户消息 | 回复 | "今天好冷" → "记得多穿衣服" |
| **问答生成** | 问题+上下文 | 答案 | "Transformer谁提出的？" → "Vaswani等人于2017年提出" |
| **代码生成** | 自然语言描述 | 代码 | "写一个快排" → def quicksort... |
| **续写/创作** | 开头文本 | 后续内容 | "很久很久以前" → "有一个..." |

#### 模型架构演变

```
Seq2Seq(RNN) → Seq2Seq+Attention → Transformer Encoder-Decoder → GPT(Decoder-Only) → ChatGPT
```

**Seq2Seq + Attention**（生成任务的奠基架构）：

```python
class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim * 2)
        self.attention = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 3, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, src, tgt):
        # Encoder: 读取源序列，产出所有隐藏状态
        enc_embeds = self.embedding(src)
        enc_outs, _ = self.encoder(enc_embeds)  # [src_len, batch, hidden*2]

        # Decoder: 逐步生成，每步attend到encoder输出
        dec_embeds = self.embedding(tgt)
        dec_outs, _ = self.decoder(dec_embeds)   # [tgt_len, batch, hidden*2]

        # Attention: decoder每步对encoder所有位置计算相关性
        # context = 加权求和encoder隐藏状态
        # output = concat(decoder隐藏状态, context) → 预测下一个词
        ...
```

**Decoder-Only（GPT系列）**：去掉Encoder，只用Transformer的Decoder部分。输入prompt，自回归预测后续token。ChatGPT、GPT-4、Claude都是这个路线。

#### 生成策略——解码方法

生成不只是"取概率最大的那个词"，还有不同策略：

| 策略 | 方法 | 特点 |
|------|------|------|
| **贪心搜索** | 每步取argmax | 快但容易重复、无多样性 |
| **Beam Search** | 保留top-k条候选路径 | 平衡质量与多样性 |
| **Top-k采样** | 从概率最高的k个词中随机采样 | 有随机性，k越大越多样 |
| **Top-p（核采样）** | 从累积概率≤p的最小词集中采样 | 自适应k，比top-k更稳定 |
| **Temperature** | 除以温度T后再softmax | T<1更确定，T>1更随机 |

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    # 把累积概率超过p的词概率清零
    mask = cum_probs - sorted_probs > p
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()  # 重新归一化
    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, sampled)
```

---

### 2.4 三大任务对比总结

| 维度 | 文本分类 | 序列标注 | 生成 |
|------|---------|---------|------|
| **输出粒度** | 文档级（1个标签） | token级（每位置1标签） | 序列级（新文本） |
| **输出空间** | 有限离散集 | 有限离散集（等长） | 无限（开放词汇） |
| **输出长度** | 固定（=类别数） | 固定（=输入长度） | 不固定（模型自定） |
| **核心挑战** | 语义理解够不够深 | 标签间依赖怎么建模 | 生成质量+多样性平衡 |
| **典型模型** | TextCNN/BERT | BiLSTM+CRF/BERT+CRF | Seq2Seq/GPT/T5 |
| **损失函数** | CrossEntropy | CRF负对数似然 | 自回归CrossEntropy |
| **评估指标** | Accuracy/F1 | F1/实体级F1 | BLEU/ROUGE/人工 |
| **速度** | 最快（1次前向） | 快（1次前向+解码） | 慢（自回归多次前向） |

### 三大任务的层次关系

```
复杂度递增 →

文本分类          序列标注              生成
"这是什么？"      "每个位置是什么？"     "创造新内容"
1个标签           等长标签序列           变长新序列
├── 情感分析      ├── NER              ├── 翻译
├── 主题分类      ├── POS              ├── 摘要
├── 意图识别      ├── 分词              ├── 对话
└── 垃圾检测      └── 槽位填充          └── 代码生成

关系：
  分类 ⊂ 标注（标注可视为对每个token做分类，但加了序列约束）
  标注 ⊂ 生成（生成可视为对每个位置做标注，但输出空间开放）
```

> **一句话总结**：文本分类是"给文章定性"，序列标注是"给每个词定性"，生成是"让模型说话"。三者复杂度递增，能力递增，但训练难度和资源需求也递增。
