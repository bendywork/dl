# Seq2Seq 英法翻译实验（Kaggle）

## 📌 实验目标
通过真实数据集动手验证 Seq2Seq + Attention 的完整流程，观察：
1. 信息瓶颈问题（无 Attention 版本 vs 有 Attention 版本）
2. Attention 权重热力图，直观理解对齐机制

---

## 数据集

Kaggle 搜索：`neural-machine-translation-with-attention`（by mateuszk013）

数据分块存储在 `data_chunks/` 目录下，共 226 个 csv 文件，列名为 `en` / `fr`。

---

## 完整代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import glob
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─────────────────────────────────────────
# 1. 数据加载与词表构建
# ─────────────────────────────────────────
DATA_DIR = '/kaggle/input/notebooks/mateuszk013/neural-machine-translation-with-attention/data_chunks/'
all_files = glob.glob(DATA_DIR + '*.csv')
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df = df[['en', 'fr']].dropna().sample(50000, random_state=42)  # 取5万条先跑

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

def build_vocab(sentences, max_vocab=10000):
    counter = Counter()
    for s in sentences:
        counter.update(s.lower().split())
    vocab = SPECIAL + [w for w, _ in counter.most_common(max_vocab - len(SPECIAL))]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

src_w2i, src_i2w = build_vocab(df['en'])
tgt_w2i, tgt_i2w = build_vocab(df['fr'])

def encode(sentence, w2i, max_len=20):
    tokens = sentence.lower().split()[:max_len]
    ids = [w2i.get(t, UNK) for t in tokens]
    ids = ids + [EOS]
    ids = ids + [PAD] * (max_len + 1 - len(ids))
    return ids

# ─────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, df, src_w2i, tgt_w2i, max_len=20):
        self.src = [encode(s, src_w2i, max_len) for s in df['en']]
        self.tgt = [encode(s, tgt_w2i, max_len) for s in df['fr']]

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        return (torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long))

dataset = TranslationDataset(df, src_w2i, tgt_w2i)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

# ─────────────────────────────────────────
# 3. 模型：Seq2Seq + Attention
# ─────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (bs, t_src)
        emb = self.embedding(x)               # (bs, t_src, embed_dim)
        outputs, (h, c) = self.lstm(emb)      # outputs: (bs, t_src, hidden)
        return outputs, h, c                  # 保留所有时间步供 Attention 使用


class Attention(nn.Module):
    def forward(self, decoder_h, encoder_outputs):
        # decoder_h:       (bs, 1, hidden)
        # encoder_outputs: (bs, t_src, hidden)
        scores = torch.bmm(decoder_h, encoder_outputs.permute(0, 2, 1))
        # scores: (bs, 1, t_src)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)
        # context: (bs, 1, hidden)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.attention = Attention()
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c, encoder_outputs):
        # x: (bs,) 当前步的词 id
        emb = self.embedding(x.unsqueeze(1))           # (bs, 1, embed_dim)
        context, attn_w = self.attention(
            h.permute(1, 0, 2), encoder_outputs)        # context: (bs, 1, hidden)
        lstm_input = torch.cat([emb, context], dim=-1)  # (bs, 1, embed+hidden)
        output, (h, c) = self.lstm(lstm_input, (h, c))
        logits = self.fc(output.squeeze(1))             # (bs, vocab_size)
        return logits, h, c, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, tgt_vocab_size, device):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device

    def forward(self, src, tgt, teacher_forcing=0.5):
        bs, t_tgt = tgt.shape
        enc_outputs, h, c = self.encoder(src)

        dec_input = torch.full((bs,), SOS, dtype=torch.long).to(self.device)
        all_logits = []

        for t in range(t_tgt):
            logits, h, c, _ = self.decoder(dec_input, h, c, enc_outputs)
            all_logits.append(logits.unsqueeze(1))
            use_tf = torch.rand(1).item() < teacher_forcing
            dec_input = tgt[:, t] if use_tf else logits.argmax(dim=-1)

        return torch.cat(all_logits, dim=1)  # (bs, t_tgt, vocab_size)

# ─────────────────────────────────────────
# 4. 训练
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_DIM, HIDDEN = 128, 256
encoder = Encoder(len(src_w2i), EMBED_DIM, HIDDEN).to(device)
decoder = Decoder(len(tgt_w2i), EMBED_DIM, HIDDEN).to(device)
model   = Seq2Seq(encoder, decoder, len(tgt_w2i), device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

print(f"设备: {device}")
print(f"src词表大小: {len(src_w2i)}, tgt词表大小: {len(tgt_w2i)}")
print(f"训练样本数: {len(dataset)}, 每epoch步数: {len(loader)}")
print("开始训练...\n")

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt)
        loss = criterion(logits.reshape(-1, len(tgt_w2i)), tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        # 每100个batch打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Step {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f">>> Epoch {epoch+1}/10 完成 | 平均Loss: {avg_loss:.4f}\n")

# ─────────────────────────────────────────
# 5. 推理 + 可视化 Attention
# ─────────────────────────────────────────
def translate(sentence, max_len=20):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([encode(sentence, src_w2i, max_len)]).to(device)
        enc_outputs, h, c = model.encoder(src)

        dec_input = torch.tensor([SOS]).to(device)
        result, attn_weights = [], []

        for _ in range(max_len):
            logits, h, c, attn_w = model.decoder(dec_input, h, c, enc_outputs)
            pred = logits.argmax(-1)
            word = tgt_i2w[pred.item()]
            if word == '<EOS>': break
            result.append(word)
            attn_weights.append(attn_w.squeeze().cpu().numpy())
            dec_input = pred

    return ' '.join(result), attn_weights

def plot_attention(src_sentence, translation, attn_weights):
    src_words = src_sentence.lower().split()
    tgt_words = translation.split()
    matrix = np.array(attn_weights)[:, :len(src_words)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(src_words))); ax.set_xticklabels(src_words, rotation=45)
    ax.set_yticks(range(len(tgt_words))); ax.set_yticklabels(tgt_words)
    ax.set_xlabel('Source (English)')
    ax.set_ylabel('Target (French)')
    ax.set_title('Attention Weights')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

# 测试翻译
test_sentences = [
    "I love you",
    "How are you today",
    "The cat is on the table",
]
for sent in test_sentences:
    translation, attn_w = translate(sent)
    print(f"EN: {sent}")
    print(f"FR: {translation}\n")
    plot_attention(sent, translation, attn_w)
```

---

## 实验观察清单

训练完成后逐项检查：

| 观察点 | 期望结果 | 说明 |
|-------|---------|------|
| Loss 是否稳定下降 | 从 ~6 降到 ~2 | 模型在正常学习 |
| 短句翻译质量 | 基本正确 | "I love you" → "je t'aime" |
| 长句翻译质量 | 有一定偏差 | 信息瓶颈仍然存在 |
| Attention 热力图 | 对角线明显 | 对齐学到了 |
| 训练时间（GPU） | 约 30~60 分钟 | T4 GPU 10个epoch |

---

## ⚠️ 注意事项

1. **梯度裁剪必须加**：`clip_grad_norm_(model.parameters(), 1.0)`，不加 loss 容易变 nan
2. **Teacher Forcing**：训练时用 0.5，推理时不用（已在 `translate` 函数中自动关闭）
3. **数据集列名确认**：加载后先 `print(df.columns)` 确认英文列是 `en`、法文列是 `fr`，不同版本可能不同
4. **GPU 确认**：Kaggle notebook 开启 GPU Accelerator，否则很慢

---

## 进阶方向

- [ ] 去掉 Attention，对比翻译质量差异（体验信息瓶颈）
- [ ] 增大数据量到20万条，观察效果提升
- [ ] 换成 Transformer 编码器，对比 LSTM vs Transformer
- [ ] 计算 BLEU 分数作为量化指标
