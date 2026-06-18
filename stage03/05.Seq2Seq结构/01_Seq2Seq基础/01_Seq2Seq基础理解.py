# -*- coding: utf-8 -*-
"""
Seq2Seq 基础理解案例
目标：理解 Encoder-Decoder 结构，以及 context vector 的作用
场景：数字序列反转（如 [1,2,3] → [3,2,1]）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ─────────────────────────────────────────────
# 直觉类比
# ─────────────────────────────────────────────
# Seq2Seq = "翻译官"模型，分两个阶段：
#
#   Encoder（理解阶段）
#     → 逐步读入输入序列，把所有信息"压缩"成一个 context vector
#     → context vector = 编码器最后时刻的隐状态 h_n
#
#   Decoder（生成阶段）
#     → 用 context vector 初始化自己的隐状态
#     → 每一步：读入上一个输出词，结合当前隐状态，预测下一个词
#     → 直到生成 <EOS>（结束符）
#
# 瓶颈：整个输入序列被压缩成"一个向量"
#       → 序列越长，信息丢失越多（这正是 Attention 要解决的问题）
# ─────────────────────────────────────────────

# ── 超参数 ──────────────────────────────────
VOCAB_SIZE  = 12   # 0=pad, 1=SOS, 2=EOS, 3-11=数字1-9
SOS_TOKEN   = 1
EOS_TOKEN   = 2
PAD_TOKEN   = 0
EMBED_DIM   = 32
HIDDEN_SIZE = 64
MAX_LEN     = 8


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        """
        x : [bs, src_len]  输入 token ids
        返回:
          outputs : [bs, src_len, hidden]  每步输出（后面 Attention 会用到）
          hidden  : [1, bs, hidden]        最终隐状态 = context vector
        """
        emb = self.embedding(x)               # [bs, src_len, embed_dim]
        outputs, hidden = self.gru(emb)        # outputs: [bs, T, H], hidden: [1, bs, H]
        return outputs, hidden


# ─────────────────────────────────────────────
# Decoder（无 Attention 版本）
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, token_id, hidden):
        """
        解码单步
        token_id : [bs]        上一步输出的 token（或 SOS）
        hidden   : [1, bs, H]  当前隐状态
        返回:
          logits : [bs, vocab_size]
          hidden : [1, bs, H]
        """
        emb = self.embedding(token_id.unsqueeze(1))      # [bs, 1, embed_dim]
        out, hidden = self.gru(emb, hidden)               # out: [bs, 1, H]
        logits = self.fc(out.squeeze(1))                  # [bs, vocab_size]
        return logits, hidden


# ─────────────────────────────────────────────
# Seq2Seq 整体模型
# ─────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src : [bs, src_len]  输入序列
        tgt : [bs, tgt_len]  目标序列（含 SOS，不含最后一个 EOS）
        teacher_forcing_ratio: 训练时用真实标签作为下一步输入的概率
        """
        bs, tgt_len = tgt.shape
        vocab_size  = self.decoder.fc.out_features

        # Encoder：压缩输入为 context vector
        _, hidden = self.encoder(src)

        # Decoder：逐步生成
        all_logits = []
        dec_input  = tgt[:, 0]    # 第一个输入：SOS

        for t in range(1, tgt_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden)
            all_logits.append(logits.unsqueeze(1))

            # Teacher forcing：随机选择用真实标签还是预测值
            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else logits.argmax(-1)

        return torch.cat(all_logits, dim=1)   # [bs, tgt_len-1, vocab_size]

    @torch.no_grad()
    def generate(self, src, max_len=MAX_LEN):
        """推理时的贪心解码"""
        _, hidden = self.encoder(src)
        dec_input = torch.full((src.size(0),), SOS_TOKEN, dtype=torch.long)
        result    = []

        for _ in range(max_len):
            logits, hidden = self.decoder.forward_step(dec_input, hidden)
            dec_input = logits.argmax(-1)
            result.append(dec_input.unsqueeze(1))
            if (dec_input == EOS_TOKEN).all():
                break

        return torch.cat(result, dim=1)   # [bs, gen_len]


# ─────────────────────────────────────────────
# 数据：数字序列反转任务
# 输入: SOS 1 2 3 EOS → 输出: SOS 3 2 1 EOS
# token: 数字 n → token_id = n + 2（0=pad, 1=SOS, 2=EOS）
# ─────────────────────────────────────────────
def make_reverse_batch(batch_size=32, seq_len=5):
    nums   = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len))   # 数字 token
    src    = torch.cat([
        torch.full((batch_size, 1), SOS_TOKEN),
        nums,
        torch.full((batch_size, 1), EOS_TOKEN)
    ], dim=1)
    tgt    = torch.cat([
        torch.full((batch_size, 1), SOS_TOKEN),
        nums.flip(1),
        torch.full((batch_size, 1), EOS_TOKEN)
    ], dim=1)
    return src, tgt


# ─────────────────────────────────────────────
# 案例 1：查看 Encoder 输出的 context vector
# ─────────────────────────────────────────────
def demo_encoder_context():
    print("=" * 55)
    print("案例1：Encoder 把序列压缩成 context vector")
    print("=" * 55)

    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)
    src, _  = make_reverse_batch(batch_size=2, seq_len=4)

    print(f"输入序列（含SOS/EOS）:")
    for i, row in enumerate(src):
        print(f"  样本{i}: {row.tolist()}")

    with torch.no_grad():
        enc_outputs, hidden = encoder(src)

    print(f"\nEncoder 每步输出 shape: {enc_outputs.shape}  [bs, src_len, hidden]")
    print(f"Context vector shape:  {hidden.shape}  [1, bs, hidden]")
    print(f"\nContext vector（第一个样本前8维）:")
    print(f"  {hidden[0, 0, :8].tolist()}")
    print("\n→ 整个输入序列的信息被压缩进这一个向量里")


# ─────────────────────────────────────────────
# 案例 2：训练序列反转任务
# ─────────────────────────────────────────────
def demo_train_reverse():
    print("\n" + "=" * 55)
    print("案例2：训练 Seq2Seq 做序列反转")
    print("=" * 55)

    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)
    model   = Seq2Seq(encoder, decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    for epoch in range(2000):
        src, tgt = make_reverse_batch(batch_size=128, seq_len=5)
        logits = model(src, tgt, teacher_forcing_ratio=0.5)
        loss   = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 400 == 0:
            print(f"  epoch {epoch+1:4d}  loss={loss.item():.4f}")

    # ── 评估 ──────────────────────────────────
    model.eval()
    src_test, tgt_test = make_reverse_batch(batch_size=10, seq_len=5)
    pred = model.generate(src_test)

    print("\n推理结果（数字部分）：")
    correct = 0
    for i in range(len(src_test)):
        src_nums  = src_test[i][1:-1].tolist()           # 去掉 SOS/EOS
        tgt_nums  = tgt_test[i][1:-1].tolist()           # 期望反转
        pred_row  = pred[i].tolist()
        # 去掉 EOS 及之后
        if EOS_TOKEN in pred_row:
            pred_row = pred_row[:pred_row.index(EOS_TOKEN)]
        ok = pred_row == tgt_nums
        correct += int(ok)
        print(f"  输入:{src_nums} 期望:{tgt_nums} 预测:{pred_row} {'✅' if ok else '❌'}")

    print(f"\n准确率: {correct}/10")


# ─────────────────────────────────────────────
# 案例 3：可视化 context vector 瓶颈
# ─────────────────────────────────────────────
def demo_bottleneck():
    print("\n" + "=" * 55)
    print("案例3：context vector 瓶颈直觉演示")
    print("=" * 55)

    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE)

    # 两个相近的序列
    seq_a = torch.tensor([[SOS_TOKEN, 3, 4, 5, EOS_TOKEN]])
    seq_b = torch.tensor([[SOS_TOKEN, 3, 4, 6, EOS_TOKEN]])   # 只有最后数字不同
    seq_c = torch.tensor([[SOS_TOKEN, 9, 8, 7, EOS_TOKEN]])   # 完全不同

    with torch.no_grad():
        _, h_a = encoder(seq_a)
        _, h_b = encoder(seq_b)
        _, h_c = encoder(seq_c)

    dist_ab = (h_a - h_b).norm().item()
    dist_ac = (h_a - h_c).norm().item()

    print(f"序列A: {seq_a[0].tolist()}")
    print(f"序列B: {seq_b[0].tolist()} （末尾一个字符不同）")
    print(f"序列C: {seq_c[0].tolist()} （完全不同）")
    print(f"\n‖context(A) - context(B)‖ = {dist_ab:.4f}  （相近序列距离小）")
    print(f"‖context(A) - context(C)‖ = {dist_ac:.4f}  （不同序列距离大）")
    print(f"\n结论：context vector 确实捕获了序列语义")
    print(f"局限：序列越长，所有信息压缩到固定大小向量，信息损失越严重")
    print(f"      → 这就是为什么需要 Attention 机制（下一节）")


if __name__ == '__main__':
    demo_encoder_context()
    demo_train_reverse()
    demo_bottleneck()
