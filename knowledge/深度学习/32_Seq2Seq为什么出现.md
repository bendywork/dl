# Seq2Seq 为什么出现

## 📌 核心问题
> RNN/LSTM/GRU 能处理序列，为什么还需要 Seq2Seq？

## 🌱 根源与动机

RNN/LSTM/GRU 的输入输出长度绑定：每个时间步进一个 token，出一个结果。但很多任务要求**输入长度 ≠ 输出长度**，且输出长度事先未知（翻译、摘要、对话）。

Seq2Seq 的解法：拆成两个 RNN——Encoder 负责读入压缩，Decoder 负责生成输出，输入输出长度完全解耦。

## 📐 结构

```
Encoder: 读完全部输入 → 压缩成上下文向量 c（最后一步隐藏状态）
Decoder: 从 c 出发 → 逐步生成输出 → 遇到 <EOS> 停止
```

```
输入: 我 爱 你 <EOS>
        ↓ ↓ ↓   ↓
      [Encoder] → c
                   ↓
      [Decoder] → I → love → you → <EOS>
```

## 💡 关键理解

Encoder 和 Decoder 是**两个独立的 RNN**，通过上下文向量 c 桥接。c 是唯一的"信息通道"，Encoder 的全部理解都压缩在这个向量里。

## 🔧 代码结构

```python
encoder = nn.LSTM(input_size=128, hidden_size=256)
decoder = nn.LSTM(input_size=128, hidden_size=256)

# Encoder
encoder_out, (h_n, c_n) = encoder(encoder_input)

# Decoder
decoder_input = torch.tensor([[<SOS>]])
hidden = (h_n, c_n)

for step in range(max_len):
    output, hidden = decoder(decoder_input, hidden)
    next_token = output.argmax(dim=-1)
    decoder_input = next_token
    if next_token == <EOS>:
        break
```

## ⚠️ 易错点与常见误解

1. **c 能装下所有信息吗？** — 这是 Seq2Seq 的瓶颈，长句子信息必然丢失。Attention 机制让 Decoder 每步都能回头看 Encoder 的每一步输出，解决了这个瓶颈。

2. **Decoder 每步输入从哪来？** — 训练时用真实答案（Teacher Forcing），推理时用自己上一步的预测，两者不一致导致暴露偏差。

## 🔗 知识延伸

- [[24_LSTM如何解决长时依赖]] — Encoder/Decoder 内部可用 LSTM
- Attention 机制 — 解决 c 信息瓶颈的问题
- Transformer — 用 Attention 完全替代 RNN 的架构

## 📚 参考资料

- Sutskever et al., "Sequence to Sequence Learning with Neural Networks", 2014
