# -*- coding: utf-8 -*-
"""
Seq2Seq Attention 结构

在原生 Seq2Seq 的结构基础上进行考虑，如果将编码器提取的特征向量 C 作为解码器的初始状态信息，
那么在长序列的解码结构中，主要存在两个方面的问题：
    - 问题 1: 在解码器 t 个时刻后，对当前时刻的解码操作中，
            解码器提取得到的特征向量受向量 C 的影响很小（类似长时依赖问题）；
    - 问题 2: 整个 Seq2Seq 中，编码器仅输出了一个特征向量 C，相当于在解码器的每个时刻
            使用的都是完全相同的编码器特征向量信息（或者说是同一个特征向量 C 经过不同转换
            后得到的不同向量）——也就是对齐问题。

解决方案：
    - 不单纯使用一个特征向量，而是将编码器的所有时刻输出特征向量均作为解码器的输入，
      由解码器自行决定使用哪些编码器特征向量；
    - 解码器的不同时刻使用到的编码器特征向量应该不一样才好，所以可以基于编码器的输出、
      每个时刻的解码器状态信息构建不同的特征向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 编码器
# ============================================================
class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=1,  # 简化写法，仅支持一层结构
            batch_first=True,
            bidirectional=True
        )
        self.ctx_feature_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, token_ids):
        # 1. token id 转换为 token embedding 向量 [bs,et] -> [bs,et,hidden_size]
        token_embed = self.embed_layer(token_ids)
        # 2. 调用 rnn 结构获取序列特征向量
        #    output: [bs,et,2*hidden_size] —— 当前是双向结构
        #    state : rnn 和 gru 的时候只有一个值；lstm 的时候有两个值（二元组）；
        #            shape 均为 [?,bs,hidden_size]
        output, state = self.rnn_layer(token_embed)
        # 3. 将每个时刻的输出特征向量转换进行维度映射后转换输出
        token_ctx_embed = self.ctx_feature_layer(output)

        return token_ctx_embed


# ============================================================
# Cross-Attention 模块
# ============================================================
class Seq2SeqCrossAttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

    def forward(self, decoder_emd, encoder_output_ctx_embed):
        """
        Attention 前向计算
            decoder_emd            : [bs,e]   解码器某个时刻的特征信息
            encoder_output_ctx_embed: [bs,t,e] 编码器输出特征信息

        NOTE:
            bs: 批次样本数目
            t : 编码器序列 token 数目，也就是序列长度
            e : 每个样本 / token 对应的向量维度大小

        RETURN:
            基于"解码器特征向量"合并的编码器特征向量信息，
            返回 shape 一般和输入的解码器信息 shape 一致 -> [bs,e]
        """
        # 1. 获取 q / k / v
        q = torch.unsqueeze(decoder_emd, dim=1)  # [bs,e] -> [bs,1,e]
        k = encoder_output_ctx_embed             # [bs,t,e]
        v = encoder_output_ctx_embed             # [bs,t,e]

        # 2. 计算 q 和 k 的相关性，这里采用点积的操作
        #    [bs,1,e] * [bs,e,t] -> [bs,1,t]
        # ** 针对每个样本的每个 query 计算和其它 t 个 key 之间的相关性；
        #    总的来讲就是：bs 个样本，每个样本 1 个 query，每个 query 和 t 个 key 的相关性的值。
        score = torch.matmul(q, torch.permute(k, dims=(0, 2, 1)))
        # [bs,1,t] -> [bs,1,t] 针对 t 个置信度计算 softmax 权重值
        alpha = torch.softmax(score, dim=-1)

        # 3. value 加权
        #    [bs,1,t] * [bs,t,e] -> [bs,1,e]
        # ** 针对每个样本的每个 query，将对应的 k 个 value 加权合并
        # *** 总的来讲就是：bs 个样本，每个样本 1 个 query，
        #     基于每个 query 和 t 个 key 的权重系数对 t 个 value 加权合并。
        value = torch.matmul(alpha, v)

        # 4. 最终结果返回
        return value[:, 0, :]  # [bs,1,e] -> [bs,e]


# ============================================================
# 解码器
# ============================================================
class DecoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.atten_layer = Seq2SeqCrossAttentionModule(hidden_size)
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=1,            # 简化任务，仅固定为一层结构
            batch_first=True,
            bidirectional=False      # 解码器只能是单向的
        )
        self.classify_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, encoder_ctx):
        """
        前向执行过程
            token_ids   : [bs,dt]                解码器输入 token ids
            encoder_ctx : [bs,et,hidden_size]    编码器提取出来的文本特征向量
        """
        if self.training:
            # 1. 针对输入数据进行 embedding 操作 [bs,dt] -> [bs,dt,hidden_size]
            token_embed = self.embed_layer(token_ids)
            bs, dt, e = token_embed.shape

            # 2. 遍历每个时刻进行处理
            hcn = None    # 初始状态
            score = []    # 保存每个时刻对应的预测置信度
            for t in range(dt):
                # 2a. 当前时刻输入特征向量
                token_embed_t = token_embed[:, t:t + 1, :]  # [bs,1,e]

                # 2a1. 基于上一个时刻的状态信息 + 编码器的特征向量合并得到一个高阶特征向量，
                #      并将这个高阶特征向量合并到当前时刻的解码器输入中
                if hcn is not None:
                    if isinstance(hcn, tuple):
                        decoder_emd = torch.mean(hcn[0] + hcn[1], dim=0)  # [?,bs,e] -> [bs,e]
                    else:
                        decoder_emd = torch.mean(hcn, dim=0)              # [?,bs,e] -> [bs,e]
                    # 计算编码器的合并 value [bs,e]
                    atten_value = self.atten_layer(decoder_emd, encoder_ctx)
                    # 将编码器的输出特征 value 作为当前时刻的输入
                    atten_value = atten_value[:, None]  # [bs,e] -> [bs,1,e]
                    token_embed_t = token_embed_t + atten_value

                # 2b. 计算当前时刻的 LSTM 输出和状态信息
                #     —— 基于当前输入以及上一个时刻的状态信息
                # output_t: [bs,1,e] 当前时刻的 RNN/LSTM 的输出
                output_t, hcn = self.rnn_layer(token_embed_t, hcn)

                # 2c. 基于当前时刻的输出决策判断属于各个类别的置信度
                #     [bs,1,e] * [e,c] -> [bs,1,c]
                score_t = self.classify_layer(output_t)
                score.append(score_t)

            # [bs,t,c] bs 个样本，每个样本 t 个 token，每个 token 对应 c 个类别的置信度
            score = torch.concat(score, dim=1)
            return score
        else:
            # ===== 针对推理预测过程，需要一个 token、一个 token 进行输入预测得到结果 =====
            # 当前代码要求输入仅一个 token
            bs, dt = token_ids.shape
            assert dt == 1, (
                f"解码器推理解码操作的时候，要求解码器原始输入必须仅包含一个特殊 token，"
                f"当前输入 shape 为: {token_ids.shape}"
            )
            hcn = None
            token_ids_t = token_ids  # t 时刻的输入 token id
            while True:
                # 2. 针对输入数据进行 embedding 操作 [bs,1] -> [bs,1,hidden_size]
                token_embed_t = self.embed_layer(token_ids_t)

                # 2a1. 基于上一个时刻的状态信息 + 编码器的特征向量合并得到一个高阶特征向量，
                #      并将这个高阶特征向量合并到当前时刻的解码器输入中
                if hcn is not None:
                    if isinstance(hcn, tuple):
                        decoder_emd = torch.mean(hcn[0] + hcn[1], dim=0)  # [?,bs,e] -> [bs,e]
                    else:
                        decoder_emd = torch.mean(hcn, dim=0)              # [?,bs,e] -> [bs,e]
                    # 计算编码器的合并 value [bs,e]
                    atten_value = self.atten_layer(decoder_emd, encoder_ctx)
                    # 将编码器的输出特征 value 作为当前时刻的输入
                    atten_value = atten_value[:, None]  # [bs,e] -> [bs,1,e]
                    token_embed_t = token_embed_t + atten_value

                # 3. 调用 rnn 结构获取序列特征向量
                #    output: [bs,1,hidden_size] 每个 token 对应的特征向量
                output, hcn = self.rnn_layer(token_embed_t, hcn)

                # 5. 预测属于各个类别的置信度
                #    [bs,1,e] * [e,c] -> [bs,1,c]
                score = self.classify_layer(output)  # [bs, 1, vocab_size]

                # 6. 获取预测类别 id
                pred_ids = torch.argmax(score, dim=2)  # [bs,1] 获取置信度最大的下标作为预测类别 id
                token_ids_t = pred_ids  # 当前时刻的预测输出 token id 作为下一个时刻的输入

                # 7. 将当前时刻的预测结果和之前的结果合并到一起
                token_ids = torch.cat([token_ids, pred_ids], dim=1)  # [bs,dt+1]

                # 8. 判断是否结束生成逻辑：一般情况下至少两个条件
                #    —— 预测到结尾 token，预测总序列长度超过一定的限制
                if token_ids.shape[1] > 10:
                    break
            return token_ids  # 预测类别 id


# ============================================================
# Seq2Seq 主模块（编码器 + 解码器）
# ============================================================
class Seq2SeqModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, decoder_vocab_size=None):
        super().__init__()

        if decoder_vocab_size is None:
            decoder_vocab_size = vocab_size
        # 编码器创建
        self.encoder = EncoderModule(vocab_size, hidden_size)
        # 解码器创建
        self.decoder = DecoderModule(decoder_vocab_size, hidden_size)

    def forward(self, encoder_token_ids, decoder_token_ids):
        # 1. 获取编码器提取出来的特征向量信息
        encoder_ctx = self.encoder(encoder_token_ids)

        # 2. 获取解码器前向执行信息
        decoder_outputs = self.decoder(decoder_token_ids, encoder_ctx)

        return decoder_outputs


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    # ---------- 编码器案例 ----------
    encoder = EncoderModule(100, 128)
    token_ids = torch.randint(0, 20, size=(2, 5))
    encoder_ctx_embed = encoder(token_ids)
    print("编码器输出 shape:", encoder_ctx_embed.shape)
    # 期望输出: torch.Size([2, 5, 128])

    # ---------- 解码器案例 - 训练 ----------
    decoder = DecoderModule(100, 128)
    decoder.train()
    token_ids = torch.randint(0, 20, size=(2, 5))
    encoder_ctx_embed = torch.rand(2, 5, 128)

    decoder_score = decoder(token_ids, encoder_ctx_embed)
    print(f"解码器输出:{decoder_score.shape} - {decoder_score.dtype}")
    # 期望输出: torch.Size([2, 5, 100]) - torch.float32

    # ---------- 解码器案例 - 推理 ----------
    decoder = DecoderModule(100, 128)
    decoder.eval()
    token_ids = torch.randint(0, 20, size=(2, 1))
    encoder_ctx_embed = torch.rand(2, 5, 128)

    decoder_pred_token_ids = decoder(token_ids, encoder_ctx_embed)
    print(f"解码器输出:{decoder_pred_token_ids.shape} - {decoder_pred_token_ids.dtype}")
    # 期望输出: torch.Size([2, 11]) - torch.int64

    # ---------- Seq2Seq 案例 ----------
    seq2seq = Seq2SeqModule(10000, 64)
    print(seq2seq)

    loss_fn = nn.CrossEntropyLoss()

    # ---------- 训练过程测试 ----------
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    decoder_token_ids = torch.tensor([[3, 102, 235, 1523, 2132, 1243]])
    decoder_target_ids = torch.tensor([[102, 235, 1523, 2132, 1243, 4]])

    seq2seq.train()
    decoder_score = seq2seq(encoder_token_ids, decoder_token_ids)
    print(decoder_score.shape)  # torch.Size([1, 6, 10000])

    loss = loss_fn(torch.permute(decoder_score, dims=(0, 2, 1)), decoder_target_ids)
    print(loss)  # 例如: tensor(9.2312, grad_fn=<NllLoss2DBackward0>)

    # ---------- 推理预测过程测试 ----------
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    decoder_token_ids = torch.tensor([[3]])

    seq2seq.eval()
    pred_token_ids = seq2seq(encoder_token_ids, decoder_token_ids)

    print(pred_token_ids.shape)               # torch.Size([1, 11])
    print(f"预测 token id:\n\t{pred_token_ids}")
