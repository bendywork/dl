# RNN二次复习案例代码
import torch


# rnn是循环神经网络
# 对比与普通全连接神经网络，RNN的结构更加灵活，没有那么死板，
# 普通的全连接每一个token是独立的，互不干扰，学习到的内容是不关联的
# 而RNN是联通的，因为他学习的是一整个语言序列的内容而不是单一token内容
# 所以RNN是讲究时间顺序的，
# 每一个输入包含 bs个样本，对应到矩阵数据就是bs行数据，一行数据就是一个样本，每一列就是一个特征
# 每一个样本包含t个时刻，每一个时刻对应一个token其实就是将一个语言序列分割为t个token，每一个token对应一个时刻
# 每一个时刻我们通过embeeding转为e维度的词向量（高维度稠密的向量）
# 所以输入是bs，t，e的shape形状
bs,t,e = 1,5,128
# rnn除了会处理当前输入的向量之外，还会处理一个隐藏状态，因为从纵向去看，每一个token处理成向量之后各自做全连接
# 但是从横向去看，每一个token与token之间还会有二次联系，这个联系就是隐层，初始状态下，隐层为0
# 因为输入是1，5，128 而对于单个token来说，其实他的shape是1，128 又因为隐层是直接与全连接之后的结果做mautl所以
# 需要看全连接之后的shape 对于单个token 每一个token输入的shape是1，128 fc之后设定输出是256，那么一次fc之后的
# 输出是1，256，那隐层是多少，是256，256吗？
#
# ── 导师批注 ──────────────────────────────────────────────────
# [✅ 正确] h（隐藏状态）的存在以及初始为0，这个理解是对的
#
# [⚠️ 偏差] "纵向FC" 和 "横向隐层" 不是两个独立操作
#   你的理解：先对 x_t 单独做FC，再把结果与隐层做某种运算
#   实际情况：x_t 和 h_{t-1} 是同时作为输入，一起参与一次线性变换
#   没有"先FC再接隐层"这个步骤，是一步完成的
#
# [❌ 根本错误] h 不是权重矩阵，它是一个状态向量（数据）
#   你的理解：隐层 h 是一个形如 (256, 256) 的权重矩阵，与FC结果做matmul
#   实际情况：h 是 shape=(bs, hidden_size) 的向量，随时间步传递，是"记忆"
#   真正是矩阵的是 W_hh，它作用在 h 上，shape 才是 (hidden_size, hidden_size)
#
# [💡 正确公式] h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
#   x_t:     (bs, e)           输入当前token
#   h_{t-1}: (bs, hidden_size) 上一时刻的记忆状态（数据，不是权重）
#   W_xh:    (e, hidden_size)  输入到隐层的权重
#   W_hh:    (hidden_size, hidden_size) 隐层到隐层的权重（这才是矩阵）
#   h_t:     (bs, hidden_size) 新的记忆状态，传给下一时刻
# ──────────────────────────────────────────────────────────────
# 原来如此，我的理解是错误的，我开始修改
# 首先需要明确，输入x与h（隐层输入）是同时进行的，所以需要先初始化与x做运算的fc函数
# 对于单个token他是bs，e所以输入就是：
hidden_size = 256
# 词表大小 1000
vaca_size = 1000
# 1000个单词，每个单词映射成e维度的向量
embeddings = torch.nn.Embedding(1000, e)
# 这句话是啥意思
token_ids = torch.randint(0, vaca_size, (bs, t))
input_x = torch.nn.Linear(e,hidden_size)
w_xh = input_x.weight
input_h = torch.nn.Linear(hidden_size, hidden_size)
w_hh, h_b = input_h.weight, input_h.bias
h0 = torch.zeros(bs, hidden_size)
# 现在写法对吗？
h_prev = h0
embeddings_new = []
for t_index in range(t):
    # 对每一个token构造embedding向量
    x_t = embeddings(token_ids[:, t_index])
    # 疑问需要解答，为什么这里是 x_t @ w_xh 维度不匹配！应该是 x_t @ w_xh.T 或直接用 input_x(x_t) 为什么w的参数的shape是这样的，需要解释说明
    h_t = torch.tanh(x_t @ w_xh.T + h_prev @ w_hh.T + h_b)
    h_prev = h_t
    embeddings_new.append(h_prev[:,None])
all_output = torch.cat(embeddings_new, dim=1)
print(f"最终 h_T shape: {h_prev.shape}")
print(f"所有时刻输出 shape: {all_output.shape}")
# print(f"最终 h_T shape: {h_prev.shape}")
# 最后的结果怎么拿到




# ── 你的理解 ─────────────────────────────────────────────────
# 1. embeddings_new = [] 循环外初始化 ✅
# 2. 循环里 append(h_prev[:,None]) 收集每时刻输出 ✅
# 3. 循环后 torch.cat(embeddings_new, dim=1) 拼接 ✅
# 4. 两个 print 验证 shape ✅
# ─────────────────────────────────────────────────────────────

# ── 导师建议 ─────────────────────────────────────────────────
# [✅✅✅] 手写 RNN 完整正确！运行后应该看到：
#   最终 h_T shape:    torch.Size([1, 256])
#   所有时刻输出 shape: torch.Size([1, 5, 256])
#
# ══ 阶段总结：手写 RNN 前向传播 ══
#   输入流程：token_ids → Embedding查表 → x_t (bs,e)
#   每时刻计算：h_t = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b)
#   两种输出：h_T (bs,hidden) 用于分类；all_output (bs,t,hidden) 用于序列任务
#
# [下一步：用 torch.nn.RNN 官方实现对比]
#   在文件末尾新写一段，用一行官方 API 复现相同功能：
#
#   rnn_layer = torch.nn.RNN(input_size=e, hidden_size=hidden_size, batch_first=True)
#   x_input = embeddings(token_ids)          # 一次取出所有时刻，shape=(bs,t,e)
#   h0_official = torch.zeros(1, bs, hidden_size)  # 官方 h0 多一个 num_layers 维
#   output, h_n = rnn_layer(x_input, h0_official)
#   print(f"官方 output shape: {output.shape}")    # (1,5,256) 对应 all_output
#   print(f"官方 h_n shape:    {h_n.shape}")       # (1,1,256) 对应 h_T
#
#   写完保存，看两段结果的 shape 是否一致
# ─────────────────────────────────────────────────────────────
# __TUTOR_SAVE__ = "2026-06-03 15:08:30"