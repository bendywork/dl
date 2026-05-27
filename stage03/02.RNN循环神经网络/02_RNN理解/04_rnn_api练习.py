import torch
from torch import nn


def test_rnn_api():
    """
    1. 循环神经网络（RNN）提取特征向量
     优点：能够处理变长输入；能够捕捉长距离依赖关系；能够捕捉序列中的位置信息
     问题：存在梯度消失和梯度爆炸问题；计算效率较低；难以并行化
     2. RNN API的使用
      - 输入数据的格式要求：输入数据的格式为[seq_len, batch_size, input_size]
        其中seq_len表示序列长度
        batch_size表示批次大小
        input_size表示输入特征向量的维度大小
      - 输出数据的格式：输出数据的格式为[seq_len, batch_size, hidden_size]
        其中seq_len表示序列长度
        batch_size表示批次大小
        hidden_size表示隐藏状态的维度大小
     3. RNN API的参数说明
      - input_size：输入特征向量的维度大小
      - hidden_size：隐藏状态的维度大小
      - num_layers：RNN的层数，默认为1
      - bias：是否使用偏置项，默认为True
      - batch_first：输入数据的格式是否为[batch_size, seq_len, input_size]，默认为False
      - dropout：RNN层之间的dropout概率，默认为0
      - bidirectional：是否使用双向RNN，默认为False
     4. RNN API的使用步骤
      - 定义RNN层，指定输入特征向量的维度大小和隐藏状态的维度大小
      - 准备输入数据，确保输入数据的格式符合RNN API的要求
      - 将输入数据传入RNN层，得到输出数据和隐藏状态
      - 根据需要对输出数据进行处理，例如取最后一个时间步的输出数据作为特征向量
     5. RNN API的应用场景
      - 语言模型：RNN可以用于语言模型的训练和推理，能够捕捉上下文信息，生成连贯的文本
      - 机器翻译：RNN可以用于机器翻译任务，能够捕捉源语言和目标语言之间的对应关系，生成准确的翻译结果
      - 语音识别：RNN可以用于语音识别任务，能够捕捉语音信号中的时间依赖关系，识别出正确的文本内容
      - 时间序列预测：RNN可以用于时间序列预测任务，能够捕捉时间序列中的趋势和周期性，预测未来的数值变化
     6. RNN API的改进和优化
      - LSTM（长短期记忆网络）：LSTM通过引入门控机制，解决了RNN中的梯度消失问题，能够更好地捕捉长距离依赖关系
      - GRU（门控循环单元）：GRU通过引入更新门和重置门，简化了LSTM的结构，能够更高效地捕捉长距离依赖关系
      - 双向RNN：双向RNN通过同时考虑正向和反向的序列信息，能够更全面地捕捉上下文信息，提升模型的性能
      - 堆叠RNN：堆叠RNN通过增加RNN层数，能够更深层次地提取特征向量，提升模型的表达能力
     7. RNN API的使用注意事项
      - 输入数据的格式必须符合RNN API的要求，否则会导致运行时错误
      - RNN的计算效率较低，尤其是在处理长序列时，可能会导致训练时间过长
      - RNN存在梯度消失和梯度爆炸问题，可能会导致模型训练不稳定，需要使用适当的优化方法
     8. RNN API的总结
      - RNN是一种能够处理序列数据的神经网络结构，能够捕捉序列中的时间依赖关系和上下文信息
      - RNN API提供了方便的接口来定义和使用RNN层，能够帮助我们快速构建RNN模型
      - RNN API的使用需要注意输入数据的格式和计算效率等问题
     9. RNN API的未来发展方向
      - 更高效的RNN结构：研究更高效的RNN结构，例如Transformer等，能够更好地捕捉长距离依赖关系，提升模型的性能
      - 更好的优化方法：研究更好的优化方法，例如AdamW等，能够更好地解决RNN中的梯度消失和梯度爆炸问题，提升模型的训练稳定性
      - 更广泛的应用场景：将RNN应用于更多的领域，例如图像处理、推荐系统等，能够更好地发挥RNN的优势
    """
    bs, t, e = 1, 5, 128 # bs个样本 每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为e
    rnn = nn.RNN(input_size=e, # 输入特征向量的维度大小为e
                 hidden_size=2 * e, # 隐藏状态的维度大小为2e
                 num_layers=1,  # RNN的层数，默认为1
                 bias=True,  # 是否使用偏置项，默认为True
                 batch_first=True, # 输入数据的格式为[batch_size, seq_len, input_size]
                 dropout=0, # RNN层之间的dropout概率，默认为0
                 bidirectional=True, # 双向RNN
                 dtype=torch.float32)
    token_embs = torch.randn(bs, t, e, dtype=torch.float32)  # 一般为上一个模块的输出特征向量
    output, hidden = rnn(token_embs) # 将输入数据传入RNN层，得到输出数据和隐藏状态
    print("output.shape:", output.shape) # 输出数据的格式为[batch_size, seq_len, hidden_size]
    print("hidden.shape:", hidden.shape) # 隐藏状态的格式为[1, batch_size, hidden_size]


if __name__ == "__main__":
    test_rnn_api()
