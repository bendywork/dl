# 二、结合分词、token向量化转换 + 机器学习(LR) / 深度学习，完成datas / text_classify文件夹中的文本分类模型的训练
# 完成：
# 训练: 如果替换一个数据集，如何在最小改动的情况下，完成新数据集的模型训练
# 推理：在训练完成的基础上，最终支持给定任何一个文本，返回该文本对应的类别id、类别名称以及预测所属概率，支持返回TopK的结果；
import math
import os.path
import pandas as pd
from sklearn.externals.array_api_compat import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim


class FcConfig(object):

    BASE_DIR = "/Users/scx/data/codeData/python/dl"

    TRAIN_PATH = os.path.join(BASE_DIR, "base", "datas", "text_classify", "train_tokens.csv")

    TEST_PATH = os.path.join(BASE_DIR, "base", "datas", "text_classify", "test.csv")

    BATCH_SIZE = 5

    VOCABULARY_SIZE = 10000

    EMBEDDING_SIZE = 128

    LAYER_SIZE = 2

    LABEL_SIZE = 10

    LEARNING_RATE = 0.01

    EPOCHS = 10

    HIDDEN_SIZE = 128



class Vocabulary(object):

    def __init__(self):
        self.vocabulary_size = 0
        self.id2word = {}
        self.word2id = {}
        self.label_size = 0
        self.label_id2label = {}
        self.label_label2id = {}

    def load_file(self, file_path):
        if not os.path.exists(file_path):
            raise Exception("The given file doesn't exist")
        data_frame = pd.read_csv(filepath_or_buffer=file_path, sep="\t", header=None)
        data = data_frame[0]
        label = data_frame[1]
        original_data_list = data.str.split(" ")
        # print(original_data_list)
        vocabulary_set = set()
        label_set = set()
        for item in original_data_list:
            for item_word in set(item):
                vocabulary_set.add(item_word)
        vocabulary_ = ["<PAD>", "<UNK>"] + list(vocabulary_set)
        for item in label:
            label_set.add(str(item))
        self.label_size = len(label_set)
        self.label_id2label = {i: label for i, label in enumerate(label_set)}
        self.label_label2id = {label: i for i, label in self.label_id2label.items()}
        self.vocabulary_size = len(vocabulary_)
        self.id2word = {i: word for i, word in enumerate(vocabulary_)}
        self.word2id = {word: i for i, word in self.id2word.items()}
        pass


class DataSet(Dataset):
    def __init__(self, file_path,  vocabulary):
        # 1. 读 CSV，拿到文本列和标签列
        csv_data = pd.read_csv(filepath_or_buffer=file_path, sep="\t", header=None)
        # 拿到文本列和标签列
        # print(csv_data.shape)
        csv_text_column, csv_label_column = csv_data[0], csv_data[1]
        # print(csv_text_column.shape, csv_label_column.shape)
        # print(type(csv_text_column))
        # 2. 遍历每行文本 → 词袋法向量化 → append 到 self.samples
        self.samples = []
        self.labels = []
        # print(vocabulary.word2id)
        for item in csv_text_column:
            vector = [0] * vocabulary.vocabulary_size
            split_list = item.split(" ")
            for it in split_list:
                 vector[vocabulary.word2id.get(it, 1)] += 1
            self.samples.append(vector)
            # self.samples.append([vocabulary.word2id.get(word, 1) for word in split_list])
        # 3. 遍历每行标签 → label2id → append 到 self.labels
        # self.labels = [vocabulary.label_label2id[str(item)] for item in csv_label_column]
        self.labels = [vocabulary.label_label2id[str(it)] for it in csv_label_column]
        # print(self.labe)
        pass

    def __getitem__(self, index):
        """
        继承自DataSet只需要实现getitem与lens方法就会自动完成后续的工作
        """
        x_sample = torch.tensor(self.samples[index], dtype=torch.float32)
        y_sample = torch.tensor(self.labels[index], dtype=torch.long)
        return x_sample, y_sample
    
    def __len__(self):
        return len(self.samples)


class FcModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FcModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features= input_size, out_features= hidden_size, bias=False,dtype= torch.float32),
            nn.ReLU(),
            nn.Linear(in_features= hidden_size, out_features= hidden_size, bias=False,dtype= torch.float32),
            nn.ReLU(),
            nn.Linear(in_features= hidden_size, out_features= output_size, bias=False,dtype= torch.float32),
        )

    def forward(self, x):
        return self.fc(x)


class Trainer(object):
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        pass

    def train(self):
        for epoch in range(self.config.EPOCHS):
            for batch_index, (batch_samples, batch_labels) in enumerate(self.train_loader):
                # 1. 前向传播
                outputs = self.model(batch_samples)
                # 2. 计算损失
                loss = self.criterion(outputs, batch_labels)
                # 3. 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 4. 打印损失
                if batch_index % 10 == 0:
                    print("Epoch: %d, Batch: %d, Loss: %f" % (epoch, batch_index, loss.item()))


class Predictor(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.model.eval()

    # def predict(self):
    #     with torch.no_grad():
    #         for batch_index, (batch_samples, batch_labels) in enumerate(self.test_loader):
    #             outputs = self.model(batch_samples)
    #             _, predicted = torch.max(outputs, dim = 1)
    #             print("预测结果:", list(predicted))
    #             print("真实标签:", list(batch_labels))
    #             print("准确率:", sum(predicted == batch_labels).item() / len(batch_labels))
    #             break

    def predict_topk(self, text, k=3):
        """
        predict_topk 是对单条文本做预测，直接传字符串进来就行。
        """
        # text 是一个字符串，比如 "帮我 播放 一首 歌"
        vector = [0] * self.vocab.vocabulary_size
        split_list = text.split(" ")
        # 词袋向量化
        for word in split_list:
            vector[self.vocab.word2id.get(word, 1)] += 1
        # vector = torch.tensor(vector, dtype=torch.float32)
        # 2. 转 tensor，shape 要是 [1, vocab_size]（加 batch 维度）
        x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        # 3. forward → logits [1, 12]
        with torch.no_grad():
            # 推理不需要计算梯度
            logits = self.model(x)
        # 4. softmax → 概率
        softmax_probs = torch.softmax(logits, dim=1)
        # 5. topk → 取前 k 个
        topk_probs, topk_ids = torch.topk(softmax_probs, k=k)
        # 6. 用 id2label 转回类别名称
        topk_labels = [self.vocab.label_id2label[id.item()] for id in topk_ids.squeeze(0)]
        return topk_labels, topk_probs



class Main(object):
    def __init__(self):
        self.config = FcConfig()
        pass

    def run(self):
        voca = Vocabulary()
        voca.load_file(file_path=FcConfig.TRAIN_PATH)
        # print("词表大小:", voca.vocabulary_size)
        # print("类别数:", voca.label_size)
        # print("类别列表:", list(voca.label_label2id.keys()))
        # data_set = DataSet(file_path=FcConfig.TRAIN_PATH, vocabulary=voca)
        train_dataset = DataSet(FcConfig.TRAIN_PATH, vocabulary=voca)
        print(len(train_dataset))  # 应该是 12100
        print(len(train_dataset[0][0]))  # 应该是词表大小
        print(train_dataset[12][1])  # 应该是一个整数（标签id）
        train_loader = DataLoader(train_dataset, batch_size=FcConfig.BATCH_SIZE, shuffle=True)
        # for batch_index, (batch_samples, batch_labels) in enumerate(train_loader):
        #     print(batch_index, batch_samples.shape, batch_labels.shape)
        #     break
        model = FcModel(
            input_size=voca.vocabulary_size,
            hidden_size=FcConfig.HIDDEN_SIZE,
            output_size=voca.label_size
        )
        print(model)
        trainer = Trainer(model= model, train_loader = train_loader, config= self.config)
        trainer.train()
        torch.save(model.state_dict(), "model.pth")

        # 推理
        predictor = Predictor(model=model, vocab=voca)
        labels, probs = predictor.predict_topk("播放 一首 歌", k=3)
        for label, prob in zip(labels, probs[0].tolist()):
            print(f"{label}: {prob:.4f}")

if __name__ == '__main__':
    Main().run()
