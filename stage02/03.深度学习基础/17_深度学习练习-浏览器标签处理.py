import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup as bs
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class bookmark_classify(nn.Module):

    def __init__(self, input_size, output_size):
        super(bookmark_classify, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
        

    def forward(self, x):
        score = self.features(x)
        if self.training:
            return score
        return torch.argmax(score, dim=1)
        # 【修复1-原】始终返回logits，不再根据training/eval模式区分返回值
        # return score




class bookmark_file_reader():
    
    def __init__(self, file_path):
        self.file_path = file_path


    '''
    读取书签文件，返回BeautifulSoup对象
    '''
    def read_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            soup = bs(f, 'html.parser')
        return soup
    
    def parse_bookmarks(self, soup):
        bookmarks = {}
        for h3 in soup.find_all('h3'):
            # 书签文件夹标题
            title = h3.get_text()
            title_couple = {}
            # 获取书签文件夹下的所有链接
            for a in h3.find_next_siblings('dl')[0].find_all('a'):
                url = a.get('href')
                title_couple[a.get_text()] = url 
            bookmarks[title] = title_couple
        return bookmarks

    def parse_bookmarks_2_list_data(self, soup):
        data = []
        for h3 in soup.find_all('h3'):
            title = h3.get_text()
            temp = {}
            temp['category'] = title
            for a in h3.find_next_siblings('dl')[0].find_all('a'):
                url = a.get('href')
                temp['url'] = url
                temp['title'] = a.get_text()
                temp['text'] = a.get_text() + ' ' + url
                data.append(temp.copy())
        return data
    
    


CATEGORY_MAP = {
    # 学习
    'AI': '学习', 'AI自学': '学习', 'JavaEE': '学习', 'JavaFX': '学习', 'JavaSE': '学习',
    'Java技术体系': '学习', 'Java面试复习': '学习', 'Flutter技术体系': '学习',
    'Golang技术体系': '学习', 'Python技术体系': '学习', 'Rust技术体系': '学习',
    'VertX技术栈': '学习', 'j2me': '学习', '前端技术体系': '学习', '基础': '学习',
    '学习': '学习', '学习项目': '学习', '其他学习': '学习', '技术学习': '学习',
    '笔记学习': '学习', '网课': '学习', '课程': '学习', '教程': '学习',
    '教程文档': '学习', '算法': '学习', '机器学习': '学习', '深兰AI': '学习',
    '音乐学习': '学习', '英语专栏': '学习', '学术': '学习',
    # 工具
    '工具': '工具', '在线工具': '工具', '开发': '工具', '代码': '工具',
    'codeSpace': '工具', '接口': '工具', '速用': '工具', '常用': '工具',
    '导航': '工具', '搜索引擎': '工具', '图床': '工具', '字体': '工具',
    '素材': '工具', '图片': '工具', '镜像': '工具', '邮箱': '工具',
    # 娱乐
    '娱乐': '娱乐', 'LOL': '娱乐', 'DOOM': '娱乐', '游戏': '娱乐',
    '音影': '娱乐', '媒体': '娱乐', '放松': '娱乐', '传奇私服': '娱乐',
    # 技术
    '技术': '技术', '技术栈': '技术', '开源': '技术', '源码': '技术',
    '建站': '技术', '服务器': '技术', '自建服务器': '技术', '运维': '技术',
    '项目': '技术', '公益项目': '技术', '我的项目': '技术', '案例': '技术', 'web3': '技术',
    # 资源
    '资源': '资源', '资讯': '资源', '软件': '资源', '下载': '资源',
    '白嫖': '资源', '激活': '资源', '激活专区': '资源', 'idea激活': '资源',
    '图书': '资源', '装机': '资源', '机场': '资源',
    # 生活
    '生活': '生活', '社交': '生活', '社区': '生活', '商城': '生活',
    '商店': '生活', '财务': '生活', '钱包': '生活', '维修': '生活',
    '出海': '生活', '公益': '生活', '账号': '生活',
    # 职业
    '工作': '职业', '简历': '职业', '简历样式': '职业', '面试': '职业',
    '私活': '职业', '私活项目': '职业', '经验': '职业', '博客': '职业',
    '他人博客': '职业', '我的': '职业',
    # 其他
    'OOO': '其他', '书签栏': '其他', '新建文件夹': '其他', '杂七杂八': '其他',
    '综合': '其他', '偏门': '其他', '小玩意': '其他', '有趣': '其他',
    '尝试': '其他', '机遇': '其他', '信息收集': '其他', '安全': '其他',
    '网络安全': '其他', '社工': '其他', '论坛': '其他',
}


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, '..', '..', 'resources', 'bookmarks.html')
    reader = bookmark_file_reader(file_path)
    soup = reader.read_file()
    data = reader.parse_bookmarks_2_list_data(soup)

    # 类别映射，过滤掉未映射的
    mapped_data = []
    for item in data:
        cat = CATEGORY_MAP.get(item['category'])
        if cat:
            mapped_data.append({'text': item['text'], 'category': cat})

    print(f"映射后数据量: {len(mapped_data)}")

    # TF-IDF 转向量
    text_list = [item['text'] for item in mapped_data]
    category_list = [item['category'] for item in mapped_data]

    vectorizer = TfidfVectorizer(max_features=500)
    x = vectorizer.fit_transform(text_list).toarray()

    encoder = LabelEncoder()
    y = encoder.fit_transform(category_list)

    print("TF-IDF特征矩阵形状:", x.shape)
    print("类别标签形状:", y.shape)
    print("各类别数量:")
    for i, cls in enumerate(encoder.classes_):
        print(f"  {cls}: {(y == i).sum()}")

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 转为tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 模型训练 input size = 500, output size = 类别数 
    net = bookmark_classify(input_size=x_train.shape[1], output_size=len(encoder.classes_))
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)
    loss_fnb = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    total_epoch = 1000
    batch_size = 16
    total_train_batch = len(x_train) // batch_size
    perm = torch.randperm(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    for epoch in range(total_epoch):
        net.train()
        for batch_idx in range(total_train_batch):
            x_batch = x_train[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            y_batch = y_train[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            score = net(x_batch)
            loss = loss_fnb(score, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        total_test_batch = len(x_test) // batch_size
        with torch.no_grad():
            x_test_batch = x_test[:total_test_batch * batch_size]
            y_test_batch = y_test[:total_test_batch * batch_size]
            pred = net(x_test_batch)
            acc = (pred == y_test_batch).float().mean().item()
            print(f"第{epoch}轮评估完成，评估准确率为: {acc:.3f}")

        print(f"第{epoch}轮训练完成，loss值为: {loss.item():.4f}")
        # ===================== 修复版本（暂不启用）=====================
        # # 【修复2】每个epoch重新shuffle
        # perm = torch.randperm(len(x_train))
        # x_train_shuffled = x_train[perm]
        # y_train_shuffled = y_train[perm]
        #
        # # 【修复3】累计全epoch平均loss
        # total_loss = 0.0
        # total_samples = 0
        # # 【修复4】向上取整，不丢尾部batch
        # total_train_batch = (len(x_train) + batch_size - 1) // batch_size
        #
        # net.train()
        # for batch_idx in range(total_train_batch):
        #     si = batch_idx * batch_size
        #     ei = min(si + batch_size, len(x_train_shuffled))
        #     x_batch = x_train_shuffled[si:ei]
        #     y_batch = y_train_shuffled[si:ei]
        #     score = net(x_batch)
        #     loss = loss_fnb(score, y_batch)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item() * len(x_batch)
        #     total_samples += len(x_batch)
        # avg_train_loss = total_loss / total_samples
        #
        # # 【修复5+6】eval用全部测试集，forward返回logits可同时算loss和acc
        # net.eval()
        # with torch.no_grad():
        #     score = net(x_test)
        #     eval_loss = loss_fnb(score, y_test)
        #     pred = torch.argmax(score, dim=1)
        #     acc = (pred == y_test).float().mean().item()
        #     print(f"第{epoch}轮评估完成，评估准确率为: {acc:.3f}，评估loss为: {eval_loss.item():.4f}")
        # print(f"第{epoch}轮训练完成，平均loss值为: {avg_train_loss:.4f}")