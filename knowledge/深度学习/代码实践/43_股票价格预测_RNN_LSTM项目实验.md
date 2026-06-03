# 43_股票价格预测 · RNN+LSTM 项目实验

## 项目目标

用 RNN + LSTM 对股票历史价格做时序预测，预测未来 N 天的收盘价走势。

通过这个项目掌握：
- 时序数据的预处理与滑动窗口构造
- RNN / LSTM / GRU 在回归任务上的应用
- 双向 LSTM 与多层 LSTM 的使用
- 模型评估与可视化

---

## 推荐 Kaggle 数据集

| 数据集 | 说明 | 链接 |
|--------|------|------|
| MAANG Companies Stock Prices | Meta/Apple/Amazon/Netflix/Google 每日股价，持续更新 | https://www.kaggle.com/datasets/nikhil1e9/maang-stock-prices |
| Huge Stock Market Dataset | NYSE 大量股票历史数据 | https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs |
| New York Stock Exchange | S&P500 成分股历史数据 | https://www.kaggle.com/datasets/dgawlik/nyse |

**推荐从 MAANG 数据集开始**，数据干净，字段清晰，文件小，适合入门。

---

## 数据字段说明

标准股票 CSV 包含以下字段：

| 字段 | 说明 |
|------|------|
| Date | 日期 |
| Open | 开盘价 |
| High | 当日最高价 |
| Low | 当日最低价 |
| Close | 收盘价（预测目标） |
| Volume | 成交量 |

---

## 实验阶段规划

### 阶段一：数据准备
- [ ] 下载数据集，加载单只股票（如 AAPL）
- [ ] 可视化收盘价历史走势
- [ ] 归一化处理（MinMaxScaler，缩放到 0~1）
- [ ] 滑动窗口构造序列样本（window_size=60，预测未来1天）
- [ ] 划分训练集 / 验证集 / 测试集（8:1:1）

**滑动窗口逻辑：**
```
输入：第 0~59 天的收盘价 → 预测第 60 天
输入：第 1~60 天的收盘价 → 预测第 61 天
...
输入序列 shape：(samples, window_size, features) = (N, 60, 1)
输出 shape：(samples, 1)
```

### 阶段二：手写 RNN 基线版本
- [ ] 用前面复习的手写 RNN 跑一遍
- [ ] 理解 input_size=1（单特征：只用收盘价）
- [ ] 观察训练曲线，记录 MSE/MAE

### 阶段三：torch.nn.RNN 官方版本
- [ ] 用 `torch.nn.RNN` 替换手写版
- [ ] 对比两者输出是否一致
- [ ] 调整 hidden_size、num_layers 观察效果变化

### 阶段四：升级到 LSTM
- [ ] 将 RNN 替换为 `torch.nn.LSTM`
- [ ] 注意 LSTM 返回 (output, (h_n, c_n))，多了 cell state
- [ ] 对比 RNN vs LSTM 的预测曲线

### 阶段五：升级到双向 LSTM + 多层
- [ ] `bidirectional=True`，hidden 维度变化
- [ ] `num_layers=2`，堆叠 LSTM
- [ ] Dropout 防止过拟合

### 阶段六：多特征输入（进阶）
- [ ] 加入 Open / High / Low / Volume，input_size=5
- [ ] 观察多特征是否比单特征收盘价效果更好

### 阶段七：可视化与评估
- [ ] 反归一化，还原真实价格
- [ ] 画预测曲线 vs 真实曲线
- [ ] 计算 RMSE / MAE / MAPE

---

## 核心模型结构

```python
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (bs, window_size, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # 只取最后一个时刻的输出
        out = self.fc(out[:, -1, :])  # (bs, output_size)
        return out
```

---

## 关键概念对应

| 项目概念 | 对应今天复习的知识 |
|---------|----------------|
| 每天的收盘价 | 每个时刻的 token |
| window_size=60 | t=60，序列长度 |
| input_size=1 | e=1，特征维度（只用收盘价） |
| hidden_size | 隐藏层维度，超参数 |
| out[:, -1, :] | 取最后时刻 h_T，对应 h_prev |
| fc 输出层 | 把 h_T 映射成预测价格 |

---

## 参考 Kaggle Notebook

- RNN + LSTM 对比：https://www.kaggle.com/code/ozkanozturk/stock-price-prediction-by-simple-rnn-and-lstm
- RNN + LSTM + GRU 三模型对比：https://www.kaggle.com/code/raoulma/ny-stock-price-prediction-rnn-lstm-gru
- LSTM 多步预测：https://www.kaggle.com/code/thibauthurson/stock-price-prediction-with-lstm-multi-step-lstm
- MAANG 股票 LSTM 预测：https://www.kaggle.com/code/nikhil1e9/stock-price-forecasting-using-lstm

---

## 开发顺序建议

```
01_data_explore.py     → 加载数据，可视化，理解字段
02_data_preprocess.py  → 归一化，滑动窗口，DataLoader
03_rnn_baseline.py     → 手写 RNN 基线
04_rnn_official.py     → torch.nn.RNN 官方版
05_lstm_model.py       → LSTM 替换 RNN
06_bilstm_model.py     → 双向多层 LSTM
07_evaluate.py         → 评估 + 可视化
```

---

## 注意事项

1. **不要用未来数据**：划分数据集时按时间顺序切，不能随机 shuffle
2. **归一化要在划分后做**：用训练集的 scaler fit，再 transform 验证集和测试集
3. **股票预测本质是困难的**：项目目标是掌握 LSTM 的使用方式，不是真的炒股
4. **LSTM 的 h0 和 c0**：都需要初始化，和 RNN 只有 h0 不同
