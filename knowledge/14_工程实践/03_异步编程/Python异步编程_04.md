# Python 异步编程：协程、多线程、多进程

## 📌 核心问题
> NLP 服务要同时处理多个请求，用多线程/多进程/协程，哪个更合适？

## 🌱 根源与动机

Python 的 **GIL（全局解释器锁）** 限制了真正的并行：
- **多线程**：适合 I/O 密集（GIL 在 I/O 等待时释放）
- **多进程**：适合 CPU 密集（绕过 GIL，各自独立内存）
- **协程**：适合高并发 I/O（单线程异步，切换开销最小）

## 📐 理论推导

### 协程（asyncio）

```python
import asyncio

async def fetch(url):           # 协程函数
    await asyncio.sleep(1)      # 挂起，让出控制权
    return f"result:{url}"

async def main():
    results = await asyncio.gather(
        fetch("url1"), fetch("url2")   # 并发执行
    )
    print(results)

asyncio.run(main())
```

原理：事件循环 → 遇到 `await` 挂起当前协程 → 执行其他就绪协程 → 等待完成后继续

### 多线程

```python
from concurrent.futures import ThreadPoolExecutor

def process(text):
    return model.predict(text)    # I/O 密集任务

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process, texts))
```

### 多进程

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_task(data):
    return heavy_compute(data)   # CPU 密集任务

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_task, dataset))
```

### 生成器/异步生成器（大数据集迭代）

```python
async def data_generator(dataset):
    for batch in dataset:
        await asyncio.sleep(0)    # 让出 I/O
        yield batch               # 逐批生成，不占用内存

async def train():
    async for batch in data_generator(dataset):
        model.train_step(batch)
```

## 💡 关键理解

| 场景 | 推荐方案 |
|------|---------|
| 模型推理（CPU/GPU 密集）| 多进程 |
| 批量 HTTP 请求 / 数据库查询 | 协程 |
| 文件 I/O / 队列消费 | 多线程或协程 |
| FastAPI 服务器 | 协程（async） |

- **协程不是并行**，是交叉执行；多进程是真正并行
- **注意 fork 安全**：多进程 + CUDA 模型要用 `spawn` 启动方式

## 🔧 代码实现

对应代码：`knowledge/工程实践/异步编程/`
- `01_协程基础案例.py`
- `02_多线程案例.py`
- `03_多进程案例.py`
- `04_协程数据迭代生成.py`

## ⚠️ 易错点与常见误解

1. **协程函数不自动运行**，必须用 `asyncio.run()` 或 `await`
2. **多进程共享模型很贵**：进程间传数据要序列化，模型重复加载用 `Manager` 或共享内存
3. **GIL 只影响 CPU 计算**，纯 PyTorch 操作其实部分释放 GIL（底层是 C++ 线程）
4. **不要在协程中调用阻塞函数**（如 `time.sleep`），用 `await asyncio.sleep`

## 🔗 知识延伸

- [[Flask与FastAPI部署]] — FastAPI 基于 asyncio
- [[LLM基础与训练]] — 大模型训练用多进程分布式

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/扩展_02_Python异步编程.pdf`
