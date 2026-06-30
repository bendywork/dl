# LangChain-ChatChat 基础安装和使用（结合 DeepSeek）

> 来源：CSDN @青花瓷 | 2025-03-18 | 阅读量约2k | CC 4.0 BY-SA
> 原文：https://blog.csdn.net/quickrubber/article/details/146341916

## 前言

LangChain-ChatChat 可以创建本地知识库，能对各种大模型进行整合。本文将本地架设的 DeepSeek 整合进来实现本地知识库功能。

## 一、下载

GitHub: https://github.com/chatchat-space/Langchain-Chatchat

## 二、Pycharm 工程配置

创建 Pycharm 工程，将 Langchain-Chatchat 文件复制到工程中，Python 3.9。

## 三、安装依赖

```bash
pip install langchain-chatchat -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 四、启动准备

### 4.1 初始化

```bash
chatchat init
```

### 4.2 修改 model_setting.yaml

- 修改默认 LLM 名称为目标模型
- 平台名称改为 `ollama`
- 修改 MODEL_PLATFORMS 参数

修改完成后运行：

```bash
chatchat kb -r
```

### 4.3 向量库 Samples 加载失败

若报"向量库 Samples 加载失败"，需下载 embedding 模型：

```bash
ollama pull quentinz/bge-large-zh-v1.5
```

### 4.4 重新加载

```bash
chatchat kb -r
```

Samples 知识库加载成功。

## 五、启动 LangChain-chatchat

```bash
chatchat start -a
```

### 5.1 RAG 报错：`TypeError: __init__() got an unexpected keyword argument 'proxies'`

原因：httpx 版本过高，需降级：

```bash
pip install httpx==0.27.0
```

降级后重新 `chatchat start -a`，RAG 对话恢复正常。

## 总结

完整流程：

1. 安装 langchain-chatchat 包
2. `chatchat init` 初始化
3. 修改 model_setting.yaml 对接 Ollama 上的 DeepSeek
4. `ollama pull` 下载 embedding 模型（bge-large-zh-v1.5）
5. `chatchat kb -r` 重建知识库
6. 解决 httpx 版本兼容问题（降级到 0.27.0）
7. `chatchat start -a` 启动并验证

## 原文截图

> CSDN 防盗链，直接嵌入不显示。在浏览器打开原文后图片才会加载，或手动下载以下 URL。

| # | 描述 | URL |
|---|------|-----|
| 1 | GitHub 页面 | https://i-blog.csdnimg.cn/direct/62916f58bf0240ca8b6a65db1bdbfb02.png |
| 2 | Pycharm 工程目录 | https://i-blog.csdnimg.cn/direct/ba9e59bdbe1246e8b9e01c99f919b23b.png |
| 3 | 安装过程 | https://i-blog.csdnimg.cn/direct/02fa1a05393e4386971058a96f30a882.png |
| 4 | 安装完毕 | https://i-blog.csdnimg.cn/direct/db9139d21c3d42b4a489477e70d01c4e.png |
| 5 | chatchat init | https://i-blog.csdnimg.cn/direct/452d16db8a82406caa7b89ebd86290ea.png |
| 6 | model_setting.yaml 修改 | https://i-blog.csdnimg.cn/direct/ed93cf1d6ea649cea94f47ae5e45682c.png |
| 7 | kb -r 执行 | https://i-blog.csdnimg.cn/direct/7363cfa615424eed9446c80afc0562ee.png |
| 8 | 向量库加载失败 | https://i-blog.csdnimg.cn/direct/7f9cadc143eb4a628de4f7ae3c79722c.png |
| 9 | ollama pull embedding | https://i-blog.csdnimg.cn/direct/65ed4ca9e3da434e99983a3429eee2f7.png |
| 10 | 模型路径 | https://i-blog.csdnimg.cn/direct/472504446b5a4a049174d5e623209aee.png |
| 11 | 加载成功 | https://i-blog.csdnimg.cn/direct/253ea157f49d4297b302d0fb39d43d87.png |
| 12 | 启动日志 | https://i-blog.csdnimg.cn/direct/5bd8c4a7dc104258b32c8de2ae9366b5.png |
| 13 | 多功能对话界面 | https://i-blog.csdnimg.cn/direct/c19f87dc8ff746eabb08eb8550147509.png |
| 14 | 知识库管理界面 | https://i-blog.csdnimg.cn/direct/5d272d2f86eb4e0fa9b86dfb5fe85027.png |
| 15 | RAG 错误 | https://i-blog.csdnimg.cn/direct/e17ea501b585400c823a77fedae75271.png |
| 16 | 调试界面错误 | https://i-blog.csdnimg.cn/direct/ae6bebb1d3614be599c73449ee346c36.png |
| 17 | 参考博客 | https://i-blog.csdnimg.cn/direct/14d4dfe88d6748909a8f00d04c34b59a.png |
| 18 | httpx 原版本 | https://i-blog.csdnimg.cn/direct/12c86b648aa84725b663f6c331a3f33c.png |
| 19 | 版本降级过程 | https://i-blog.csdnimg.cn/direct/ba28e243bc5e4a8190610063086cd9f5.png |
| 20 | 降级后版本 | https://i-blog.csdnimg.cn/direct/f9c486f8076044b1a358aa4c95517e18.png |
| 21 | 重新启动成功 | https://i-blog.csdnimg.cn/direct/790cc5cc6cad44eb82cf290f48cd4061.png |
| 22 | RAG 正常 | https://i-blog.csdnimg.cn/direct/1f702adc3af745e3bda6dc1325204f98.png |
| 23 | 截图23 | https://i-blog.csdnimg.cn/direct/b16f78e5d6114af69defdb68ffbee8a8.png |
| 24 | 截图24 | https://i-blog.csdnimg.cn/direct/b07329ff702c418182e941cf53df6f56.png |
| 25 | 截图25 | https://i-blog.csdnimg.cn/direct/07b9c526786443ce995d287db571de83.png |
| 26 | 截图26 | https://i-blog.csdnimg.cn/direct/8ed60d607b5042229b3a94f02df0cb0d.png |
