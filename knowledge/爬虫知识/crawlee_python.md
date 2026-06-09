# Crawlee Python 爬虫库文档

> 整理时间：2026-06-09  
> 基于项目：`nlp-scx/stage01/crawl`

---

## 一、概述

Crawlee 是 Apify 出品的现代化爬虫框架，Python 版本支持异步并发、请求队列、自动重试等核心能力。

### 核心特性
- 异步 IO（基于 asyncio）
- 内置请求队列与去重
- 路由系统（按 URL label 分发 handler）
- 自动重试与并发控制
- 支持 HTTP / Browser（Playwright）两种模式

### 安装

```bash
pip install crawlee
# 如需 Playwright 支持
pip install crawlee[playwright]
playwright install
```

---

## 二、核心类速查

| 类名 | 模块 | 用途 |
|------|------|------|
| `HttpCrawler` | `crawlee.crawlers` | HTTP 请求爬虫，轻量高效 |
| `BeautifulSoupCrawler` | `crawlee.crawlers` | 内置 BS4 解析 |
| `PlaywrightCrawler` | `crawlee.crawlers` | 浏览器渲染爬虫 |
| `HttpCrawlingContext` | `crawlee.crawlers` | handler 接收的请求上下文 |
| `Router` | `crawlee.router` | URL 路由分发器 |

---

## 三、Router 路由系统

Router 是 Crawlee 的核心设计，将不同 URL 分发给不同 handler 处理。

```python
from crawlee.router import Router
from crawlee.crawlers import HttpCrawlingContext

router = Router()

# 默认 handler：所有未匹配 label 的请求
@router.default_handler
async def default_handler(context: HttpCrawlingContext):
    print(context.request.url)

# 命名 handler：通过 label 匹配
@router.handler("detail")
async def detail_handler(context: HttpCrawlingContext):
    print("detail page:", context.request.loaded_url)
```

也可以用方法方式注册（适合封装到类中）：

```python
class MyCrawler:
    def __init__(self):
        self.router = Router()
        # 将实例方法注册为 handler
        self.router.handler("catalogue")(self.detail_handler)
        self.router.default_handler(self.list_handler)

    async def list_handler(self, context: HttpCrawlingContext):
        pass

    async def detail_handler(self, context: HttpCrawlingContext):
        pass
```

---

## 四、HttpCrawler 启动方式

```python
from crawlee.crawlers import HttpCrawler

crawler = HttpCrawler(
    router=router,
    max_requests_per_crawl=0,   # 0 = 不限制请求数
    max_concurrency=10,          # 最大并发数
)

# 传入起始 URL 列表
await crawler.run(["https://example.com"])
```

---

## 三、Router 路由系统

Router 是 Crawlee 的核心调度机制，根据请求的 `label` 将 URL 分发到不同 handler。

```python
from crawlee.router import Router
from crawlee.crawlers import HttpCrawlingContext

router = Router()

# 默认 handler（无 label 的请求走这里）
@router.default_handler
async def default_handler(context: HttpCrawlingContext):
    print(context.request.url)

# 命名 handler（label="detail" 的请求走这里）
@router.handler("detail")
async def detail_handler(context: HttpCrawlingContext):
    print("detail page:", context.request.loaded_url)
```

### 动态注册（类方法场景）

```python
class MyCrawler:
    def _setup_routes(self):
        # 用法等价于装饰器，适合类封装
        self.router.handler("catalogue")(self.detail_handler)
        self.router.default_handler(self.list_handler)
```

### 添加带 label 的请求

```python
await context.add_requests([
    {"url": "https://example.com/item/1", "label": "detail"},
    {"url": "https://example.com/page/2"},          # 无 label → default_handler
])
```

---

## 四、HttpCrawler 初始化参数

```python
from crawlee.crawlers import HttpCrawler

crawler = HttpCrawler(
    router=router,               # Router 实例
    max_requests_per_crawl=0,    # 0 = 不限制请求数
    max_concurrency=10,          # 最大并发数
)

await crawler.run(["https://example.com"])
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `router` | `Router` | 路由分发器 |
| `max_requests_per_crawl` | `int` | 最大请求总数，0 为不限 |
| `max_concurrency` | `int` | 并发协程数 |

