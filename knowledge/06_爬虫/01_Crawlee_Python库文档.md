# Crawlee Python 爬虫库文档

> 整理时间：2026-06-09
> 基于项目：`nlp-scx/stage01/crawl`

---

## 一、概述

Crawlee 是 Apify 出品的现代化爬虫框架，Python 版本基于 asyncio，支持异步并发、请求队列、自动重试、URL 路由等核心能力。

### 核心特性

- 异步 IO（asyncio）
- 内置请求队列与去重
- 路由系统（按 label 分发 handler）
- 自动重试与并发控制
- 支持 HTTP / Playwright 两种模式

### 安装

```bash
pip install crawlee
pip install crawlee[playwright]   # 浏览器模式
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

Router 根据请求的 `label` 将 URL 分发到不同 handler。

```python
from crawlee.router import Router
from crawlee.crawlers import HttpCrawlingContext

router = Router()

@router.default_handler
async def default_handler(context: HttpCrawlingContext):
    print(context.request.url)

@router.handler("detail")
async def detail_handler(context: HttpCrawlingContext):
    print("detail:", context.request.loaded_url)
```

### 类方法动态注册

```python
class MyCrawler:
    def _setup_routes(self):
        self.router.handler("catalogue")(self.detail_handler)
        self.router.default_handler(self.list_handler)
```

### 添加带 label 的请求

```python
await context.add_requests([
    {"url": "https://example.com/item/1", "label": "detail"},
    {"url": "https://example.com/page/2"},  # 无 label → default_handler
])
```

---

## 四、HttpCrawler 初始化参数

```python
from crawlee.crawlers import HttpCrawler

crawler = HttpCrawler(
    router=router,
    max_requests_per_crawl=0,   # 0 = 不限制请求总数
    max_concurrency=10,          # 最大并发协程数
)

await crawler.run(["https://example.com"])
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `router` | `Router` | 路由分发器 |
| `max_requests_per_crawl` | `int` | 最大请求总数，0 为不限 |
| `max_concurrency` | `int` | 并发协程数 |

---

## 五、HttpCrawlingContext 上下文对象

handler 函数唯一参数，包含请求与响应的所有信息。

```python
async def my_handler(context: HttpCrawlingContext):
    # 请求信息
    context.request.url          # 原始 URL
    context.request.loaded_url   # 重定向后实际 URL
    context.request.label        # 路由 label

    # 响应内容
    body = context.http_response.read().decode()

    # 向队列添加新请求
    await context.add_requests(["https://example.com/page/2"])

    # 添加带 label 的请求（路由到指定 handler）
    await context.add_requests([
        {"url": "https://example.com/item/1", "label": "detail"}
    ])
```

---

## 六、项目实战结构

基于 `nlp-scx/stage01/crawl` 项目的工程化分层：

```
crawlee_data/
├── config.py          # Pydantic Settings 配置（读 .env）
├── crawlers/
│   ├── base.py        # BaseCrawler：封装 HttpCrawler + StoragePipeline
│   └── example.py     # ExampleCrawler：books.toscrape.com 示例
├── models/
│   └── article.py     # Peewee ORM 数据模型
├── parsers/
│   └── extractor.py   # CSS/XPath 提取工具函数
├── pipelines/
│   └── storage.py     # StoragePipeline：批量写库（500条/批）
└── utils/
    ├── db.py          # PooledMySQLDatabase：DBUtils 连接池
    └── logger.py      # Loguru 日志配置
```

---

## 七、BaseCrawler 封装模式

将 `HttpCrawler` 封装为基类，子类只需实现 `_setup_routes`：

```python
class BaseCrawler:
    def __init__(self, name: str, start_urls: list[str]):
        self.name = name
        self.start_urls = start_urls
        self.router = Router()
        self.storage = StoragePipeline()
        self._setup_routes()

    def _setup_routes(self):
        @self.router.default_handler
        async def default_handler(context: HttpCrawlingContext):
            pass

    async def run(self):
        crawler = HttpCrawler(
            router=self.router,
            max_requests_per_crawl=0,
            max_concurrency=settings.crawl_concurrency,
        )
        try:
            await crawler.run(self.start_urls)
        finally:
            await self.storage.flush()
```

子类示例（列表页 + 详情页双 handler）：

```python
class ExampleCrawler(BaseCrawler):
    def _setup_routes(self):
        self.router.handler("catalogue")(self.detail_handler)
        self.router.default_handler(self.list_handler)

    async def list_handler(self, context: HttpCrawlingContext):
        sel = Selector(text=context.http_response.read().decode())
        for link in sel.css("article h3 a::attr(href)").getall():
            await context.add_requests([base_url + link])

    async def detail_handler(self, context: HttpCrawlingContext):
        sel = Selector(text=context.http_response.read().decode())
        item = {"url": context.request.loaded_url, "title": sel.css("h1::text").get()}
        await self.storage.process_item(item)
```

---

## 八、StoragePipeline 批量存储

缓冲区满 `batch_size` 条时自动 flush，爬虫结束后调用 `flush()` 清尾。

```python
pipeline = StoragePipeline(batch_size=500)
await pipeline.process_item({"url": "...", "title": "..."})
await pipeline.flush()  # 手动刷尾（run() 结束后调用）
```

---

## 九、配置管理（Pydantic Settings）

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    mysql_host: str = "localhost"
    crawl_concurrency: int = 10
    crawl_retry_times: int = 3

settings = Settings()
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `mysql_host` | `localhost` | 数据库地址 |
| `crawl_concurrency` | `10` | 并发数 |
| `crawl_retry_times` | `3` | 重试次数 |
| `crawl_request_delay` | `0.5` | 请求间隔(秒) |
| `db_pool_min` | `3` | 连接池最小连接 |
| `db_pool_max` | `10` | 连接池最大连接 |

---

