# Crawlee Python 框架详解

## 📌 核心问题
> Crawlee Python 是什么？它和 Scrapy 有什么区别？如何用它构建现代化的异步爬虫？

## 🌱 根源与动机

### 为什么会有 Crawlee？

| 框架 | 诞生年份 | 异步模型 | 设计哲学 |
|------|---------|---------|---------|
| Scrapy | 2008 | Twisted（回调式） | 大而全，约定优于配置 |
| pyspider | 2014 | 多进程+消息队列 | 脚本化快速原型 |
| **Crawlee** | 2022 | **原生 async/await** | 现代化、渐进式 |

Scrapy 的核心问题是 **Twisted 不是标准 Python 异步**：
- 回调地狱（callback hell）难以调试
- 与 Python 生态中其他 async 库（httpx、asyncio）难以协同
- 学习曲线陡峭（Settings、Signals、Middleware、Item Pipeline...）

Crawlee 由 [Apify](https://apify.com) 维护（2026 年增长最快的爬虫框架），用标准 `async/await` 构建，开发者只需懂 Python 异步就能上手。

## 📐 核心概念

### 1. Router — 路由分发

Crawlee 用 Router 对不同页面类型分发不同处理逻辑：

```python
from crawlee.crawlers import HttpCrawler, HttpCrawlingContext
from crawlee.router import Router

router = Router()

@router.handler("detail")          # 详情页处理器
async def detail_handler(context: HttpCrawlingContext):
    # 解析商品详情
    pass

@router.default_handler             # 默认处理器（列表页）
async def list_handler(context: HttpCrawlingContext):
    # 翻页 + 收集详情页链接
    for link in detail_links:
        await context.add_requests([link], label="detail")

crawler = HttpCrawler(router=router)
await crawler.run(["https://example.com/list"])
```

**关键理解：** Router 不是 URL 匹配，是 **label 匹配**。你用 `add_requests` 时打 label，Router 根据 label 分发到对应 handler。

### 2. HttpCrawlingContext — 请求上下文

每个 handler 收到一个 `context`，它包含了当前请求的所有信息：

```python
async def handler(context: HttpCrawlingContext):
    context.request.url           # 原始请求 URL
    context.request.loaded_url    # 实际加载的 URL（可能有重定向）
    context.http_response.read()  # 响应体 (bytes)
    context.http_response.status_code  # HTTP 状态码
    
    # 添加新请求到爬取队列
    await context.add_requests([new_urls])
    # 提取数据并送入管道
    await context.push_data({"title": "..."})
```

### 3. 并发控制

Crawlee 内置并发控制，不需要手动管理：

```python
crawler = HttpCrawler(
    router=router,
    max_concurrency=10,           # 最大并发数
    max_requests_per_crawl=0,     # 0 = 不限制总数
    request_handler_timeout=30,   # 单个请求超时（秒）
)
```

### 4. 自动限速与反爬

```python
crawler = HttpCrawler(
    router=router,
    max_request_retries=3,          # 失败重试次数
    request_provider=RequestProvider(
        # 自动 robots.txt 遵守
    ),
    # 自动延迟（避免高频请求）
    max_requests_per_minute=60,
)
```

## 💡 本项目中的实际使用

本项目实际使用的 Crawlee 特性：

### BaseCrawler 封装（来自 `core/crawlers/base.py`）

```python
class BaseCrawler:
    """封装 Crawlee Router + StoragePipeline"""
    
    def __init__(self, name: str, start_urls: list[str]):
        self.name = name
        self.start_urls = start_urls
        self.router = Router()
        self.storage = StoragePipeline()  # 自定义存储管道
    
    async def run(self):
        crawler = HttpCrawler(
            router=self.router,
            max_concurrency=10,          # 从配置读取
        )
        await crawler.run(self.start_urls)
        await self.storage.flush()       # 结束后刷新缓冲区
```

### ExampleCrawler 使用（来自 `core/crawlers/example.py`）

```python
class ExampleCrawler(BaseCrawler):
    def _setup_routes(self):
        self.router.handler("catalogue")(self.detail_handler)
        self.router.default_handler(self.list_handler)
    
    async def list_handler(self, context: HttpCrawlingContext):
        sel = Selector(text=context.http_response.read().decode())
        for link in sel.css("article.product_pod h3 a::attr(href)").getall():
            await context.add_requests([full_url], label="catalogue")
        # 翻页
        next_page = sel.css("li.next a::attr(href)").get()
        if next_page:
            await context.add_requests([next_url])
    
    async def detail_handler(self, context: HttpCrawlingContext):
        sel = Selector(text=context.http_response.read().decode())
        item = {
            "url": context.request.loaded_url,
            "title": sel.css("h1::text").get().strip(),
        }
        await self.storage.process_item(item)  # 进入批量写入管道
```

## ⚠️ Crawlee vs Scrapy 对比速查

| 维度 | Crawlee Python | Scrapy |
|------|---------------|--------|
| **异步模型** | 原生 async/await | Twisted 回调 |
| **学习成本** | ⭐⭐ 低（懂 asyncio 即可） | ⭐⭐⭐⭐ 高 |
| **项目结构** | 自由（渐进式复杂度） | 固定（CLI 生成项目） |
| **反爬内置** | ✅ 指纹生成、代理轮换、限速 | ❌ 需要中间件 |
| **浏览器集成** | ✅ Playwright 开箱即用 | ⚠️ 需要 Scrapy-Splash |
| **去重** | RequestQueue 自动去重 | 内置 RFPDupeFilter |
| **管道** | 自定义（极灵活） | 固定 Pipeline 体系 |
| **社区规模** | 🔥快速增长（2026） | ✅ 最大最成熟 |
| **中文文档** | ⚠️ 较少 | ✅ 丰富 |

## 🔧 完整可运行示例

```python
"""基于 Crawlee 的最小可用爬虫"""
import asyncio
from crawlee.crawlers import HttpCrawler, HttpCrawlingContext
from crawlee.router import Router
from parsel import Selector

router = Router()

@router.default_handler
async def handler(context: HttpCrawlingContext):
    sel = Selector(text=context.http_response.read().decode())
    
    for quote in sel.css("div.quote"):
        text = quote.css("span.text::text").get()
        author = quote.css("small.author::text").get()
        print(f"「{text}」— {author}")
    
    # 翻页
    next_page = sel.css("li.next a::attr(href)").get()
    if next_page:
        await context.add_requests([f"http://quotes.toscrape.com{next_page}"])

async def main():
    crawler = HttpCrawler(router=router, max_concurrency=5)
    await crawler.run(["http://quotes.toscrape.com"])

asyncio.run(main())
```

## ⚠️ 易错点与常见误解

1. **Router handler 的 label 不匹配** → 如果用 `@router.handler("detail")` 但 `add_requests` 时没指定 label，请求会走到 `default_handler`
2. **忘记 `await storage.flush()`** → 缓冲区里的最后一批数据在爬虫结束前可能未写入，必须 flush
3. **`context.http_response.read()` 只能读一次** → 多次调用返回空 bytes，应该先读到变量再使用
4. **Crawlee 不等于"不需要反爬"** → 它内置了基础反爬（User-Agent 轮换等），但面对东方财富这种人机验证，仍然需要 Playwright + 手动干预
5. **并发数不是越高越好** → 并发过高触发反爬导致全部被封，建议从 5-10 开始调

## 🔗 知识延伸

- [[企业级爬虫架构设计]] — 三层架构中 Crawlee 处于调度层
- [[企业级反爬虫技术体系]] — Playwright 浏览器反爬
- [[爬虫文本清洗与数据提取]] — Router handler 中的 HTML 解析

## 📚 参考资料

- Crawlee Python 文档：https://crawlee.dev/python
- 本项目 Crawlee 封装：`core/crawlers/base.py`
- 本项目 Crawlee 示例：`core/crawlers/example.py`
