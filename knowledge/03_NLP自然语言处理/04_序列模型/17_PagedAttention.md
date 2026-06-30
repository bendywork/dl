# 06 · PagedAttention（分页注意力，vLLM）

PagedAttention 是 vLLM（LLM 推理框架）的核心创新。它不改变 Attention 公式，而是**复用操作系统虚拟内存的分页思想来管理 KV cache**。

## 问题：KV Cache 的内存碎片

LLM 推理时为每个 token 存 K 和 V 用于后续 decode，最大预留上下文 = 显存占用：

```
用户A: 请求 2048 tokens → 预留 2048 位置的 KV cache
用户B: 请求 128 tokens  → 预留 2048 位置的 KV cache（浪费 1920）
用户C: 请求 8192 tokens → 预留 8192 位置的 KV cache

三个请求共享一张 GPU → 预留了大块连续内存但大量未使用 → 内存碎片
```

原生方式是**每个请求预先分配一片连续的 KV cache 空间**——就像程序直接 malloc 一大块"最大可能的内存"却只用了前几个字节。

## 解法：PagedAttention

```
KV cache 不分配连续的大块 → 拆成"页"（Page），按需分配

一个请求的 KV cache:
  逻辑上: [token_0, token_1, ...... token_n]
  物理上: [Page_0] → [Page_3] → [Page_7] → [Page_1]
           ↑
    页表维护映射关系，token 0~31 在 Page_0，token 32~63 在 Page_3...

当请求长度超过当前分配 → 分配新 Page
当请求被删 → 页回收，空闲页可用于新请求
```

## 效果

| | 原生 | PagedAttention |
|------|------|------|
| 内存浪费 | 大（预留 = 最大可能长度） | 小（按需 + 页共享） |
| GPU 利用率 | 低（碎片化） | 高（接近 100%） |
| 并行请求数 | 少（每请求独占大片显存） | 多（页粒度共享） |

## 工业应用

| 系统 | 说明 |
|------|------|
| **vLLM** | PagedAttention 定义者 |
| **TensorRT-LLM** | 吸收 KV cache 块管理思想 |
| **SGLang** | RadixAttention（更激进的 Prefix Caching） |

## 一句话

> 写操作系统的虚拟内存页表搬到 KV Cache 管理 → 显存利用率翻倍。vLLM 靠这个变成 LLM 推理的标配框架。
