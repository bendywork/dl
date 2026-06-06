# ============================================================
# 文件: 06_张量拼接与分割.py
# 主题: PyTorch 张量拼接与分割 API 精讲
# 涵盖: torch.cat / torch.stack / torch.split /
#        torch.chunk / torch.unbind
# 重点: cat vs stack 的区别（是否新增维度）
#        split vs chunk 的区别（均分 vs 指定大小）
# ============================================================

import torch

SEP = "=" * 55
def demo_cat():
    """
    torch.cat(tensors, dim=0)
      沿指定已有维度拼接，不新增维度
      所有张量除拼接轴外的其他维度必须相同
    常用场景: 将多个 batch 合并, 拼接特征通道
    """
    print(SEP)
    print("1. torch.cat --- 沿已有维度拼接（不增加维度）")
    print(SEP)

    a = torch.ones(2, 3)
    b = torch.zeros(4, 3)
    c = torch.cat([a, b], dim=0)  # 沿行拼接
    print(f"  a shape={a.shape}, b shape={b.shape}")
    print(f"  cat([a,b], dim=0) shape={c.shape}:")
    print(f"  {c}")

    x = torch.randn(3, 2)
    y = torch.randn(3, 5)
    z = torch.cat([x, y], dim=1)  # 沿列拼接
    print(f"  cat(dim=1): {x.shape} + {y.shape} -> {z.shape}")

    # NLP 中拼接词向量特征
    emb1 = torch.randn(8, 10, 128)  # (batch, seq, emb1)
    emb2 = torch.randn(8, 10, 64)   # (batch, seq, emb2)
    emb_cat = torch.cat([emb1, emb2], dim=-1)
    print(f"  词向量拼接: {emb1.shape}+{emb2.shape} -> {emb_cat.shape}")

def demo_stack():
    """
    torch.stack(tensors, dim=0)
      新建一个维度来堆叠，所有张量 shape 必须完全相同
      n 个 shape=(a,b) 的张量 stack(dim=0) -> shape=(n,a,b)
    cat vs stack 核心区别:
      cat  不新增维度: (2,3)+(4,3) -dim=0-> (6,3)
      stack 新增维度:  (3,)+(3,)  -dim=0-> (2,3)
    常用场景: 将 list of tensors 打包成 batch
    """
    print(SEP)
    print("2. torch.stack --- 新增维度堆叠（vs cat 的关键区别）")
    print(SEP)

    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    t3 = torch.tensor([7, 8, 9])

    s0 = torch.stack([t1, t2, t3], dim=0)
    print(f"  stack([t1,t2,t3], dim=0): shape={s0.shape}")
    print(f"  {s0}")

    s1 = torch.stack([t1, t2, t3], dim=1)
    print(f"  stack([t1,t2,t3], dim=1): shape={s1.shape}")
    print(f"  {s1}")

    # 对比 cat
    cat_result = torch.cat([t1, t2, t3], dim=0)
    print(f"  cat dim=0 结果:   shape={cat_result.shape}  {cat_result}  <-- 不新增维度")
    print(f"  stack dim=0 结果: shape={s0.shape}  <-- 新增了 dim=0")

    # DataLoader 中，每次 collate 就是 stack
    samples = [torch.randn(3, 224, 224) for _ in range(4)]  # 4张图
    batch = torch.stack(samples, dim=0)
    print(f"  4张(3,224,224)图像 stack -> batch shape={batch.shape}")

def demo_split():
    """
    torch.split(tensor, split_size_or_sections, dim=0)
      split_size: int -> 按指定大小均等分割（最后一块可能更小）
      sections:   list -> 按指定列表各自的大小分割
    返回: tuple of tensors（视图，共享内存）
    """
    print(SEP)
    print("3. torch.split --- 按大小分割张量")
    print(SEP)

    t = torch.arange(10)
    print(f"  原始: {t}  shape={t.shape}")

    # 按固定大小分割
    parts = torch.split(t, 3, dim=0)  # 每份3个
    print(f"  split(3): {[p.shape for p in parts]}")
    for i, p in enumerate(parts):
        print(f"    部分{i}: {p}")

    # 按指定列表分割
    parts2 = torch.split(t, [2, 3, 5], dim=0)
    print(f"  split([2,3,5]): {[p.shape for p in parts2]}")
    for i, p in enumerate(parts2):
        print(f"    部分{i}: {p}")

    # 2D 张量按列分割（常见于多头注意力）
    q = torch.randn(4, 12)  # (batch, heads*d_head)
    heads = torch.split(q, 4, dim=1)  # 每4列一个头
    print(f"  多头: (4,12) split(4,dim=1) -> {[h.shape for h in heads]}")

def demo_chunk():
    """
    torch.chunk(tensor, chunks, dim=0)
      将张量均分为 chunks 块（若不能整除，最后一块更小）
      vs split: chunk 指定块数; split 指定每块大小
    """
    print(SEP)
    print("4. torch.chunk --- 均分为N块（vs split 的区别）")
    print(SEP)

    t = torch.arange(10)
    print(f"  原始: {t}  shape={t.shape}")

    c3 = torch.chunk(t, 3, dim=0)  # 分3块
    print(f"  chunk(3): {[p.shape for p in c3]}")
    for i, p in enumerate(c3):
        print(f"    块{i}: {p}")

    c4 = torch.chunk(t, 4, dim=0)  # 分4块，最后一块可能更小
    print(f"  chunk(4): {[p.shape for p in c4]}")

    print("")
    print("  split vs chunk 对比:")
    print("    split(t, 3): 每块3个 -> [3,3,3,1] 共4块")
    print("    chunk(t, 3): 均分3块 -> [4,3,3]  共3块")

def demo_unbind():
    """
    torch.unbind(tensor, dim=0)
      沿指定维度拆解为 tuple of tensors（去掉该维度）
      类似 Python 的 list(t) 但是针对张量维度
      返回的每个张量 shape 少一个维度
    常用场景: 将序列张量按时间步拆解
    """
    print(SEP)
    print("5. torch.unbind --- 沿维度拆解为独立张量")
    print(SEP)

    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"  原始 shape={t.shape}:")
    print(f"  {t}")

    rows = torch.unbind(t, dim=0)
    print(f"  unbind(dim=0) -> {len(rows)} 个 shape={rows[0].shape} 的行向量:")
    for i, r in enumerate(rows):
        print(f"    行{i}: {r}")

    cols = torch.unbind(t, dim=1)
    print(f"  unbind(dim=1) -> {len(cols)} 个 shape={cols[0].shape} 的列向量:")
    for i, c in enumerate(cols):
        print(f"    列{i}: {c}")

    # 序列处理: (T, N, H) 拆成 T 个 (N, H)
    seq = torch.randn(5, 4, 8)  # (T=5, batch=4, hidden=8)
    steps = torch.unbind(seq, dim=0)
    print(f"  seq (T=5,N=4,H=8) unbind(0) -> {len(steps)} 步, 每步 shape={steps[0].shape}")


if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    demo_cat()
    demo_stack()
    demo_split()
    demo_chunk()
    demo_unbind()
    print(SEP)
    print("所有张量拼接与分割演示完毕！")
    print(SEP)