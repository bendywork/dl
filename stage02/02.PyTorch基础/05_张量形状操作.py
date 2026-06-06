# ============================================================
# 文件: 05_张量形状操作.py
# 主题: PyTorch 张量形状操作 精讲
# 涵盖: reshape / view / squeeze / unsqueeze / permute /
#        transpose / flatten / contiguous / expand / repeat
# 重点: view vs reshape 的区别（连续内存 contiguous）
# ============================================================

import torch

SEP = "=" * 55
def demo_view_vs_reshape():
    """
    view:    要求张量内存必须连续（contiguous），否则报错
             与原张量共享内存（零拷贝）
    reshape: 优先返回 view（共享内存），若不连续则自动复制
             更安全，但有时会产生拷贝

    什么是内存连续？
      张量元素在内存中按行优先（C 顺序）紧密排列
      transpose/permute 后，内存布局不变但步长(stride)改变
      → 此时张量不连续，view 会报错
    """
    print(SEP)
    print("1. view vs reshape --- 连续内存的关键区别")
    print(SEP)

    t = torch.arange(12)
    print(f"  原始: {t}  shape={t.shape}")

    v = t.view(3, 4)
    print(f"  view(3,4): shape={v.shape}")
    print(f"  {v}")

    r = t.reshape(2, 6)
    print(f"  reshape(2,6): shape={r.shape}")
    print(f"  {r}")

    # 演示 view 在非连续张量上报错的场景
    t2 = t.view(3, 4).transpose(0, 1)  # 转置后非连续
    print(f"  transpose 后 is_contiguous={t2.is_contiguous()}")
    try:
        t2.view(12)
    except RuntimeError as e:
        print(f"  view() 报错: {str(e)[:80]}")
    t2_reshaped = t2.reshape(12)  # reshape 自动处理
    print(f"  reshape(12) 成功: {t2_reshaped}")

def demo_squeeze_unsqueeze():
    """
    squeeze:   删除大小为 1 的维度（压缩虚假维度）
                squeeze(dim) 只压缩指定 dim（若该 dim=1）
    unsqueeze: 在指定位置插入大小为 1 的新维度
                dim 可为负数（从末尾数）
    典型用途:
      unsqueeze(0) -> 添加 batch 维: (C,H,W) -> (1,C,H,W)
      squeeze()    -> 去掉 batch 维: (1,C,H,W) -> (C,H,W)
    """
    print(SEP)
    print("2. squeeze / unsqueeze --- 增删大小为1的维度")
    print(SEP)

    t = torch.zeros(1, 3, 1, 4)
    print(f"  原始 shape: {t.shape}")

    s = t.squeeze()
    print(f"  .squeeze()    -> shape={s.shape}  (全部 size=1 维度被删除)")

    s1 = t.squeeze(0)
    print(f"  .squeeze(0)   -> shape={s1.shape}  (只删 dim=0)")

    s2 = t.squeeze(2)
    print(f"  .squeeze(2)   -> shape={s2.shape}  (只删 dim=2)")

    # unsqueeze
    base = torch.randn(3, 4)
    print(f"  base shape: {base.shape}")

    u0 = base.unsqueeze(0)
    print(f"  .unsqueeze(0) -> shape={u0.shape}  (加 batch 维)")

    u1 = base.unsqueeze(1)
    print(f"  .unsqueeze(1) -> shape={u1.shape}  (中间插维)")

    u_1 = base.unsqueeze(-1)
    print(f"  .unsqueeze(-1)-> shape={u_1.shape} (末尾插维)")

def demo_permute_transpose():
    """
    permute(dims): 任意维度重排（通用版 transpose）
      dims: 新维度顺序的元组，长度必须等于原张量维度数
    transpose(dim0, dim1): 只交换两个指定维度（permute 的简化版）
    两者都返回非连续视图，后续 view 前需 .contiguous()
    """
    print(SEP)
    print("3. permute / transpose --- 维度重排")
    print(SEP)

    # 图像张量: (N, H, W, C) -> (N, C, H, W) PyTorch 标准格式
    img = torch.randn(2, 32, 32, 3)
    print(f"  img (N,H,W,C) shape: {img.shape}")

    img_nchw = img.permute(0, 3, 1, 2)
    print(f"  permute(0,3,1,2) -> (N,C,H,W) shape: {img_nchw.shape}")
    print(f"  is_contiguous after permute: {img_nchw.is_contiguous()}")

    # transpose: 只交换两个维度
    mat = torch.randn(3, 4)
    print(f"  mat shape: {mat.shape}")
    mat_T = mat.transpose(0, 1)
    print(f"  .transpose(0,1) -> shape: {mat_T.shape}  (等价于 mat.T)")

    # 常见陷阱: permute/transpose 后需 contiguous() 才能 view
    try:
        mat_T.view(-1)
    except RuntimeError as e:
        print(f"  transpose 后直接 view 报错: {str(e)[:60]}")
    safe = mat_T.contiguous().view(-1)
    print(f"  .contiguous().view(-1) shape: {safe.shape}")

def demo_flatten():
    """
    flatten(start_dim, end_dim): 将指定范围的维度合并为一个
      默认 start_dim=0, end_dim=-1 -> 彻底展平为 1D
      CNN -> 全连接层之间常用: (N,C,H,W) -> (N, C*H*W)
    """
    print(SEP)
    print("4. flatten --- 展平维度")
    print(SEP)

    t = torch.randn(2, 3, 4)
    print(f"  t shape: {t.shape}")

    f_all = t.flatten()
    print(f"  .flatten()          -> shape: {f_all.shape}  (完全展平)")

    f_12 = t.flatten(start_dim=1)
    print(f"  .flatten(start_dim=1)-> shape: {f_12.shape}  (保留batch维)")

    # CNN -> FC 典型用法
    feature_map = torch.randn(8, 64, 7, 7)  # (N,C,H,W)
    fc_input = feature_map.flatten(start_dim=1)
    print(f"  CNN特征图 (8,64,7,7) flatten(1) -> {fc_input.shape}")
    print(f"  等价于 reshape(8,-1) -> {feature_map.reshape(8,-1).shape}")

def demo_expand_repeat():
    """
    expand: 将大小为 1 的维度扩展（广播），不复制内存
      只能扩展 size=1 的维度，扩展后大小可指定
      -1 表示该维度保持原大小不变
    repeat: 沿各维度重复张量（真正复制数据）
      参数是每个维度重复的次数
    区别: expand 是虚扩展（零拷贝），repeat 是真拷贝
    """
    print(SEP)
    print("5. expand / repeat --- 维度扩展与复制")
    print(SEP)

    bias = torch.tensor([[1.0], [2.0], [3.0]])  # shape (3,1)
    print(f"  bias shape: {bias.shape}")

    exp = bias.expand(3, 4)
    print(f"  .expand(3,4) -> shape={exp.shape}  (零拷贝,共享内存)")
    print(f"  {exp}")
    print(f"  expand 前后内存是否共享: {bias.data_ptr() == exp[0:1, 0:1].data_ptr()}")

    rep = bias.repeat(1, 4)
    print(f"  .repeat(1,4) -> shape={rep.shape}  (真实复制)")
    print(f"  {rep}")

    # expand 常见广播场景: 为每个 batch 加同一个 bias
    x = torch.randn(4, 3)   # batch=4, features=3
    b = torch.tensor([0.1, 0.2, 0.3])  # (3,)
    b_exp = b.unsqueeze(0).expand(4, -1)  # (1,3) -> (4,3)
    result = x + b_exp
    print(f"  x+bias broadcast: x{x.shape} + b_exp{b_exp.shape} = result{result.shape}")


if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    demo_view_vs_reshape()
    demo_squeeze_unsqueeze()
    demo_permute_transpose()
    demo_flatten()
    demo_expand_repeat()
    print(SEP)
    print("所有张量形状操作演示完毕！")
    print(SEP)