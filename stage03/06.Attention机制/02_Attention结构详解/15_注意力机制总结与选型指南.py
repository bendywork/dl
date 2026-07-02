"""
15_注意力机制总结与选型指南
=============================
40+ 种 Attention 变种, 如何选? 本文件提供复杂度对比表和选型决策树.

复杂度速查表:
┌─────────────────────────┬───────────────┬───────────────┬──────────────────────┐
│ 类型                     │ 时间复杂度     │ 空间复杂度     │ 代表模型              │
├─────────────────────────┤───────────────┤───────────────┤──────────────────────┤
│ Standard MHA             │ O(n^2 * d)    │ O(n^2 + n*d)  │ Transformer, BERT     │
│ MQA (Multi-Query)        │ O(n^2 * d)    │ O(n*d)        │ PaLM, Falcon          │
│ GQA (Grouped-Query)      │ O(n^2 * d)    │ O(G*n*d)      │ LLaMA 2/3, Mistral    │
│ MLA (DeepSeek)           │ O(n^2 * d)    │ O(n*d_latent) │ DeepSeek-V2/V3        │
│ FlashAttention            │ O(n^2 * d)    │ O(n*d)        │ Most LLMs (GPU)       │
│ Local Window             │ O(n*w*d)      │ O(n*w + n*d)  │ Mistral, Longformer   │
│ Longformer               │ O(n*(w+g)*d)  │ O(n*(w+g)+n*d)│ Longformer-4096       │
│ BigBird                  │ O(n*(w+r+g)*d)│ O(n*(w+r+g))  │ BigBird-Pegasus       │
│ Linformer                │ O(n*k*d)      │ O(n*d + k*d)  │ Linformer             │
│ Performer/FAVOR+         │ O(n*m*d)      │ O(n*d + m*d)  │ Performer             │
│ Linear Attention          │ O(n*d^2)      │ O(n*d + d^2)  │ cosFormer, Linear-T   │
│ Nyströmformer            │ O(n*m*d)      │ O(n*d + m*d)  │ Nyströmformer         │
│ Sparse/K-Means Routing   │ O(n*sqrt(n)*d)│ O(n*sqrt(n))  │ Routing Transformer   │
│ Axial                    │ O(n*max(H,W)*d)│ O(n*d)       │ Axial Transformer     │
│ Deformable               │ O(N_q*K*d)    │ O(N_q*K)      │ Deformable DETR       │
│ Infini-Attention         │ O(n*d^2)      │ O(d^2)        │ Infini-Transformer    │
│ PagedAttention           │ O(n^2*d)      │ O(n_blocks*B*d)│ vLLM                 │
│ Ring Attention            │ O(n^2*d/N)    │ O(n*d/N)      │ Ring-Attn (分布式)    │
└─────────────────────────┴───────────────┴───────────────┴──────────────────────┘

符号: n=seq_len, d=d_model, w=window, g=global, r=random, k=proj_dim, m=features

选型决策树:
  1. 序列长度 n < 2048?
     → Yes: Standard MHA + FlashAttention (最成熟)
     → No: 继续

  2. 是推理还是训练?
     → 训练: 继续
     → 推理: 先选 GQA/MQA/MLA 压缩 KV Cache, 再叠加 FlashAttention / PagedAttention

  3. 需要全局依赖?
     → Yes (如文档理解): Longformer / BigBird / Infini-Attention
     → No (局部即可): Local Window + dilated (Mistral 风格)

  4. 是图像/视频?
     → 图像: Swin (window + shift) / Axial / Deformable
     → 视频: TimeSformer (分离时空) / Video RoPE
     → 多模态: Gated Cross-Attention (Flamingo 风格)

  5. 需要超长上下文 (>100K)?
     → Ring Attention (多GPU) / Infini-Attention (单GPU压缩记忆)
     → 位置编码: RoPE with NTK/YaRN scaling

  6. 图/集合数据?
     → 图: GAT / GATv2
     → 集合: Set Transformer (PMA + ISAB)
     → 分子/蛋白质: SE(3)-Transformer, Equiformer

企业级推荐组合 (2024-2025):
  LLM 训练: GQA + RoPE + FlashAttention 3 + Ring Attention
  LLM 推理: GQA + RoPE + FlashAttention 3 + PagedAttention + Prefix Caching
  多模态:  Cross-Attn + Gated + 可选的 Deformable (视觉端)
  长文档:  Infini-Attention 或 Longformer 模式
  低资源:  Linear Attention (cosFormer) 或 Linformer

工程落地注意:
  - FlashAttention 需要 CUDA, 纯 PyTorch 无法复现速度
  - GQA 是性价比最高的推理优化 (几乎无损, Cache 显著缩小)
  - MLA 实现复杂但有极致压缩比, 适合自研大模型
  - RoPE 是当前事实标准, 新项目首选
  - 稀疏方案 (Local/BigBird) 常需要定制 CUDA kernel, 工程成本高
  - 所有推理优化最终要服务于: 更高吞吐 × 更低延迟 × 更长上下文
"""

# 可选的交互式选型函数
def recommend_attention(seq_len: int, is_inference: bool, domain: str,
                        has_multi_gpu: bool = False):
    """
    TODO: 根据输入参数打印推荐方案
    domain: "nlp" | "cv" | "video" | "graph" | "multimodal"
    返回: (attention_type, position_encoding, kv_cache_strategy)
    """
    pass


if __name__ == "__main__":
    # TODO: 运行 recommend_attention 测试几个典型场景
    # TODO: 绘制复杂度曲线图: seq_len∈[512, 128K], 对比各方案理论FLOPs
    print("注意力机制选型指南已加载. 运行 recommend_attention() 获取推荐.")
