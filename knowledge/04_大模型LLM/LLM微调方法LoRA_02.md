# LLM 微调方法：LoRA / QLoRA

## 📌 核心问题
> 全量微调 70B 模型需要数百 GB 显存，怎么用消费级 GPU 微调大模型？

## 🌱 根源与动机

全量微调（Full Fine-tuning）：所有参数都更新，显存 = 参数量 × 16（FP16存储 + 梯度 + 优化器状态）。

**LoRA（Low-Rank Adaptation）** 的核心假设：**预训练权重的更新量是低秩的**。

不修改原始权重 W，而是在旁边加两个小矩阵：
```
W' = W + ΔW = W + BA
其中 B ∈ R^{d×r}, A ∈ R^{r×k}，r << min(d,k)
```
训练时只更新 B 和 A，原始 W 冻结。

## 📐 理论推导

### LoRA 数学原理

```
原始前向传播：h = Wx
LoRA 前向传播：h = Wx + BAx = (W + BA)x
```

参数量对比（以 d=4096, k=4096, r=8 为例）：
- 原始 W：4096 × 4096 = 16M 参数
- LoRA：4096×8 + 8×4096 = 65K 参数（约 0.4%）

**初始化**：A 用随机高斯，B 用零初始化 → 训练开始时 ΔW = BA = 0

**缩放因子**：`ΔW = (α/r) × BA`，`α` 是超参，通常设为 r 或 2r

### QLoRA

QLoRA = 4bit 量化 + LoRA：
```
原始权重 → INT4 量化（NF4格式）+ 双重量化
LoRA 适配器 → BF16 精度训练
→ 65B 模型可在单张 48GB A100 上微调
```

关键技术：
- **NF4**（Normal Float 4）：针对正态分布权重的最优 4bit 表示
- **双重量化**：对量化常数再量化，进一步压缩内存
- **分页优化器**：防止显存峰值 OOM

## 💡 关键理解

- LoRA 几乎不损失性能，但参数量减少 1000 倍以上
- **r 的选择**：r=4~64，越大越接近全量微调效果，但参数量也越多
- **应用 LoRA 的层**：通常加在 Q、V 投影矩阵上；也可加在所有线性层
- **推理时可合并**：`W_merged = W + BA`，推理无额外开销

## 🔧 代码实现

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,                          # 秩
    lora_alpha=16,                # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用到哪些层
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM  # 任务类型
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

## ⚠️ 易错点与常见误解

1. **LoRA 不是蒸馏**，模型架构不变，只是减少可训练参数
2. **target_modules 要匹配模型架构**：LLaMA 是 `q_proj/v_proj`，GPT2 是 `c_attn`
3. **合并后精度是 FP32**，推理前最好再量化一次
4. **QLoRA 的 INT4 基座不能再全量微调**，LoRA 适配器保持 BF16

## 🔗 知识延伸

- [[LLM基础与训练]] — 微调处于哪个阶段
- [[LLM模型量化]] — 量化原理
- [[BERT预训练模型]] — 小模型微调对比

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/LLM_02_微调方式.pdf`
- LoRA 论文：Hu et al. 2022 "LoRA: Low-Rank Adaptation of Large Language Models"
