# 目标模型算子清单：MiniMind

> **来源：** 对 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的实际 ONNX 导出，非人工快照。
> **生成日期：** 2026-09-07 ｜ **模型：** `MiniMindForCausalLM`，8 层 / hidden 768 / 8 heads / 4 kv_heads / head_dim 96 / vocab 6400 / 63.91M 参数
> **导出：** torch 2.14.0+cpu，legacy TorchScript exporter，opset 17，`flash_attn=False`（导出显式 attention 而非 SDPA 融合）
> **重跑：** `out/venv/bin/python python/tools/export_minimind_onnx.py` 然后 `python/tools/onnx_op_inventory.py`
>
> 本文是**前端覆盖差距的度量**，不是支持矩阵。当前能力以 [Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md) 与 `test/nlp_validation/transformer_capability_matrix.json` 为准。
> 目标模型的选择理由与阶梯见[项目目标](PROJECT_GOAL.md) §2.2。MiniMind 是 MiniMind-O 的 Thinker（技术报告记 63.91M，与本次导出一致）。

## 核心结论

**导出配置对算子清单的影响，大于模型本身。** 同一个模型三种导出方式，实算节点从 2354 降到 650，缺失算子从 17 种降到 11 种：

| 导出配置 | 总节点 | 常量折叠后实算节点 | 算子种类 | 导入器缺失 |
|---|---:|---:|---:|---:|
| 动态轴（`dynamic_axes`） | 3634 | 2354 | 25 | **17** |
| 静态形状 | 2053 | 778 | 20 | **13** |
| 静态形状 + 非原地 mask | 1141 | **650** | **18** | **11** |

三条结论：

1. **`Equal` / `Where` / `ConstantOfShape` / `Trilu` / `Identity` 在静态导出下常量折叠后全部归零。** 它们不是模型语义，是 torch 导出 `Tensor.expand()` 与 causal mask 的固定惯用法（已核实：`Where` 输入 0 的生产者 80/80 是 `Equal`，`Expand` 输入 1 的生产者 80/80 是 `Where`，且整条链输入均为常量）。
2. **`ScatterND` ×16 与 `Shape` ×32 由一处原地切片赋值引起**：`scores[:, :, :, -seq_len:] += ...`。改为非原地加法后两者同时归零。补丁见 `python/tools/minimind_noninplace_mask.patch`。
3. **静态 prefill 只差 11 个算子即可完整导入，其中没有一个需要 shape-as-value 或 KV cache 基座。** 见下方 L1a。

## 折叠后实算算子（静态 + 非原地 mask）

prefill 650 个实算节点 / decode 666 个，18 种算子：

| ONNX 算子 | prefill | decode | 导入器 | 来源 |
|---|---:|---:|:---:|---|
| `Add` | 73 | 73 | ✅ | 残差、RMSNorm eps、RoPE |
| `Cast` | 57 | 57 | ❌ | RMSNorm 的 `.float()` / `.type_as()` |
| `Concat` | 16 | 32 | ✅ | RoPE `rotate_half`；decode 多 16 处 KV 追加 |
| `Div` | 41 | 41 | ❌ | attention 缩放、RMSNorm |
| `Expand` | 16 | 16 | ❌ | GQA `repeat_kv`（目标形状为常量，**静态下不需 shape-as-value**） |
| `Gather` | 1 | 1 | ✅ | token embedding |
| `MatMul` | 73 | 73 | ✅ | qkv/o 投影、注意力、FFN、lm_head |
| `Mul` | 114 | 114 | ❌ | RMSNorm 权重、RoPE、SwiGLU 门控 |
| `Neg` | 16 | 16 | ❌ | RoPE `rotate_half` |
| `Pow` | 33 | 33 | ❌ | RMSNorm 的 `x.pow(2)` |
| `ReduceMean` | 33 | 33 | ❌ | RMSNorm |
| `Reshape` | 48 | 48 | ❌ | 头维度展开与合并 |
| `Sigmoid` | 8 | 8 | ❌ | SwiGLU 的 SiLU |
| `Slice` | 32 | 32 | ✅ | RoPE `rotate_half` 前后半切分 |
| `Softmax` | 8 | 8 | ✅ | 注意力 |
| `Sqrt` | 33 | 33 | ❌ | RMSNorm `rsqrt` |
| `Transpose` | 32 | 32 | ✅ | 头维度换位 |
| `Unsqueeze` | 16 | 16 | ❌ | RoPE 广播（静态下等价 reshape） |

**已可达 7 种**：`Add` `Concat` `Gather` `MatMul` `Slice` `Softmax` `Transpose`
**缺失 11 种**：`Cast` `Div` `Expand` `Mul` `Neg` `Pow` `ReduceMean` `Reshape` `Sigmoid` `Sqrt` `Unsqueeze`

导入器已映射但本模型不用（视觉链遗留）：`Conv` `Flatten` `Gemm` `GlobalAveragePool` `LayerNormalization` `MaxPool` `Relu`

## 动态轴导出的额外负担

声明 `dynamic_axes` 后，导出器插入形状重建链，额外引入 7 种算子且无法折叠：

| 算子 | 动态轴 | 静态 | 说明 |
|---|---:|---:|---|
| `Shape` | 289 | 0 | **形状重建链的主体，shape-as-value 的直接来源** |
| `Unsqueeze` | 356 | 16 | 形状拼接 |
| `Gather` | 162 | 1 | 从 shape 张量取维度 |
| `Concat` | 120 | 16 | 拼接形状 |
| `Range` | 64 | 0 | 位置索引 |
| `Equal` / `Where` | 80 / 80 | 0 / 0 | `expand` 的 -1 维处理 |
| `ConstantOfShape` | 88 | 0 | 形状常量物化 |

**这不构成"必须先做 shape-as-value"的结论**：形状重建链是导出器的实现选择。变长能力可以由本仓库的有界动态形状合同承担，而非导入这条链。二者的取舍是 M3 的输入。

## 对实施计划的影响

| 级别 | 范围 | 新增需求 | 状态 |
|---|---|---|---|
| **L1a** | 静态固定序列长度 prefill，单张图 650 节点 | 上述 11 个算子。**无需 KV cache，无需 shape-as-value** | 第一个可达的完整模型端到端目标 |
| **L1b** | prefill + 多步 decode | KV cache（M2）、变长（M3）、host 侧采样与生成循环 | L1a 之后 |

### 与第一波三条线的关系

- **B 线**选的 8 个 ONNX 名称中，`Cast` `Div` `Mul` `Sqrt` `ReduceMean` `Reshape` **6 个直接命中** L1a 缺口；`Constant` 是导入前提；`Sub` 本模型不用（保留，属通用能力）。
- **C 线**的 `Equal`：在静态导出下折叠后为 0 节点，在动态轴导出下为 80 节点。**它的价值取决于最终走哪条导出路径**，不因本清单作废，但不应被计为 L1a 的必需项。
- L1a 相对 B 线还差 5 个：`Expand` `Neg` `Pow` `Sigmoid` `Unsqueeze`。全部是逐元素或静态形状变换，`Pow` 已在 [M5](implementation/M5_ELEMENTWISE_OPS.md) 计划内。

### 采样不在图内

`MiniMindForCausalLM.generate()` 的 temperature / top-p / top-k / repetition_penalty 全部在 host 侧 Python 中，导出图不含 `ArgMax` / `TopK` / 多项式采样。生成循环的驱动方式（host 循环 vs. 有界 While）是 L1b 的待决项。
