# M9：MiniMind 模型目标与导出验收

`PROJECT_GOAL.md` 已经把 Transformer 的目标模型定为 MiniMind 系列：MiniMind-O 是北极星，纯文本 MiniMind 是当前验收目标。第一波 G0/G1 已完成执行观测、Equal 闭环和一组静态 ONNX 导入，但还没有证明 MiniMind 图可以从导出一路运行到生成。现在仓库有实际导出脚本和产物：`python/tools/export_minimind_onnx.py` 可生成 prefill/decode 两张图，`out/minimind_onnx_noscatter/` 保存了一组静态、非原地 mask 的对照产物，元数据记录了 opset 17、8 层、hidden 768、4 个 KV heads 和 `[batch, seq, kv_heads, head_dim]` 布局。动态 decoder 是否能稳定导出 `past_key_values`、`past`/`present` 轴和可复现的图，仍未验收。

本模块把“目标模型是什么”和“导出的图是否就是要支持的图”变成可重复的入口。L1 先完成纯文本 MiniMind 的固定形状 prefill，再完成同一会话的多步 decode；L2 的 MiniMind-V 和 L3 的 MiniMind-O 只复用并扩展这条证据链。它不把一份 ONNX 节点统计表当成编译器能力，也不在导出失败时用隐式 Python 推理或临时算子绕过编译器。

> 状态：待实施，下一波的模型入口门禁。第一波证据见 [G0](G0_BASELINE.md) 和 [G1](G1_RECORD.md)，并行安排见 [WAVE_2](WAVE_2.md)。目标阶梯以 [PROJECT_GOAL.md](../PROJECT_GOAL.md) §2.2 为准，节点统计以 [OP_TODO.md](../OP_TODO.md) 为准。

## 当前事实与要做的模块

| 模块 | 当前情况 | 要做的事 |
|---|---|---|
| E0 导出可复现 | 已有 MiniMind 源码、导出脚本、opset 17 静态和 dynamic_axes 产物 | 锁定源码版本、配置、PyTorch/ONNX 版本，重新导出并保存元数据与 SHA256；动态 `past_key_values` 作为通过/失败门禁 |
| E1 L1a prefill | 静态 prefill 图已生成；常量折叠后的实算缺口记录在 OP_TODO | 用真实图完成 importer → Relay → LLVM → RuntimeSession，逐项补齐实际需要的算子 |
| E2 L1b decode | 静态 decode 图已生成，包含外部 past/present 张量；运行时还没有 KV 更新合同 | 交给 M2 绑定同一会话的缓存、有效长度和多步 decode |
| E3 有界变长 | dynamic_axes 原始图会引入大量 Shape/Unsqueeze/ConstantOfShape 链 | 交给 M3 选择“有界 shape 合同”或受限 shape-as-value 子图；不得因导出器插入链而开放任意动态 rank |
| E4 生成驱动 | `generate()` 的采样逻辑在 host 侧，不在 ONNX 图中；M10 的 `ControlRuntimeSession` 只支持静态 CPU 控制流，不能直接维护 KV/extent | 先定义最小 host 循环、token 选择、停止条件和每步输入/输出合同，验证 prefill 后连续 decode；再依据 M10 C2 决定是否评估固定上限的图内 `While` |
| E5 MiniMind-O 演进 | Thinker 复用 MiniMind；Talker、SenseVoice、Mimi、CAM++ 尚未接入 | 先做算子/状态/实时预算盘点；Conv1d/ConvTranspose1d、Mimi ring buffer、双自回归调度不得提前写进 L1 合同 |

## E0：锁定导出合同

1. 记录 MiniMind 源码 commit、模型配置、随机种子、PyTorch/ONNX 版本、opset、`flash_attn`、`dynamo`、样例 batch/prefill/past 长度。
2. 分别生成 prefill 和 decode 的 static 与 dynamic 版本。每张图旁边写 `export_metadata*.json`，并计算 SHA256；产物只放 `out/` 或 CI artifact。
3. 用 `python/tools/onnx_op_inventory.py` 生成原始节点、输入/输出数量、属性和 opset 表。`OP_TODO.md` 同时区分原始图统计与常量折叠后的实算统计，不能把二者混成一个“支持算子数”。
4. 对 dynamic decode 逐项检查 `past_k_i/past_v_i` 的 batch/past 轴、`present_k_i/present_v_i` 的 batch/total 轴、`total = past + current` 关系及输出顺序。任何轴丢失、别名错位、不可复现或导出器异常都记录为门禁失败。
5. 非原地 mask 补丁只能作为导出实验输入，必须保存补丁和原始图的差异；它不能改变 causal mask 语义，也不能把补丁后的节点数直接当成生产能力。

## E1：L1a 静态 prefill

1. 选定一张固定 batch/sequence 的 prefill 图，读取真实 ONNX，而不是手写等价 Relay 图。
2. 先做节点级导入审计，再按缺口由 M4/M5 接通 `Cast`、`Div`、`Expand`、`Mul`、`Neg`、`Pow`、`ReduceMean`、`Reshape`、`Sigmoid`、`Sqrt`、`Unsqueeze` 等实际需要的语义；每个算子都要经过属性校验、生产 lowering、LLVM 数值和负例。
3. 权重、常量、RoPE、GQA、RMSNorm、SwiGLU 和 attention 的中间值按独立参考比较。`ReduceMean keepdims=1` 的中间轴缺陷沿用 G1 记录，未修复前不得选该配置作通过证据。
4. 通过后发布一个带模型配置、导出 SHA、opset、输入输出签名和能力矩阵格子的 L1a receipt。没有 receipt 的图不能被 M2/M3 当作固定 ABI。

## E2/E3：为 decode 和变长交接明确边界

- M2 负责缓存的物理容量、层/头布局、cursor/valid extent、append 后的 present 绑定和失败后的 session 状态。
- M3 负责 batch/sequence/past/total 的有限范围、整除和广播证明，以及 Shape/Reshape/Expand 等控制值来源；不让 RuntimeSession 依赖 compiler 的 ShapeProgram。
- M4 负责 ONNX 节点顺序、常量控制输入和多输出名的稳定重建；`present_k_i/present_v_i` 的顺序必须保持。
- 动态导出若无法通过，先用静态 L1a 和受控外部-KV fixture 继续推进，同时把 dynamic export 标为 blocked-by-evidence；不能假设它已经是可用合同。

## E4：最小生成验收

先实现 host 侧确定性 greedy 循环，避免把采样器和编译器混在一起：prefill 返回 logits 与 KV，选择一个 token；之后每一步只输入新 token 和同一 session 的 cache，直到固定步数或 EOS。每步记录 input token、past/total extent、selected token、输出 checksum 和 profile run id。temperature/top-k/top-p/repetition penalty 作为后续可替换策略，不能在首个 L1 证据中偷偷改变图合同。M10 的 gate-on `While` 只作为独立静态/合成能力和后续候选；在 M2/M3 尚未交付 state/extent ABI 前，不得把它写成 L1b 已使用的生成循环。

## E5：L2/L3 预研出口

L1b 通过后，才对 MiniMind-V 的 256×256、64 patch token 静态视觉链做算子复用核对；再对 MiniMind-O 的 Thinker/Talker、SenseVoice、SigLIP2、Mimi、CAM++ 分别导出和盘点。Mimi 每层 ring buffer、Talker 12.5 Hz（80 ms）预算和 barge-in/近双工会话需要新的状态/服务决策，不能用 KV cache 的通过项代替。

## 代码落点与验证

| 位置 | 责任 |
|---|---|
| `python/tools/export_minimind_onnx.py` | 可复现导出入口和元数据 |
| `python/tools/onnx_op_inventory.py`、`docs/OP_TODO.md` | 原始图/折叠图统计与差距记录 |
| `out/minimind_onnx*` | 本地/CI 产物，不入库 |
| `python/kxc_onnx/`、`src/frontend/onnx_importer.cc` | 真实节点导入与多输出顺序 |
| `test/` 中新增 MiniMind smoke/e2e | 导出可复现、签名、数值、负例和生成循环 |

```bash
out/venv/bin/python python/tools/export_minimind_onnx.py --static --out out/minimind_onnx
out/venv/bin/python python/tools/onnx_op_inventory.py \
  out/minimind_onnx/minimind_prefill_static.onnx \
  out/minimind_onnx/minimind_decode_static.onnx \
  --label prefill --label decode
```

退出条件：导出门禁有完整元数据和失败证据；L1a 至少一张真实 prefill 图通过 LLVM RuntimeSession 数值；L1b 通过同一 session 的 prefill + 三步 decode；非法 past/total、shape、dtype、输出数量在 launch 前失败；每项结果都写入能力矩阵。任何一项未完成都不能写成“MiniMind-O 已端到端支持”。
