# M9 E0 导出 receipt（MiniMind-L1）

> 状态：已完成（2026-09-07）。任务书见 [M9_MINIMIND_TARGET.md](M9_MINIMIND_TARGET.md)；E2 decode 签名审计见 [M9_E2_SIGNATURE.md](M9_E2_SIGNATURE.md)。产物目录：worktree `out/`（不入库，SHA 是唯一入库证据）。

## 1. 锁定的导出合同

| 项 | 值 |
|---|---|
| 模型源码 | `github.com/jingyaogong/minimind` @ `6fc918b`（`out/minimind`） |
| 模型配置 | hidden 768 / 8 层 / 8 heads / **4 kv_heads** / head_dim 96 / vocab 6400 / silu / rms_norm_eps 1e-6 / rope_theta 1e6 / tie_word_embeddings / flash_attn=false / 参数 63.91M |
| 权重 | 随机初始化，`torch.manual_seed(0)`（只关心图结构与算子集合） |
| 框架 | torch 2.14.0+cpu、onnx 1.22.0、protobuf 7.36.1、Python 3.12 |
| 导出 | `torch.onnx.export`，opset **17**，`dynamo=False`，`do_constant_folding` 默认 |
| 样例形状 | batch=1、prefill_seq=16、decode_past=16；RoPE max_pos 2048 |
| KV 布局 | `[batch, seq, kv_heads, head_dim]`；prefill/decode 各输出 8 组 `(k, v)`，k/v 交错扁平序 |

## 2. 产物与可复现性（SHA256）

| 产物 | SHA256（前 16 位） |
|---|---|
| `out/minimind_onnx/minimind_prefill.onnx`（动态轴） | `85f1d7ba2bf3971f` |
| `out/minimind_onnx/minimind_decode.onnx`（动态轴） | `6d3f4262b09ac562` |
| `out/minimind_onnx/minimind_prefill_static.onnx` | `c140702fee9b1280` |
| `out/minimind_onnx/minimind_decode_static.onnx` | `4ff0f2714a987669` |
| `out/minimind_onnx_noninplace/minimind_prefill_static.onnx`（非原地 mask 补丁） | `eb1c60c73980b0e5` |
| `out/minimind_onnx_noninplace/minimind_decode_static.onnx`（非原地 mask 补丁） | `e0343a87ed47a67d` |

**可复现性证据**：全部 6 个产物在重新导出后 SHA256 逐字节复现（其中动态 decode 图曾在一次磁盘写满事故中被截断损坏，重新导出后精确恢复为记录值——导出是确定性的）。完整 SHA 见 `out/audit/first_run_sha256.txt` 与 `out/audit/noninplace_sha256.txt`。

## 3. 动态导出门禁：**通过**

**结构检查**（`out/audit/audit_decode_dynamic.txt`，动态 decode 图 3656 节点、opset 17）：

- 输入顺序正确（17 个）：`input_ids [batch, 1]`，随后逐层交错 `past_k_i / past_v_i`，均为 `[batch, past, 4, 96]`；
- 输出顺序正确（17 个）：`logits [batch, seq, 6400]`，随后逐层交错 `present_k_i / present_v_i`，均为 `[batch, total, 4, 96]`；
- `total = past + current`（decode current=1）在轴声明中成立。

**数值检查**（`onnx.reference.ReferenceEvaluator` vs PyTorch，`out/audit/run_dynamic_numeric.py` / `run_dynamic_prefill_numeric.py`）：

| 用例 | logits max&#124;Δ&#124; | present max&#124;Δ&#124; | present 形状（验证 total） |
|---|---:|---:|---|
| decode past=16, batch=1 | 2.6e-06 | 4.2e-06 | [1, **17**, 4, 96] = past+1 |
| decode past=8, batch=1 | 1.7e-06 | 2.8e-06 | [1, **9**, 4, 96] |
| decode past=33, batch=1 | 2.7e-06 | 4.6e-06 | [1, **34**, 4, 96] |
| decode past=16, batch=2 | 2.9e-06 | 5.2e-06 | [2, 17, 4, 96]（batch 轴动态生效） |
| prefill seq=12/16/24, batch=1 | ≤3.6e-06 | ≤5.5e-06 | [1, seq, 4, 96] |

RoPE 位置语义随 past 长度正确平移（不同 past 下 logits 均与 torch 一致）。

**结论**：动态 `past_key_values` 导出可用，E2 可据此锁定签名；L1b 不再 blocked-by-export。E3 的受限 shape 子集（Shape/Range/ConstantOfShape 链是否照搬）由 M3 按图实测决定。

## 4. 三种导出配置的算子清单对比

详见 [OP_TODO](../OP_TODO.md)（原始图统计与常量折叠后实算统计分开列）：

| 导出配置 | 总节点 | 常量折叠后实算 | 算子种类 | importer 缺失 |
|---|---:|---:|---:|---:|
| 动态轴 | 3634 | 2354 | 25 | 17 |
| 静态形状 | 2053 | 778 | 20 | 13 |
| 静态 + 非原地 mask | 1141 | 650 | 18 | **11** |

Inventory 明细：`out/minimind_onnx/inventory/{prefill,decode}_{static,dynamic}.md`、`out/minimind_onnx_noninplace/inventory_*_noninplace.md`。

## 5. 给 D 线（M4/M5）的算子需求清单

L1a（静态 + 非原地 mask，650 实算节点）相对第一波 importer 还差 **5 个 ONNX 算子**：

| ONNX 算子 | 静态图出现次数 | 输入/输出 | 关键约束（以 inventory/OP_TODO 为准） |
|---|---:|---|---|
| `Expand` | 96（原始图） | 2/1 | 目标 shape 来自受限形状表达式；与广播规则对齐 |
| `Neg` | 16 | 1/1 | 一元逐元素，float32 |
| `Pow` | 33 | 2/1 | RMSNorm/SwiGLU 的 `x²` 与 silu 指数；float32；语义按 M5 S2（禁 `exp(b·log a)` 替代） |
| `Sigmoid` | 8 | 1/1 | 一元逐元素，float32 |
| `Unsqueeze` | 80（原始图） | 2/1 | axes 来自常量；静态时可规范化为 reshape（与 M4 S2 一致）；axes 输入是常量张量 |

其余 6 个缺失名称（`ConstantOfShape` `Erf` `Shape` `Split` `Squeeze` + 动态 `Gather` 链）不在 L1a 静态+非原地路径的实算集内或属 M3/M4 后续切片：`Shape`/`ConstantOfShape`/`Squeeze` 由 M3 的形状值切片承接；`Split`（1 处）与 `Erf`（原始图 6 处，常量折叠后实算集中是否保留以 inventory 为准）按 M4 S3/M5 S3 推进。

## 6. 非原地 mask 补丁的边界记录

`python/tools/minimind_noninplace_mask.patch` 只改变 mask 的实现方式（避免原地 scatter 类算子），不改变 causal mask 语义；补丁前后静态图的节点差异见 `out/minimind_onnx_noninplace/inventory_*_noninplace.md` 与原始 inventory 的对比。补丁后的图是 L1a 的目标图；它不是生产能力声明。
