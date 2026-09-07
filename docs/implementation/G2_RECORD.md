# G2 第二波整体验收记录

> 状态：**部分通过（2026-09-08）**。本文按[第二波并行计划](WAVE_2.md)第 4 节逐项记录实测证据。代码集成点：`dev` @ `975429e`。
> WAVE_2 §4 允许在 L1 端到端未打通时只宣布已达成的部分，并写明阻塞项，不得改写目标；本文按此执行。

## 1. 五条线的合入状态

六个分支全部并入 `dev`，均领先 0 个提交：

| 线 | 分支 | 交付 |
|---|---|---|
| A：模型入口 | `line-a2/m9-export` | M9 E0 导出合同锁定 + E2 decode 签名审计（[E0 receipt](M9_E0_RECEIPT.md)、[E2 签名](M9_E2_SIGNATURE.md)） |
| B：KV 状态 | `line-b2/m2-kv-state` | M2 S1 容量/cursor/valid extent/append/read，动态 stateful plan 合同 v1，调用合同 ABI 升至 v3 |
| C：形状与注意力 | `line-c2/m3-shape-values` | M3 S1 `shape_of` + 版本化 extent ABI；S2 受限形状值链与六个形状算子 |
| D：前端/算子 | `line-d2/m4m5-ops` | `Neg`/`Sigmoid`/`Pow`/`Expand` 挂载，`Unsqueeze` 归一到 `reshape`，protobuf e2e |
| E：结构化控制流 | `line-e2/m10-control-flow` | M10 C0 审计 + C1 gate-on LLVM receipt + C2 决策（[M10 receipt](M10_CONTROL_RECEIPT.md)） |
| —（缺陷修复） | `line-f2/reduce-keepdims-fix` | G1 记录的 `reduce_mean keepdims=1` 轴映射缺陷已修 |

## 2. G2 清单逐项结果

| # | 验收项 | 结果 | 证据 / 阻塞 |
|---|---|---|---|
| 1 | 真实 ONNX prefill → importer → Relay → LLVM → RuntimeSession，数值对齐参考 | **未通过** | 见 §3 |
| 2 | 同一 session prefill 后 ≥3 步 decode，地址不变、extent 递增、哨兵不污染 | **部分通过** | `kv_state_llvm_test` 的 `s1_append_read_progression`、`s1_in_place_address_and_sentinels`、`s2_causal_attention_decode_loop` 全部通过，但图是**合成注意力声明**，不是 E0 锁定的真实 decode 图 |
| 3 | 两个合法 bounded shape 同产物执行，无隐式编译；非法输入零 launch | **通过** | `shape_value_llvm_test::bounded_shape_to_reshape_production`：同一产物在 `[4,5,3]`/`[6,7,3]` 上执行，前后 primitive cache 统计不变；越界 extent 与整除 guard 违规均在 launch 前拒绝且 cache 统计不变。另有 `bounded_dynamic_graph_llvm_test` |
| 4 | 最小 greedy token 序列；bundle 按 export receipt / run_id / kernel / state extent 关联 | **未通过** | 仓库中无 greedy 采样或 token 序列测试；bundle 与 export receipt 的关联字段未接入。依赖第 1 项 |
| 5 | M10 控制流 gate-on 证据独立完成 | **通过** | [M10 receipt](M10_CONTROL_RECEIPT.md) C1：`If` 双分支 launch 计数 1/0 与 0/1；`While` 0/1/3 次迭代 carried value 正确；越 `max_trip_count` 抛错；gate-off 拒绝路径保留 |
| 6 | 完整回归 + 全部检查器 + 能力矩阵 | **通过（回归与检查器）** | 见 §4。能力矩阵逐格更新尚未随本波结果刷新，随第 1 项一并处理 |

## 3. 第 1 项的确切阻塞点

**算子覆盖已经不是瓶颈。** L1a 静态 prefill（常量折叠后 650 个实算节点）所需的 18 种 ONNX 算子，`python/kxc_onnx/importer.py` **已全部映射 18/18**：

```
Add Cast Concat Div Expand Gather MatMul Mul Neg Pow
ReduceMean Reshape Sigmoid Slice Softmax Sqrt Transpose Unsqueeze
```

实测导入 E0 锁定的真实图（`out/minimind_onnx_noninplace/minimind_prefill_static.onnx`，1141 节点 / 60 initializer，静态形状 + 非原地 mask 补丁）：

```
UnsupportedONNXOpError: Unsupported ONNX op 'Identity' in node 'Identity_388'
```

`Identity` 在[模型算子清单](../OP_TODO.md)的实算表中不出现，因为常量折叠后它归零（原始图 31 个，折叠后 0 个）；但导入器读的是**未折叠的原始图**，因此仍会遇到。这是一个透传语义缺口，不是新计算能力：

- 归属：M4 导入层（D 线 owns `python/kxc_onnx/`），处置方式应与 `Unsqueeze → reshape` 的归一化一致；
- 这是**第一个**错误，不是唯一一个。修掉后需要重跑导入以暴露后续节点级缺口（属性边界、多输入 `Concat`、`Gather` 常量索引限制等在 G1 已记录的既有限制都可能再次命中）；
- 在第 1 项通过之前，第 2 项的真实图绑定、第 4 项的生成循环与 bundle 关联都不具备前置条件。

## 4. 回归与检查器实测

| 项 | 结果 |
|---|---|
| `ctest` 默认 CPU/LLVM（`dev-ninja-cpu`） | **46/46** |
| `ctest` bounded 专项（`bounded-llvm`，全部 gate 开启） | **50/50** |
| `pytest`（`test/` + `python/`） | **215/215** |
| `check_relay_op_contract` | PASS（36 个算子，新增 7 个形状值算子全部 `tested`） |
| `check_pass_contract` / `check_nlp_gpu_validation` | PASS |
| `check_include_layers` / `check_public_headers --compile` / `check_docs` | PASS |
| `git diff --check` | PASS |

## 5. 本波解决的跨线冲突（记录以免重复踩）

- **`expand` 算子名碰撞**：D 线的静态 `expand`（单输入，目标折进 attrs，承载 ONNX `Expand`）与 C 线的受限形状值 `expand`（双输入，目标由 shape 表达式控制）语义与元数都不同。按 C 线已建立的 `reshape` / `reshape_dynamic` 先例，后者改名 `expand_dynamic`，两者并存。静态目标是动态形态在全部元素均为常量时的特例，是否统一由后续立项。
- **`CanonicalTEScheduleContract` 的双参数**：B 线的 `body_only_runtime_extents`（调用方声明的下标，用于 stateful 路径）与 C 线的 `body_consumed_extents`（lowering 扫描 compute body 得出的 buffer 集合，用于 bounded 路径）服务互斥分支，且 bounded 分支显式拒绝前者。两个参数并存，未合并语义。
- **调用合同 ABI**：B 线的 state-sourced extent 把 `ModuleRuntimeExtentScalar::expression` 变为 `optional` 并把 canonical 编码升到 `KXC_MODULE_INVOKE_V3`。形状值单元沿用同一版本化编码，未另立一套。
- **generated 文件**：`contracts/relay_op_contract.json` 按两边算子集的并集重建（无重叠修改），`src/relay/generated/*` 由生成器重新产出，未手工拼接。

## 6. 下一波的入口

1. **补 `Identity` 透传并重跑真实图导入**，把节点级缺口一次性列全 —— 这是 L1a 唯一的当前阻塞。
2. L1a 通过后发布 E1 receipt，M2/M3 再按真实图的 ABI 绑定，替换当前的合成 fixture。
3. 第 4 项的 host 生成循环与 bundle 关联字段随 E1/E4 一并推进。
4. 能力矩阵在第 1 项通过后统一刷新逐格证据，不以基座能力就绪代替端到端运行证据。
