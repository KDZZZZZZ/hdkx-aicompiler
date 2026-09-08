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

## 3. 第 1 项的确切阻塞点（2026-09-08 实测更新）

**算子覆盖不是瓶颈。** L1a 静态 prefill 所需的 18 种 ONNX 算子，`python/kxc_onnx/importer.py` 已全部映射。真实图（`out/minimind_onnx_noninplace/minimind_prefill_static.onnx`，1141 节点 / 60 initializer；decode 图 1173 节点，形态一致）的完整节点级缺口如下。

### 3.1 原始（未折叠）图的四个缺口

| 缺口 | 节点数 | 性质 | 状态 |
|---|---:|---|---|
| `Identity` 未映射 | 31 | 导出器对逐字节相同的 initializer 去重后的**权重别名**：输入全是 initializer，无一喂图输出，也不串联 | **已解决**（`dbe7b6b`，归一为参数别名，零 Relay 节点、零拷贝 kernel） |
| `ConstantOfShape` 未映射 | 16 | Relay 侧算子 `constant_of_shape` 已由 M3 S2 交付，只差导入映射 | 待办 |
| `Expand` 控制形状来自 Shape 链（非 initializer） | 16 | 正是 M3 S2 `expand_dynamic` 受限形状值链的目标场景，编译侧能力已具备 | 待办 |
| `Trilu` 未映射 | 8 | 因果 mask 上三角，Relay 侧无对应算子 | 待办 |

### 3.2 关键发现：这三个待办缺口都是**常量子图**

对全图做静态可达性分析（值只依赖 initializer/Constant，不依赖任何图输入）：

- 16 个 `ConstantOfShape` 输出 **全部静态**；
- 8 个 `Trilu` 输出 **全部静态**（`Trilu(ConstantOfShape(…), k, upper=1)`，两个输入都来自 `Constant`）；
- 16 个 `Expand` 的**控制形状输入全部静态**（数据输入才是运行时值）；
- 全图 1141 节点中 **491 个是纯静态可折叠节点**（含 388 个 `Constant`、31 个 `Identity`、以及整条 mask 子图 `ConstantOfShape`/`Mul`/`Equal`/`Where`/`Trilu`）。

**折叠后剩 650 个实算节点，18 种算子全部已映射，节点级约束冲突只剩 1 处**（该 650 与 [E0 receipt](M9_E0_RECEIPT.md) 记录的实算规模一致）：

```
Mul 114  Add 73  MatMul 73  Cast 57  Reshape 48  Div 41  Pow 33
ReduceMean 33  Sqrt 33  Slice 32  Transpose 32  Neg 16  Concat 16
Unsqueeze 16  Expand 16  Softmax 8  Sigmoid 8  Gather 1
```

| 折叠后仅存的缺口 | 节点数 | 说明 |
|---|---:|---|
| `Gather` 运行时索引 | 1 | token embedding 查表 `embed_tokens.weight[input_ids]`；导入器现要求 initializer-backed 常量索引。[M3 任务书](M3_SHAPE_VALUES.md) S3 已预告：通用 Gather 的运行时索引需要独立范围检查，不能沿用「常量索引已证明安全」的结论 |

### 3.3 由此修正的实施路线

打通 L1a **不需要新增 `ConstantOfShape` / `Trilu` 两个算子，也不需要为 Expand 接形状值链**。更小且更正确的做法是：

1. 在导入侧加一个**常量折叠 pass**（折叠 491 个纯静态节点）。它一次性消解 3.1 表里剩下的三个缺口：`ConstantOfShape` 与 `Trilu` 整体消失，`Expand` 的控制形状变成 initializer-backed 常量，直接落到 D 线已挂载的静态 `expand`。
2. 之后唯一需要真正新增的能力是 **带运行时索引的 `Gather`**（含索引范围检查），只有 1 个节点，但它是 embedding 查表，无法绕开。

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

1. ~~补 `Identity` 透传~~ —— **已完成**（`dbe7b6b`）。
2. **导入侧常量折叠 pass**（491 个纯静态节点）。这是当前性价比最高的一步：一次消解 `ConstantOfShape`(16)、`Trilu`(8)、`Expand` 形状链(16) 三个缺口，且不新增任何算子。折叠后须重跑导入确认实算算子集与 §3.2 的 650 节点一致。
3. **带运行时索引的 `Gather`**（含索引范围检查）—— 折叠后唯一剩余的真实能力缺口，embedding 查表无法绕开。
4. 前两项通过后跑通 L1a 端到端并发布 E1 receipt；M2/M3 再按真实图 ABI 绑定，替换第 2 项的合成 fixture。
5. 补第 4 项：host greedy 生成循环 + bundle 与 export receipt / run_id 的关联字段。
6. 能力矩阵在第 1 项通过后统一刷新逐格证据，不以基座能力就绪代替端到端运行证据。

> 记账口径提醒：§3 的缺口是**逐个暴露**的——导入器在拓扑序上先撞到 `Gather`，`ConstantOfShape`/`Trilu` 在其后。四个缺口是静态审计一次列全的，不是靠反复试错得到的；后续每改一处仍须重跑导入，确认没有新的节点级约束（属性边界、多输入 `Concat` 等）被激活。
