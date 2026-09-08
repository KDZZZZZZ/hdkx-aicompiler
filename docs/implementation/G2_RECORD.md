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
| 1 | 真实 ONNX prefill → importer → Relay → LLVM → RuntimeSession，数值对齐参考 | **数值通过，规模受限** | 1/2/3 层真实图逐元素对齐（4.9e-06 / 6.9e-06 / 8.2e-06），完整 8 层受编译耗时超线性所限暂未跑完。见 §3 |
| 2 | 同一 session prefill 后 ≥3 步 decode，地址不变、extent 递增、哨兵不污染 | **部分通过** | `kv_state_llvm_test` 的 `s1_append_read_progression`、`s1_in_place_address_and_sentinels`、`s2_causal_attention_decode_loop` 全部通过，但图是**合成注意力声明**，不是 E0 锁定的真实 decode 图 |
| 3 | 两个合法 bounded shape 同产物执行，无隐式编译；非法输入零 launch | **通过** | `shape_value_llvm_test::bounded_shape_to_reshape_production`：同一产物在 `[4,5,3]`/`[6,7,3]` 上执行，前后 primitive cache 统计不变；越界 extent 与整除 guard 违规均在 launch 前拒绝且 cache 统计不变。另有 `bounded_dynamic_graph_llvm_test` |
| 4 | 最小 greedy token 序列；bundle 按 export receipt / run_id / kernel / state extent 关联 | **未通过** | 仓库中无 greedy 采样或 token 序列测试；bundle 与 export receipt 的关联字段未接入。依赖第 1 项 |
| 5 | M10 控制流 gate-on 证据独立完成 | **通过** | [M10 receipt](M10_CONTROL_RECEIPT.md) C1：`If` 双分支 launch 计数 1/0 与 0/1；`While` 0/1/3 次迭代 carried value 正确；越 `max_trip_count` 抛错；gate-off 拒绝路径保留 |
| 6 | 完整回归 + 全部检查器 + 能力矩阵 | **通过（回归与检查器）** | 见 §4。能力矩阵逐格更新尚未随本波结果刷新，随第 1 项一并处理 |

## 3. 第 1 项：已达成与仍受限的部分（2026-09-08 实测）

### 3.1 导入链路已全部打通

E0 锁定的真实图（prefill 1141 节点 / decode 1173 节点）现在完整导入：prefill 650 个 Relay 节点、decode 666 个、各 446 个参数。650 与 [E0 receipt](M9_E0_RECEIPT.md) 记录的折叠后实算规模一致。四个原始缺口的处置：

| 缺口 | 节点数 | 处置 |
|---|---:|---|
| `Identity` | 31 | 归一为**参数别名**（`dbe7b6b`）。全部是导出器对逐字节相同 initializer 去重后的权重别名，零 Relay 节点、零拷贝 kernel |
| `ConstantOfShape` | 16 | 由**导入前常量折叠**消解（`9996f63`），未新增算子 |
| `Trilu` | 8 | 同上。`Trilu(ConstantOfShape(…), k, upper=1)` 两个输入都来自 `Constant`，整条 mask 子图是常量 |
| `Expand` 控制形状来自 Shape 链 | 16 | 折叠后变成 initializer 常量，直接落到既有静态 `expand`，未接形状值链 |

折叠掉 491 个纯静态节点，只物化 386 个前沿值共 257 KiB。求值用 ONNX 自己的 `ReferenceEvaluator`，折叠语义即 ONNX 语义。

静态审计之外，逐个跑出来的还有四处（审计只覆盖了算子种类和几类结构约束，没覆盖 dtype 与推导链）：

- `Gather` 运行时索引（1 个，embedding 查表）：导入期只固定形状/dtype/axis 合同，值域由 lowering 后的 `GatherCompute` 守卫承担；常量索引仍走更强的逐值域证明。Python 与 C++ 两侧对齐。
- `Add`(73) / `Transpose`(32) / `Softmax`(8) 不在逐算子推导链里：补齐，并带可解析预检，避免把 resnet18 这类图从能导入变成失败。
- 57 个 float32→float32 恒等 `Cast`（RMSNorm 的 `.float()`）：放开，落成一次拷贝。

### 3.2 数值结论成立

`test/minimind_l1a_llvm_test.cpp` 用真实图跑 importer → Relay → LLVM → RuntimeSession，与 ONNX 参考实现逐元素比较全部输出：

| 规模 | Relay 节点 | kernel 调用 | 最坏逐元素差 | 编译+执行耗时 |
|---|---:|---:|---|---:|
| 1 层 | 90 | 90 | 4.91738e-06 | 1s |
| 2 层 | 170 | 170 | 6.92904e-06 | 3s |
| 3 层 | 250 | 250 | 8.16584e-06 | 149s |

误差量级是 float32 舍入噪声，且随层数平稳增长。参考路径与被测路径完全独立。

### 3.3 仍受限：编译规模，根因是一个系统性模式

完整 8 层（650 kernel）暂时跑不完。本轮定位到的不是三个孤立缺陷，而是**同一个系统性模式的三个实例**：Relay 的表达式是 DAG，而这些地方按**树**遍历它，共享子表达式每被引用一次就重做一次，代价随共享点数量指数增长。Transformer 的残差与 QKV 共享正是最坏情况。

| # | 位置 | 表现 | 状态 |
|---|---|---|---|
| 1 | IR 文本打印器 | `ToText`：1 层 2.3 MB，2 层 **784 MB**（337 倍）。`RunInstrumentedPass` 每个 pass 无条件渲染前后两次全图文本，3 层即 OOM（峰值 9.4 GB） | 已修：DAG 感知打印，共享节点只展开一次 |
| 2 | Relay mutator 基类 | `RelayPass::Mutate` 无共享记忆，`fold_constant` 自身指数级，且其输出把 DAG **展开成树**，继续把膨胀喂给下游每个阶段 | 已修：按节点指针记忆化 |
| 3 | ANF 归一化前的类型校验器 | `RequireTyped`（`src/relay/transforms/normalize_to_anf.cc`）无访问集合，且**每层还拼接 path 字符串**做诊断。指数次访问 × 与深度成正比的字符串构造 | 已修：加访问集合，每个节点只校验一次 |

第 3 处尤其说明问题：它**一分钱内存都不多占**，因此在此前所有 RSS 曲线里完全隐身，只有在标准流水线下按耗时才暴露。

**修复效果**：内存已线性（约 117 MiB/层）；3 层耗时 181s → **42s**（本线独立复测，与并行会话报告的 44s 一致），数值不变仍为 8.16584e-06 且通过。

**但曲线仍超线性**：2 层 3s → 3 层 42s 仍是十余倍跳变，4 层未在可接受时间内结束。因此**至少还有第四处未定位**。

> 根因收口建议：逐处打补丁治标。真正该做的是给 `RelayPassFunctor` 那一层的遍历基类提供**默认带记忆的遍历**，让新写的 pass 和分析默认就是 DAG 安全的——否则每加一个按树遍历的地方，就重新引入一次同样的指数缺陷。

> 记账口径：3.3 的三处修复由并行会话 `hdkx-aicompiler-bb` 提交，不在本线的提交里；本文记录是因为它们直接决定第 1 项的可达规模。耗时数字由本线独立复测确认。

### 3.4 顺带记录的插桩缺陷

`RunInstrumentedPass`（`src/relay/transforms/pipeline.cc:126` 与 `src/tir/transforms/pipeline.cc:136` 两处同构）无条件渲染前后两次全图 IR 文本，只为算一个变更 hash 和 `ir_*_bytes` 指标；文本仅在 `ShouldCaptureIR` 为真时才需要落 artifact。这是上面指数缺陷能打到编译热路径的直接原因。改成惰性会动到 M1 的 profiling 契约（`ir_changed` 依赖 hash、hash 依赖文本，且 `test/profile_bundles` 有录好的 bundle），本轮不动，留档待立项。

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
2. ~~导入侧常量折叠 pass~~ —— **已完成**（`9996f63`）。491 个静态节点折掉，`ConstantOfShape`/`Trilu`/`Expand` 形状链三个缺口一次消解，未新增算子。
3. ~~带运行时索引的 `Gather`~~ —— **已完成**（`9996f63` / `9d99941`，Python 与 C++ 两侧）。
4. ~~L1a 端到端与数值对齐~~ —— **数值结论已成立**（`9d99941`），1/2/3 层真实图逐元素对齐。
5. **消除编译耗时的超线性**（当前唯一挡住完整 8 层的问题，见 §3.3）。已修三处，3 层耗时降 4 倍，但曲线仍超线性，至少还有第四处未定位。收口做法是给遍历基类提供默认记忆化，而不是继续逐处打补丁。做完才能跑完整 8 层并发布 E1 receipt。
6. E1 receipt 发布后，M2/M3 按真实图 ABI 绑定，替换第 2 项的合成 fixture。
7. 补第 4 项：host greedy 生成循环 + bundle 与 export receipt / run_id 的关联字段。
8. 把 `RunInstrumentedPass` 的 IR 文本渲染改成惰性（见 §3.4），需要连带处理 M1 profiling 契约。
9. 能力矩阵在第 1 项完整通过后统一刷新逐格证据，不以基座能力就绪代替端到端运行证据。

> 方法论提醒：§3.1 的四个原始缺口是静态审计一次列全的，但 3.1 末尾那四处（`Gather` 运行时索引、三个缺失的推导链、恒等 `Cast`）是逐个跑出来的——审计只覆盖了算子种类和结构约束，没覆盖 dtype 与元数据推导。后续每改一处仍须重跑导入。同理，§3.3 的耗时超线性是**在标准 pass 流水线下才第一次暴露**的：早期用 `KXC_MINIMIND_RELAY_PASSES=""` 绕过流水线做的测量根本没走到 mutator，得出的结论不完整。测量条件必须和验收条件一致。
