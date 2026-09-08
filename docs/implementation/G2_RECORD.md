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
| 1 | 真实 ONNX prefill → importer → Relay → LLVM → RuntimeSession，数值对齐参考 | **通过** | E0 锁定的完整 8 层图：650 kernel 调用，全部输出与 ONNX 参考逐元素对齐，最坏差 8.82e-06。见 §3 |
| 2 | 同一 session prefill 后 ≥3 步 decode，地址不变、extent 递增、哨兵不污染 | **未通过**（证据等级已升一档） | 真实 decode 图三步链已端到端验证，但**每步需要一个独立编译产物**，与「同一产物、runtime 期间不编译」直接冲突。见 §3.5 |
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

### 3.2 数值结论：完整模型通过

`test/minimind_l1a_llvm_test.cpp` 用真实图跑 importer → Relay → LLVM → RuntimeSession，逐元素比较**全部输出**（logits 加每层 present_k/present_v），并校验输出签名、秩与形状。参考值由 ONNX 自己的 `ReferenceEvaluator` 算出，与被测路径完全独立，因此是交叉验证而非自证。

| 规模 | Relay 节点 | kernel 调用 | 最坏逐元素差 | 耗时 |
|---|---:|---:|---|---:|
| 1 层 | 90 | 90 | 4.91738e-06 | 1s |
| 2 层 | 170 | 170 | 6.92904e-06 | 2s |
| 3 层 | 250 | 250 | 8.16584e-06 | 3s |
| 4 层 | 330 | 330 | 8.70228e-06 | 3s |
| **8 层（E0 锁定配置）** | **650** | **650** | **8.82149e-06** | **7s** |

8 层峰值 RSS 1.49 GB。650 个 kernel 调用与 [E0 receipt](M9_E0_RECEIPT.md) 记录的折叠后实算规模精确一致。误差随层数平稳增长且量级不变，是 float32 舍入噪声——差异只来自 MatMul 的归约顺序。测试容差取 1e-4，比实测高一个数量级。

### 3.3 曾经的规模限制：一个系统性缺陷，已解除

完整 8 层一度跑不完（OOM 峰值 >10 GB，4 层耗时 >900s 未结束）。根因不是四个孤立 bug，而是**同一个系统性模式的四个实例**：Relay 的表达式是 DAG，而这些地方按**树**遍历它，共享子表达式每被引用一次就重做一次，代价随共享点数量指数增长。Transformer 的残差与 QKV 共享正是最坏情况。

| # | 位置 | 表现 |
|---|---|---|
| 1 | IR 文本打印器 | `ToText`：1 层 2.3 MB，2 层 **784 MB**（337 倍）。`RunInstrumentedPass` 每个 pass 无条件渲染前后两次全图文本，3 层即 OOM（峰值 9.4 GB） |
| 2 | Relay mutator 基类 | `RelayPass::Mutate` 无共享记忆，`fold_constant` 自身指数级，且其输出把 DAG **展开成树**，继续把膨胀喂给下游每个阶段 |
| 3 | ANF 归一化前的类型校验器 | `RequireTyped` 无访问集合，且每层还拼接 path 字符串做诊断：指数次访问 × 与深度成正比的字符串构造 |
| 4 | ANF 名称收集 | `ANFNormalizer::CollectNames` 同一文件同一模式。该类本就有 `atom_cache_` 按 `const Object*` 处理 DAG 共享，只是这个遍历没享受到同等待遇 |

四处修复后曲线彻底平坦（见 §3.2 的耗时列），内存也线性。

两点值得记住：

- 第 3、4 处**不多占任何内存**，因此在此前所有 RSS 曲线里完全隐身，只有在标准流水线下按耗时才暴露。早期用 `KXC_MINIMIND_RELAY_PASSES=""` 绕过 pass 流水线做的测量根本没走到 mutator，据此得出的结论不完整——测量条件必须与验收条件一致。
- 同一模式出现四次，说明「按树遍历 Relay DAG」在这个代码库里是**系统性的**。逐处打补丁治标：每新加一个按树遍历的地方，就会重新引入一次同样的指数缺陷。

> **建议单独立项**：给 `RelayPassFunctor` 那一层的遍历基类提供**默认带记忆的遍历**，让新写的 pass 与分析默认就是 DAG 安全的。这是本轮唯一未做的根因收口。

> 记账口径：3.3 的四处修复由并行会话 `hdkx-aicompiler-bb` 提交（`8a27ef7`），不在本线的提交里；本文记录是因为它们直接决定第 1 项能否达成。§3.2 的全部数字由本线独立复测确认。

### 3.5 第 2 项：真实 decode 图已验证，但判未通过

真实 decode 图（666 kernel）端到端通过，且不是孤立测一张图——**past 取自 prefill 参考的 present**，`present[1,16,4,96]` 与 `past[1,16,4,96]` 精确对接，因此验证的是一次真实的自回归续接。三步链实测：

| 步 | 图 | 误差 |
|---|---|---|
| prefill | 650 kernel | 8.82149e-06 |
| decode（past 16→17） | 666 kernel | 6.85453e-06 |
| decode（past 17→18） | 666 kernel | 5.54e-06 |

decode 单步 4s / 1.49 GB（本线独立复测确认 666 kernel、6.85453e-06）。

**但这不是第 2 项。** past 长度被烤进静态导出的形状：`--past 16` 是 `past[1,16,…]→present[1,17,…]`，`--past 17` 是 `past[1,17,…]→present[1,18,…]`，节点数同为 1173 但形状不同，**每步需要一个独立编译产物**。而第 2 项要求同一 session、同一产物、cache 地址不变、runtime 期间不发生隐式编译——直接冲突。

因此现在拿到的是 [WAVE_2](WAVE_2.md) §4 明确允许的「真实图 + 外部-KV 受控证据」：比原先的合成 attention fixture 强一档，但不能替代第 2 项本身。`kv_state_llvm_test` 的合成 `AttentionDeclaration` 未改动——M2 的状态机是声明式的、不消费任意 Relay 图，硬塞真实图会变成为通过而通过。

**一个对 M2 有决定意义的新事实**：`present` 的前缀与输入 `past` **逐位相同**（`array_equal` 成立，不是容差意义上的接近）。全部 16 对 (past, present)、past 长度 16 对 present 17，无一例外。也就是说真实图对 cache **只追加、不改写历史**。

这里的「逐位」不是措辞讲究：绑同一块 state 之后，图若偷偷改写了历史，靠数值容差阈值是发现不了的，只有精确比较能兜住。

该性质由两条独立路径确认——生成侧比对内存中的数组，本线比对落盘后的 `in_*.bin` / `ref_*.bin` 字节。两条路的数据源不同，因此结论不依赖于 fixture 生成代码本身是否有缺陷。

这解掉了原先「要么加一次拷贝、要么把 past 输入与 present 输出绑到同一块 state」的二选一：**后者的语义前提已经成立**，那次拷贝是冗余的，剩下的是 runtime 绑定的工程问题而非语义风险。

要真正通过第 2 项，二选一（详见 [M9 E2 签名](M9_E2_SIGNATURE.md) §4）：

1. **有界动态 past**：一个产物覆盖所有步，依赖 M3 的受限 shape 绑定；
2. **定容 state + 有效长度掩码**：依赖 [M2](M2_KV_STATE.md) §3 第 1、2 条。

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

已完成：

1. ~~`Identity` 透传~~（`dbe7b6b`）。
2. ~~导入侧常量折叠~~（`9996f63`）。491 个静态节点折掉，三个缺口一次消解，未新增算子。
3. ~~运行时索引 `Gather`~~（`9996f63` / `9d99941`）。
4. ~~L1a 端到端与数值对齐~~（`9d99941`）。完整 8 层 650 kernel，最坏差 8.82e-06。
5. ~~消除编译规模限制~~（`8a27ef7`）。四处 DAG 遍历修复。
6. ~~发布 M9 E1 receipt~~（`282534e`）。覆盖 L1a 静态 prefill。
7. ~~真实 decode 图端到端~~（`c60beeb`、`32fc222`）。666 kernel，past 取自 prefill 的 present。

仍待办：

8. **通过 G2 第 2 项**（当前最主要的缺口，见 §3.5）。静态导出每步换产物的路走不通，二选一：
   - **有界动态 past**——一个产物覆盖所有步，依赖 M3 的受限 shape 绑定；
   - **定容 state + 有效长度掩码**——依赖 M2 §3 第 1、2 条。

   §3.5 已确认 `present` 前缀逐位等于输入 `past`，真实图对 cache 只追加不改写，因此把 past 输入与 present 输出绑到同一块 state 在语义上安全，无需冗余拷贝。这是走第 2 条路的前提，已经成立。

   实操提示（避免重复摸索）：`export_minimind_onnx.py` 已有 `--past N`，「每步一个产物」这条现状随时可复现，不需要新工具。真正的工作量在 runtime 侧——M2 的 `KvStatePlanDeclaration` 是声明式生成 plan 的，接真实图需要**新增一条绑定路径**把 `past` 输入与 `present` 输出绑到同一块 state，而不是去改那个声明。
9. **G2 第 4 项**：host greedy 生成循环 + bundle 与 export receipt / run_id 的关联字段。依赖第 8 项拿到可复用的多步产物。
10. **根因收口（建议单独立项）**：给 `RelayPassFunctor` 遍历基类提供默认记忆化，见 §3.3。本波修的四处是同一模式的四个实例；不收口的话，每新增一个按树遍历就重新引入一次指数缺陷。**这项独立于 L1，不阻塞任何人，但拖得越久新写的 pass 越多。**
11. **插桩惰性化**：`RunInstrumentedPass` 无条件渲染两次全图 IR 文本（Relay/TIR 两处同构），见 §3.4。8 层 7 秒说明 TIR 侧没有同样的爆炸，优先级低于第 10 项。
12. 能力矩阵在第 8、9 项完成后统一刷新逐格证据。

> 方法论提醒（本波三次踩到）：
> - §3.1 的四个原始缺口是静态审计一次列全的，但其余四处（`Gather` 运行时索引、三个缺失的推导链、恒等 `Cast`）是逐个跑出来的——审计能列出结构性缺口，列不全语义缺口。
> - §3.3 的第 3、4 处不多占任何内存，在所有 RSS 曲线里隐身，只有在标准流水线下按耗时才暴露；早期用 `KXC_MINIMIND_RELAY_PASSES=""` 绕过流水线的测量根本没走到 mutator，据此得出的结论不完整。**测量条件必须与验收条件一致。**
> - §3.5 的 decode 三步链单看数值全部通过，只有追问「同一产物吗」才发现不满足第 2 项。**通过的数值不等于通过的验收项。**
