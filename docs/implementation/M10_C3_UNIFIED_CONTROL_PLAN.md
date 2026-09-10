# M10 C3 实施计划：把结构化控制流并入主执行链

> 状态：**大部分实施（2026-09-11）**。PR1（统一静态 `If`）、PR2（统一 `While` 与迭代值作用域/生命周期）、plan identity 覆盖、PR3（region-aware bounded admission）、PR4（region 边界状态提交）与 PR6（清退第二执行权威）已落地并通过验证；仅 PR5（真实 MiniMind 图内循环）未完成，见 §7.2。本文的 §1–§6 保留完整设计与分阶段任务；§7 记录实际落地状态。核对基线：源码审查基于 `origin/dev@cc09d06`。相关现状见 [M10 结构化控制流](M10_STRUCTURED_CONTROL.md)；目标阶梯见 [项目目标](../PROJECT_GOAL.md) §2.2，架构分层见 [架构总览](../ARCHITECTURE.md) §3。

## 1. 问题定义

仓库现在有**两条发布与执行链**：

```text
Compiler::Compile / CompileBounded
        └─ CompilePrimitiveUnits → AssembleCompiledGraph
                └─ CompiledGraph { CompiledModule, ExecutablePlan }
                        └─ RuntimeSession            ← 主链（唯一权威）

Compiler::CompileControlFlowExact
        └─ CompilePrimitiveUnits → AssemblePrimitiveModule
                └─ CompiledControlFlowGraph { ControlExecutionPlan }
                        └─ ControlRuntimeSession      ← 第二套产物与执行权威
```

第二条链功能完整、默认关闭（`KXC_ENABLE_CONTROL_RUNTIME`）、gate-on 下已有真实 LLVM 数值证据，但它：

- 有自己发布的 plan 类型 `ControlExecutionPlan` 和产物类型 `CompiledControlFlowGraph`；
- 有自己分配输出、保有 kernel 绑定的执行会话 `ControlRuntimeSession`；
- 与 `ExecutablePlan` / `RuntimeSession` 零耦合，核心没有生产调用者；
- 明确拒绝 runtime extent、持久 state、CUDA、非默认流。

本次改造的目标是**取消第二条链**：让结构化控制流成为唯一 `ExecutablePlan` / `RuntimeSession` 上的一个可选拓扑，复用同一原语编译、模块组装、状态提交与完成管理。改造后只保留一条发布与执行链：

```text
Compiler::Compile / Compiler::CompileBounded
        └─ 共同的准备、原语编译、模块组装
                └─ CompiledGraph { CompiledModule, ExecutablePlan }
                        └─ RuntimeSession
                             ├─ 线性 calls 路径（现状）
                             └─ structured_schedule 路径（新增，可选）
```

**这不是重写动态 shape，也不是给 `ControlRuntimeSession` 继续加功能。**

### 1.1 范围校正（开工前必须统一的口径）

源码审查澄清了三点，避免把改造范围定错：

1. **bounded 与持久状态已经在主链路组合**。`ExecutablePlanMode::kBoundedStatefulExternalV1`（`include/kxc/runtime/executable_plan.h`）、`CompiledGraph::BindBoundedStateOutputs`（`src/compiler/compiler.cc`）、`RuntimeSession` 的 `states_by_value` / `state_prefixes_by_value` / `length_book` / `CopyStateRange`（`src/runtime/internal/session_node.h`、`src/runtime/session.cc`）都是现成的，并有真实 MiniMind 的 prefill 交接、四步 bounded decode 与 16 份 KV 地址不变证据（[bounded 状态报告](M2_BOUNDED_STATE_REPORT.md)）。
   因此准确的问题不是“三种能力互相排斥”，而是：**bounded + state 已在主链路成立；structured control 还在另一条执行链，无法加入这个组合。**
   → 以现有 `RuntimeSession` 的状态实现为基础接控制流，不重新设计 KV owner。

2. **控制流已共用一部分编译设施**。`production_control_flow.cc` 复用 `CompilePrimitiveUnits`（与主链相同）与 `AssemblePrimitiveModule`。但产品组装从 `AssemblePrimitiveModule` 起与主链的 `AssembleCompiledGraph`（内部走 `BuildCompiledModule`）**完全分叉**。
   → 改造重点是**统一产物组装与执行权威**，保留结构化 lowering 与 region/Phi/loop-carried 表示。

3. **exact route 与 bounded 不应为“统一”强行合并**。`ExactProfileRouteTable` 发布的已是普通 `CompiledGraph`，负责“在已准备的精确变体中选择”；bounded 负责“一份产物覆盖一个形状范围”。二者是目标文档区分的两层。
   → 本次统一的是**合同来源、产物和执行器**，不是要求所有 shape 策略只有一个类。shape 模块大重构不纳入本计划。

另有一条需保留的现状：**host greedy loop 继续作为可用路径与对照基线**，它不是架构错误。M10 当前明确采用 host loop（见 [M10 receipt](M10_CONTROL_RECEIPT.md) C2）。本次补的是**图内循环能力**，不是让既有生成路径失效。

**本轮范围锁定**：CPU/LLVM、默认流、固定 rank、有界 shape、CPU 标量 bool 谓词、固定 batch、固定容量、每轮追加 `C=1`。CUDA、非默认流、请求批处理与图内循环的组合后续逐项验收，不在本轮同时放开。

## 2. 核心设计决策

### 2.1 控制拓扑不是互斥 mode，而是 plan 上的可选方面

**不新增 `ExecutablePlanMode::kControlFlow`。** 现在 `ExecutablePlanMode`（`kStatic` / `kDynamicFreshOutputV1` / `kDynamicStatefulV1` / `kStaticStatefulExternalV1` / `kBoundedStatefulExternalV1`）承载的是分配与状态合同。再加一个互斥 control mode，只是把“两套 plan 互斥”变成“同一 plan 内不同 mode 互斥”，组合问题依旧。

改为在 `ExecutablePlan` 上增加可选的 `structured_schedule`：

```text
ExecutablePlan
├── values                     现有值与存储合同
├── calls                      静态 kernel 调用点表
├── input/output/state bindings
├── mode                       分配与状态合同（保持现有枚举）
└── structured_schedule        新增，可选的结构化执行拓扑
    └── regions
        ├── kernel(call_index)
        ├── branch(predicate, then, else, phis)
        └── loop(condition, body, carried, max_trip_count)
```

**必须固定的两个含义**：

- **`mode` 只管分配与状态合同，拓扑由 `structured_schedule` 决定，二者正交。** 每个现有 mode 都允许（但不要求）携带 `structured_schedule`。这一点写入 `ExecutablePlanMode` 注释与 `Validate()`。
- **`calls()` 是编译期调用点，不是运行时调用次数。** loop body 中的 kernel 无论运行多少次，都只有一个调用点、一个模块入口和一个 artifact pin。`CompiledGraph` 现有“module entry / call / pin 一一对应”的校验保留，不得为接控制流删除。

### 2.2 region walker 只决定执行哪个调用点

`RuntimeSession` 内的 region walker 只负责“下一步执行哪个调用点”，复用现有参数准备、模块执行、状态提交与完成管理通道。它**不得**：

- 持有第二份 KV、第二个 length book、第二套常量快照；
- 拥有独立的 kernel launch 通道或输出分配器；
- 反向依赖 profiling（观测继续经 `ExecutionObserver` 接入）。

### 2.3 未定义面的前置补齐（本计划相对口头方案的关键补强）

以下三点若不先定义，PR1/PR3/PR4 会在错误抽象上返工，故作为设计前置：

**A. region × 分区模型（PR1 就会撞上）。**
现有 `PartitionValueGraph`（`src/compiler/graph/partition.cc`，声明于 `src/compiler/internal/compilation_unit.h`）把图压平成**有序、全覆盖**的 `PrimitiveUnit` 列表（`src/compiler/internal/primitive_unit.h`），`calls()` 即该顺序；`GraphTemplate.ordered_units()` 与分区 unit 的数量/顺序被强校验（`src/compiler/shape/dynamic_shape_contract.cc:410-418`）。
一个含 `If` 的图会产生**互斥分支**：两个分支各自的 kernel 如何进入同一个 unit 列表？“有序且全覆盖”在有互斥分支时如何定义？
本计划要求在 PR1 之前先确定并记录：
- 每个 branch region / loop body region 是**一个还是多个** partition unit；
- `ordered_units()` 在结构化图下的语义（建议：unit 列表为**所有 region 的静态闭包并按固定确定性顺序排列**，执行顺序完全由 region 决定；cardinality 校验改为“unit 集合相等 + region 引用的 call_index 均存在”，不再要求线性拓扑）；
- 共享调用（同一 kernel 被多 region 引用）只产生一个调用点。

**B. 迭代存储分配规则（PR2 就需要，不能拖到 PR6）。**
编译期存储/别名规划是线性的：`ValueSpec` 的 storage 绑定与 `src/runtime/memory_plan.cc` 的 liveness 都按 `calls()` 线性顺序。一旦同一静态 value id 在不同迭代 frame 有不同绑定，必须定义 loop body 内每个值的存储策略。
本计划要求 PR2 明确产出（首版可以保守）：
- loop body 内值**每轮独立 buffer，不做跨迭代复用**的最小规则；
- carried 值跨迭代存活，不得与当轮临时值别名；
- region/iteration 维度如何映射到 storage id，或明确首版允许每轮新分配并记录该选择的代价。

**C. 状态更新 region 的 ABI 与版本（PR4 随合同一起版本化）。**
PR4 给状态绑定增加“所属状态更新 region”。该字段**必须在 PR4 随合同一起进 identity/version**，否则会出现“行为变了 identity 没变”的错误 cache reuse（违反 [能力挂载总则](../../.agents/skills/kxc-capability-mounting/SKILL.md) 的完成定义）。需要明确：是扩展 `StateOutputBinding` 还是 plan 级映射；`experimental_identity.cc` 中哪些字节追加。

## 3. 分阶段任务（六个 PR）

每个 PR 必须有真实消费者与可执行验收，不以“新增 schema”本身为交付。

### PR 1：让普通 `Compiler::Compile → RuntimeSession` 跑通静态 `If`

**目标**：第一步就产生可运行的统一链路，不是先合一个无人使用的新 schema。

**主要修改位置**

| 文件 | 修改内容 |
|---|---|
| `include/kxc/runtime/executable_plan.h`、`src/runtime/executable_plan.cc` | 增加可选 `structured_schedule`；kernel task 引用现有 `call_index`；把公共合同校验与线性/结构化拓扑校验拆开 |
| `src/compiler/control_flow/control_plan.*`、`relay_control_plan.cc` | 保留编译侧 region/Phi 表示，复用现有 lowering |
| `src/compiler/control_flow/production_control_flow.cc` | 将控制流编译结果组装为普通 `CompiledGraph`（改走 `AssembleCompiledGraph` 路径） |
| `src/compiler/compiler.cc`、`include/kxc/compiler/compiler.h` | 普通编译入口在门禁允许时接收控制流，复用准备结果与共同组装流程 |
| `src/runtime/session.cc` | 增加私有 region 执行分派，调用现有参数准备与模块执行机制 |
| `src/compiler/identity/experimental_identity.cc` | 第一版起纳入结构化 plan 的兼容性编码 |
| `src/compiler/graph/partition.cc`、`src/compiler/internal/compilation_unit.h` | 按 §2.3-A 实现 region 感知分区与 unit 校验 |

**必须遵守**

- 同一准备流程内**明确选择**允许的控制能力（现有 `ControlFlowPolicy::StaticOnly()` 硬编码于 `compiler.cc`；`NativeExact()` 用于控制流），根据准备结果选择 lowering；**不要“先静态编译、抛异常后再试控制流”，也不要递归调用公开 `Compiler::Compile` 重做整套准备**。
- 控制 task 不再经 `BoundControlKernel` 持有执行入口，而是引用 `CompiledModule` 中经过正常验证的入口；artifact provenance、signature、launch metadata、pin 校验不得绕过。
- 不得把两个分支拍平成“全部执行”的 `calls()` 序列。静态调用点表可含两边 kernel，执行顺序由 region 决定。
- `ExecutablePlan::Validate()` 检查 region 可达性、值作用域和 Phi 来源，替换纯线性 producer-before-consumer 规则。
- 所有分支的模块入口与常量合同在发布/构造时验证；运行时只分配并执行选中分支。**不能因某分支本次未走就允许其缺 artifact。**

**验收**

- `Compiler::Compile` 返回 `CompiledGraph`，交给普通 `RuntimeSession`，执行 `If` 两分支与一个嵌套 `If`，取得真实 LLVM 数值结果；
- 未选中分支 kernel submit 数为零；
- 门禁关闭时在执行前明确拒绝控制流；门禁打开时不破坏原有纯数据流路径；
- 本 PR 不开放控制流与 bounded/state 的组合。

### PR 2：并入 `While`，解决迭代值的作用域与生命周期

**目标**：迁移循环执行语义，不是把旧 `ExecuteLoop` 复制进 `session.cc`。

**主要修改位置**：`src/runtime/internal/value_table.h`、`src/runtime/session.cc`、`src/runtime/control_session.cc`、plan 结构化校验、现有控制流测试；必要时新增私有 `src/runtime/internal/execution_frame.h`（只承载 region 局部绑定）。

**实质性不兼容**：主 `ValueTable::Bind/Allocate` 禁止同一 value id 重复绑定（`src/runtime/internal/value_table.h`）；旧执行器允许覆盖并把 tensor 放进 `retained_values` 保活（`src/runtime/control_session.cc`）。前者适合一次线性执行，后者可循环但不能原样成为长生成循环的内存管理方式。

**实现要求**

- 引入**每次 region 调用的局部值 frame**，由同一次 `RuntimeSession` 执行上下文管理：
  ```text
  Run execution context
  ├── 常量、外部输入、会话 state 引用
  └── loop activation
      ├── 当前 carried 值
      ├── 本轮 condition frame
      └── 本轮 body frame
  ```
- 同一静态 value id 在不同迭代 frame 可有不同绑定，但**同一 frame 内仍保持单次绑定约束**；不删除 `ValueTable` 的重复绑定检查。
- Phi 与 loop-carried forwarding 表达“选中哪个值”，不伪装成 `ValueWriteMode::kInPlace`。更新 backedge 时先收集完整新 carried tuple，再一起安装，避免 `(a,b) ← (b,a)` 被顺序覆盖破坏。
- 旧迭代临时值在相关操作完成且不再被 carried/output 引用后释放；不计入复杂跨分支存储复用，但不得因顶层 `retained_values` 让每轮全部临时值保留到循环结束。
- 按 §2.3-B 明确迭代存储分配规则。

**保留现有循环语义**：condition-before-body；条件为假即退出；条件为真但已达 `max_trip_count` 时拒绝下一轮。`max_trip_count` 是安全上限，不是静默截断条件；真实 `max_new_tokens` 应进正常循环条件。

**验收**

- 覆盖 0 次、1 次、多次迭代、恰好正常停止、超上限、嵌套控制流、carried tuple 交换；
- 循环内同一调用点多次执行，artifact 数量不随迭代增加；
- 编译结果释放后 session 仍可运行，输出与 completion 保活正确；
- 循环临时内存不因历史迭代数持续累积。

**决策门（止损点）**：PR2 完成后，`If` + `While` 已在普通 `CompiledGraph` / `RuntimeSession` 上统一（不含 shape/state）。在此确认“统一”路线在架构上成立、代价可接受，再投入 PR3/PR4。最坏情况停在 PR2，不损失已完成的统一。**后续禁止再向旧 `ControlRuntimeSession` 添加能力。**

### PR 3：让 bounded admission 与 shape 证明认识 region

**目标**：同一份带控制流的产物服务多个合法 shape，不引入第二套 shape evaluator。

> **区域 × 分区设计定稿（2026-09-11，实现前置）**
>
> 本节固定 PR3 的模型与类型改动。核查结论：`GraphTemplate::Verify()`（`src/compiler/shape/shape_specialization.cc`）强制线性“生产者先于消费者”，而互斥分支的 unit 在扁平顺序上无法表达；`BuildValueGraph`（`src/compiler/graph/value_graph.cc:166,170`）直接拒绝 `If`/`While`；`BuildTemplate`（`src/compiler/shape/shape_exact.cc:165`）用 `BuildStaticExecutablePlan` 取输出边界，对控制图失败。
>
> **1. unit 顺序与分区。** 控制 lowering（`LowerPreparedRelayToControlPlanWithSidecar`）已经产出**稠密、按 lowering 顺序**的 `PrimitiveUnit` 列表和 region 结构。PR3 以该列表作为 `ordered_units` 的规范顺序，不再为控制图重建 `ValueGraph`；`PartitionedGraph` 直接从 control plan 构造：`units = primitive_units`，`calls[i] = KernelCall(unit.symbol, unit.boundary_input_value_ids, unit.output_value_ids)`，`input/constant/output ids` 取 `plan.graph_inputs` / `plan.constant_values` / `plan.graph_outputs`。每个 unit 恰好一个调用点，与运行时 `structured_schedule.call_index` 一致。
>
> **2. GraphTemplate 的最小结构化扩展。** 给 `shape::GraphTemplate` 增加**可选** `synthesized_value_names`（由控制拓扑而非某个单元产出的值：Phi 结果、loop result/body argument）。默认空，线性行为逐字节不变。`Verify()` 仅在此基础上放宽一处：允许 shape-program output 由 synthesized 值满足，不要求 unit 生产者。**不改**任何线性规则。
>
> **3. 区域感知的形状证明。** 扩展受限形状解析器（`shape_value_resolver.cc`）遍历 `If`/`While`/`Tuple`/`TupleGetItem`/`Let`，规则：
>   - `If`：两分支逐叶必须同 kind/同 rank/同符号维表达式，结果取合并后的证明；两支各自的调用都进入证明序列（都要编译，不能只证明走到的分支）。
>   - `While`：loop var 以 initial 的叶形状绑定（循环不变量）；condition 必须是单一 data 叶（CPU 标量 bool）；body 逐叶必须与 initial 同 dims（本步不允许 backedge 改变 tensor shape）。
>   - `Let`/`Tuple`/`TupleGetItem`：结构透明，只做叶对齐。
>   - 解析器额外产出**按值/调用可查的符号维**，供 ShapeProgram 的每个命名值使用；命名沿用 `ValueName(id)`，与 control plan 的 value id 对齐。
>
> **4. 证明序列与 unit 对齐。** 解析器的遍历顺序与 control lowering 的 unit 顺序必须一致，或改为**按值 id 查表**而非按位置对齐。定稿选择后者：解析器输出 `value_id -> dims`，`PrepareBoundedCompile` 按 control plan 的 value id 组装 ShapeProgram 与逐 unit 合同，消除两条遍历顺序耦合的风险。
>
> **5. 运行时 extent。** 结构化 bounded plan 携带 `structured_schedule`；walker 对每个 kernel task 沿用线性路径的 `ModuleInvocationContract` 求值：region 内 kernel 的 runtime extent 只引用图输入轴（fresh-output 范围），不引用持久状态（PR4）。
>
> **6. 本步不放开**：持久 state、region 边界的状态更新、CUDA、请求批处理；`ValueGraph` 本身仍只服务线性静态路径，控制图走 control plan 的独立分区入口。

**主要修改位置**

| 文件 | 修改内容 |
|---|---|
| `include/kxc/compiler/shape_specialization.h` | `GraphTemplate` 增加可选 `synthesized_value_names`；`Verify()` 只放宽合成值一处 |
| `src/compiler/shape/shape_value_resolver.{h,cc}` | 遍历 region（If/While/Tuple/TupleGetItem/Let），产出按 value id 可查的符号维与结构化拓扑 |
| `src/compiler/shape/restricted_symbolic_shape.cc` | 接受控制 representative；用 value-id 查表组装 ShapeProgram |
| `src/compiler/shape/dynamic_shape_contract.{h,cc}` | `PrepareBoundedCompile` 结构化入口：从 control plan 构造 PartitionedGraph + 逐 unit 合同 + 结构化 schedule |
| `src/compiler/compiler.cc` | `CompileBounded` 在结构化时把 schedule 装进 plan |
| `src/compiler/control_flow/control_plan.*` | 拆开“结构正确”与“必须 static exact”校验 |

**不是删两个 `StaticOnly()` 就完成**：`PrepareBoundedCompile` 对 representative 与 logical boundary 都走静态准备，再构造**平面** `PartitionedGraph`，unit 数量/顺序/`GraphTemplate` 校验均建立于此（`dynamic_shape_contract.cc:357,410-418`）；公开 symbolic adapter 也明确排斥控制流。region 感知分区（上节定稿）是本 PR 的前提。

**第一版允许范围**

- `If` 两边对应结果同 rank、dtype、device，并满足同一组符号形状约束；
- 普通 tensor 的 loop-carried shape 在一次循环执行内保持不变，可在不同 Run 间取不同合法 shape；
- region 内 kernel 可消费 bounded 输入、产生 bounded 中间结果。
- **状态有效长度随迭代增长留到 PR4**；本步不承诺任意 tensor shape 可在 backedge 变化。

**证明义务（相对口头方案上调优先级的重点）**

编译侧必须证明：未选中分支同样合法、loop backedge 满足声明不变量，且**对整个符号范围成立而非 representative 样例**。representative 只用于具体化与核对，不能把“representative 上走到的分支”当成唯一需编译的分支。这是新增证明机制，建议本 PR 单独安排“证明机制设计”步骤。运行时只消费 `ModuleInvocationContract`，可求值已 lower 的表达式，但不得重读 Relay、调用编译器 `ShapeProgram` 或临时编译适配当前长度的 body。

**验收**：新增 `test/bounded_control_flow_llvm_test.cpp`，同一 compiled graph 在最小/中间/最大合法 shape 下执行两分支与循环，检查数值、输出形状、artifact/pin 不变；共享符号冲突、越界、未知 rank、不支持的 backedge 形状变化明确拒绝。入口可查的零启动拒绝；依赖执行结果的检查在对应消费/写入前完成——不把两者混成“所有错误全图零启动”。

### PR 4：将现有 bounded state 绑定扩展到循环体

**这是整次改造最关键的一步。难点不是让循环“看见 KV”，而是改变状态更新的执行边界。**

当前 stateful 路径大体在 `Run` 开头准备追加、执行平面调用、末尾复制并提交长度。`Run` 内含多轮 decode 时，下一轮必须读到上一轮追加的 KV 与新长度，不能等循环结束才交接。

**主要修改位置**：`include/kxc/runtime/executable_plan.h`、`src/runtime/executable_plan.cc`、`src/runtime/session.cc`、`src/runtime/internal/session_node.h`、bounded 请求准备与 plan identity。

**4.1 状态绑定从“整图一次”扩展为“明确 region 边界上的一次”**

- 复用现有状态/prefix/present/append 绑定关系，新增其所属的**状态更新 region**。
- 一个 decode body 的 16 个 K/V 追加绑定：在 body 成功执行并完成全部复制后，才算一次状态转换。不是每遇一个 kernel 更新，也不是每返回一层嵌套 region 自动追加。
- 首版限制：condition region 只读状态，decode body 在声明边界更新状态；不明确的条件写入、多重写入与别名先拒绝。
- **状态声明必须在循环形状证明前可见**：对结构化 bounded 请求，在现有准备阶段提供可选状态绑定声明，使循环证明知道哪些长度是受容量限制的 state extent。该声明随后生成 plan 合同；编译阶段不分配状态，也不建立第二个状态 owner。现有平面图的编译后绑定 API 可保留。
- 按 §2.3-C，本字段随合同一起版本化并进 identity。

> **§2.3-C state-extent ABI 定稿（2026-09-11，PR4 实现前置）**
>
> 问题：循环迭代时状态的**有效长度**推进，但图内 tensor 的 rank 与符号 shape 不得改变（否则违反 PR3 的循环形状不变量）。定稿如下。
>
> 1. **状态容量与有效长度是两个不同的量。** 容量是 `ValueSpec.state_capacity` / `state_extent_axis`，进 plan 合同与 identity；有效长度是运行时数据，由会话 `length_book` 持有，结构化执行期间由 walker 的临时 `working_extent` 推进。二者都不进 kernel ABI。
> 2. **kernel 永远看到同一符号 rank/shape；只有运行时 extent 变。** 状态输入在逻辑边界上把 extent 轴表示为 `-1`（PR3 已如此）。每个 kernel 的 runtime extent 由 `ModuleInvocationContract` 从**该次调用的输入轴**解析，不是编译期常量。因此一次迭代读取 `[B, P_t, F]`、下一次读取 `[B, P_{t+1}, F]` 是同一条已编译调用点，无需重编译、无隐式 shape 求值。
> 3. **提交发生在明确的 region 边界。** `StateOutputBinding` 增加 `update_region`（-1 = 线性整图末尾提交，保持旧行为）。结构化模式下每一轮 body region 正常结束后：把该 region 产生的 append source 复制到 `state[working_extent : working_extent+append_count]`，推进 `working_extent`，并用新长度重铺 prefix 视图，供下一轮读取。condition region 只读状态、不提交。
> 4. **对外提交边界不变。** 一次 `Run` 失败则不提交并 poison；成功完成后 `length_book` 一次落定最终长度。`working_extent` 只是本次执行的临时进度，不是第二套权威。
> 5. **identity 覆盖 `update_region`**：它改变行为（何时提交），必须随 plan ABI 版本进入 identity。
> 6. **本步边界**：固定容量、固定 batch、每次追加 `C=1`；backedge 不改变 tensor shape（状态增长只体现在 extent 与 prefix 长度上）；条件写入、多重写入、别名拒绝；CUDA 与请求批处理不放开。

**4.2 KV 与长度继续只归现有 session 所有**

复用 `RuntimeSessionNode::states_by_value`、`state_prefixes_by_value`、`length_book`、`state_mutex`、`state_completion`（`src/runtime/internal/session_node.h`）。这些已承担容量存储、prefix 工作区、已提交长度与失败标记，不复制成 `ControlStateStore` 或另一组 cursor。loop-carried 状态引用最终指向现有 state id，不每轮把整份 KV 当普通 tensor 重新分配传递。

**4.3 明确内部迭代长度与对外提交长度**

```text
进入 Run
    校验输入、状态一致性、容量与 profile
    取得状态执行权限
    working_extents = 当前已提交长度

每一轮
    condition 读取本轮 working_extents
    检查本轮 shape / capacity
    按有效长度准备 prefix
    执行 decode body
    等待 body 与全部 KV 追加复制完成
    推进 working_extents

整个 Run 成功
    一起发布最终 committed extents

执行开始后失败
    不返回成功结果
    停止后续执行
    session 标记 poisoned，不承诺回滚
```

`working_extents` 只是本次执行的临时进度，不是第二套持久状态权威。已有代码对 stateful 执行开始后的失败采用 poisoned、要求重建策略（`StatefulLengthBook::poisoned`，`src/runtime/internal/session_node.h`）；第一版延续它，不暗中承诺 KV 事务回滚。

**4.4 保留现有 prefix 整理**

物理缓存形如 `[B, capacity, ...]`，kernel 需要连续 `[B, P, ...]`；`B > 1` 时直接建较短 shape 的 view 不等价（batch stride 不同）。继续复用 prefix 工作区与 `CopyStateRange`（`src/runtime/session.cc`），不引入错误的“零拷贝优化”。

**4.5 先实现可证明的状态增长合同**

第一组组合限定固定 batch、固定容量、每轮追加 `C=1`：

```text
P_next = P_current + 1
所有层 K/V 有效长度一致
所有实际使用的 P、total 均在编译 profile 内
```

入口可证的最大追加量提前检查；乘加溢出拒绝。不以“可能提前 EOS”为由省掉容量与访问范围保证。

**验收**：同一 `RuntimeSession` 内执行有条件多轮循环，KV 地址不变，下一轮确实读到上一轮追加结果。覆盖零轮无追加、连续多轮、容量边界、跨层 extent 不一致、第二轮及以后复制失败后的 poisoned 行为。首版只开放经过验证的 bounded external-state 组合，不顺手放开所有 alias/donation/in-place。

### PR 5：用真实 MiniMind 验收三者组合

**目标**：证明“控制流 × bounded shape × KV”在**同一次执行**里成立，而非三项分别通过。

复用 `test/minimind_bounded_decode_llvm_test.cpp` 的模型、prefill 初始化、参考输出与状态检查，新增 `test/minimind_control_decode_llvm_test.cpp`：

```text
真实 bounded prefill
    ↓ InitializeState 显式初始化
一个持有 KV 的 decode RuntimeSession
    ↓
图内 While
    ├── 真实 bounded decode body
    ├── token 选择与停止条件
    ├── KV 追加
    └── 下一轮
```

prefill 可仍是另一份显式编译产物；**统一 owner 不意味着把 prefill 与 decode 强合成一个 kernel 或一个模型图**（现有报告本就是两份计划通过初始化复制交接）。

**两个易形成“假完成”的点**：

- **不能在 loop task 里放调用 host/Python 的任意 callback。** 若当前 argmax、EOS 比较尚不能经编译链表达，把缺失算子列成**独立前置切片**补齐并明确 tie-breaking 等语义。**注意这不是“小切片”**：词表 6400 的 greedy argmax 是完整带索引归约（现有 41 个 Relay 算子中无 argmax），EOS 是标量 bool 比较，且 argmax 结果要回喂 body 输入，构成 loop-carried 数据依赖——这正是 backedge 设计的核心。建议在 PR5 之前单独确认 argmax 的 lowering/TE 支持与 tie-breaking。固定 token 序列可先验证循环与状态，不能代替最终 greedy 反馈验收。
- **不强行把 ONNX 控制流导入作为前置**。可先用真实导入的 decode 计算体，在编译期构造 Relay 控制包装；报告必须写明这是“编译器构造的控制包装 + 真实模型计算体”，不得声称导入了带 `While` 的原始 ONNX 模型。

**验收**：数值与现有显式 host greedy loop 对照，检查每轮 logits、token、16 个 KV 前缀与最终长度，覆盖提前停止与不同初始 `P`；明确记录模型 fixture 已加载与真实 kernel 执行证据（现有 bounded decode 测试未配置目录时只跑无下载部分，**不得把 fixture 缺失当成模型通过**）。

### PR 6：清退第二套执行权威，补齐 identity、观测与集成边界

**目标**：不是新链路跑通就结束，而是不留两套长期竞争的实现。

**清理旧路径**（所有控制流消费者迁移后删除或退役）：

- `CompiledControlFlowGraph`（`include/kxc/compiler/compiler.h`）；
- 独立发布的 `ControlExecutionPlan` 类型（`include/kxc/runtime/control_execution_plan.h`）；
- `ControlRuntimeSession` 的独立执行实现（`include/kxc/runtime/control_session.h`、`src/runtime/control_session.cc`）；
- `BoundControlKernel` 的独立执行绑定与快照路径（`src/runtime/control_execution_plan.cc`）。

编译侧 `ControlPlan` 与 runtime 的 region/Phi/loop 数据结构可保留；**要移除的是第二个完整的产物与执行所有者，不是删除全部控制流类型**。默认按仓库内调用者直接迁移；确有外部兼容需求时，旧入口只能是普通 `CompiledGraph` / `RuntimeSession` 的薄封装，不得再分配状态、保有独立常量副本或自行提交 kernel。

**identity（从 PR1 起纳入，本步完整审计）**：结构化拓扑、Phi/backedge 连接、循环上限、状态更新边界；有序 extent ABI、容量、prefix/present 配对与写入合同；对应的 graph semantics、plan compatibility 与 variant identity。实际谓词值、运行时 cursor、实际迭代次数不进编译身份；局部 value/region id 纯重编号不造成语义不同（现有 identity 已区分 cursor 与 capacity：`src/compiler/identity/experimental_identity.cc`）。

**审计所有把 `plan.calls()` 当执行顺序的消费者**：module/plan 校验、内存规划（`src/runtime/memory_plan.cc` 按 `calls()` 线性 liveness）、profiling、热替换与执行桥接。结构化计划不能误用线性存储存活区间，也不能把实际执行三十次 kernel 记成十个调用点。

**观测**：区分「静态调用点：`call_index`/`task_id`」与「动态执行实例：本次序号/region/iteration」。继续经现有 `ExecutionObserver`，不让 runtime 反向依赖 profiling，不把旧控制流的字符串 events 发展成另一套观测系统。

**热替换**：先只允许在完整 Run 之间发生——整个循环固定使用进入 Run 时验证的 module/代际，不在每次迭代重新选择。结构化计划尚未支持的请求批处理、分布式桥接明确拒绝，不因返回类型统一就自动宣称支持。

**文档与测试收口**：更新 [架构总览](../ARCHITECTURE.md)、[M10_STRUCTURED_CONTROL](M10_STRUCTURED_CONTROL.md)、相关 receipt、能力矩阵与 CMake 测试注册；旧控制流测试迁移到主入口，不再以“旧 session 测试通过”为最终验收。exact route 整理另开小任务（明确与 adaptive route 的身份转换边界，复用既有 evaluator/identity），**不作为控制流合并的阻塞项，也不新增第三套路由器**。

## 4. 合并门禁

最终验收表如下（**拟定门禁，非本次已跑结果**）：

| 门禁 | 必须证明的内容 |
|---|---|
| 主链路回归 | 现有静态、bounded、stateful 与 bounded MiniMind-V 路径不因控制流接入退化 |
| 编译入口统一 | 普通 `Compile` 能发布静态控制图；`CompileBounded` 能发布声明范围内的 bounded 控制图 |
| 执行权威统一 | 控制流无独立 allocator、KV owner、length book 或 kernel launch 通道 |
| 控制语义 | 双分支、嵌套、零轮、carried 同时更新、正常停止与超上限拒绝正确 |
| 三者真实组合 | 真实 decode 在图内循环中使用变化的 `P` 和同一份持久 KV，而非三项分别通过 |
| 失败语义 | 可前置错误零启动且状态可继续；执行后错误停止提交、poison，不伪装成功或回滚 |
| 生命周期 | module、常量、状态与临时值保活正确；历史迭代不全量滞留 |
| 无隐式编译 | Run、分支选择、循环迭代与形状变化都不触发 primitive 编译或未声明的变体选择 |
| 证据完整 | 真实模型 fixture、输入、每轮结果、调用实例、状态长度与失败位置均可追溯 |

构建至少区分四种配置：普通 CPU/LLVM、bounded CPU/LLVM、control CPU/LLVM、**bounded + control 同开**。已有 CUDA 路径做回归，但新的 CUDA 控制流支持另立验收，不混写。

每轮改动按 [能力挂载总则](../../.agents/skills/kxc-capability-mounting/SKILL.md) 完成“声明 → 校验 → 真实 consumer → identity/version → 正例 → 负例”，并跑通公共检查：

```bash
python3 tools/architecture/check_docs.py --root .
python3 tools/architecture/check_include_layers.py --root .
python3 tools/architecture/check_public_headers.py --root . --compile
python3 python/tools/check_relay_op_contract.py --root .
python3 python/tools/check_pass_contract.py --root .
git diff --check
```

## 5. 执行顺序与风险

推荐顺序即 PR 编号：

**统一静态 `If` → 统一 `While` 与值作用域 → region-aware bounded 证明 → region 边界的状态更新 → 真实 MiniMind 图内循环 → 清退旧执行权威。**

最先动手的不是新建 `ControlState`，也不是重排 shape 文件，而是：

> **让现有静态 `If` 经普通 `Compiler::Compile` 返回普通 `CompiledGraph`，再由普通 `RuntimeSession` 用同一模块调用路径执行。**

该纵向切片先证明“统一不是换个外壳”。

**已识别的风险与对应前置**：

| 风险 | 前置 |
|---|---|
| region 如何进入分区模型（PR1 即撞上） | §2.3-A，PR1 前定稿 |
| 迭代存储分配在 PR2 缺位 | §2.3-B，PR2 产出 |
| 状态更新 region 未随合同版本化 | §2.3-C，PR4 随 ABI 版本化 |
| PR3 证明义务被低估 | 单独安排证明机制设计步骤 |
| PR5 argmax/EOS 被误当“小切片” | 独立前置切片，明确 tie-breaking |
| shape 在证明层不可行 | PR2 后决策门止损，保留 host loop 基线 |

## 6. 非目标

- 不重写动态 shape，不合并 bounded 与 exact route；
- 不新增互斥 `kControlFlow` mode；
- 不为统一把 prefill 与 decode 合成单图；
- 不放开 CUDA、非默认流、请求批处理与图内循环的组合（后续逐项验收）；
- 不改变 host greedy loop 作为可用路径与对照基线；
- 不引入第二套 state owner、shape evaluator 或观测系统。

## 7. 实施状态（2026-09-10）

### 7.1 已完成并通过验证

**PR1 + PR2 + plan identity。** 分支 `feat/m10-c3-unified-control`。

| 交付 | 内容 |
|---|---|
| `ExecutablePlan` 可选 `structured_schedule` | 正交方面；`mode` 仍管分配/状态，`calls()` 仍是每调用点一个模块入口/pin；kernel task 以 `call_index` 引用调用点 |
| 结构化计划校验 | `ValidateStructuredSchedule`：region 可达、task/Phi 作用域、每个调用点恰好被执行一次；仅在存在 schedule 时放宽线性 producer-before-consumer |
| `ValueTable` frame 栈 | 同一静态 value id 可在不同迭代 frame 重绑定，frame 内仍单次绑定；storage 全 run 保活 |
| `RuntimeSession` region walker | 复用 `PrepareCallArguments` 与 `InvokeOrderedModuleEntry`；无第二 allocator / launch 通道 / 状态 owner；backedge 先收集全部 carried 再安装 |
| `Compiler::Compile` 显式选择策略 | 用 `ProfileRelayControlCapabilities` 廉价探测残余控制拓扑，再决定 `StaticOnly`/`NativeExact`；先在准备阶段失败，不“先静态编译再重试” |
| plan ABI identity | 结构化拓扑（region 形状、Phi/backedge 接线、循环上限、调用点选择）进入 `BuildPlanAbiFingerprint`；region/task id 为本地定位符按首现重编号，实际谓词值与迭代次数仍是运行时数据 |
| PR6 清退第二执行权威 | 删除 `CompiledControlFlowGraph`、`Compiler::CompileControlFlowExact`、`ControlExecutionPlan`、`ControlRuntimeSession`、`BoundControlKernel` 与其私有 access/spec、`production_control_flow.cc` 及两个旧控制流测试；include-layer 检查器移除专用数据面例外 |

验证命令与结果（gate-on `out/build/adaptive-bounded-llvm`）：

- `m10_unified_control_llvm_test`：嵌套 `If`（三条路径各一次、每 run 恰好一次 kernel submit）、`While` 0/1/3 次迭代（body 每迭代一次提交）、carried tuple 交换、超 `max_trip_count` 经普通 session 失败、与独立 reference 一致、拓扑 identity 区分；
- 全量 CTest **76/76**（删除两个旧控制流测试后）；Python **362/362**；Relay/Pass/include-layer/文档检查通过；
- gate-off 构建（`out/build/bounded-cuda`，`KXC_ENABLE_CONTROL_RUNTIME=OFF`）编译通过，`Compiler::Compile` 仍按静态策略拒绝残余控制，`m10_unified_control_llvm_test` 正确输出 SKIP；
- `grep` 全仓库已无 `ControlExecutionPlan`/`ControlRuntimeSession`/`CompiledControlFlowGraph`/`BoundControlKernel`/`CompileControlFlowExact` 的残留引用。

### 7.2 未完成：PR4–PR5 与具体阻塞点

**PR3（region-aware bounded admission）已完成。** 采用 §PR3 设计定稿的“控制计划直接分区”路线，不改造 `ValueGraph`：

- 新 `PrepareStructuredBoundedCompile`（`src/compiler/shape/structured_bounded_compile.cc`）在控制能力许可下准备代表图，用受限形状解析器产出符号维（按单位顺序与 `unit_output_dimensions` 对齐），再对解析器重写后的程序调用控制 lowerer 得到稠密 unit 与 region，最后组装 `ShapeProgram`、`GraphTemplate`（拓扑产生的 Phi/loop 值经 `synthesized_value_names` 标记）、逐 unit `DynamicUnitShapeContract`、图输入 guard 与 `StructuredSchedule`。
- 受限形状解析器新增 `If`/`While`/`Tuple`/`TupleGetItem`/`Let` 的结构化遍历：`If` 两臂逐叶同 kind/同 dims，`While` 循环变量绑定 initial 叶形状且 backedge 必须保持形状不变量（本步不允许 backedge 改形状）；`Resolution` 暴露按 rewritten 节点键控的叶维证明。
- `shape::GraphTemplate` 增加可选 `synthesized_value_names`，`Verify()` 仅对合成值放宽“每输出有 unit 生产者”一处；线性模板字节不变。
- 运行时：`ExecutablePlanMode::kDynamicFreshOutputV1` 现可携带 `structured_schedule`；`RuntimeSession` 结构化路径在动态模式下对 region kernel 走 `InvokeDynamicCall`，并使拓扑产生的值在计划校验中可用。
- `Compiler::CompileBoundedStructured` 发布普通 `CompiledGraph`；`test/bounded_control_flow_llvm_test.cpp` 验证同一 artifact 服务 N=1/3/5/8 的 bounded `If` 与 bounded `While`，以及越界形状在 launch 前拒绝。gate-on CTest 77/77、Python 362/362、检查器通过。

**PR4（region 边界状态更新）已完成。** 按上方 §2.3-C 定稿实现：

- `StateOutputBinding` 增加 `update_region`（-1 = 线性末尾提交）；结构化 bounded 计划要求它指向 schedule 中的真实 region，并随 plan ABI identity 版本化。
- `RuntimeSession` 的结构化 walker 在 `body` region 正常结束后提交该 region 的 append：把 append source 复制到 `state[working_extent : working_extent+C]`，推进 `working_extent`，重铺 prefix 视图；动态 kernel 调用用会话持有的 live prefix 替换状态输入，使下一轮读到新长度。condition region 只读、不提交；`Run` 失败不提交并 poison。
- `BindBoundedStateOutputs` 在结构化计划下允许 append source 是拓扑产生的 loop 值（无单一 kernel 产生者）。
- 支撑改动：`CloneRelaySnapshot` 支持 `While` 深拷贝；控制 lowering 增加有界入口（接受 `-1` 轴）；`ControlPlan::ValidateBounded` 与 `LeafTypes` 在 admission 下接受 fixed-rank 通配轴；`FlattenLogicalTensorTypes` 的 admission 重载被结构化路径使用。

验证：`test/bounded_control_flow_llvm_test.cpp` 的 `bounded_state_append_at_loop_region` 证明一次循环迭代恰好提交一行（extent 1→2），且追加行落在推进后的 extent；gate-on CTest 77/77、Python 362/362、全部检查器通过。

**PR5（真实 MiniMind 图内循环）未完成**，依赖 PR4（已完成）；其独立前置（argmax/EOS 编译链表达）尚未开始。

**PR6（清退第二执行权威）已完成。** `CompileControlFlowExact` 的消费者是既有控制流测试；随 PR1/PR2 已把这些测试迁移到普通 `Compiler::Compile`/`RuntimeSession`，旧入口与其私有类型、两个旧测试均已删除。

### 7.3 结论

唯一执行权威的统一**已完成**：`Compiler::Compile` 现在把带 `If`/有界 `While` 的图发布成普通 `CompiledGraph`，由普通 `RuntimeSession` 执行，无第二 allocator/launch/state owner，拓扑进入 identity，且第二套产物与执行权威（PR6）已删除。计划 §3 的决策门（PR2 后确认统一路线成立）已通过。

控制流与 bounded shape / 持久状态的三者组合（PR3 + PR4）已完成：同一份带控制流的产物服务多个合法 shape，并在循环 region 边界推进会话状态的有效长度。

PR5（真实 MiniMind 图内循环）未完成：需要把真实 decode 计算体接入图内 `While`，并补齐 token 选择/停止条件的编译链表达。
