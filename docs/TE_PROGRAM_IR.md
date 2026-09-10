# TE Program 设计与实现边界

> **当前状态（2026-09-09）：M8 静态 v1 首切片已实现并通过三套完整门禁。**
>
> 不可变 Program、静态单 Call 生产接入、完整候选缓存身份，以及 CPU:0 优化等级 3 的 `sqrt(add(x,y))` 融合已接通。方法与实际 LLVM/Profile Bundle 证据见 [M8 技术报告](implementation/M8_TE_PROGRAM_REPORT.md)。Program v1 不承载动态 extent、KV state 或自动候选选择；现有 bounded/stateful 路径继续使用其显式合同。
>
> 下文保留原始设计约束与验收计划；其中“拟议”“当前缺口”等描述属于提案时的历史背景。已实现范围与最新证据以技术报告、[架构总览](ARCHITECTURE.md)、代码和测试为准。新 agent IR 的设计仍按项目目标延后。

## 目标与非目标

目标是在不改变 Relay、Target、Pass、缓存、执行计划、运行时会话和形状控制面既有所有权的前提下，增加完整的 `te::Program`：它是算法—硬件候选的可验证表示，并以确定性规则 Lower 到 TIR。第一验收切片是静态精确的 `sqrt(add(x, y))`。

本文不提议：

- 把 Relay 降格为实现细节，或由 TE 重新定义算子数学语义；
- 新建独立的 Math、Dependence 或 Fusion registry；
- 改变 `Target`、`PassContext`、原语缓存、`ExecutablePlan`、`RuntimeSession` 或 `ShapeProgram` 的 owner；
- 隐式编译、运行时调度选择、自动调优、通用动态形状，或放宽当前 CUDA 子集。

## 语义、事实与候选

三层表示各自只回答一个问题：

| 层 | 权威内容 | 不承担 |
|---|---|---|
| Relay | 数学语义：算子、属性、类型、值边界及可观察语义 | 循环结构、内存层次和硬件映射选择 |
| TE Compute DAG | 依赖事实：张量、读写关系、计算定义、形状与数据类型 | 把一个候选实现误写成唯一语义 |
| `te::Program`（拟议） | 一份完整、可验证的算法—硬件候选：DAG 引用、阶段、循环域/顺序、存储与并行映射、调度决定和 ABI 边界 | 重新推导 Relay 语义或成为第二个图级执行计划 |

Relay 是数学语义层；TE Compute DAG 显式暴露某个候选中的 producer、consumer、索引与归约依赖。不同算法候选可以拥有不同 DAG，因此 DAG 不是独立于候选的第二份数学真相；每个 Program 都必须重新接受 Relay 输出类型、边界语义和数值契约校验。`te::Program` 必须同时引用或内含足以复核的 DAG 与调度选择，不能只保存零散 schedule mutation。这样，候选内的依赖事实可以被规范化、比较、验证和缓存，而 Relay 继续定义外部可观察答案。

不增加平行 registry：算子语义继续来自 Relay 算子契约与已解析的 Call，候选依赖从 TE Compute DAG 推导，跨 Call 的归属继续由 `PrimitiveUnit` 与 partition 决定。`te::Program` 只是这些既有权威在一个编译单元内的候选载体。

## 当前实现缺口

本提案不是对现有类型的重新命名，当前代码尚有以下明确缺口：

- `src/compiler/graph/partition.cc` 仍强制一个普通 Relay Call 对应一个 `PrimitiveUnit` 和一个 `KernelCall`；
- `src/compiler/lowering/lowered_graph.cc` 的 `LowerPrimitiveUnit` 只调用一个已解析 Call 的 TE callback，不能直接形成跨 Call DAG；
- 当前 TE→TIR 入口消费输出 Tensor 与独立 `Schedule`，尚无完整、不可变且可 canonicalize 的 Program 候选；
- `src/codegen/llvm/codegen_llvm_stmt.cc` 的 `GenFor` 尚未消费 `ForType::Parallel`、`Vectorized` 或 `Unrolled` 的性能含义，`GenAllocate` 当前主要发射 `malloc/free`；因此 TIR 标注不能被报告为已实现的 CPU SIMD、并行或寄存器放置；
- `tir::BindCudaThreads` v4 已支持静态多维独占输出、线程内串行归约、可证明的多阶段私有存储及有 guard 的单阶段只读 Gather；仍拒绝未证明的间接 Load、跨线程/未初始化读取及超限临时存储，见 [归一化报告](implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md) 与 [Gather/Pow 报告](implementation/GPU_GATHER_POW_REPORT.md)；
- 通用 `RuntimeSession` 的 kernel、分配和拷贝路径尚未形成完整 profiling/correlation 闭环。

这些限制决定首切片只能声称减少 runtime kernel 边界，不能声称已消除所有内部物化、已实现 CPU 向量/并行或已支持 CUDA fusion。

## 拟议的 `te::Program` 契约

每个 `te::Program` 必须是完整且不可变的候选，并以与构造顺序、对象地址和临时名称无关的规范形式表达：

1. **边界**：有序 ABI 输入、常量、输出及其静态 shape、dtype、设备和别名/写入限制；
2. **依赖**：拓扑有序的 TE Compute DAG，包含每个 tensor 的 producer、consumer、索引与归约依赖；
3. **算法结构**：每个 stage 的迭代域、循环顺序、归约轴、计算位置和中间值可见性；
4. **硬件候选**：已选择的 split/reorder/vectorize/unroll/parallel 及其参数，以及已验证的存储空间、并行层级和启动约束；
5. **证明输入**：目标能力快照、适用的规范化 TIR pipeline、需要满足的不变量和该候选的版本化 schema；
6. **可序列化身份**：完整 canonical bytes 与 digest。摘要只用于索引，完整规范内容决定相等性。

候选只能使用已实现并经目标验证的调度含义。当前可用的 TE 调度原语仍仅为 `split`、`reorder`、`vectorize`、`unroll`、`parallel`；`fuse`、`tile`、`bind`、`thread_axis`、`compute_at`、tensorization 和 autotuning 不因本提案而成为已实现能力。CUDA 线程绑定仍只能由现有 `tir::BindCudaThreads` 在其静态独占输出及线程内归约子集内证明与生成启动元数据。

## 确定性 `te::Program` → TIR

Lowering 输入是已经验证、已 canonicalize 的 `te::Program` 和不可变 `Target`，输出为一个确定的 `tir::PrimFunc`（或该单元明确声明的有序 `PrimFunc` 集）。同一 canonical program、目标能力快照和规范 pipeline 必须产生相同的 TIR 语义与同一身份输入；实现不得依赖哈希表遍历、对象地址、线程时序、缓存命中或运行时负载。

Lowering 至少必须逐项验证：边界和输出契约、producer/consumer 拓扑、循环域与静态 extent、索引合法性、归约写入、调度前提、目标能力、符号唯一性及 ABI 一致性。验证失败、无法 canonicalize、无法证明映射或 TIR pipeline 破坏声明不变量时，必须 fail-closed：拒绝该候选，不退回到另一种未声明实现，也不在运行时重新编译。

现有默认 TE 调度和 TE→TIR 路径在实现前仍是当前事实；它们不是 `te::Program` 已存在或可用的证据。

## 跨 Relay Call fusion 的最小接入

跨 Relay Call fusion 是后续 `te::Program` 的图内候选范围，而非新图系统。只扩展以下既有边界：

- **`PrimitiveUnit`**：从“恰有一个 root Relay Call”的当前契约，扩展为一个有序、连通、可验证的 Relay Call region；保留冻结的输入、输出、常量、设备、副作用、别名和语义键边界。
- **partition**：在准备后的 `ValueGraph` 上决定哪些相邻、同设备且满足纯度、类型、别名和依赖前提的 Call 形成一个 region；不能融合的 Call 保持独立单元。
- **`lowered_graph`**：为一个 region 建立共享的 TE Compute DAG，构造一个完整 `te::Program`，并按既定确定性规则 Lower；`ExecutablePlan::KernelCall` 仍按该单元的冻结边界发射。

这不要求独立 fusion registry，也不把 `ExecutablePlan` 变为融合决策 owner。融合是否合法由已有 Relay/值图事实和 `PrimitiveUnit` 边界验证；候选如何实现由 `te::Program` 验证。任何副作用、别名、设备不一致、非静态契约、未支持控制流或无法证明的依赖都阻止融合。

## 所有权与身份

以下 owner 不变：

| 组件 | 保持的职责 |
|---|---|
| `Target` | 后端选择和不可变能力快照的唯一权威 |
| `PassContext`/规范流水线 | 单次调用上下文、Pass 解析、顺序与不变量契约 |
| 原语缓存 | 请求合并、失败传播、背压、淘汰、发布和 `ArtifactPin` 生命周期 |
| `ExecutablePlan` | 与运行时实现无关的值/存储/调用顺序契约 |
| `RuntimeSession` | 计划与模块边界校验、分配绑定和执行；不编译、不选择候选 |
| `ShapeProgram` 与精确 profile 路由 | 符号形状决策和有限已发布精确产物的路由；不拥有 TE 候选 |

`te::Program` 的 canonical contract 必须进入 artifact identity，不能只放在调试元数据中。拟议的 `PrimitiveArtifactKey` 输入为：单元语义、目标能力指纹、规范 pipeline、ABI 版本、**canonical TE program contract**、后端版本。现有 schedule contract 不能同时以另一个独立且可能漂移的权威表达同一候选；迁移时应由 canonical program contract 生成或取代该身份字段，并提升身份 schema/ABI 以隔离旧缓存项。

`UnitSemanticKey` 仍表达与目标和实现策略无关的单元语义；`te::Program` contract 则表达目标相关的候选。图编号、存储 ID、对象地址、链接符号、请求热度和可变缓存状态都不得进入等价性。

## 第一切片

第一切片只覆盖静态精确、同设备、纯逐元素的 `sqrt(add(x, y))`：两个 Relay Call 构成一个候选 region，`add` 的结果只在 region 内消费，外部 ABI 为 `x`、`y` 与 `sqrt` 输出。它必须证明：

- Relay 仍是 `sqrt(add(x, y))` 的唯一数学语义来源；
- TE Compute DAG 显式保留 `add → sqrt` 依赖；
- `te::Program` 完整记录该 region 的单一候选和边界；
- 同一输入、Target 和 pipeline 得到确定性 TIR 与稳定 artifact identity；
- 融合前后在受支持 dtype/静态 shape 上数值等价，且输出 ABI 不变。

它不证明广播、归约、多输出、常量折叠、别名写入、控制流、动态形状、任意多 Call 融合、CUDA 并行化或自动候选搜索。

## 证明等级、fail-closed 与测试门禁

证据按强度分级；高等级不应跳过低等级：

| 等级 | 必须证明 | 最低测试门禁 |
|---|---|---|
| P0：构造 | Relay region、TE DAG、Program 边界和 canonical bytes 完整且稳定 | 单元测试：非法/不完整 Program、非规范顺序、边界漂移均拒绝 |
| P1：Lowering | Program 到 TIR 的结构、符号、ABI、循环/依赖和声明的 pipeline 不变量成立 | TIR 结构断言与同输入重复 lowering 的字节/身份稳定性测试 |
| P2：后端 | 经过目标验证的 TIR 可由该后端构建为与签名匹配的内核 | LLVM 可用时的真实 LLVM 编译和数值测试；CUDA 仅在 Toolkit 与设备可用时作真实构建/启动测试 |
| P3：端到端 | `ExecutablePlan` 与 `RuntimeSession` 按冻结 ABI 执行并得到数值等价结果 | 首切片的 `sqrt(add(x,y))` 融合/非融合参考比较、缓存命中/未命中一致性及失败路径测试 |

所有门禁均 fail-closed。缺失能力、无效候选、目标不匹配、无 CUDA 工具链/设备或无法建立所需证明，必须明确标记为拒绝或验证缺口；不得记为该后端已验证成功。合成 CUDA 目标测试只可证明契约行为，不能替代 P2/P3 的硬件执行证据。

当前事实必须保持准确：LLVM 是主要 CPU 后端，且仅在发现 LLVM >= 20 时可用；CUDA 是实验性后端，依赖 CUDA Toolkit 与设备，只开放保守调度子集，TE 循环当前保持串行；性能分析目前覆盖编译准备、原语编译、汇总组装、Pass、形状/自适应编译以及已安装 Span 的分布式计划/通信，可选 CUPTI 可关联受支持 CUDA 活动。不得宣称通用 `RuntimeSession`、分配或拷贝路径已有完整 profiling 覆盖，也不得把合成工作台或合成 CUDA 测试称为后端生产证据。

## T1–T9 依赖与验收顺序

以下是不带日期的依赖检查点，用于保证每个切片只建立在已验证边界上，而不是当前实现状态。它们不是允许独立合并的空框架 PR：T1–T7 的最小字段必须随首个 `sqrt(add(x, y))` 纵向切片一起形成真实 production consumer；只有测试、重构或文档可以在不新增未消费 API 的前提下单独提交。

1. **T1 — Program 数据契约：** 为首切片定义最小不可变 `te::Program`、版本化 canonicalization 和 P0 验证器；这些类型必须立即被 T2/T3 的真实 Lowering 消费。
2. **T2 — DAG 捕获：** 从已验证 `PrimitiveUnit` 生成完整 TE Compute DAG 与 Program，冻结边界并验证依赖。
3. **T3 — 确定性 Lowering：** 仅以 canonical Program、Target 和规范 pipeline Lower 到 TIR，建立 P1 稳定性与结构测试。
4. **T4 — 身份与缓存：** 将 canonical TE program contract 接入 `PrimitiveArtifactKey`，完成 schema/ABI 隔离和缓存一致性测试。
5. **T5 — 单 Call 回归：** 让既有单 Call 单元经 Program 路径获得与当前受支持语义等价的结果，保留 fail-closed 行为。
6. **T6 — Region 单元：** 最小扩展 `PrimitiveUnit`、partition 和 `lowered_graph`，但仍只允许已证明的纯静态 region。
7. **T7 — 首切片：** 实现并验证 `sqrt(add(x, y))` 的跨 Call region、确定性 TIR、ABI 和 P3 数值等价。
8. **T8 — 后端证据：** 在可用 LLVM 环境完成真实 P2/P3；CUDA 只在满足既有限制、Toolkit 和设备时增加相应证据，不扩大其调度承诺。
9. **T9 — 观测与回归门禁：** 为 Program 构造、Lowering、缓存和执行接入已有 profiling Span 的适当边界，并固定 P0–P3、拒绝路径和文档检查为提交门禁。

## 逐文件影响范围

下表列出实现该提案时预期会触及的边界；列出不表示文件已经修改或能力已经实现。

| 文件 | 拟议影响范围 |
|---|---|
| `include/kxc/te/` 下的新/现有 Program 头文件 | 声明 `te::Program`、canonical contract、验证与只读访问边界；不引入独立语义或 fusion registry。 |
| `src/te/` | 实现 Program 构造、canonicalization、P0 验证，以及由现有 Compute DAG/调度事实形成候选。 |
| `src/compiler/internal/primitive_unit.h` | 将单 root Call 表达扩展为有序 Call region，同时冻结原有边界契约。 |
| `src/compiler/analysis/primitive_unit.cc` | 验证 region 连通性、顺序、纯度/别名/设备前提及稳定边界。 |
| `src/compiler/graph/partition.cc` | 仅在已证明的静态纯 region 上生成融合 `PrimitiveUnit`；保留未融合回退为独立单元。 |
| `src/compiler/internal/lowered_graph.h`、`src/compiler/lowering/lowered_graph.cc` | 从 unit/region 生成共享 DAG 与 Program，并以确定性 Program→TIR 替换直接的零散 schedule 路径。 |
| `src/compiler/internal/te_to_tir.h` 及 TE→TIR 实现 | 接受并验证 Program 的规范调度/映射契约，维持现有 TIR 证明与目标限制。 |
| `src/compiler/primitive/primitive_compiler.cc` | 在既有 `CompilePrimitiveUnits` 路径内传递 Program contract、执行 P1 检查并保持缓存 owner 不变。 |
| `include/kxc/compiler/identity.h`、`src/compiler/identity/identity.cc` | 将 Program canonical contract 纳入 `PrimitiveArtifactKey`，升级身份 schema/ABI 并避免双重 schedule 权威。 |
| `src/compiler/compiler.cc` 及组装相关测试 | 维持一个 unit 对应一个就绪 artifact 和冻结 `KernelCall` 边界；验证 region 后的计划装配。 |
| `src/codegen/`、`src/tir/` | 只在确有 Program 映射验证或 P2 测试需要时调整；不借机扩展 LLVM/CUDA 支持范围。 |
| `tests/` 中 TE、compiler、identity、codegen、runtime 测试 | 覆盖 P0–P3、确定性、缓存隔离、`sqrt(add(x,y))` 数值/ABI 等价、以及所有 fail-closed 拒绝路径。 |
| `docs/ARCHITECTURE.md`、能力矩阵及本文件 | 仅随已落地的公共边界、机器契约和可复现测试更新当前状态；本提案本身不改变架构权威中的当前能力。 |
