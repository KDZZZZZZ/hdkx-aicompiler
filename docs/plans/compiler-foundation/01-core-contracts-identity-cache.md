# 01：共同基础——契约、identity 与 artifact cache

> **状态：** 阻塞（本轨可实现项完成；仅待 02–06 分支消费 CoreContract v1）
> **所属路线：** [编译器基础路线图](README.md)  
> **权威输入：** [`docs/COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md`](../../COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)  
> **前置：** 无；本轨的最小冻结接口应允许 02–06 使用 mock/fake 并行开发。

## 1. 目标

本轨建立所有后续能力共享、可审计且 fail-closed 的编译基础：

1. 在 partition 前验证“Relay IR 可表达”是否等于“当前目标可执行”。
2. 将 operator schema、pass schema 与生产 pipeline 配置收敛为单一权威来源和一个解析结果。
3. 严格分离图内连线身份、编译语义身份、artifact 身份、dispatch 身份、链接 symbol 与 storage 身份。
4. 让缓存命中携带不可变 artifact/pin；同 key 请求可 singleflight，失败、取消、预算和背压有独立状态。
5. 收敛 legacy whole-graph lowering 与文档能力声明，使生产 per-unit 主链成为唯一能力判断标准。
6. 冻结足够小的接口，使 shape、adaptive、dynamic graph、region、NLP/GPU 轨道无需等待实现即可写 mock/fake 和测试。

## 2. 非目标

- 不实现 `DimExpr`、constraint、`ShapeProgram`、bucket、polymorphic kernel 或动态 output；这些属于 [02 shape](02-shape-system-and-specialization.md)。
- 不实现后台编译策略、热点模型、版本替换或服务级调度；这些属于 [03 adaptive hot swap](03-adaptive-compilation-hot-swap.md)。
- 不实现 `If`/`Let`/函数值执行、task DAG、region fusion、具体 Transformer op 或 CUDA schedule。
- 不恢复按 `cached_dimension >= query_dimension` 选择模块的 fuzzy cache。
- 不改变 `RuntimeSession(module, plan)` 只消费 runtime 数据的单向依赖边界。
- 不在本计划中写实现代码、改变现有行为或将接口草案视为已稳定 ABI。

## 3. 现状证据与问题陈述

| 事实 | 证据 | 本轨结论 |
|---|---|---|
| `OperatorSpec` 已包含 arity、attrs、type relation、effect、alias、lowering kind/key | `include/kxc/relay/op.h` | 可以作为 schema 收敛的输入，但 JSON、注册/default 与文档仍可能多源漂移。 |
| `PassSpec` 已声明 dialect、scope、phase、invariant/analysis 与 implementation key | `include/kxc/pass/pass.h` | metadata 方向正确，但生产 policy 仍由 `Compiler::RelayPassPolicy`/`TIRPassPolicy` 等位置维护。 |
| `Compiler::Compile` 输出 module + runtime-neutral plan | `include/kxc/compiler/compiler.h` | 必须保留 runtime 不反向依赖 compiler 的边界。 |
| plan 的 `ValueSpec` 保存 value/storage id、静态 shape、dtype/device；`KernelCall` 以 symbol + value ids 连线 | `include/kxc/runtime/executable_plan.h` | value id 是 graph locator，不能升级为跨图 kernel identity。 |
| `kDynamicDimension = -1` 仅标注动态输入，动态 output 被拒绝 | `include/kxc/runtime/kernel_abi.h` | sentinel 是 legacy ABI 验证适配，不是本轨的 shape 语义。 |
| partition 的 structural hash 写入 input/output graph-local value id | `src/compiler/graph/partition.cc::BuildStructuralHash` | 必须从 unit semantic key 中移除 value id，同时保留其在 plan/诊断中的用途。 |
| primitive cache 是 256 项进程内 LRU，命中返回值副本；backend 阶段可再次 peek | `src/compiler/cache/primitive_cache.cc` | 需要 artifact pin、请求状态和 singleflight，不能把 cache miss 当作生命周期状态。 |
| session 验证 module/plan/signature 并按顺序执行，且只允许单设备 | `src/runtime/session.cc` | core contract 只能向 session 交付已冻结的 artifact/plan，不在 session 内做 cache/compile 策略。 |

权威审查还指出：whole-graph `LowerToTIR` 与生产 per-unit lowering 并存；`If`、`Let` 等可由 IR 表达但缺可执行 capability；operator/pass contract 具有多权威来源；cache hit 被压缩为 bool 后可能在后续 peek 前被淘汰。这些是本轨必须先处理的事实收敛问题。

## 4. 设计约束

- canonical identity 只包含影响数学语义、边界 tensor contract、ABI、target、pipeline、schedule/backend 的规范化内容；不含对象地址、span、value id、unit id、storage id、`global_symbol` 或 profile 文本。
- hash 只用于索引；完整 canonical bytes 或结构化等价比较才可决定命中。
- symbol 是 module 链接/寻址身份，plan call id 是图内调用身份；任一都不能单独代表 artifact 语义。
- artifact cache 只收录成功、完整、不可变且已验证的 artifact；in-flight、失败、取消和退避属于协调器状态。
- 淘汰只能移除 cache 可发现性，不能使已命中编译事务、已组装 plan 或异步 launch 失去其强引用。
- capability 缺失在入口、图 Pass 后、partition 前报告节点定位和缺失能力，不允许等待 lowering 的偶然异常。
- legacy adapter 可以保留，但新接口不得接收或传播 `-1` 作为通用 dynamic shape。

## 5. 最小冻结接口（供其他轨道 mock/fake）

本节定义的是语义合同和数据字段，不是 C++ 实现或公开 ABI 承诺。M1 审查后才允许变为头文件/API；在此之前，跨轨测试应以同名 fake 数据对象对接。

### 5.1 Capability verifier

| 项 | 草案 |
|---|---|
| 输入 | `CapabilityRequest { graph_or_unit_locator, relay_node_kind, OperatorSpec, checked_type, target_capabilities, pipeline_fingerprint, requested_mode }` |
| 输出 | `CapabilityResult { supported, missing_capabilities[], diagnostic_locator, normalized_requirements }` |
| 规则 | 不支持节点、未绑定 symbolic shape、缺 lowering、target/dtype/layout 不支持均 `supported=false`；不猜测 fallback。 |
| 调用点 | Compiler 入口、图 Pass 后、partition 前；region/dynamic graph 可增加更严格的同类 verifier。 |
| fake | `FakeCapabilityVerifier` 按显式 request key 返回预设结果；不得按 op 名隐式放行。 |

`requested_mode` 至少区分当前 static exact、未来 shape specialization、control-flow/region；这样 02–05 可测试“请求被拒绝”的集成路径，而无需实现真实 backend capability。

### 5.2 Schema source 与 `PipelineResolver`

| 项 | 草案 |
|---|---|
| `ContractSource` | 唯一机器可读源，描述 operator/pass 的 schema version、规范字段、implementation binding key、支持矩阵和文档锚点。 |
| operator 产物 | `OperatorContract`：显式 arity、attrs、type/shape rule key、effect/alias、lowering/target capability；新字段缺失 fail closed。 |
| pass 产物 | `PassContract`：dialect、scope、phase、required/produced invariants、analysis preserve/invalidate、target 条件与 implementation key。 |
| 输入 | `PipelineRequest { target, opt_level, named_pipeline, enabled[], disabled[], requested_scope }` |
| 输出 | `NormalizedPipeline { ordered_passes[], invariant_transitions[], target_requirements, fingerprint, contract_versions }` |
| 规则 | `PipelineResolver` 是生产顺序的唯一解析者；binding table 仅把 implementation key 映射到函数，不能复制 metadata。 |
| fake | `FakePipelineResolver` 返回固定 normalized pipeline 和 fingerprint，可注入 invariant/target 失败。 |

在单源生成完成前，可采用一个 source 加严格双向 checker 的过渡方案；不得继续依赖 op 名 category 推断或 default 补全关键契约。

### 5.3 Identity/key 模型

| 名称 | canonical 内容 | 消费者 | 明确排除 |
|---|---|---|---|
| `GraphValueLocator` | graph revision、value id、output index | plan routing、诊断、profile | semantic cache、backend identity |
| `UnitSemanticKey` | normalized unit IR、canonical attrs、boundary tensor 语义、effect/alias、shape-rule version | partition、region、artifact family | value id、symbol、storage id、对象地址 |
| `ArtifactKey` | unit semantic key、target capability、normalized pipeline fingerprint、ABI version、schedule/backend version | artifact cache、coordinator | graph locator、请求热度 |
| `DispatchKey` | artifact family、已绑定 shape/layout/valid-extent 条件、variant policy/version | 02/03 的选择逻辑 | primitive 语义本身 |
| `PlanVariantKey` | graph template revision、selected artifact generations、concrete profile、physical memory-plan version | plan registry、profile | mutable cache state |
| `LinkSymbol` | artifact entry 或安全重定位 alias | module lookup、launch | cache equivalence |
| `StorageId` | 单个 plan 内物理存储分配 | memory planner、lifetime verifier | value/semantic identity |

当前 `CompilationUnit::structural_hash` 的职责应迁移为 `UnitSemanticKey`；图内 `unit_id` 和 `KernelCall` value ids 继续保留，但只作为定位/连线数据。若 backend 不能重定位 entry，则由 artifact key 派生 artifact symbol，plan 单独保存 call locator；不能重新把 symbol 绑回 semantic key。

### 5.4 Artifact、pin 与 cache observer

| 项 | 草案 |
|---|---|
| `ArtifactHandle` | 不可变 `{ artifact_key, executable, signature, launch_metadata, provenance, byte_size, validation_record }` 的强引用。 |
| `ArtifactPin` | 由成功 lookup 或 compile result 返回；持有 handle，保证后续 backend/plan 阶段不会因 cache eviction 失效。 |
| `ArtifactLookup` | 输入完整 `ArtifactKey`；结果为 `Hit { pin }` 或 `Miss`，不返回裸指针、bool 或仅 hash。 |
| `ArtifactStore` | 只接受已验证 ready handle；重复 store 必须比较完整 key/contract，冲突显式失败。 |
| `CacheObserver` | 只报告 hit/miss、bytes、eviction、pin count、age；不暴露可变内部 map 元素。 |
| fake | `FakeArtifactStore` 以 canonical key 映射 immutable fake handle，并可模拟 eviction 但不得破坏已发出的 pin。 |

该接口只定义 ready artifact 的生命周期。编译请求的排队、singleflight、失败 TTL、取消和背压由下一节的协调器合同承载，避免 “not in cache” 同时表示五种不同状态。

### 5.5 Compile request 协调合同

| 项 | 草案 |
|---|---|
| `CompileRequest` | `{ artifact_key, dispatch_key?, priority, budget_class, request_origin, cancellation_token }`；shape 字段必须是 02 的显式 dispatch contract。 |
| `CompileTicket` | 同 key callers 共享的只读 future/result；记录 merged waiter count，不泄露任务所有权。 |
| `CompileOutcome` | `Ready { pin }`、`Failed { category, retry_after, diagnostic }`、`Cancelled`；unsupported 是失败而非重试信号。 |
| 状态机 | `Absent -> Queued -> Compiling -> Validating -> Ready`；任意未完成态可至 `Cancelled`，验证/编译错误至 `Failed(retry_after)`。 |
| singleflight | 相同完整 artifact+dispatch key 只建一个 ticket；hash collision 必须通过 canonical key 比较分流。 |
| 背压 | 全局、per-model、per-target/backend 并发和 bytes 预算；低价值 prewarm 可被拒绝，显式请求不得静默丢失。 |
| fake | `FakeCompileCoordinator` 用可控 promise、clock 和队列结果模拟合并、失败、取消、饱和与发布顺序。 |

03 将实现此合同的生产协调器；01 只冻结状态、输入/输出和可观测字段，确保 02/05/06 可以先构造 deterministic fake artifact 流。

### 5.6 版本与 plan 交接合同

| 项 | 草案 |
|---|---|
| `SelectedArtifact` | `{ artifact_pin, generation, applicability_proof, signature_digest }`；generation 在没有 03 时可固定为 `0`。 |
| `FrozenPlanInput` | graph/template locator、value routing、selected artifacts、logical/physical/valid-extent contract、memory-plan version。 |
| 验证 | assembler 必须比较 signature digest、target、ABI、layout/capacity/workspace；不兼容只能生成新 plan variant。 |
| session 边界 | session 仅接收已冻结结果；不得触发 lookup、compile、shape prediction、generation 选择或 policy fallback。 |
| fake | `FakePlanAssembler` 接收 fake selected artifacts，生成可检查的调用列表和引用保活记录，不执行 kernel。 |

## 6. 与其他轨道的并行协议

| 轨道 | 可立即使用的 mock/fake | 01 完成前禁止的耦合 | M1 集成检查 |
|---|---|---|---|
| 02 shape | fake resolver、fake artifact store、fixed generation 0 | 将 `-1` 写入 semantic/artifact key；直接依赖 cache 内部 map | shape binding 改变仅进入 dispatch/profile 字段 |
| 03 hot swap | fake `ArtifactHandle`、ticket state machine、fake clock | 在 session 内实现 coordinator；把 symbol 当 key | publish 前后 pin 与 generation 保活可证明 |
| 04 dynamic graph | fake capability matrix、normalized pipeline | 绕过 capability verifier 直接接受 IR 节点 | 每种 executable node 有明确 capability/error |
| 05 region | fake unit semantic key、fake plan assembler | 用 unit id/value id 作为 region cache identity | 无关重编号不改变 semantic key |
| 06 validation | fake target capability、cache observer、deterministic failures | 以成功 importer/launch 推断 capability 已支持 | 报告可区分拒绝、cache miss、compile failure |

M1 的冻结原则是“窄而可替换”：每个 mock/fake 只实现表中的输入/输出语义，不复制 Compiler、Relay 或 RuntimeSession 的私有状态。

## 7. 实施步骤

1. **建立事实基线与 capability matrix。** 从现有 operator/pass contract、production compiler path、lowering 入口和 runtime session 整理 supported/unsupported capability；把 `If`、`Let`、dynamic output、unknown symbolic dim 和不支持 target 的拒绝路径写成稳定诊断要求。
2. **定义并接入 capability verifier。** 在 compiler 入口、graph pass 后、partition 前调用同一 verifier；错误包含 graph/unit locator、节点、目标和缺失 capability。先覆盖现有 static exact 主链，不扩大支持集合。
3. **选择 schema 单源并构建迁移 checker。** 指定 `ContractSource`，使 operator/pass metadata、binding key、default pipeline 和文档锚点可生成或严格核验；移除关键字段的名称推断/default 填充依赖。
4. **实现 `PipelineResolver` 的纯解析阶段。** 用 `PipelineRequest` 生成 deterministic `NormalizedPipeline`、fingerprint 与 invariant transition；`Compiler::*PassPolicy` 和默认 pass order 改为兼容入口或删除重复事实源。
5. **拆分 identity。** 从 partition canonical serialization 中删除 graph-local value id、unit id、symbol、span 等非语义字段；新增 locator/key 的显式序列化与完整等价比较；确定 artifact symbol/重定位策略。
6. **改造 ready artifact cache。** lookup 直接产生 `ArtifactPin`，编译状态持有 pin/handle，不再以 hit bool 后二次 peek；以 bytes、引用状态、重编译成本替代纯 256-entry LRU 作为目标策略。
7. **冻结 request state 与 observer。** 接入或先以 adapter 暴露 ticket、failure、retry、queue/budget、merge、pin/evict 事件；生产 singleflight 可在 03 落地，但接口与 fake 先通过测试。
8. **收敛 legacy lowering 和文档。** whole-graph `LowerToTIR` 标记 compatibility/testing；能力矩阵、扩展指南、runtime/profiling 文档标 current/target/archived，并链接 production `Compiler::Compile` 证据。
9. **完成跨轨 M1 review。** 02–06 使用相同 fake contracts 编译其测试；确认无私有 header、object address、value id、symbol 或 sentinel 泄漏进入新公共合同。

## 8. 测试与实现证据（2026-07-23）

| 类别 | 正例 | 反例/并发例 | 断言 |
|---|---|---|---|
| capability | 当前 19-op static exact 主链 | `If`/`Let`、缺 lowering、未绑定 symbol、动态 output、不支持 target | partition 前稳定失败，含 locator 与缺失能力 |
| schema 单源 | 每个 registered operator/pass 有完整合同与 binding | schema 字段遗漏、重复名称、binding 不存在、默认顺序漂移 | checker fail closed，resolver 不产生 pipeline |
| pipeline | 同 request 产生相同 order/fingerprint | scope/phase 逆序、invariant/target 不满足、enable/disable 冲突 | 无重复 policy；transition 可审计 |
| identity | 无关 value id/symbol 重编号仍等价 | attrs/dtype/layout/ABI/target/pipeline/backend 改变 | 前者同 semantic key，后者安全 miss |
| hash 安全 | 相同 hash 的不同 canonical bytes | 人工 collision 或截断 hash | 桶内完整比较，绝不误命中 |
| pin/eviction | hit 后 assemble/launch 期间保持 artifact | cache 满、并发 store/evict、已有 pin | lookup artifact 可用，eviction 不破坏 pin |
| singleflight | N 个同完整 key request | 同 hash 不同 key、取消 waiter、验证失败、retry TTL | 一次 compile；状态/失败/重试可观测 |
| backpressure | 高优先级显式请求 | 饱和、低价值 prewarm、per-target budget 耗尽 | 有结构化拒绝/排队，不产生 compile storm |
| legacy/docs | production per-unit path 与文档能力表 | whole-graph 入口被误标 production | 文档链接 feature gate；legacy 仅 compatibility/testing |
| mock contract | 02–06 的 fake resolver/store/coordinator/assembler | fake 使用私有类型或绕过 key | 仅依赖 M1 冻结字段，行为可重复 |

测试应同时覆盖 CPU/LLVM 的现有静态回归；CUDA、sanitizer、NLP 端到端属于后续轨道的集成矩阵。任何并发缓存/发布实现须加入 pin 生命周期、失败 TTL 和取消的确定性测试，不能只验证 hit/miss 计数。

## 9. Done 条件

- [x] capability verifier 在三个规定边界 fail closed，并有可定位的正反例。
- [x] operator/pass 的关键 metadata、binding、default pipeline 与文档锚点由 JSON -> generated C++ -> checker 单向链管理。
- [x] `PipelineResolver` 成为生产 pipeline 顺序与 fingerprint 的唯一解析点；兼容 policy 只委托 resolver。
- [x] `GraphValueLocator`、`UnitSemanticKey`、`ArtifactKey`、`DispatchKey`、`PlanVariantKey`、symbol、storage id 有独立 canonical 定义和测试。
- [x] 无关 graph-local 重编号或 symbol 变化不影响 unit semantic/artifact identity；所有语义/ABI/target 变化安全 miss。
- [x] ready cache lookup 返回 immutable pin/handle；淘汰只移除可发现性，已发 pin 和 executable 强引用继续有效。
- [x] production same-key singleflight、failure/retry、bounded bytes/in-flight/backpressure，以及取消/预算/observer 的 CoreContract v1 fake 均有确定性并发/合同测试。
- [x] whole-graph lowering 明确降为 compatibility/testing，生产能力声明以 per-unit `Compiler::Compile` 为准。
- [ ] 02–06 各自分支已实际消费 CoreContract v1 mock/fake，且没有绕过 runtime/compiler 单向依赖的私有耦合（跨 worktree 硬阻塞；本轨已提供 conformance fake）。
- [x] 文档状态、feature gate、测试证据一致；没有把 `-1`、fuzzy cache 或历史 adaptive runtime 宣称为当前 dynamic shape/hot swap 能力。

## 10. 风险、决策门与状态

| 风险 | 防护 | 决策门 |
|---|---|---|
| canonical key 漏掉 ABI/layout/schedule 字段导致误复用 | 结构化 canonical bytes、差分/负例、完整比较 | M1 前审查 key field matrix |
| 过早固定公共 C++ ABI 阻碍 02–05 | 先冻结语义合同与 fake，不冻结对象布局 | M1 后才提头文件/API |
| schema 迁移期间多源漂移 | source-of-truth 标记与 CI checker，旧表只作生成输入 | 每个 contract 只允许一个 writer |
| cache 改造引入全局锁/资源膨胀 | observer、bytes budget、pin 计数、明确淘汰语义 | 03 接入前进行并发/预算 review |
| legacy 路径继续被新增功能使用 | production/compatibility 标签与 capability test | 新特性不得只接 legacy lowering |
| 其他轨道等待实现而失去并行性 | fake resolver/store/coordinator/assembler 合同 | M1 review 验证各轨 mock 已可运行 |

本轨源码、单元/并发测试、generated contract 和文档收敛已完成。状态保持“阻塞”仅因为独立的 02–06 worktree 尚未逐轨提交 CoreContract v1 消费证据；本轨不修改其他 worktree。完整提交、测试和后续集成要求见 `docs/handoffs/compiler-foundation/core.md`。
