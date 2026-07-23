# 05：Region 执行计划与运行时

> **状态：** 规划中；不表示当前已实现  
> **所属路线：** [编译器基础路线图](README.md)  
> **权威输入：** [编译器基础架构审查](../../COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)  
> **范围：** 从 per-Call partition policy 演进为 region、task DAG 与 dependency-aware runtime；不实现代码。
## 1. 当前基线与演进原则
当前链路将普通 compute `Call` 分成一个 `CompilationUnit`、一个 `PrimFunc` 与一个
kernel entry。`ExecutablePlan` 以稳定 value id、`ValueSpec` 和有序 `KernelCall` 描述
runtime-neutral ABI；`RuntimeSession(module, plan)` 校验 module/plan/signature，绑定输入
常量、分配中间/输出，并在单设备、单 stream 的顺序中启动。
这是一条正确的静态执行基线。演进不应把 `Call = CompilationUnit = KernelCall` 固化为
永久模型，也不能把 Compiler、Relay、缓存策略或 shape predictor 倒灌进 `RuntimeSession`。
新的永久不变量是：每个 unit/region 都有显式边界、可验证 ABI、语义 identity、effect/alias
契约与可追踪 locator；提交给 session 的版本不可变。
## 2. 目标、非目标与术语
### 2.1 目标
1. 将 `CompilationUnit` 从当前 `PartitionPolicy::PerCall` 的产物演进为可验证 region。
2. 支持 fusion region、library region 和未来 control-flow region，同时保留 per-Call 回退。
3. 将 ordered calls 演进为可验证 task DAG，显式表达 kernel、copy、event、shape-eval、
   allocate、sync 与控制流 task。
4. 以依赖和完成事件而非 call 序号计算 memory liveness，扩展到多 stream、多 device。
5. 保持 `RuntimeSession` 作为静态、强类型的 module+frozen-plan executor。
### 2.2 非目标
- 不在本计划恢复历史 adaptive runtime、fuzzy cache 或按“buffer 更大”运行的 fallback。
- 不使 RuntimeSession 解释 Relay、选择 variant、编译 miss 或按 operator name 决策。
- 不承诺任意 fusion、任意跨设备 placement、通信后端或 data-dependent/ragged allocation。
- 不把 storage id 当作 value identity，也不把 graph-local value id/symbol 当作 artifact key。
- 不在 task DAG 可验证前启用并发执行；单 stream 拓扑执行是安全基线。
### 2.3 术语
| 名称 | 含义 |
|---|---|
| region | 具有 nodes、live-in/live-out、effect/alias、target/layout/workspace 契约的编译边界 |
| fusion region | 经合法性证明的多个内部 op，共享单个或少量 kernel artifact 的 region |
| library region | 以外部库 ABI/descriptor 调用的 region，而非假装为 TE kernel |
| task | plan 中一个可调度动作：kernel/copy/shape-eval/allocate/event/sync/control |
| dependency | 数据、effect、alias、allocation 或 event 产生的必须先后关系 |
| frozen plan variant | 已选择 symbols/generations、shape binding、physical memory plan 的不可变执行版本 |
## 3. 冻结接口与数据结构草案
这些结构是跨 04/05/06 的协议草案。实际类型可后置，但字段语义、验证与版本号先冻结。
```text
CompilationUnit {
  UnitLocator locator; UnitSemanticKey semantic_key;
  UnitKind {PerCall, Fusion, Library, ControlFlow};
  nodes; live_ins; live_outs; constants;
  target, layout, workspace, EffectSummary, AliasSummary;
}
Task {
  TaskId id; TaskKind {Kernel, Copy, ShapeEval, Allocate, Event, Sync, Control};
  inputs, outputs, dependencies; Device device; StreamClass stream;
  optional<UnitLocator> unit; optional<ArtifactGeneration> generation;
}
ValueContract {
  ValueId id; logical_shape; physical_shape; valid_extent;
  dtype, layout, alignment, memory_scope, alias_class, storage_id;
}
ExecutionPlanVariant {
  version; tasks; values; output_values; chosen_artifacts;
  shape_bindings; memory_assignments; event_edges; plan_fingerprint;
}
```
`UnitSemanticKey` 来自规范化 region IR、边界 tensor 语义、attrs、target、ABI、pipeline 与
backend/schedule version；不包含 `value_id`、locator、symbol、object address 或 storage id。
`UnitLocator` 只服务 plan routing、profile 与诊断。artifact entry symbol 与 graph call identity
必须可分离或通过安全重定位绑定。
`ValueContract` 将当前 `ValueSpec(shape,dtype,device,storage)` 的 exact 特例扩展为 logical、
physical、valid extent 与 layout。logical 参与数学语义；physical 参与 allocation/schedule；
valid extent 是 padded/bucket kernel 的读写边界，三者不可互换。
## 4. 硬依赖、软依赖与可并行任务
### 4.1 硬依赖
- region boundary verifier：live-in/out、常量捕获、重复 logical operand 与多输出必须完整。
- effect/alias contract：否则 fusion、task reorder 与 storage reuse 不可证明。
- exact logical/physical/valid-extent contract 和 plan validator：否则 allocation/copy task 无 ABI。
- dependency/event representation：否则多 stream liveness 仍会错误依赖线性 call order。
- immutable artifact/generation ownership：否则 task 运行期间 module 可能被替换或释放。
### 4.2 软依赖
- 04 的 structured control-flow IR 只在接入 `Control` task 时需要；kernel/copy DAG 可先独立落地。
- `CompileCoordinator`、bucket/polymorphic dispatch 仅决定生成哪个 frozen plan variant，不阻塞
  per-Call exact DAG。
- multi-device、通信库、动态输出和 region fusion 都可在单 device、single-stream DAG 之后推进。
- 06 的 Transformer fixtures、CPU/LLVM/CUDA profile 可先消费 per-Call fake/feature-gated plan。
### 4.3 可并行任务
| 工作包 | 可独立条件 | 与其他包的接口 |
|---|---|---|
| region partition/verifier | 以 IR fixture 验证，不启动 runtime | `CompilationUnit` v1 |
| task-DAG validator/fake executor | 以手写 frozen fake plan 验证 | `Task`/`ValueContract` v1 |
| memory planner | 消费 task dependencies 与 contract | `memory_assignments`/event edges |
| library region ABI adapter | 以 mock library entry 验证 | `UnitKind::Library`、descriptor contract |
| 04 control-flow lowering | 只产出 `Control` task/CFG | `Control` task v1 |
| 06 fixtures/profile gates | feature-gated，禁止宣称支持 | plan/profile schema v1 |
### 4.4 集成点
04 只提交验证通过的 structured region/CFG、control task、Phi 与 effect/alias metadata；
05 不 include compiler IR。05 向 04 提供 fake executor 的 trace、task validator 与 branch/loop
liveness 结果。06 只调用稳定的 plan submission、profiling events 和 capability gates；它不得
通过测试专用入口改变 partition 或 executor 行为。
## 5. 分阶段步骤
### Phase A：将 per-Call 明确为 policy
1. 把当前 one ordinary compute Call = one unit 标记为 `PerCall` policy，而非结构不变量。
2. 为所有 unit 生成显式 nodes、live-in/out、constants、target/layout/workspace、effect/alias。
3. 固化 region semantic key 与 locator 的分工；缓存只使用前者。
4. 保留现有 N ordinary Call 对应 N unit 的 characterization test 作为回退证据。
**测试：** 无关 value-id/symbol 重编号或插入独立 Call 不改变等价 unit key；attrs、dtype、
layout、target、ABI、pipeline 变化必须 miss。重复 operand、常量、多输出和共享 producer 的
边界/ABI 必须与当前 per-Call 结果一致。
### Phase B：保守 fusion 与 library region
1. 先允许无 effect、无冲突 alias、兼容 target/layout 的 elementwise producer-consumer fusion。
2. 对无法证明合法的边界稳定回退 `PerCall`，而非猜测 fusion 或改变数值顺序。
3. library region 持有显式 descriptor、workspace、stream 与 error ABI；它不是伪装的 kernel。
4. 记录 fusion 前后 unit locator、semantic key、artifact provenance 和 profile 对照。
**测试：** fused/unfused 数值、ABI、effect trace、共享常量和多输出一致；跨 alias/effect、
不兼容 layout、未知 workspace 和跨 device fusion 必须拒绝或回退。mock library 的失败与资源
保活必须可观察。
### Phase C：从 ordered calls 到单 stream task DAG
1. 为现有 `KernelCall` 生成 `KernelTask`，并加入 `AllocateTask`、`CopyTask`、`EventTask`、
   `SyncTask`、`ShapeEvalTask` 的 schema。
2. plan validator 检查 task id 唯一、依赖存在且无环、每个 value producer/consumer 完整、
   device/stream 匹配、输出可达及 artifact signature/generation 匹配。
3. first executor 仅按拓扑序在一个 stream 执行；结果必须等于当前 ordered RuntimeSession。
4. 使用手写 fake plan/fake executor 验证 schema，避免 05 等待 04 或真实 backend。
**测试：** diamond DAG、copy 后 kernel、多个 independent task、缺 producer、cycle、错误
stream/device、错误 generation、未同步读取与输出不可达。保留静态 module+plan Session 测试。
### Phase D：dependency-aware memory
1. 将 allocation candidate 绑定 `ValueContract`：physical shape、layout、alignment、scope、
   workspace 与 alias class 必须兼容。
2. liveness 从“最后一个 call index”变为 producer 到所有 consumer/retire event 的闭区间。
3. 分支互斥 storage 复用需要 effect/alias 与 event 证明；loop-carried 值需要 backedge
   lifetime，均由 04 的 control task 输入表达。
4. 任何未证明 completion 的 async value 都不得重用或释放。
**测试：** diamond join、async consumer、branch Phi、loop ping-pong、workspace、alignment
冲突和 alias-in-place。比较无复用 reference 的数值，验证峰值和无非法重叠；Sanitizer 测试是
后续实现门禁，不是本计划的实现动作。
### Phase E：多 stream 与多 device
1. 在 DAG 已稳定后为 task 声明 stream class、event record/wait 与 device placement。
2. copy/compute overlap 仅由显式 event edge 授权；同步失败时安全退回单 stream 拓扑执行。
3. 跨 device 必须使用显式 copy/communication task；physical device、VirtualDevice 与 worker id
   不可混用。
4. memory planner 按真实 completion event 计算 retire，不以提交顺序或 worker 号推断。
**测试：** single-stream 与 multi-stream bitwise/tolerance 一致、copy/compute overlap trace、
missing event rejection、跨 device copy correctness、in-flight artifact generation 保活。
## 6. RuntimeSession 的稳定职责
`RuntimeSession` 继续只消费 `CompiledModule + ExecutablePlan` 或等价的纯 runtime
`FrozenPlanVariant`。构造期验证 module entry、plan value、signature、constant mapping 与
selected generation；运行期验证实际 NDArray 对 logical/physical/valid-extent 的契约，再执行。
它不应：调用 Compiler；查询 primitive cache；等待/提交编译；解释 Relay；根据 op 名、模型格式
或“较大 capacity”选择 kernel；在运行中原地替换 executable。上层控制面负责 shape binding、
variant dispatch、CompileCoordinator 和将选择结果冻结为 plan。in-flight `AsyncOperation` 必须
强引用 selected artifact/module 与 storage，使新 generation 仅影响新请求。
## 7. 可观测性与验收数据
每个 task/region 至少记录 graph revision、unit semantic key、locator、task kind、device/stream、
logical/physical/valid extent、dependency wait、allocation bytes、artifact generation、queue/launch
时间与 retired storage。profile 不得把尚未实现的 background compile 或 dynamic scheduling 当事实。
06 可据此采集端到端 latency/throughput、compile/cache、peak memory、bucket padding、copy/
compute overlap 与 CPU/LLVM/CUDA 证据。字段缺失时对应 feature gate 不得开放。
## 8. Done 条件
1. `CompilationUnit` 可表示 PerCall、Fusion、Library（ControlFlow 接口预留且由 04 接入），
   所有 region 有可验证边界、effect/alias 与分离的 identity。
2. frozen task DAG 可独立于 Compiler 验证无环、value 完整、ABI/generation/device 正确；fake
   executor 和单 stream executor 与当前静态顺序结果一致。
3. memory reuse 由 dependency/event、physical contract 与 alias 证明；不再依赖 call-index
   假设，也绝不将 capacity 当 logical shape。
4. 多 stream/multi-device 仅在显式 event/copy 契约和测试后启用，失败可安全回退。
5. RuntimeSession 仍是静态执行器，没有 compiler/Relay/cache policy 依赖；04、05、06 可按
   冻结接口并行开发并以集成 contract suite 收敛。
