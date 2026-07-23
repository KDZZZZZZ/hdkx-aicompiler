# 04：动态图与控制流

> **状态：** 规划中；不表示当前已实现
> **所属路线：** [编译器基础路线图](README.md)
> **权威输入：** [编译器基础架构审查](../../COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)
> **范围：** 把当前静态 data-flow 路径扩展为可验证的结构化控制流；不在本计划实现代码。
## 1. 背景与问题界定
当前 `ValueGraph` 只将参数、常量、`Call`、tuple 与 tuple field 建成稳定 value id。
它要求 checked type，拒绝 free/unbound `Var`，也不支持 `If`、`Let` 与函数值的完整执行契约。
当前 `ExecutablePlan` 是 runtime-neutral 的有序 `KernelCall` 序列，
`RuntimeSession(module, plan)` 只做强类型的静态、单设备、顺序执行。
因此“Relay 可以构造控制流”不等于“编译器可以执行控制流”；入口必须 fail closed。
本计划处理四个常被混淆的问题，它们必须保留不同的 IR 与运行时语义：
| 问题 | 语义 | 不能替代为 |
|---|---|---|
| `Let` / ANF 归一化 | 绑定计算结果、显式求值顺序与共享 | runtime branch 或动态 shape |
| 数据依赖 `If` / loop | 条件或迭代次数由运行时数据决定的控制流 | host 端预先选择一个 kernel |
| shape-dependent variant dispatch | 已知 graph/template 下按已绑定 shape 选择 exact/bucket/polymorphic 版本 | 表达任意 `If` 或 loop body |
| eager/tracing dynamic graph | 每次执行构图，或录制一次实际路径 | 编译 IR 的通用控制流语义 |
特别地，`-1` 仍只是 legacy input-ABI validation sentinel，绝不是控制流或动态 shape
机制。未知维度必须成为显式 `DimExpr`/constraint，或在 frontend 拒绝。
## 2. 目标与非目标
### 2.1 目标
1. 在 partition 前提供 executable-capability verifier，并对不支持的节点给出节点定位与缺失能力。
2. 使用 structured region 作为前端/中端控制流模型，并可显式降为可验证 CFG。
3. 让 `Branch`、`Loop`、`Phi`、loop-carried value、effect/alias 成为计划级显式事实。
4. 使 branch/loop 的 allocation、liveness、event 和 executor 语义可证明，而不是复用顺序 call-index 假设。
5. 保持 per-Call 静态路径为可回退 policy；将控制流作为增量 capability gate。
### 2.2 非目标
- 不将 `RuntimeSession` 改为持有 `Compiler`、Relay registry、ShapePredictor 或后台编译器。
- 不把 eager/tracing 当成 AOT/Relay 控制流的替代实现；它们仅是独立 frontend 接入策略。
- 不以 host 端偷偷展开未知 loop，或以“较大维度可运行”的 fuzzy fallback 处理分支。
- 不在本阶段实现 ragged/data-dependent dynamic output、异常处理、递归、闭包或通用函数值。
- 不把 variant dispatch 写成按 op 名称分支；它只消费冻结的 shape/dispatch contract。
## 3. 冻结接口与数据结构草案
以下是语义草案，不是当前 API 承诺。先冻结字段、验证规则与序列化，再选择具体 C++ 类型。
```text
ControlRegion {
  RegionId id; RegionKind {Sequence, If, Loop};
  values: [ValueId]; live_ins: [ValueId]; live_outs: [ValueId];
  effects: EffectSummary; aliases: AliasSummary;
}
BranchTerminator {
  ValueId predicate; RegionId then_region; RegionId else_region;
  [PhiBinding] results;
}
LoopTerminator {
  RegionId body; [ValueId] initial_carried; [PhiBinding] carried;
  ValueId condition; optional<ValueId> max_trip_count;
}
PhiBinding { ValueId result; ValueId from_then_or_backedge; ValueId from_else_or_init; }
EffectSummary { reads, writes, allocates, host_callbacks, device_sync; }
AliasSummary { must_alias, may_alias, no_alias; }
```
`PhiBinding` 只合并同一逻辑 tensor contract 的候选值；dtype、device、logical shape、
layout、valid extent 和 alias 规则不兼容时必须在 verifier 失败。loop-carried 值有两份
角色明确的来源：初始值与 backedge 值；不得用同一 `value_id` 隐含覆盖。
控制流 lowering 的目标可选择 CFG：basic block、block argument/phi、branch edge、loop
backedge 与 terminator。structured region 是优化和诊断的主模型；CFG 是执行、无环/回边
验证以及未来 backend lowering 的规范表示。两者必须有可逆的 source locator。
shape variant 仍使用独立的 `DispatchKey`：unit semantic key、target、dtype/layout、
shape binding/bucket、shape-ABI 与 backend/schedule version。它可以在 region 进入前选择
被冻结 `PlanVariant`，却不能读取任意 tensor predicate 来替代 `BranchTerminator`。
## 4. 硬依赖、软依赖与并行边界
### 4.1 硬依赖
- 01 的 capability verifier；否则 `If`/`Let` 会在 partition/lowering 晚期失败。
- 当前 static exact value contract、统一 effect/alias contract；首版只允许两分支和 loop-carried 值具有完全相同的静态 dtype/shape/layout/device。
- 冻结的控制流 plan/task interface 与 plan validator；executor 不应猜测 region 含义。

### 4.2 软依赖
- 02 的 `DimExpr`、constraint、logical/physical/valid-extent contract 只在动态 shape Phi、shape-changing loop 和动态输出接入时成为硬依赖；静态 exact `If`/loop IR 与 verifier 可先并行完成。
- GraphTemplate、exact shape profile、CompileCoordinator 与 KernelSlot 有助于 region 内 specialization，但首个解释/静态 body 闭环不依赖热替换。
- region fusion、library region、多 stream 和动态 allocation 可后续接入；首版可保守地单 stream、exact allocation、无 fusion。
- eager/tracing frontend 可并行探索，但不应成为 Relay executable dialect 的门槛。
### 4.3 可并行任务
| 任务 | 可并行原因 | 交付物 |
|---|---|---|
| frontend `Let`/ANF 与 capability verifier | 只产出冻结 structured IR | 正反例、节点诊断、IR 序列化 |
| `If`/loop region 与 CFG verifier | 不调用真实 executor | region/CFG fixture、Phi/effect/alias 规则 |
| 05 的 task-DAG executor | 只消费 frozen fake tasks | fake plan、拓扑/event/memory contract 测试 |
| shape dispatch | 在 region 边界只消费 bindings | guard/bucket 选择测试 |
| 06 的 fixture/profile | 以 feature gate 标明未支持 | reference 数据、指标 schema |
### 4.4 集成点
04 向 05 交付版本化 `ControlRegion`/CFG 与 `ControlTask` 契约、value/effect/alias
metadata、以及 fake executor trace；05 不包含 Relay/TE/TIR 头。05 返还 plan validator、
branch/loop task 生命周期和 executor 结果。06 只通过 capability gate、profile schema 和
fixture adapter 消费这些接口，不能绕过 verifier。
## 5. 分阶段步骤
### Phase A：归一化与拒绝路径
1. 将 `Let` 归一化为 ANF：每个可执行计算有唯一绑定、显式共享与稳定 source span。
2. 在 graph pass 后、partition 前运行 executable-capability verifier。
3. 当前未开放的 `If`/loop 保持 fail closed；错误包含节点、所需 capability 与建议 gate。
4. 确认归一化不把 effectful/aliasing expression 重排或重复求值。
**测试：** nested Let、共享 producer、dead binding、tuple binding；未归一化或未绑定 Var
必须稳定失败。静态 per-Call graph 的 value id、ABI、数值结果不得变化。
### Phase B：受限 structured `If`
1. 定义 predicate 为标量 bool/明确 device-to-host 或 device-resident contract，禁止隐式拷贝。
2. 建立 then/else region、显式 live-in/live-out 和 `PhiBinding`。
3. verifier 检查两分支的输出数、类型、shape/layout/extent、effect 与 alias 兼容性。
4. 降为 CFG；先允许纯计算、无 host callback、无跨分支写同一 alias class 的分支。
**测试：** true/false、嵌套 If、共享 live-in、多输出、常量捕获、shape 不兼容、分支 alias
冲突、predicate device 不匹配。每例与 reference interpreter 比较数值和未执行分支的副作用。
### Phase C：受限 loop 与 loop-carried values
1. 定义 while/for 的初始 carried 值、body condition、backedge 与 exit `PhiBinding`。
2. 要求终止条件、最大 trip count 或独立静态证明；无证明的 loop 先拒绝。
3. body 的 effect/alias summary 必须闭合：carried value 的读写、in-place alias 与 workspace
   均显式列出。
4. 支持固定/shape-bound trip count 的 exact profile；data-dependent output 继续拒绝。
**测试：** zero-trip、one-trip、多-trip、多个 carried 值、body tuple 输出、break condition、
非法 backedge type/extent、alias 写冲突和 non-terminating rejection。验证每次迭代的 Phi
绑定、输出及 storage lifetime。
### Phase D：与 shape dispatch 组合
1. 在 region 入口以已绑定 shape 选择 exact/bucket/polymorphic body variant。
2. bucket body 必须接收 physical capacity 与 valid extent，并证明 tail-safe；不允许以容量
   大小推断适用性。
3. variant guard 失败时等待/编译/报错；控制流 predicate 仍由 `BranchTerminator` 执行。
4. 保留 region semantic identity 与 graph-local locator 分离，value id 不进入 artifact key。
**测试：** branch 两侧不同合法 bucket、loop 内 tail、guard 域外拒绝、同 region 重编号的
semantic key 稳定性，以及 dispatch 不改变控制流 trace。
### Phase E：frontend eager/tracing 接入边界
1. eager 模式必须把每次构图的 effect、alias、shape binding 显式导出，不能假设静态图。
2. tracing 只录制已走路径；遇到数据依赖 branch/loop 时须 graph break、记录 structured
   control flow，或明确拒绝。
3. trace cache key 包含 guards 与 shape contract；不能从一次路径推广到未证明路径。
4. frontend adapter 仅生成 frozen IR/template，不调用 RuntimeSession 内部策略。
**测试：** trace 中 tensor predicate、不同分支重放、loop trip 变化、guard 失效与 graph
break。所有失败消息要说明是 tracing 局限还是 compiler capability 缺失。
## 6. branch/loop memory 与 runtime executor 要求
- `If` 的互斥分支 storage 可在 effect/alias、device、physical layout 和 completion event
  都兼容时复用；live-out Phi 不得引用已退役分支 storage。
- loop-carried storage 可 ping-pong 或 in-place，但必须由 alias summary 声明并在每次
  backedge 后延长 lifetime；禁止按线性 call index 提前复用。
- 分支条件、shape eval、allocation、kernel、copy、event/sync 都必须是显式 task；完成事件
  决定跨 stream 的 liveness。
- executor 只解释已验证 CFG/task DAG 和 frozen variant；它不做 Relay pattern matching、
  cache 选择或 runtime 编译。
- 首版可由 05 的 fake executor 以单 stream 拓扑序验证 trace；真实 RuntimeSession 仍保持
  静态 module+plan executor，控制面在其上组装冻结执行版本。
## 7. Done 条件
1. `Let`/ANF、If、loop 的支持矩阵与 capability gate 一致；未支持形式在 partition 前失败。
2. structured region 与 CFG 均可独立验证：边界完整、Phi 正确、无非法 backedge、effect/alias
   合法，且 source locator 可追踪。
3. 受限 If/loop 的 true/false、zero/multi-trip、multi-output、alias 和内存复用测试通过，
   并与 reference 的数值和 effect trace 一致。
4. shape dispatch、控制流和 eager/tracing 的职责没有混淆；无 `-1`、fuzzy capacity 或 op-name
   fallback 作为动态语义。
5. 04 与 05 可独立开发：04 通过 fake executor fixture 验证 IR，05 通过 frozen fake plan
   验证执行；集成仅依赖版本化接口测试。
