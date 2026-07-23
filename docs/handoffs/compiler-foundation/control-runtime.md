# Compiler Foundation / W2 Track 04/05 Control Runtime 交接

> 分支：`feature/compiler-foundation-control-runtime`
>
> 状态：**default-OFF、CPU:0、static-exact 的 resolved control runtime 已闭环**
>
> 边界：真实 Relay/TE/backend control artifact resolver 尚未接入；当前按允许的 fallback 使用 typed `CompiledModule` launcher fixture 验证生产可校验 schema、adapter、executor 和生命周期。

## 1. 本次交付

新增了一条与 `ControlPlan v2` preparation、`FrozenTaskPlan v1` 静态 DAG 分离的路径：

```text
ControlPlan v2 (unresolved, compiler preparation)
  -> api::BindControlPlanForRuntime(typed module-entry bindings)
  -> runtime::ControlExecutionPlan v1 (resolved, immutable)
  -> runtime::ControlRuntimeSession (CPU:0/default stream)
```

关键接口：

- `include/kxc/runtime/control_execution_plan.h`
  - immutable `ControlExecutionPlan v1`；
  - `BoundControlKernel` 在 adapter 边界直接快照并强持有 ready `CompiledModule`
    entry executable、该 entry 的 `KernelSignature`/launch metadata/constant table 和显式
    `generation > 0`；后续 launch 不再按 symbol 回查 module；
  - resolved task 不含 `kernel_ref`、Relay op 名或可供 executor 解释的 symbol 字段。
- `include/kxc/compiler/control_flow.h`
  - `ControlKernelBinding` 提供 task id、typed module entry、generation 和按 ABI
    排列的 non-output value ids；
  - `BindControlPlanForRuntime` 首先完整验证 `ControlPlan v2`（包括 locator/provenance
    非空约束），然后按 task id 绑定；不解释 `kernel_ref`，也不将其用于 artifact 选择/dispatch。
- `include/kxc/runtime/control_session.h`
  - 独立 `ControlRuntimeSession`，数据面不 include Relay、Compiler、TE、cache 或后台线程；
  - 同步/异步结果包含 outputs、deterministic trace 和 typed loop iteration count。

`Compiler::Compile` 未修改，默认 static-dataflow capability gate 继续拒绝 Relay `If`。
本次没有把控制流塞进 ordered `ExecutablePlan`，也没有重解释 `FrozenTaskPlan v1` 的 DAG/generation 字段。

## 2. Resolved schema 与 verifier

`ControlExecutionPlan v1` 只接受：

- 来源 provenance 为 `ControlPlan v2`；
- 全图 CPU:0、task `default` stream；
- exact dtype/rank/shape，所有维度非负；
- 全局 `PureNoAliasV1` effect model；
- structured region tree、完整 live-in/live-out、同 region 前序 dependency；
- Branch predicate 必须是 CPU scalar bool，Phi 两侧与结果 contract 完全相同；
- condition-before-body、有显式 `max_trip_count` 的 Loop；
- static-exact loop initial/body argument/backedge/result 与 lexical body-argument scope；
- 每个 kernel 已绑定 ready immutable module entry、同一 signature/metadata object identity、
  `generation > 0` 和精确 ABI role/value mapping；output ABI 顺序必须与 task outputs 完全一致且不可重复；
- constant 必须由 module constant role/key 精确绑定，跨 entry 的同一 logical constant 必须保持
  payload identity，并在 binding 时验证 Storage range/null-data/alignment。

显式拒绝：

- unresolved kernel、`kernel_ref` execution、缺失/重复/额外/non-kernel binding；
- generation 0、entry/signature/ordered-output/device 漂移；同 contract constant 的 ABI 重排也拒绝；
- `-1`/symbolic/dynamic shape、dynamic output、shape-changing loop；
- must/may alias、host callback、write/allocation/sync effect；
- CUDA/device predicate copy、multi-device、multi-stream、copy/event/ShapeEval；
- unbounded loop。

## 3. Executor 与 liveness

V1 采用最保守但可证明的 control-aware lifetime：

- 只有实际选择的 branch 才执行和分配 output；未选 branch 不 launch；
- graph input/predicate/constant 在使用前验证 dtype/shape/device、Storage range/null-data；
- kernel output 在实际 invocation 前按 signature alignment lazy allocate；
- Phi 只在 value table 中绑定 selected source，不声明一般 alias 能力；
- loop 每轮重新绑定 current/body argument，condition 完成后读取 predicate，backedge storage
  跨下一轮 condition/body 保活；
- 超过 `max_trip_count` 时抛出明确 `std::runtime_error`，不展开、不 fallback；
- V1 不做 control storage reuse，也不按线性 call index 提前 release；每次 invocation 的历史
  storage 都保留到 completion；
- launcher 返回的 completion 必须 defined、设备一致、使用 canonical default stream，且 completed/event 状态合法；
- completion context 强持有 frozen plan、`BoundControlKernel`、module/artifact、generation、输入、
  常量、中间值、selected branch、Phi、loop carried/backedge、outputs 和所有先前 operation；
- final completion 从 retained prior-operation list 中移除，避免 completion/context ownership cycle。

trace 与 Track 04 reference executor 使用同一稳定事件：

```text
read:<task>:<value>
task:<task>
branch:<task>:then|else
loop:<task>:iteration:<n>
write:<task>:<value>
```

## 4. Feature gate 与 CI

新增 CMake gate：

```cmake
KXC_ENABLE_CONTROL_RUNTIME=OFF
```

- 默认 OFF：resolved plan 仍可在 compiler side 组装/验证，但 `ControlRuntimeSession`
  构造与公开 `BoundControlKernel::Launch` 都立即拒绝且不 launch；
- ON：开放本页限定的 CPU static-exact executor；
- 与 `KXC_ENABLE_REGION_TASK_DAG` 独立；GitHub CPU CI 使用 OFF/ON × OFF/ON 组合矩阵，
  防止两个实验 gate 相互污染。

新增 `control_runtime_integration_test`，同时带 `control-flow`、`control-runtime`、
`runtime-plan`、`cpu` labels；`run_control_flow_tests` 和 `run_runtime_plan_tests` 都依赖它。

focused evidence：

- gate OFF：compiler default gate、adapter/resolved verifier、ordered ABI、malformed Storage、direct/session gate rejection **7/7 passed**；
- gate ON：compiler default gate、multi-Phi/multi-carried Branch/Loop differential、termination、stream/ABI/lifetime negative **7/7 passed**；
- CPU CTest label：gate OFF **39/39 passed**，gate ON **39/39 passed**；
- focused control-flow suite：两种 gate 均 **6/6 passed**；runtime-plan suite均 **4/4 passed**；
- include-layer check：passed；
- public-header self-compile：passed；
- `git diff --check`：passed。

测试使用 typed `CompiledModule` + `KernelSignature` + `KernelLauncher` fixture。production
runtime 从不根据 `kernel_ref` 或 op 名 dispatch；测试会把 `kernel_ref` 改成无效 decoy 并证明
resolved binding/输出不变。

## 5. 双 oracle 覆盖

Track 04 `ControlPlanReferenceExecutor` 与 resolved runtime 比较：

- true/false Branch、multiple Phi output 和完整 trace；
- 未选 branch launcher count 为 0；
- Loop zero/one/multi-trip、multiple carried/backedge output、完整 trace 和 iteration count；
- max-trip exhaustion 的拒绝类别；
- duplicate logical input operand 与 ordered unique output ABI；
- provenance、lexical scope、predicate、placement、dynamic shape、effect/alias、binding/signature/input
  contract 负例均 fail closed；malformed predicate/constant Storage 在 kernel launch 前失败；
- session/module/plan/input 局部 owner 销毁后 completion 仍保活 module/artifact/storage，释放
  completion 后 artifact 正常析构，无引用环。

## 6. 尚未开放的生产门禁

本次采用 typed module fixture fallback，因此仍需后续 compiler production adapter：

1. 保留 normalized Relay Call/attrs/CompilationUnit/ValueInfo sidecar，不能从 `kernel_ref` 恢复；
2. 调用真实 TE hook并复核 flattened output count、definedness、dtype、rank、每一 exact extent；
3. 使用 Core semantic key + target/ABI/pipeline/backend fingerprint 解析/编译 artifact；
4. 将 Track 03 authority-issued generation lease 转移为 runtime opaque keepalive；当前 schema 已冻结
   generation 和 immutable module-entry边界，但 fixture generation 不声称是 production publication；
5. 增加真实 LLVM numeric、pending CUDA retention、device predicate Copy/Event/Sync、sanitizer 和
   observability 证据后，才可扩大 backend 或考虑默认开启；
6. dynamic Shape/Phi/output、shape-changing loop、storage reuse、multi-stream/device 继续 fail closed。

在这些门禁完成前，能力声明严格限定为：**feature-gated CPU static-exact resolved control
runtime，typed CompiledModule fixture binding；不是通用 dynamic graph/runtime，也不是 production
Relay control compilation。**
