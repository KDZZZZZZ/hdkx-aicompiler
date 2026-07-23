# Compiler Foundation：Runtime manifest / observability 交接

> **分支：** `feature/compiler-foundation-runtime-observability`
>
> **基线：** `525950a`（default-OFF Region Task DAG）
>
> **状态：** W2 Track05 generation-0 static-exact manifest、可审计 fallback 与同步 observer 已实现；feature gate 仍默认关闭
>
> **非能力声明：** 本交接不声明 non-zero generation、dynamic shape、Fusion/Library/ControlFlow 执行、多 stream/device、后台调度或真实 pending CUDA retention 已完成。

## 1. 架构边界

依赖方向保持为：

```text
Core / Compiler / upper control plane
        -> runtime::PlanVariant / FrozenTaskPlan manifest
        -> RuntimeSession + CompiledModule + AsyncOperation
```

`RuntimeSession` 仍是静态数据面：

- Runtime 不 include Compiler、Relay、primitive cache 或 adaptive coordinator；
- session 构造后不 lookup、compile、选择 generation 或原地替换 executable；
- fallback 只在构造期选择；task 执行开始后任一失败直接传播，不 replay per-Call oracle；
- task executor 与 observer 不创建后台线程；
- `KXC_ENABLE_REGION_TASK_DAG` 默认仍为 `OFF`。

Compiler 可以向下依赖纯 Runtime DTO。生产 `Compiler::Compile` 现在生成
`CompiledGraph::variant`，其 manifest identity 直接来自完整 Core
`ArtifactKey::canonical_bytes()`，同时以 opaque retention token 持有全部
production `ArtifactPin`。Runtime 不解释该 canonical identity，也不反向依赖 Core。

## 2. Runtime-only selected-artifact manifest

文件：

- `include/kxc/runtime/task_plan.h`
- `src/runtime/task_plan.cc`

### 2.1 DTO

`SelectedArtifactBinding` 为每个 ordered call 或 kernel task 保存：

- `ArtifactBindingKind::{kCall,kTask}` 与 plan-local `invocation_id`；
- 完整、不为空的 opaque `artifact_identity` canonical bytes；
- selected `generation`；
- Runtime 规范化的 `exact_abi_fingerprint`；
- `entry_symbol`；
- 将 identity、generation、ABI、entry 与 invocation locator 一起规范化的
  `entry_binding`。

`SelectedArtifactManifest` 保存：

- schema version `kSelectedArtifactManifestVersion == 1`；
- 每 call/task 恰好一个且 locator 唯一的 binding；
- `plan_fingerprint`；
- 可选 opaque retention token。生产 Compiler 总是提供 token；runtime fake/contract
  fixture 可不提供。

`PlanVariant` 冻结 `ExecutablePlan + per-call manifest`；`FrozenTaskPlan` 可通过
`AttachSelectedArtifacts` 冻结 per-kernel-task manifest。`PlanTaskMemory` 必须先于
selected-artifact freezing；已冻结 manifest 的 plan 不再允许重做 memory planning。

### 2.2 Canonical fingerprints

Runtime canonicalizer 使用 length-delimited fields，再以完整 canonical bytes 的 hex
编码形成可打印 fingerprint；它不是依赖 Compiler 的摘要，也不把 graph-local symbol
当作 artifact identity。

- call/task exact ABI 覆盖实际 `CompiledModule` 的 ordered `KernelSignature`、launch
  metadata、constant key/NDArray contract、ordered invocation value contract、storage id，
  以及 task output allocation alignment；
- entry binding 覆盖 binding kind/id、完整 artifact identity、generation、exact ABI 与
  entry symbol；
- plan fingerprint 覆盖 values、storage/memory contract、calls 或 tasks/dependencies、
  regions/effect/alias、graph inputs/constants/outputs 和全部 selected bindings；
- task/value/region 等集合按稳定 id canonicalize；ABI-sensitive input/output 顺序保持不变。

上层可以提供 Core canonical artifact bytes，但 session 始终用实际 module/plan 重新计算
Runtime ABI、entry binding 与 plan fingerprint，不信任调用方提供的文本摘要。

### 2.3 generation 边界

DTO/fake 仍可表达非负 generation，以保留跨轨 schema；当前真实
`RuntimeSession` 对 `PlanVariant` 和 `FrozenTaskPlan` 都只接受 generation `0`，并要求
`TaskSpec::artifact_generation` 与 manifest 一致。任何 non-zero generation 在 launch 前
拒绝。这是本任务明确保留的门禁，不表示 Track03 hot-swap 已接通。

## 3. RuntimeSession 构造期复验与生命周期

新增/扩展 API：

```cpp
RuntimeSession(CompiledModule, ExecutablePlan,
               RuntimeExecutionMode, RuntimeObserver = {});
RuntimeSession(CompiledModule, PlanVariant,
               RuntimeExecutionMode, RuntimeObserver = {});
RuntimeSession(CompiledModule, FrozenTaskPlan, RuntimeObserver = {});

TaskDAGSelectionResult TaskDAGSelection() const;
SelectedArtifactManifest artifact_manifest() const;
```

构造期依次验证：

1. module ready、plan/schema 完整、单 device/stream 边界；
2. module entry 与 call/task entry symbol 精确绑定；
3. ordered signature role/arity/shape/dtype/device/alignment；
4. launch backend/device/grid/block/shared-memory contract；
5. constant key、constant NDArray 与 plan constant value 映射；
6. task Allocate alignment、storage id、dependency-aware memory plan；
7. manifest 每 invocation 完整、generation 为 0；
8. Runtime 重算 exact ABI、entry binding 和 plan fingerprint 与 manifest 完全相等。

成功的 `RunAsync` completion 强持有：

- `CompiledModule`；
- ordered plan 或 `FrozenTaskPlan`；
- `SelectedArtifactManifest` 及 opaque pin/lease token；
- per-run `ValueTable`；
- 最终操作之前的全部 `AsyncOperation`；
- current 与 retired/reused storage。

因此 session、输入 handle、局部 manifest 或 Compiler 返回对象可在 completion 前释放。
当前确定性 retention 证据为 CPU fake；真实 pending CUDA 仍是默认开启前的硬门禁。

## 4. 可审计 Task-DAG selection / fallback

`TaskDAGSelectionResult` 记录 requested/selected mode、`FallbackReason` 和稳定 diagnostic。
observer 同时收到一个 `RuntimeEventKind::kFallback` 事件。已分类原因：

| 原因 | 行为 |
|---|---|
| `kFeatureDisabled` | gate OFF，选择已验证 per-Call oracle |
| `kMissingArtifactManifest` | bare `ExecutablePlan` 不能伪造 artifact identity，选择 per-Call |
| `kUnsupportedDynamicInput` | 任一 `-1`/dynamic value 不进入 static-exact DAG |
| `kUnsupportedAlias` | alias value 或 conservative alias region 不进入当前 executor |
| `kUnsupportedFusion` | 缺 verified fusion provenance，直接 frozen construction 拒绝 |
| `kUnsupportedLibrary` | 缺 library descriptor/workspace/stream/error ABI，拒绝 |
| `kUnsupportedControlFlow` | 当前 RuntimeSession 不执行 control-flow region，拒绝 |
| `kUnsupportedShapeEvaluation` | 缺 ShapeProgram/runtime output contract，拒绝 |
| `kAdapterBug` | 已通过显式 preflight 后 adapter/validator invariant 失败；发 diagnostic 后 fail closed，不降级执行 |

原来的 `catch (const std::invalid_argument&) { /* silent fallback */ }` 已删除。已知
unsupported 条件在 adapter 前显式检查；adapter 内异常统一视为 bug 并抛出
`logic_error`。直接 frozen plan 没有 ordered oracle 可选，因此 library/control/fusion 等
事件表示带结构化原因的构造拒绝，而不是执行了另一路径。

Gate ON 时，只有带完整 manifest 的 `PlanVariant` 可以适配为 task DAG。旧
`RuntimeSession(module, ExecutablePlan, kTaskDAG)` 保持 API 兼容，但不会从 symbol 猜造
artifact identity；它以 `kMissingArtifactManifest` 回到 per-Call。

## 5. 同步 observer event schema

文件：

- `include/kxc/runtime/task_executor.h`
- `src/runtime/task_executor.cc`
- `src/runtime/session.cc`

`RuntimeEvent` 的稳定 kind：

- `kTaskWait`：消费一个已完成 scheduler dependency；带 dependency task id；
- `kGeneration`：kernel submission 前记录 selected generation、artifact identity 和 entry；
- `kTaskLaunch`：task host action/backend submission 成功返回后记录；失败 submission 不发
  successful launch；
- `kAllocation`：allocation 成功后记录 value/storage、bytes、alignment；
- `kRelease`：所有 consumer host actions 完成后的 **logical retire/reuse eligibility**；不声称
  async backend physical free，storage 仍由 completion 保活；
- `kTaskComplete`：当前同步 host action 完成，不表示 GPU event 已完成；
- `kFallback`：构造期 selection/rejection reason 与 diagnostic。

observer 在调用 `RuntimeSession`/`RunAsync` 的线程同步调用，不使用 dispatcher、queue 或
后台线程。observer 异常被隔离，不能改变 selection、task submission 或 no-replay 语义。
`ExecuteTasksDeterministically` 返回同一事件序列并可同时注入 observer；topological tie-break
仍为 task id，测试不依赖时钟或 sleep。

## 6. 测试与 CI

聚焦测试：

- `task_plan_test`
  - per-task/per-call manifest completeness、generation/task matching、entry binding、plan
    fingerprint drift、immutable DTO；
- `task_executor_test`
  - wait/generation/launch/allocation/release exact deterministic trace、manifest identity、
    dependency-aware release、failure stop/no replay；
- `runtime_session_test`
  - gate OFF/ON selection query与 fallback observer；
  - missing manifest、dynamic、alias、library、control 分类；
  - module ABI tamper 与 non-zero generation 拒绝；
  - task events、observer exception isolation、failed submission 无 false launch；
  - manifest token 经 `AsyncOperation` 保活；
  - multi-entry/repeated-entry task bindings 与执行开始后不 replay；
- `operator_compilation_test`（LLVM build）
  - production `CompiledGraph::variant` binding 与 production artifact pins 一一对应。

`.github/workflows/ci.yml` 保留 CPU gate `OFF/ON` matrix，并新增 gate-ON
ASan+UBSan runtime-plan job；LLVM job以 gate ON 构建并运行
`operator_compilation_test` 与 `runtime_session_test`。CUDA/Compute Sanitizer 不在无设备 CI
中伪报通过。

本地 CPU-only（GCC 13.3，LLVM/CUDA disabled）结果：

- gate OFF：`cpu` label **38/38 passed**；
- gate ON：`cpu` label **38/38 passed**；
- gate ON ASan+UBSan：`runtime-plan` label **3/3 passed**，使用
  `abort_on_error=1` / `halt_on_error=1`；
- 两套 CPU suite 均包含 include-layer 与 public-header self-compile。

没有联网、安装依赖、push 或 merge。

## 7. 默认开启前仍未完成

1. authoritative non-zero generation lease 与 Track03 immutable slot handoff；
2. LLVM/CUDA task-DAG 数值矩阵和真实 pending CUDA manifest/pin retention；
3. Compute Sanitizer 0 error；
4. Fusion legality/provenance、Library ABI、ControlFlow executor、ShapeEval contract；
5. multi-stream event completion 与 physical retirement；
6. 性能门禁（p50/p95、launch/sync count、peak bytes）与设备 profile evidence。

在这些门禁完成前，`KXC_ENABLE_REGION_TASK_DAG` 保持默认 `OFF`，static per-Call oracle
永久保留。
