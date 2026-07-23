# Compiler Foundation：Runtime manifest / observability 交接

> **分支：** `feature/compiler-foundation-runtime-observability`
>
> **基线：** `525950a`（default-OFF Region Task DAG）
>
> **状态：** W2 Track05 generation-0 static-exact trusted declaration、可审计 fallback 与同步 observer 已实现；feature gate 仍默认关闭
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
`CompiledGraph::variant`：Compiler 从完整 Core `ArtifactKey::canonical_bytes()` 声明
manifest identity，并以 opaque retention lease 持有全部 production `ArtifactPin`。
Runtime 不解释 identity，也不反向依赖 Core。

**信任边界：** `SelectedArtifactManifest`、`PlanVariant`、`ArtifactSelection`、
`MakePlanVariant` 及 fingerprint helper 都是公共可构造 API。Runtime 将其视为 trusted
upper-control-plane observability declaration；caller 可以提交任意非空 identity、generation
0 和任意 `shared_ptr<const void>`，并生成结构自洽的 manifest。Runtime 不能证明 identity
来自真实 selected/pinned `ArtifactPin`，也不能证明 lease 与 identity 关联。上述关联只由当前
production Compiler 组装路径建立，不是 Runtime 可验证的 provenance 或安全能力。

## 2. Runtime-only selected-artifact declaration

文件：

- `include/kxc/runtime/task_plan.h`
- `src/runtime/task_plan.cc`

### 2.1 DTO

`SelectedArtifactBinding` 为每个 ordered call 或 kernel task 保存：

- `ArtifactBindingKind::{kCall,kTask}` 与 plan-local `invocation_id`；
- caller 声明的、完整且不为空的 opaque `artifact_identity` 文本；production Compiler
  在自己的可信路径中使用 Core canonical bytes；
- caller 声明的 selected `generation`；
- `exact_abi_fingerprint` 与 `entry_binding` 字段：builder helper 从实际 module/plan
  派生；public DTO direct constructor 也允许 caller 提供，故字段本身不是来源证明；
- `entry_symbol`；
- `RuntimeSession` 会根据实际 module/plan 重算 exact ABI，并复核 identity、generation、
  ABI、entry、invocation locator 的 canonical entry binding。

`SelectedArtifactManifest` 保存：

- schema version `kSelectedArtifactManifestVersion == 1`；
- 每 call/task 恰好一个且 locator 唯一的 binding；
- `plan_fingerprint`；
- 可选 opaque `retention_lease`。它只做 type-erased lifetime retention，不是 credential、
  validation token 或 provenance proof。生产 Compiler 总是提供持有 pins 的 lease；runtime
  fake/contract fixture 可不提供。

`PlanVariant` 冻结 `ExecutablePlan + per-call declaration`；`FrozenTaskPlan` 可通过
`AttachSelectedArtifacts` 附加 per-kernel-task declaration。`PlanTaskMemory` 必须先于
artifact declaration attachment；已附加 manifest 的 plan 不再允许重做 memory planning。

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

上层可以声明 Core canonical artifact bytes。session 用实际 module/plan 重算 exact ABI，并
复核 entry binding 与 plan fingerprint 的结构一致性；这能发现 declaration 与当前
module/plan 的 drift，但不会认证 caller 提供的 artifact identity、generation、lease 来源或
identity↔lease 关联。fingerprint 是确定性 observability/consistency 格式，不是签名或 MAC。

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
7. caller declaration 每 invocation 完整、generation 为 0；
8. Runtime 重算 exact ABI，并验证 entry binding 和 plan fingerprint 与 declaration 结构一致；
   这不构成 artifact identity provenance 验证。

成功的 `RunAsync` completion 强持有：

- `CompiledModule`；
- ordered plan 或 `FrozenTaskPlan`；
- `SelectedArtifactManifest` 及 opaque retention lease；
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
| `kMissingArtifactManifest` | bare `ExecutablePlan` 没有 upper-plane artifact declaration，选择 per-Call；retention lease 不是 admission 条件 |
| `kUnsupportedDynamicInput` | 任一 `-1`/dynamic value 不进入 static-exact DAG |
| `kUnsupportedAlias` | alias value 或 conservative alias region 不进入当前 executor |
| `kUnsupportedFusion` | 当前没有 fusion execution contract，直接 frozen construction 拒绝 |
| `kUnsupportedLibrary` | 缺 library descriptor/workspace/stream/error ABI，拒绝 |
| `kUnsupportedControlFlow` | 当前 RuntimeSession 不执行 control-flow region，拒绝 |
| `kUnsupportedShapeEvaluation` | 缺 ShapeProgram/runtime output contract，拒绝 |
| `kAdapterBug` | 已通过显式 preflight 后 adapter/validator invariant 失败；发 diagnostic 后 fail closed，不降级执行 |

原来的 `catch (const std::invalid_argument&) { /* silent fallback */ }` 已删除。已知
unsupported 条件在 adapter 前显式检查；adapter 内异常统一视为 bug 并抛出
`logic_error`。直接 frozen plan 没有 ordered oracle 可选，因此 library/control/fusion 等
事件表示带结构化原因的构造拒绝，而不是执行了另一路径。

Gate ON 时，只有带完整 trusted declaration 的 `PlanVariant` 可以适配为 task DAG。旧
`RuntimeSession(module, ExecutablePlan, kTaskDAG)` 保持 API 兼容；bare plan 没有
observability declaration，因此以 `kMissingArtifactManifest` 回到 per-Call。retention lease
始终可选，只影响 lifetime，不是 admission credential。这是接口要求，不是防伪或 provenance
证明。

## 5. 同步 observer event schema

文件：

- `include/kxc/runtime/task_executor.h`
- `src/runtime/task_executor.cc`
- `src/runtime/session.cc`

`RuntimeEvent` 的稳定 kind：

- `kTaskWait`：消费一个已完成 scheduler dependency；带 dependency task id；
- `kGeneration`：kernel 的 `kTaskStart` 前记录 accepted upper-plane declaration 中的 generation、
  artifact identity 和 entry；这些字段是 observability，不是 Runtime-authenticated provenance；
- `kTaskStart`：调用 task host action/backend submission 前记录；即使 submission 抛错也保留；
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
  - per-task/per-call declaration completeness、generation/task matching、entry binding、plan
    fingerprint drift、immutable DTO；公共 caller 任意 identity 被明确当作 trusted declaration；
- `task_executor_test`
  - wait/generation/start/launch/complete exact deterministic lifecycle、declared identity、
    allocation/release、dependency-aware release，以及失败 submission 有 start 但无 launch/complete；
- `runtime_session_test`
  - runtime-plan-scoped gate OFF/ON selection query、fallback observer 与 manifested multi-entry
    numeric equality；
  - missing declaration、dynamic、alias、library、control 分类；
  - module ABI drift 与 non-zero generation 拒绝；
  - task events、observer exception isolation、failed submission 无 false launch；
  - retention lease 经 `AsyncOperation` 保活；
  - multi-entry/repeated-entry task bindings 与执行开始后不 replay；
- `operator_compilation_test`（仅 `KXC_USE_LLVM=1`）
  - production `Compiler::Compile -> CompiledGraph::variant -> RuntimeSession(..., kTaskDAG)`；
  - `UsesTaskDAG`/fallback、Compiler declaration 与 production pins 一一对应、数值 `9.0f`；
  - `CompiledGraph` 和 session 释放后 completion 继续保活 production pin lease，completion
    ownership 释放后 active production pins 归零。

### 6.1 本地证据

本地环境为 GCC 13.3 / CMake 3.28.3，`llvm-config` 不存在，LLVM/CUDA 均关闭。实际结果：

- gate OFF：`run_runtime_plan_tests` **3/3 passed**；`cpu` label **38/38 passed**；
- gate ON：`run_runtime_plan_tests` **3/3 passed**；`cpu` label **38/38 passed**；
- gate ON ASan+UBSan：`runtime-plan` label **3/3 passed**，使用
  `ASAN_OPTIONS=abort_on_error=1:detect_leaks=1` 与
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`；
- `operator_compilation_test.cpp` 以 `KXC_USE_LLVM=1`、gate OFF/ON 分别做了
  `-fsyntax-only`（并在屏蔽现有公共头 warning 后通过 `-Werror`）；这只证明 E2E 测试分支
  可编译，不是 LLVM link/JIT/数值执行证据；
- 两套 CPU suite 均包含 include-layer 与 public-header self-compile。

没有联网、安装依赖、push 或 merge。

### 6.2 CI 注册边界（不是已通过证据）

`.github/workflows/ci.yml` 保留 CPU gate `OFF/ON` matrix 和 gate-ON ASan+UBSan
runtime-plan job。LLVM job明确以 gate ON 构建 `operator_compilation_test`，并用独立
`ctest --no-tests=error --tests-regex '^operator_compilation_test$'` 执行，避免测试未注册时被其他
LLVM tests 掩盖；随后执行 `runtime_session_test` 等 LLVM checks。当前本地任务没有 CI run
URL/result，因此只能声明 **CI 已注册**，不能声明 LLVM E2E 已绿色。CUDA/Compute Sanitizer
不在无设备 CI 中伪报通过。

## 7. 公共 C++ source / binary compatibility

仓库没有为此新增 public surface 声明跨版本预编译 C++ ABI。该变更向 public headers 添加
`CompiledGraph::variant`、Runtime DTO/event 字段与构造重载，并在本次 review fix 中删除
`SelectedArtifactManifest::retention_token()`、改名为 `retention_lease()`，以避免 capability
含义。已有 branch consumer 必须修改 accessor 调用，并使用匹配版本的 headers 与 library 做
源码重编译；这既不 source-compatible，也不 binary-compatible。不得把变更前编译的 C++
object/binary 与变更后 library（或反向）混用。这里的 manifest schema version 与 fingerprint
format version 只版本化数据/规范化格式，不是 C++ binary ABI 兼容承诺。

## 8. 默认开启前仍未完成

1. authoritative non-zero generation lease 与 Track03 immutable slot handoff；
2. LLVM-enabled CI 对已注册 production E2E 的实际绿色记录、其余 LLVM/CUDA task-DAG
   数值矩阵和真实 pending CUDA pin retention；
3. Compute Sanitizer 0 error；
4. Compiler-side Fusion legality evidence、Runtime fusion execution contract、Library ABI、
   ControlFlow executor、ShapeEval contract；
5. multi-stream event completion 与 physical retirement；
6. 性能门禁（p50/p95、launch/sync count、peak bytes）与设备 profile evidence。

在这些门禁完成前，`KXC_ENABLE_REGION_TASK_DAG` 保持默认 `OFF`，static per-Call oracle
永久保留。
