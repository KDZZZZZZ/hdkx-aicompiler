# Compiler Foundation：Region / ExecutionPlan / Runtime 交接

> **分支：** `feature/compiler-foundation-runtime-plan`
> **状态：** W1 静态 exact 基线已完成；生产 feature gate 默认关闭
> **范围：** frozen Region/task-DAG DTO、validator、dependency-aware memory planner、deterministic fake executor，以及单设备/单 stream 的 feature-gated `RuntimeSession` 执行
> **非能力声明：** 本交接不声明 dynamic shape、control flow、fusion/library ABI、多 stream 或多 device 已完成

## 0. Supervisor fix 后的预留/证据状态

| 项目 | 当前准确状态 |
|---|---|
| `semantic_key` | **仅为 DTO 预留字段。** 本分支不生成、不规范化、不验证生产 semantic key，也没有 artifact cache 消费它；测试中的非空字符串只是 fixture 标签。 |
| `artifact_generation` | **仅为 DTO 预留字段。** validator 只检查非负；`RuntimeSession` 只接受 `0`，尚无 selected-generation manifest、artifact binding 或 pin。 |
| pending GPU retention | **未验证。** 当前 retention 证据来自同步 CPU fake；本分支没有真实 pending CUDA task-DAG、提前释放 handle 或 Compute Sanitizer 证据。 |

## 1. 架构边界

本轨保持了原有单向依赖：

```text
Compiler / Relay / partition
        -> CompiledModule + ExecutablePlan / FrozenTaskPlan
        -> RuntimeSession + NDArray + DeviceStream + AsyncOperation
```

`RuntimeSession` 仍是静态、强类型数据面执行器：

- 不 include 或调用 Compiler、Relay、ShapePredictor、primitive cache；
- 不做 variant 选择、compile miss、缓存策略或后台编译；
- 不创建后台线程；
- 只执行构造期已验证的 `CompiledModule + ExecutablePlan/FrozenTaskPlan`；
- 原 `RuntimeSession(module, ExecutablePlan)` 始终保留为 per-Call 路径；
- task-DAG 失败只允许在执行前回退，任务开始后不重放 per-Call 执行。

## 2. Feature gate 与公共 API

### 2.1 构建开关

```cmake
-DKXC_ENABLE_REGION_TASK_DAG=ON
```

`KXC_ENABLE_REGION_TASK_DAG` 默认 `OFF`。DTO、validator、memory planner 和 deterministic fake executor 始终构建并可独立测试；只有 `RuntimeSession` 的 frozen task 执行由该开关启用。

### 2.2 RuntimeSession API

文件：`include/kxc/runtime/session.h`

```cpp
enum class RuntimeExecutionMode : int32_t {
    kPerCall = 0,
    kTaskDAG = 1,
};

RuntimeSession(api::CompiledModule module, ExecutablePlan plan);
RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
               RuntimeExecutionMode mode);
RuntimeSession(api::CompiledModule module, FrozenTaskPlan plan);

bool UsesTaskDAG() const;
```

行为：

| 调用 | gate OFF | gate ON |
|---|---|---|
| 两参数 `module + ExecutablePlan` | per-Call | per-Call |
| `mode = kPerCall` | per-Call | per-Call |
| `mode = kTaskDAG` | 安全回退 per-Call | 将 static-exact ordered calls 转成 PerCall regions/task DAG；`-1` 或不支持的 alias contract 回退 per-Call |
| 直接 `module + FrozenTaskPlan` | 构造失败并指出 gate | 构造期完整校验后仅执行 static-exact `PerCall` region |

`UsesTaskDAG()` 可用于测试和发布审计，避免把静默回退误报为 task-DAG 命中。

## 3. Frozen contract v1

文件：

- `include/kxc/runtime/task_plan.h`
- `src/runtime/task_plan.cc`

版本：`kFrozenTaskPlanVersion = 1`。

### 3.1 RegionSpec

`RegionSpec` 分离 runtime locator 与 semantic identity：

- `region_id`：plan-local locator，只用于 routing/诊断；
- `semantic_key`：仅预留未来由 01 轨提供规范 key 的承载位置；当前不生成或验证 key，也没有 cache 接线；空字符串表示 unkeyed，测试非空值也不构成生产 identity 证据；
- `RegionKind`：`PerCall / Fusion / Library / ControlFlow`；
- `RegionEffect`：`Pure / Ordered`；
- `RegionAlias`：`NoAlias / Conservative`；
- `task_ids / live_in_value_ids / live_out_value_ids / constant_value_ids`。

validator 从 task/value producer-consumer 关系反推 region boundary，拒绝遗漏或多报的 live-in/live-out/constants。多个 `Ordered` region 必须由显式依赖形成全序；`Conservative` alias region 涉及的值不得参与 storage reuse。

`Fusion`/`Library` 在 v1 仅为 frozen schema/fake 分类，不代表已有 fusion policy 或 library descriptor/调用 ABI。真实 `RuntimeSession` 当前只接受 `PerCall`，明确拒绝 `Fusion`、`Library` 与 `ControlFlow` region。

### 3.2 TaskSpec

`TaskKind`：

- `Kernel`：module symbol、artifact generation、inputs/outputs；
- `Copy`：一个 exact-contract input 到一个 output；
- `Event`：v1 单 stream dependency marker，不是跨 stream record/wait event；
- `ShapeEval`：contract/fake 支持，真实 `RuntimeSession` 在构造期 fail closed；
- `Allocate`：一个 output 与 power-of-two alignment；
- `Sync`：v1 同步当前提交 stream。

公共字段还包括 `task_id`、显式 `dependency_task_ids`、`device`、`stream_id` 和预留的 `artifact_generation`。v1 强制所有 value/task 位于同一物理 device 且 `stream_id == 0`。generation 在 DTO/fake 中只校验非负；真实 `RuntimeSession` 仅接受 `0`，不表示 generation 选择或保活已经接通。

### 3.3 FrozenTaskPlan

`FrozenTaskPlan` 保存：

- version；
- exact `ValueSpec` 列表；
- tasks、regions；
- ordered graph inputs/constants/outputs。

当前 `ValueSpec(shape, dtype, device, storage_id)` 是 static exact 特例，不把 `-1` 升格为完整 dynamic shape，也不将 allocation capacity 当作 logical shape。

### 3.4 Validator 门禁

构造与 `Validate()` 检查：

1. version、task/value/region id 唯一且非负，并拒绝任一 value shape 中的 `-1`，保持 W1 static-exact；
2. task kind 的 payload、arity、symbol、预留 generation/alignment 局部合法；
3. dependency 全部存在、无 self-edge、无环；
4. deterministic topological order 以 `task_id` 作稳定 tie-breaker；
5. source value 不可被 produce/allocate；
6. 每个非 source value 恰有一个 data producer 和一个 `Allocate` task；
7. producer 必须直接依赖其 allocation，consumer 必须在 producer 之后；
8. repeated logical operand 保持原顺序，不在 kernel ABI 中去重；
9. `Copy` 两端 exact shape/dtype/device contract 一致；
10. graph output 可达；
11. region boundary、effect order 和 conservative alias 完整；
12. storage sharing 只在 tensor contract 相同且全部先前 consumers happens-before 后续 allocation 时成立；
13. input/output/constant、alias、async-live 和 conservative-alias 值不得被不安全复用。

## 4. Dependency-aware memory planning

文件：

- `include/kxc/runtime/task_executor.h`
- `src/runtime/task_executor.cc`

API：

```cpp
FrozenTaskPlan PlanTaskMemory(const FrozenTaskPlan& plan);
size_t EstimateTaskPeakLiveBytes(const FrozenTaskPlan& plan);
```

复用判定不使用 call index。对候选旧值 `A` 与新值 `B`，只有 `A` 的所有 consumer（无 consumer 时为 producer）均 happens-before `B` 的 `Allocate` task，且 exact shape/dtype/device、alignment、alias/async/output contract 兼容时，`B` 才可复用 `A` 的 storage slot。

这保证 diamond/independent branches 不会因某个任意拓扑序而误复用；graph output、async-live 和 conservative alias 始终保留独立 storage。`EstimateTaskPeakLiveBytes` 按 deterministic single-stream trace 计算 produced-value 峰值，执行溢出检查；它不是多 stream completion 模型。

## 5. Deterministic fake executor

API：

```cpp
using DeterministicTaskAction = std::function<void(const TaskSpec&)>;

Array<TaskTraceEvent> ExecuteTasksDeterministically(
    const FrozenTaskPlan& plan,
    const DeterministicTaskAction& action);
```

行为：

- validate 后按稳定拓扑序同步执行；
- 不创建线程，不访问 Compiler、cache、设备或 RuntimeSession；
- action 可为手写 kernel/copy/shape-eval fake；
- trace 记录 `TaskStart / Allocate / TaskComplete / Release`；
- release 发生在所有 consumer 完成后；
- action 抛错后立即停止，不 replay 已执行任务。

这使 02/03/04/06 可在不 include 本轨私有 header、不等待真实 backend 的情况下消费 v1 contract。

## 6. RuntimeSession task executor

文件：`src/runtime/session.cc`。

### 6.1 构造期验证

除 `FrozenTaskPlan::Validate()` 外，session 还检查：

- module ready，所有引用 symbol 存在；
- 同一 module entry 可由多个 task invocation 复用，task id 不与 symbol 混作 identity；
- kernel signature 与 value role/shape/dtype/device/arity 完全一致；
- kernel output 的 `Allocate.alignment >= KernelArgSpec.alignment`；
- constants 与 module constant key/NDArray 匹配；
- 当前只接受 `artifact_generation == 0`；这只是拒绝未接线 generation，不是 selected-artifact 校验；
- 真实 executor 只接受 `PerCall` region；拒绝缺 provenance 的 `Fusion`、缺 descriptor/workspace/stream/error ABI 的 `Library`、`ControlFlow` 和 `ShapeEval`；
- module 可包含 frozen variant 未选择的 entry；session 不把 module symbol 集合当作 task identity 集合。

### 6.2 执行语义

- `Allocate`：通过 per-run `ValueTable` 分配或实现已验证的 storage reuse；
- `Kernel`：按 `[input][constant][output]` signature 顺序组装 NDArray 并 launch；
- `Copy`：在同一 v1 stream 上调用 `NDArray::CopyFromAsync`；
- `Event`：单 stream ordering marker，无额外 backend event；
- `Sync`：同步当前 stream；
- `ShapeEval`：构造期已拒绝。

当前 ordered `ExecutablePlan` 的 adapter 为每个 Call 生成一个 `PerCall` region、每个 output 一个 `Allocate` task 和一个 `Kernel` task，并显式串行化 regions，因此 task 模式与旧顺序执行保持等价。任一 `-1` dynamic dimension 或其他 adapter/validator 构造期不支持项均回退原 per-Call plan；gate ON 也不会误标为 task-DAG 命中，执行开始后不做 replay fallback。

### 6.3 AsyncOperation retention

原 retention 语义未削弱。task completion 强持有：

- selected `CompiledModule`；
- frozen task plan；
- per-run `ValueTable`；
- 最终操作之前的全部 `AsyncOperation`；
- current 与 retired/reused Storage；
- 最终 backend operation 自身保留的 executable/storage。

因此销毁 `RuntimeSession`、输入 handle 或局部 task binding 后，completion 仍拥有执行所需状态。这里的已验证证据仅是同步 CPU fake：它覆盖 session 销毁后的 completion 与 input/constant/intermediate/output retention。真实 pending CUDA task-DAG retention **未验证**，仍是开放生产门禁，不能由同步 CPU fake 代替。

## 7. 测试证据

### 7.1 Supervisor fix 聚焦回归

本次修复后，CPU-only gate `OFF` 与 `ON` 分别通过 `run_runtime_plan_tests`：

```text
task_plan_test       passed
task_executor_test   passed
runtime_session_test passed
```

两套构建均为 3/3 CTest 通过；gate ON 的同一 3/3 suite 还在 ASan+UBSan 配置下通过。

### 7.2 Gate ON 全量 CPU 回归

配置：

```bash
cmake -S . -B out/build/runtime-plan-dag -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_REGION_TASK_DAG=ON \
  -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/build/runtime-plan-dag -j2
```

通过的 20 个本机构建测试：

```text
object_test
packed_func_test
registry_test
type_registration_test
pass_pipeline_test
device_info_test
device_runtime_test
executable_plan_test
task_plan_test
task_executor_test
graph_partition_test
infer_type_test
profile_bundle_test
compiler_contract_test
operator_compilation_test
compiler_extension_contract_test
kernel_signature_test
compiled_module_test
runtime_session_test
cuda_schedule_test
```

契约/架构检查：

- Relay op contract：19/19；
- Pass contract：19/19；
- include-layer check：213 files；
- public-header self-compile：83 headers。

### 7.3 Gate OFF 回退

`out/build/runtime-plan-cpu` 以 `KXC_ENABLE_REGION_TASK_DAG=OFF` 构建；`task_plan_test`、`task_executor_test`、`runtime_session_test`、`executable_plan_test` 和 type registration 均通过。测试确认请求 `kTaskDAG` 时仍执行 per-Call，直接 frozen plan constructor 明确失败。

### 7.4 ASan + UBSan

配置：GCC 13.3，`-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`，gate ON、CPU-only。

以下测试在 `ASAN_OPTIONS=abort_on_error=1:detect_leaks=1`、`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` 下通过：

- `task_plan_test`；
- `task_executor_test`；
- `runtime_session_test`。

### 7.5 可重复测试与未运行矩阵

`task_plan_test`、`task_executor_test`、`runtime_session_test` 已注册为 CTest label `runtime-plan`，并可用 `run_runtime_plan_tests` 一次构建/执行。CI 的 CPU job 以 gate `OFF/ON` matrix 实际运行该 suite，不安装额外测试框架或软件包。

本 worktree 未启用 LLVM/CUDA，因此未运行 LLVM numeric、CUDA numeric、Compute Sanitizer 或真实 pending CUDA task-DAG retention。没有下载、安装、联网、push 或 merge。

## 8. 提交

| Commit | 内容 |
|---|---|
| `3508db7` | `feat(runtime): add frozen region task DAG contract` |
| `c18c880` | `feat(runtime): plan and execute task DAGs deterministically` |
| `c73442e` | `feat(runtime): execute frozen task plans behind gate` |
| `4092f5c` | `fix(runtime): harden task DAG execution contracts` |
| `57d83d8` | `docs(runtime): hand off region task DAG baseline` |
| 本次原子修复 | `fix(runtime): close task DAG merge blockers` |

## 9. 正确性与性能门禁

### 9.1 已满足的 W1 正确性门禁

- frozen DTO/validator 可脱离 Compiler 独立构造和反例测试；
- cycle、missing producer/dependency、错误 boundary、错误 stream/device、copy contract、region 内外 effect order、alias reuse 和 alignment fail closed；
- FrozenTaskPlan 拒绝任一 `-1`；adapter 在 gate ON/OFF 都保留 per-Call fallback 且不误报 task DAG；
- RuntimeSession 对 Fusion/Library/ControlFlow/ShapeEval 和非零预留 generation fail closed；
- deterministic fake 覆盖 kernel/copy/event/shape-eval/allocate/sync schema；
- dependency-aware memory 覆盖 linear、diamond、independent branches、async-live、conservative alias；
- per-Call 与 task-DAG multi-entry CPU fake 结果等价；
- gate OFF 行为不变；
- AsyncOperation 保留 module/plan/prior operations/current+retired storage；
- ASan/UBSan focused suite 通过。

### 9.2 打开默认 gate 前仍需满足

1. LLVM 与 CUDA 数值等价，包括 multi-entry、copy 和 storage reuse；
2. 真实 pending CUDA：session/module/input handle 提前释放、多个 pending kernel/copy、失败清理与 completion retention；
3. Compute Sanitizer 0 error；如引入并发 scheduler，再增加 TSan；
4. 生产 plan fingerprint、selected artifact generation manifest 与 profile 字段完整；
5. unsupported/fallback 原因可观测，不把 adapter bug 静默计为正常回退；
6. static per-Call correctness oracle 永久保留。

### 9.3 性能门禁

当前没有性能提升声明，gate 默认关闭。开启前至少记录并比较：

- per-Call 与 task-DAG 的 p50/p95 end-to-end latency；
- task validation/adapter/topological scheduling 开销；
- kernel/copy launch 数及 sync 数；
- unique-allocation reference 与 dependency planner 的 peak bytes；
- storage reuse 次数、retired storage bytes；
- CPU/LLVM/CUDA 分别报告，禁止只用 fake trace 宣称加速。

建议首个 release gate：task-DAG p95 不得比同一 static exact per-Call oracle 回退超过 5%，且任何 memory reduction 必须同时通过数值、ASan/UBSan/Compute Sanitizer 与 pending-retention 门禁。未达到时保持 gate OFF，不通过加入 RuntimeSession 策略绕过。

## 10. 跨轨接线

| 轨道 | 本轨提供 | 对方接入要求 |
|---|---|---|
| 01 Core | `RegionSpec/TaskSpec/FrozenTaskPlan v1`、unkeyed 与 `semantic_key` 预留字段 | 提供并接线规范 `UnitSemanticKey`、plan fingerprint、effect/alias contract 和 selected artifact manifest；当前测试 key 不得入 cache |
| 02 Shape | `ShapeEval` task schema与 deterministic fake hook | 提供 logical/physical/valid-extent、ShapeProgram 与 runtime output contract；完成前 RuntimeSession 继续拒绝 ShapeEval |
| 03 Hot swap | module/task plan由 completion 强持有；generation 仅为预留字段 | 提供 immutable generation -> artifact binding/pin；当前 RuntimeSession 只接受 generation 0，不自行查 slot/cache，也没有 generation retention 证据 |
| 04 Control flow | region kind 与通用 dependency vocabulary | 提供 Control task/CFG、Phi、branch/loop effect 与 liveness；当前 RuntimeSession 拒绝 ControlFlow |
| 06 NLP/GPU | stable DTO、fake trace、peak-live estimate、feature gate | 只按 gate/能力矩阵消费；不得通过 NLP op 名或 fixture 特判修改 executor |
| Compiler partition | private per-Call adapter 作为回退 oracle | region partition 产物必须先通过 boundary/effect/alias verifier；不可把 graph-local id/symbol 当 artifact semantic key |

跨轨只应 include `include/kxc/runtime/task_plan.h` 和 `task_executor.h`，不要 include `src/runtime/internal/*`。

## 11. 明确硬阻塞 / 后续工作

以下能力在当前依赖未冻结前有意不接入真实 RuntimeSession：

1. **dynamic shape / ShapeEval：** 缺 logical/physical/valid-extent 与 ShapeProgram；
2. **control flow：** 缺 Control task、Phi、backedge 和 branch/loop lifetime；
3. **fusion policy：** 缺 01 的生产 semantic key、完整 effect/alias、provenance 与 compiler region verifier；`semantic_key` 当前仅预留，RuntimeSession 拒绝 Fusion；
4. **library region：** 缺 descriptor、workspace、stream/error ABI；RuntimeSession 拒绝 Library，不以普通 kernel/copy 冒充；
5. **artifact generation：** 字段仅预留，缺 03 的 immutable selected-artifact manifest/binding/pin；
6. **multi stream：** 缺 event record/wait identity与真实 completion-aware retire；当前 Event 仅单 stream marker；
7. **multi device：** 缺显式跨 device copy/communication contract；physical device、VirtualDevice、worker id 尚未接线；
8. **生产 profiling：** 缺 region/task wait、launch、allocation、retire、generation 与 fallback reason 的稳定事件 schema；
9. **GPU 生命周期证据：** 缺本分支真实 pending CUDA task-DAG retention 与 Compute Sanitizer 结果。

这些是打开后续 gate 的硬门禁，不应通过在 `RuntimeSession` 中加入编译、预测、cache policy、后台线程，或通过 fuzzy/bigger-buffer fallback 绕过。

## 12. 交接结论

本轨已完成可独立测试的 frozen DTO/verifier/memory planner/deterministic fake executor，并以默认关闭的 feature gate 仅将 static-exact `PerCall` kernel/copy/event-marker/allocate/sync task 接入 `RuntimeSession`；旧 per-Call 路径和 AsyncOperation retention 保留。`semantic_key` 与 generation 仍仅为未接线的 DTO 预留字段，真实 pending GPU retention 未验证。其余工作均已收敛为上节的明确跨轨硬依赖，因此当前分支可作为 W1 contract baseline 交接，但不能标记为多 stream、多 device、dynamic shape、control flow、library ABI 或 production region fusion 完成。
