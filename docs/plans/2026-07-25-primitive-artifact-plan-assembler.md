# Primitive Artifact Reassembly Foundation Implementation Plan

> **For implementer:** REQUIRED SKILL: `executing-plans`，按任务顺序实现并在每个提交后验证。

**Goal:** 复用现有 compiler 中间结果，让普通编译、Shape exact 和 Adaptive 共享一个最小的 `AssembleCompiledGraph()`；Adaptive 只编译指定 primitive，并以完整不可变 `CompiledGraph` generation 发布。

**Architecture:** 不引入新的 template、selection、assembler class 或 policy hierarchy。继续使用现有 `PreparedCompilerGraph`、`CompiledPrimitiveBatch`、`PrimitiveArtifactPin` 和 `CompiledGraphAccess`；先删除 runtime module entry 中未使用的 TIR，再让完整 ordered pin 列表足以组装 module、plan 和不可变 `CompiledGraph`。Adaptive 当前只接受显式 replacement unit 请求；自动性能/Shape 策略等真实生产者出现后再接入同一个入口。

**Tech Stack:** C++17、CMake/Ninja、现有 Relay/TE/TIR pipeline、primitive cache、`CompiledModule`、`ExecutablePlan`、`CompiledGraph`、Adaptive Hot Swap v2。

---

## 0. 执行前提

- 从最新 `origin/compiler-foundation-acceptance-cleanup` 创建干净、独立的
  worktree。
- 不在当前含有其他人 header/documentation 修改的工作树中执行本计划。
- 每个任务只暂存任务列出的文件；禁止 `git add .`、`git add include` 或
  `git add src`。
- 不引入新依赖。

---

## 1. 最小目标模型

```text
Relay Function + CompileConfig
             │
             ▼
PrepareCompilerGraph                         // 复用现有类型/函数
             │
             └─ PreparedCompilerGraph
                    - GraphSemanticKey
                    - PreparedStaticGraph
                    - Target
                    - CompilerExecutionContract snapshot
                    - profiling diagnostics
             │
             ▼
CompilePrimitiveUnits(..., requested_ids)    // 全量或子集
             │
             └─ CompiledPrimitiveBatch
                    - unit_id
                    - PrimitiveArtifactPin
                    - optional compile diagnostics
                    - constants compiled by this batch
             │
             ▼
MergeOrderedPins(baseline, replacements)     // Adaptive 才需要
             │
             ▼
AssembleCompiledGraph(prepared, pins, constants)
             │
             ├─ CompiledModule
             ├─ ExecutablePlan
             └─ immutable CompiledGraph
                         │
                         ▼
              AdaptiveController publication
```

### 1.1 唯一新增生产入口

只新增一个 free function，不新增 class：

```cpp
CompiledGraph AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const std::vector<PrimitiveArtifactPin>& ordered_pins,
    const Map<String, runtime::NDArray>& constants);
```

它只负责：

1. 验证 pin 数量等于 prepared primitive 数量。
2. 按 vector 下标解释 `PrimitiveUnitId`，不排序、不建立第二套 slot id。
3. 验证每个 pin 的 unit semantic key 和 target 与 prepared graph 一致。
4. 从 pin 中读取 signature、launch metadata 和 executable。
5. 构造 `CompiledModule`。
6. 从现有 `PreparedStaticGraph` 构造 `ExecutablePlan`。
7. 通过现有 `CompiledGraphAccess::Create` 完成最终交叉校验和不可变发布。

`AssembleCompiledGraph()` 不运行 Relay pass、TE/TIR lowering、backend
codegen、cache lookup 或 Adaptive publication。

### 1.2 普通、Shape 和 Adaptive

```text
Normal:
  prepared = PrepareCompilerGraph(graph, config, contract)
  batch = CompilePrimitiveUnits(prepared.graph.units, all_ids)
  return AssembleCompiledGraph(prepared, Pins(batch), batch.constants)

Shape exact:
  prepared = existing frozen PreparedCompilerGraph
  batch = CompilePrimitiveUnits(prepared.graph.units, all_ids)
  graph = AssembleCompiledGraph(prepared, Pins(batch), batch.constants)
  return ExactPlanVariant(graph, shape_profile_key, plan_variant_key)

Adaptive:
  prepared = PrepareCompilerGraph(request.graph, request.config, contract)
  replacements = CompilePrimitiveUnits(..., requested_ids)
  pins = request.baseline_graph.artifact_pins converted to ordered internal pins
  replace pins[unit_id] for every replacement
  candidate = AssembleCompiledGraph(
      prepared, pins, request.baseline_graph constants)
  if candidate ordered artifact keys == current ordered artifact keys:
      return current lease
  ValidateSamePlanAbi(current, candidate)
  Publish(candidate)
```

第一版允许 Adaptive request 重新执行 graph preparation；backend 只编译指定
unit。只有测量证明 graph preparation 成为实际瓶颈时，才在 route 中缓存
`PreparedCompilerGraph`。

### 1.3 重编译意图

当前唯一入口仍是：

```cpp
AdaptiveHotSwapController::Submit(CompileRequest)
```

调用方显式提供 replacement unit ids 和已有 `CompileConfig`。Controller
负责 admission、singleflight、取消、ABI 验证和 publication。

本轮不新增：

- `AdaptiveObservation`
- `RecompileIntent`
- `RecompilePolicy`
- `PrimitiveVariantCompiler`
- `AdaptiveControllerAccess`

未来 Shape guard、profiler 或 autotuner 需要自动触发时，只调用同一个
`Submit()`，不能直接取得 primitive compiler 或 publication authority。

### 1.4 对“新 kernel”的诚实约束

- 不允许只修改 identity 字符串来伪造新 artifact。
- replacement config 必须实际解析出不同的 pipeline/schedule/backend
  contract，才能形成不同 `PrimitiveArtifactKey`。
- 如果重编译命中相同 artifact key，Controller 返回当前 generation，不发布
  空洞的新 generation。
- 当前没有真实 autotuner、bucket Shape 或 schedule 搜索；这些不是本计划的
  验收前提。

### 1.5 控制流边界

`CompileControlFlowExact()` 继续使用 `CompilePrimitiveUnits()` 和
`AssemblePrimitiveModule()`，因为它发布的是 `ControlExecutionPlan`，不是
静态 `ExecutablePlan`。

因此：

- primitive 编译和 module entry 构造是所有拓扑共享的。
- `AssembleCompiledGraph()` 只统一 Normal、Shape exact 和 Adaptive 的静态
  `CompiledGraph` 发布。
- 不为了“名字上唯一”强迫控制流经过错误的 plan typestate。

### 1.6 非目标

- mutable `KernelSlot` 或运行中原地改函数指针。
- profiler/Shape 自动触发策略。
- autotuning、schedule 搜索或收益预测。
- bucket/polymorphic/dynamic-output Shape。
- route 级 prepared graph cache。
- 新 public fake、`ForTesting()` 或只有一个实现的 virtual interface。
- 将控制流强行包装成静态 `CompiledGraph`。

---

## 2. 验收不变量

1. 不新增 `PreparedPlanTemplate`、`PreparedCompilerProgram`、
   `PrimitiveArtifactSelection` 或 `PlanAssembler` class。
2. `PrimitiveArtifactPin` 是 signature、launch metadata、executable 和
   生命周期的唯一权威。
3. ordered pin vector 的下标就是 `PrimitiveUnitId`；缺失、重复映射和排序
   层不存在。
4. 普通编译和 Shape exact 调用同一个 `AssembleCompiledGraph()`。
5. Adaptive worker 不调用完整 `Compiler::Compile()`。
6. Adaptive backend compilation 只访问 requested unit ids。
7. 未替换 pin 的 owner 跨 generation 保持相同。
8. replacement 可以改变 artifact key 和 executable，但不得改变 Plan ABI。
9. 相同 ordered artifact keys 不增加 generation。
10. 新请求只在请求边界选择 generation；kernel launch loop 不增加 lookup。
11. 旧 generation lease 在新 generation 发布后仍可执行。
12. 控制流继续使用其正确的 resolved-plan assembler。

---

### Task 1: 删除 runtime module entry 中未使用的 TIR

**Files:**

- Modify: `src/runtime/internal/compiled_module_node.h`
- Modify: `src/compiler/primitive_compiler.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/control_flow/production_control_flow.cc`
- Modify: `test/compiled_module_test.cpp`
- Modify: `test/runtime_session_test.cpp`
- Modify: `test/control_runtime_integration_test.cpp`
- Modify: `test/compiler_identity_test.cpp`
- Modify: `test/adaptive_preparation_v2_test.cpp`

**Step 1: 锁定 runtime 行为**

在 `test/compiled_module_test.cpp` 使用现有 signature、metadata 和 executable
构造 module entry，并执行一次 module invocation。测试不得提供 TIR：

```cpp
internal::CompiledModuleEntry entry{
    signature,
    metadata,
    executable,
    nullptr,
};
```

**Step 2: 运行编译确认失败**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target compiled_module_test -j 4
```

Expected: FAIL，因为当前 aggregate 的第一个字段仍是 `tir::PrimFunc`。

**Step 3: 删除字段**

从 `CompiledModuleEntry` 删除：

```cpp
tir::PrimFunc prim_func;
```

更新所有 aggregate initializer。不要增加替代 diagnostic 字段；TIR 继续留在
compiler lowering/diagnostic 对象中，不进入 runtime module。

`invocation_contract` 保持不变；为空时继续由
`BuildCompiledModule()` 调用现有 `StaticContract(signature)`。

**Step 4: 运行相关测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiled_module_test runtime_session_test control_runtime_integration_test `
  compiler_identity_test adaptive_hot_swap_v2_test -j 4
```

Expected: 全部构建并 PASS。

**Step 5: Commit**

```bash
git add src/runtime/internal/compiled_module_node.h src/compiler/primitive_compiler.cc src/compiler/compiler.cc src/compiler/control_flow/production_control_flow.cc test/compiled_module_test.cpp test/runtime_session_test.cpp test/control_runtime_integration_test.cpp test/compiler_identity_test.cpp test/adaptive_preparation_v2_test.cpp
git commit -m "runtime: remove unused TIR from compiled module entries"
```

---

### Task 2: 让现有 primitive compiler 支持子集

**Files:**

- Modify: `src/compiler/internal/primitive_compiler.h`
- Modify: `src/compiler/primitive_compiler.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/control_flow/production_control_flow.cc`
- Test: `test/operator_compilation_test.cpp`
- Test: `test/primitive_cache_test.cpp`

**Step 1: 写失败测试**

在 `test/operator_compilation_test.cpp` 准备两个 primitive units，只请求第二个：

```cpp
const CompiledPrimitiveBatch batch = CompilePrimitiveUnits(
    units, values, config, contract, {PrimitiveUnitId{1}});

CHECK(batch.primitives.size() == 1);
CHECK(batch.primitives[0].unit_id == 1);
```

再验证重复和越界请求失败：

```cpp
CHECK_THROWS(CompilePrimitiveUnits(
    units, values, config, contract, {1, 1}));
CHECK_THROWS(CompilePrimitiveUnits(
    units, values, config, contract, {2}));
```

**Step 2: 运行测试确认失败**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target operator_compilation_test -j 4
```

Expected: FAIL，因为当前接口只支持 dense 全量编译。

**Step 3: 实现最小 overload**

保留现有全量入口，并增加：

```cpp
CompiledPrimitiveBatch CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract,
    const std::vector<PrimitiveUnitId>& requested_unit_ids);
```

行为：

- 先验证完整 `units` 仍然 dense ordered。
- `requested_unit_ids` 必须严格递增、无重复且全部有效。
- 只 lower、optimize、cache lookup 和 backend compile 请求的 units。
- 原全量 overload 生成 `[0, unit_count)` 并调用新 overload。
- 不新增 selection wrapper。

**Step 4: 收窄 `CompiledPrimitive`**

删除能从 unit 或 `PrimitiveArtifactPin` 派生的最终权威副本：

```text
symbol
semantic_key
artifact_key
signature
launch_metadata
kernel
public ArtifactPin
```

保留：

```cpp
struct CompiledPrimitive final {
    PrimitiveUnitId unit_id{-1};
    tir::PrimFunc diagnostic_tir;
    PrimitiveArtifactPin pin;
    bool cache_hit{false};
};
```

调用方需要 ABI/executable 时读取 `primitive.pin.artifact()`；需要公共 pin 时
调用现有 `ArtifactPinAccess::Wrap()`。

**Step 5: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  operator_compilation_test primitive_cache_test compiler_contract_test `
  control_runtime_integration_test -j 4
```

Expected: 全部 PASS；subset case 只增加一个 primitive cache miss。

**Step 6: Commit**

```bash
git add src/compiler/internal/primitive_compiler.h src/compiler/primitive_compiler.cc src/compiler/compiler.cc src/compiler/control_flow/production_control_flow.cc test/operator_compilation_test.cpp test/primitive_cache_test.cpp
git commit -m "compiler: compile requested primitive units only"
```

---

### Task 3: 用一个 free function 组装静态 CompiledGraph

**Files:**

- Modify: `src/compiler/internal/execution_contract.h`
- Modify: `src/compiler/compiler.cc`
- Test: `test/compiler_contract_test.cpp`

**Step 1: 写失败测试**

在 `test/compiler_contract_test.cpp` 使用现有 preparation 和 primitive compiler：

```cpp
const PreparedCompilerGraph prepared =
    PrepareCompilerGraph(graph, config, contract);
const CompiledPrimitiveBatch batch = CompilePrimitiveUnits(
    prepared.graph.partitioned.units,
    prepared.graph.partitioned.value_graph.values,
    config, contract);

const CompiledGraph compiled = AssembleCompiledGraph(
    prepared, Pins(batch), batch.constants);

CHECK(compiled.plan().calls().size() == batch.primitives.size());
CHECK(compiled.artifact_pins().size() == batch.primitives.size());
```

同时验证：

- 少一个 pin 会失败。
- 交换两个 pin 会因 semantic key/symbol 不匹配失败。
- 正常结果可以由 `RuntimeSession` 执行。

**Step 2: 运行测试确认失败**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target compiler_contract_test -j 4
```

Expected: FAIL，因为 `AssembleCompiledGraph()` 尚不存在。

**Step 3: 实现 free function**

在现有 `execution_contract.h` 声明，在 `compiler.cc` 实现：

```cpp
CompiledGraph AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const std::vector<PrimitiveArtifactPin>& ordered_pins,
    const Map<String, runtime::NDArray>& constants);
```

实现规则：

- 不排序；vector 下标必须对应同 id 的 prepared unit。
- 验证 pin defined、unit semantic key 和 target fingerprint。
- 每个 module entry 直接取 `pin.artifact()` 的 signature、metadata、kernel。
- 调用现有 `BuildCompiledModule()`。
- 调用现有 `BuildStaticExecutablePlan(prepared.graph)`。
- 用 `ArtifactPinAccess::Wrap()` 生成 public pins。
- 最后调用现有 `CompiledGraphAccess::Create()`；不复制其中已有 ABI 校验。

**Step 4: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test runtime_session_test primitive_cache_test -j 4
out/build/dev-mingw-adaptive/compiler_contract_test.exe
out/build/dev-mingw-adaptive/runtime_session_test.exe
out/build/dev-mingw-adaptive/primitive_cache_test.exe
```

Expected: 全部 PASS。

**Step 5: Commit**

```bash
git add src/compiler/internal/execution_contract.h src/compiler/compiler.cc test/compiler_contract_test.cpp
git commit -m "compiler: assemble static graphs from ordered artifact pins"
```

---

### Task 4: 迁移普通编译与 Shape exact

**Files:**

- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/internal/execution_contract.h`
- Modify: `src/compiler/shape_exact.cc`
- Test: `test/compiler_contract_test.cpp`
- Test: `test/shape_production_exact_test.cpp`
- Test: `test/shape_specialization_test.cpp`

**Step 1: 写跨路径测试**

对同一 concrete graph：

```cpp
const CompiledGraph normal = Compiler::Compile(graph, config);
const ExactPlanVariant exact = AssembleExact(graph, config);

CHECK(normal.graph_semantic_key() ==
      exact.plan_variant_key().graph_semantic_key());
CHECK(SameOrderedArtifactKeys(normal, exact));
CHECK(SamePlanAbi(normal, exact));
```

**Step 2: 迁移 `Compiler::Compile`**

普通编译改为：

```cpp
const auto contract = ResolveCompilerExecutionContract(config);
const auto prepared = PrepareCompilerGraph(graph, config, contract);
const auto batch = CompilePrimitiveUnits(
    prepared.graph.partitioned.units,
    prepared.graph.partitioned.value_graph.values,
    config, contract);
return AssembleCompiledGraph(prepared, Pins(batch), batch.constants);
```

保留现有 profiling span；不要再让 `CompileResult` 成为 module/pins 发布权威。

**Step 3: 迁移 Shape exact**

`ProductionExactShapeAdapter::AssembleExactPlan()` 对已经冻结的
`PreparedCompilerGraph`：

1. 编译全部 units。
2. 调用同一个 `AssembleCompiledGraph()`。
3. 保留现有 Shape profile 和 `PlanVariantKey` 验证。

**Step 4: 删除旧静态双轨**

零引用后删除：

- `FinishCompilerGraph`
- `CompilePreparedPrimitiveUnits` 的最终 artifact/module 搬运职责
- `AssembleModule`

`AssemblePrimitiveModule()` 继续供控制流使用。

**Step 5: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test compiler_identity_test runtime_session_test `
  shape_system_test shape_specialization_test shape_production_exact_test `
  control_runtime_integration_test -j 4
```

逐个运行，Expected: 全部 PASS。

**Step 6: Commit**

```bash
git add src/compiler/compiler.cc src/compiler/internal/execution_contract.h src/compiler/shape_exact.cc test/compiler_contract_test.cpp test/shape_production_exact_test.cpp test/shape_specialization_test.cpp
git commit -m "compiler: share static graph assembly with exact shape"
```

---

### Task 5: Adaptive 只编译并替换指定 units

**Files:**

- Modify: `include/kxc/compiler/adaptive_production_experimental.h`
- Modify: `src/compiler/adaptive/production_path_experimental.cc`
- Modify: `include/kxc/compiler/adaptive_hot_swap_v2.h`
- Modify: `src/compiler/adaptive/adaptive_hot_swap_v2.cc`
- Test: `test/adaptive_preparation_v2_test.cpp`

**Step 1: 写生产路径失败测试**

使用两 primitive graph：

```cpp
const auto baseline = controller.CompileAndPublish(
    MakeInitialRequest(graph, baseline_config));
const auto candidate = controller.CompileAndPublish(
    MakeReplacementRequest(
        graph, replacement_config, {PrimitiveUnitId{1}}));

CHECK(SamePinOwner(baseline, candidate, 0));
CHECK(!SameArtifactKey(baseline, candidate, 1));
CHECK(baseline->plan_abi() == candidate->plan_abi());
CHECK(Execute(*baseline) == expected);
CHECK(Execute(*candidate) == expected);
```

`replacement_config` 必须使用现有、实际不同的 normalized compiler pipeline；
不得仅改 diagnostic/policy 字符串。

再增加同 key no-op：

```cpp
const auto same = controller.CompileAndPublish(
    MakeReplacementRequest(
        graph, baseline_config, {PrimitiveUnitId{1}}));
CHECK(same->generation() == baseline->generation());
```

**Step 2: 重塑现有 request，避免新 wrapper hierarchy**

让现有 production request 持有一个不可变 baseline graph，并增加：

```cpp
const CompiledGraph& baseline_graph() const noexcept;
const std::vector<PrimitiveUnitId>& requested_unit_ids() const noexcept;
```

构造时规范化并验证 ids。删除 request 中单独保存的
`verified_artifact_pins_`；pins 和 constants 都从 immutable
`baseline_graph_` 读取，避免一份 request 重复持有两套 retention authority。
仅在 route/ABI identity 确实需要时保留轻量的 ordered identity snapshot。

继续使用现有 graph、config 和 baseline `CompiledGraph`；不新增
`PreparedAdaptiveProgram`、
`PrimitiveVariantCompiler` 或 policy interface。

**Step 3: 改写 worker**

替换当前 whole-graph：

```cpp
Compiler::Compile(request.graph(), request.config())
```

为：

1. `PrepareCompilerGraph()`。
2. `CompilePrimitiveUnits(..., requested_unit_ids)`。
3. 从 request 的 baseline `CompiledGraph::artifact_pins()` 得到完整 ordered
   internal pins。
4. 按 `unit_id` 替换对应 pin。
5. constants 使用 baseline module 的现有 internal immutable borrow。
6. 调用 `AssembleCompiledGraph()`。
7. ordered artifact keys 未变化时返回当前 lease。
8. 否则执行现有 candidate validation 和 publication transaction。

不得改变：

- admission/singleflight
- final cancellation/deadline observation
- route key 和 Plan ABI 检查
- generation authority
- quarantine tombstone
- old lease retention
- transactional route swap

**Step 4: 删除 whole-graph production adapter**

生产 worker 零引用后删除 `ProductionPathCompilerAdapter` 及其 public constructor
注入。Controller 测试改用真实小图和已有 cancellation/failure 输入，不增加新的
public fake。

**Step 5: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  adaptive_hot_swap_v2_test compiler_contract_test primitive_cache_test `
  runtime_session_test -j 4
```

逐个运行，Expected:

- replacement config 只增加一个 primitive cache miss。
- unit 0 pin owner 不变。
- unit 1 artifact key 改变。
- old/new lease 均可执行。
- same-key request 不增加 generation。
- production worker 无 `Compiler::Compile()` 调用。

**Step 6: Commit**

```bash
git add include/kxc/compiler/adaptive_production_experimental.h src/compiler/adaptive/production_path_experimental.cc include/kxc/compiler/adaptive_hot_swap_v2.h src/compiler/adaptive/adaptive_hot_swap_v2.cc test/adaptive_preparation_v2_test.cpp
git commit -m "compiler: reassemble adaptive graphs from primitive replacements"
```

---

### Task 6: 删除旧词汇并完成验收

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `tools/architecture/check_public_headers.py`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`
- Modify: `docs/DYNAMIC_COMPILED_MODULE_ABI.md`

**Step 1: 搜索禁止残留**

Run:

```powershell
rg -n "FinishCompilerGraph|AssembleModule|ProductionPathCompilerAdapter|PreparedPlanTemplate|PreparedCompilerProgram|PrimitiveArtifactSelection|class PlanAssembler|RecompilePolicy|PrimitiveVariantCompiler|compiler->Compile\\(f->request\\)" CMakeLists.txt include src test docs
```

Expected:

- 生产代码零引用。
- 本计划的非目标说明可以出现。
- `AssemblePrimitiveModule` 只保留在 primitive compiler 和控制流路径。

**Step 2: 更新架构文档**

文档明确：

- primitive unit 是 backend 重编译粒度。
- ordered pin vector 是完整 artifact selection。
- `CompiledGraph` generation 是发布粒度。
- Normal、Shape exact、Adaptive 使用同一个 free assembler。
- 控制流保留正确的 resolved-plan assembler。
- 当前 trigger 是显式 Submit；自动策略尚未实现。

**Step 3: 本地启用态回归**

Run:

```powershell
cmake -S . -B out/build/dev-mingw-adaptive -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug `
  -DKXC_ENABLE_LLVM=OFF `
  -DKXC_ENABLE_CUDA=OFF `
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON `
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON `
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON `
  -DKXC_BUILD_PASS_TESTS=ON `
  -DKXC_BUILD_CODEGEN_TESTS=OFF

cmake --build out/build/dev-mingw-adaptive --target `
  compiled_module_test `
  compiler_contract_test `
  compiler_identity_test `
  operator_compilation_test `
  primitive_cache_test `
  shape_system_test `
  shape_specialization_test `
  shape_production_exact_test `
  restricted_symbolic_shape_test `
  adaptive_hot_swap_v2_test `
  runtime_session_test `
  control_runtime_integration_test `
  check_include_layers `
  check_public_headers `
  -j 4
```

逐个运行二进制，Expected: 全部 PASS。

**Step 4: 合同和 diff 检查**

Run:

```powershell
python python/tools/check_relay_op_contract.py --root . --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . --matrix contracts/pass_contract.json
git diff --check
```

Expected: 合同检查和 diff check 全部 PASS。

**Step 5: Omen 验收**

在确认 Omen worktree 干净后，按 `AGENTS.md` 的 CUDA 12.9 配置构建上述测试。
另外使用 ASan/UBSan 构建并运行：

```text
object_test
primitive_cache_test
adaptive_hot_swap_v2_test
```

Expected: 无 sanitizer 报告；CUDA 环境缺口必须单独记录。

**Step 6: Commit**

```bash
git add CMakeLists.txt tools/architecture/check_public_headers.py docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md docs/DYNAMIC_COMPILED_MODULE_ABI.md
git commit -m "docs: record minimal primitive reassembly architecture"
```

---

## 3. 完成定义

- [ ] 未新增 speculative template/selection/policy/interface hierarchy。
- [ ] runtime `CompiledModuleEntry` 不再保存未使用的 TIR。
- [ ] primitive compiler 支持严格验证的 requested unit ids。
- [ ] pin 是最终 ABI/executable/lifetime 唯一权威。
- [ ] Normal 和 Shape exact 共用 `AssembleCompiledGraph()`。
- [ ] Adaptive worker 不调用完整 `Compiler::Compile()`。
- [ ] Adaptive 只 backend 编译 requested units。
- [ ] unchanged pin owner 跨 generation 复用。
- [ ] same-key request 不增加 generation。
- [ ] old/new lease 均可执行。
- [ ] runtime launch hot path 没有 mutable slot lookup。
- [ ] 控制流 typestate 未被错误合并。
- [ ] 本地、架构门禁、合同、Omen 和 sanitizer 验收有完整记录。
