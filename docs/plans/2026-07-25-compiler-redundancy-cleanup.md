# Compiler Redundancy Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 删除当前 Compiler 中重复保存、事后重放和仅供测试使用的生产层抽象，使普通静态编译只有一条从 prepared graph 到 primitive artifacts 再到 compiled graph 的权威路径。

**Architecture:** 先完成现有 `2026-07-25-primitive-artifact-plan-assembler.md`，让普通编译、Shape exact 和 Adaptive 共用 primitive artifact 组装基础；本计划不复制其中任务。随后按“删除优先”顺序移除 `CompileResult` 阶段重放、test-only whole-graph lowering、无生产消费者的 capability facade、重复 Relay preflight、空 plan wrapper 和未闭环 multi-device compiler surface。最后把 `CompileConfig` 变成创建后不可变的 target/config snapshot，消除 defensive clone 和重复校验的根因。

**Tech Stack:** C++17、CMake/Ninja、Relay/TE/TIR、现有 primitive cache、`CompiledModule`、`ExecutablePlan`、Windows MinGW CPU 验证与 Omen CUDA 12.9 验证。

---

## 0. 执行前提

- 从最新 `origin/compiler-foundation-acceptance-cleanup` 创建干净、独立的 worktree。
- 不在带有 header 注释、文档草稿或其他未提交修改的工作树中执行。
- 先完整执行：
  `docs/plans/2026-07-25-primitive-artifact-plan-assembler.md`。
- 每个任务只暂存任务列出的文件；禁止 `git add .`、`git add include`、
  `git add src`。
- 不引入新依赖、新 factory、新 interface、新 policy hierarchy 或 compatibility
  shim。
- 删除 public API 前确认本分支仍处于允许 breaking cleanup 的阶段；若已经发布
  source/ABI compatibility promise，停止对应删除任务并单独制定迁移版本。

## 1. 目标主链

完成后，普通静态编译只有以下权威链：

```text
Function + immutable CompileConfig
              |
              v
PrepareRelayProgram
              |
              v
BuildValueGraph                 // 一次遍历完成结构验证和 ResolveRelayCall
              |
              v
PartitionValueGraph
              |
              v
CompilePrimitiveUnits           // lowering/TIR/signature/cache/backend
              |
              v
AssembleCompiledGraph           // module + static plan + pins
              |
              v
immutable CompiledGraph
```

控制流继续使用独立 topology：

```text
PreparedRelayProgram
    -> LowerPreparedRelayToControlPlanWithSidecar
    -> CompilePrimitiveUnits
    -> BindControlPlanForRuntime
    -> CompiledControlFlowGraph
```

必须保留：

- `LogicalValueContract`、`ValueGraph`、`PrimitiveUnit` 的分层语义；
- `PrimitiveCacheLease` 的 singleflight、failure、backpressure 和 owner guard；
- public `ArtifactPin` 与 internal `PrimitiveArtifactPin` 的所有权边界；
- unresolved compiler `ControlPlan` 与 bound runtime `ControlExecutionPlan`；
- runtime ABI、dtype、shape、alignment、constant 和 lifetime 校验；
- 静态执行与控制流执行的 runtime hot-path 分离。

本计划不实现：

- dynamic shape、bucket、polymorphic output；
- adaptive 自动重编译策略；
- multi-device kernel execution；
- 新 pipeline DSL；
- 为已删除类型增加 deprecated alias。

---

### Task 1: 完成 primitive artifact assembler 前置计划

**Files:**

- Follow: `docs/plans/2026-07-25-primitive-artifact-plan-assembler.md`

**Step 1: 在独立 worktree 执行前置计划**

按前置计划 Task 1 到 Task 6 顺序完成，不跳过中间测试和 commit。

**Step 2: 验证静态编译已不再通过旧 assembler 发布**

Run:

```powershell
rg -n "FinishCompilerGraph|CompilePreparedPrimitiveUnits|AssembleModule" `
  src/compiler include/kxc/compiler test
```

Expected:

- `FinishCompilerGraph`、旧 `AssembleModule` 和旧 artifact 搬运路径零生产引用；
- `AssemblePrimitiveModule` 只用于控制流；
- 普通编译和 Shape exact 使用同一个 `AssembleCompiledGraph`。

**Step 3: 验证前置测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test operator_compilation_test primitive_cache_test `
  shape_production_exact_test adaptive_hot_swap_v2_test `
  control_runtime_integration_test check_public_headers -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(compiler_contract_test|operator_compilation_test|primitive_cache_test|shape_production_exact_test|adaptive_hot_swap_v2_test|control_runtime_integration_test)$"
```

Expected: 全部 target 构建并运行通过。

**Step 4: 确认工作树干净**

Run:

```powershell
git status --short
```

Expected: 无未提交修改后再进入 Task 2。

---

### Task 2: 删除事后重放的 CompileResult 状态机

`CompilePrimitiveUnits` 已一次性产生完整 backend 结果；不得再把结果拆成多组
vector 后重放 `kLowered -> kTIROptimized -> kSignatureBuilt ->
kBackendCompiled`。

**Files:**

- Delete: `src/compiler/internal/compile_state.h`
- Delete: `src/compiler/compile_state.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/internal/execution_contract.h`
- Modify: `CMakeLists.txt`
- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/type_registration_test.cpp`
- Modify: `docs/DEVICE_MODEL.md`
- Modify: `docs/ARCHITECTURE_STATUS.md`
- Modify: `docs/REPO_IMPLEMENTATION_OVERVIEW.md`
- Modify: `docs/COMPILER_FOUNDATION_BASELINE_REVIEW.md`

**Step 1: 记录删除前引用**

Run:

```powershell
rg -n "CompileResultNode|PrimitiveCompileState|CompileStage|AfterLowering|AfterTIROptimization|AfterSignatures|AfterBackends" `
  CMakeLists.txt src include test docs
```

Expected: 命中旧状态机实现、compiler 搬运逻辑、focused tests 和旧文档。

**Step 2: 先把生产发布改成只消费 batch**

在 `src/compiler/compiler.cc` 中：

- profiling 字段直接从 `PreparedCompilerGraph` 和
  `CompiledPrimitiveBatch` 读取；
- 不再构造 `PrimitiveCompileState`；
- 不再复制 optimized TIR、signature、metadata、kernel、cache-hit 和 pin
  vector；
- 不再调用任何 `After*` transition；
- module/plan/pin 发布只经过前置计划提供的 `AssembleCompiledGraph`。

不要新增替代状态机。需要诊断时使用一个局部循环读取
`batch.primitives`。

**Step 3: 缩减 PreparedCompilerGraph**

从 `PreparedCompilerGraph` 删除 `CompileResult optimized`。

Relay Function 的唯一来源改为：

```cpp
prepared.graph.partitioned.value_graph.function
```

Target、pipeline fingerprint 和 graph identity 继续由 prepared graph /
execution contract 提供，不增加第二个 holder。

**Step 4: 删除状态机 focused test**

从 `test/compiler_contract_test.cpp` 删除只验证手工调用 `After*` 的
`TestMultiPrimitiveCompileStateIdentity`。

把其中仍有价值的断言迁移到真实 `Compiler::Compile` 测试：

- plan call 数量等于 primitive 数量；
- pin 数量等于 primitive 数量；
- symbol、unit id 和 artifact key 顺序一致；
- cache-hit 后仍返回完整可执行 graph。

**Step 5: 删除 Object type registration**

从 `test/type_registration_test.cpp` 删除
`"kxc.api.CompileResultNode"` 断言。

**Step 6: 删除实现和构建项**

删除 `compile_state.h/.cc`，并从 `CMakeLists.txt` 移除
`src/compiler/compile_state.cc`。

**Step 7: 更新架构文档**

删除“CompileResult 状态机不可删除”的旧结论。文档改为：

- preparation、primitive compilation 和 final assembly 是真实阶段；
- primitive compiler 内部 phase 通过调用栈和错误上下文表达；
- 完整 backend batch 返回后不再伪造可观察中间状态。

**Step 8: 验证删除**

Run:

```powershell
rg -n "CompileResultNode|PrimitiveCompileState|enum class CompileStage|AfterRelayOptimization|AfterLowering|AfterTIROptimization|AfterSignatures|AfterBackends" `
  CMakeLists.txt src include test
```

Expected: 零命中。Adaptive v2 自己的 `v2::CompileResult` 名称允许保留，但必须
带 namespace，且不能依赖已删除 compiler state。

**Step 9: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test compiler_identity_test operator_compilation_test `
  primitive_cache_test shape_production_exact_test `
  control_runtime_integration_test type_registration_test -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(compiler_contract_test|compiler_identity_test|operator_compilation_test|primitive_cache_test|shape_production_exact_test|control_runtime_integration_test|type_registration_test)$"
```

逐个运行，Expected: 全部 PASS。

**Step 10: Commit**

```bash
git add CMakeLists.txt src/compiler/compiler.cc src/compiler/internal/execution_contract.h test/compiler_contract_test.cpp test/type_registration_test.cpp docs/DEVICE_MODEL.md docs/ARCHITECTURE_STATUS.md docs/REPO_IMPLEMENTATION_OVERVIEW.md docs/COMPILER_FOUNDATION_BASELINE_REVIEW.md
git add -u src/compiler/internal/compile_state.h src/compiler/compile_state.cc
git commit -m "refactor: remove replayed compiler state machine"
```

---

### Task 3: 把 whole-graph lowering fixture 移出生产模型

生产编译只需要 `PrepareStaticGraph`、`LowerPrimitiveUnit` 和
`BuildStaticExecutablePlan`。`LoweredGraph`、`LowerGraph` 和
`LowerPreparedStaticGraph` 当前只由测试消费。

**Files:**

- Modify: `src/compiler/internal/lowered_graph.h`
- Modify: `src/compiler/lowering/lowered_graph.cc`
- Modify: `test/support/primitive_lowering.h`
- Modify: `test/operator_compilation_test.cpp`
- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/executable_capability_test.cpp`
- Modify: `test/shape_production_exact_test.cpp`
- Modify: `docs/REPO_IMPLEMENTATION_OVERVIEW.md`
- Modify: `docs/COMPILER_FOUNDATION_BASELINE_REVIEW.md`

**Step 1: 将测试 helper 改为明确的 fixture**

在 `test/support/primitive_lowering.h` 定义 test-only aggregate：

```cpp
struct PrimitiveLoweringFixture final {
    api::internal::PreparedStaticGraph prepared;
    std::vector<relay::LoweredFunction> lowered;
};
```

提供一个 test-only free function：

```cpp
PrimitiveLoweringFixture LowerPrimitivesForTest(Function function);
```

实现只组合现有生产 building blocks：

1. `PrepareStaticGraph`；
2. 遍历 `prepared.partitioned.units`；
3. 对每个 unit 调用 `LowerPrimitiveUnit`；
4. 返回 test fixture。

不把 fixture 放回 `src/compiler`。

**Step 2: 迁移 operator lowering tests**

把 `test/operator_compilation_test.cpp` 中的 `LoweredGraph` 使用改成：

- 需要 graph/unit 信息时读取 `fixture.prepared.partitioned`；
- 需要 PrimFunc/constant 时读取 `fixture.lowered[index]`；
- 不复制 production plan 或 constants aggregate。

**Step 3: 迁移 capability tests**

失败路径分别使用最靠近真实边界的入口：

- static graph eligibility：`PrepareStaticGraph`；
- primitive lowering：`LowerPrimitiveUnit`；
- end-to-end executable proof：`Compiler::Compile`。

不要保留一个新的 whole-graph test adapter。

**Step 4: 迁移 shape test**

`test/shape_production_exact_test.cpp` 如需检查 frozen prepared graph，只遍历
`PreparedStaticGraph.partitioned.units` 并逐 unit 调用
`LowerPrimitiveUnit`，不构造 `LoweredGraph`。

**Step 5: 删除生产 aggregate 和入口**

删除：

- `LoweredPrimitive`
- `LoweredGraph`
- `LowerPreparedStaticGraph`
- `LowerGraph`
- `ValidateLoweredGraph`

保留：

- `LowerPrimitiveUnit`
- `PrepareStaticGraph`
- `BuildStaticExecutablePlan`

**Step 6: 搜索验证**

Run:

```powershell
rg -n "LoweredGraph|LoweredPrimitive|LowerPreparedStaticGraph|LowerGraph\\(|ValidateLoweredGraph" `
  src include test
```

Expected: 生产代码零命中；test-only fixture 使用自己的
`PrimitiveLoweringFixture` 名称。

**Step 7: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  operator_compilation_test compiler_contract_test `
  executable_capability_test shape_production_exact_test -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(operator_compilation_test|compiler_contract_test|executable_capability_test|shape_production_exact_test)$"
```

逐个运行，Expected: 全部 PASS。

**Step 8: Commit**

```bash
git add src/compiler/internal/lowered_graph.h src/compiler/lowering/lowered_graph.cc test/support/primitive_lowering.h test/operator_compilation_test.cpp test/compiler_contract_test.cpp test/executable_capability_test.cpp test/shape_production_exact_test.cpp docs/REPO_IMPLEMENTATION_OVERVIEW.md docs/COMPILER_FOUNDATION_BASELINE_REVIEW.md
git commit -m "refactor: keep whole graph lowering fixtures in tests"
```

---

### Task 4: 删除 PreparedProgramPlan 空 wrapper

`PreparedStaticPlan` 和 `PreparedControlPlan` 都只复制
`PreparedRelayProgram::typed_anf()`；两个生产调用者只使用
`std::holds_alternative`。

**Files:**

- Modify: `src/compiler/internal/relay_program.h`
- Modify: `src/compiler/analysis/relay_program.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/control_flow/production_control_flow.cc`
- Modify: `test/executable_capability_test.cpp`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`

**Step 1: 锁定 topology 选择行为**

保留以下测试场景：

- ordinary dataflow：`requires_control_topology() == false`；
- constant-folded `If`：`false`；
- residual `If`：`true`；
- residual bounded `While`：`true`。

测试直接读取 `prepared.residual_profile().requires_control_topology()`，不再
构造 plan wrapper。

**Step 2: 删除 wrapper 和函数**

从 `relay_program.h/.cc` 删除：

- `PreparedStaticPlan`
- `PreparedControlPlan`
- `PreparedProgramPlan`
- `PlanRelayProgram`

不新增 `RelayPlanKind`，因为 capability bitset 已经是唯一权威。

**Step 3: 修改两个生产入口**

普通编译：

```cpp
if (prepared.residual_profile().requires_control_topology()) {
    throw std::logic_error(
        "Compiler::Compile cannot publish a structured-control plan");
}
```

控制流编译：

```cpp
if (!prepared.residual_profile().requires_control_topology()) {
    Fail("has no residual control topology after preparation; use "
         "Compiler::Compile for the static fast path");
}
```

**Step 4: 更新架构文档**

文档明确：

- topology 仍然只在 compiler preparation 后选择一次；
- choice 从 residual capability set 的 `any()` 派生；
- 不保存第二个 bool、enum、variant 或 Function wrapper；
- runtime hot path 不重新分析 topology。

**Step 5: 验证**

Run:

```powershell
rg -n "PreparedStaticPlan|PreparedControlPlan|PreparedProgramPlan|PlanRelayProgram" `
  src include test
cmake --build out/build/dev-mingw-adaptive --target `
  executable_capability_test compiler_contract_test `
  control_runtime_integration_test -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(executable_capability_test|compiler_contract_test|control_runtime_integration_test)$"
```

Expected: 搜索零命中，测试全部 PASS。

**Step 6: Commit**

```bash
git add src/compiler/internal/relay_program.h src/compiler/analysis/relay_program.cc src/compiler/compiler.cc src/compiler/control_flow/production_control_flow.cc test/executable_capability_test.cpp docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md
git commit -m "refactor: derive compiler topology directly from capabilities"
```

---

### Task 5: 删除无生产消费者的 public CapabilityVerifier

`CapabilityVerifier::Verify` 通过完整 Compiler 路径证明 executable；仓库生产
代码没有消费者。`CapabilityBoundary` 三个值不改变验证行为，只影响错误文本。

**Files:**

- Delete: `include/kxc/compiler/capability.h`
- Delete: `src/compiler/capability.cc`
- Delete: `test/compiler_capability_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `test/nlp_validation/transformer_capability_matrix.json`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`
- Modify: `docs/ARCHITECTURE_STATUS.md`
- Modify: `docs/REPO_IMPLEMENTATION_OVERVIEW.md`

**Step 1: 保留真正的 capability coverage**

在删除 public facade 前确认
`test/executable_capability_test.cpp` 覆盖：

- untyped Relay；
- free variable；
- dynamic/negative shape；
- unsupported `If`/`While` policy；
- bad attrs、arity、nested tuple；
- stateful/effect/alias rejection；
- target/device placement rejection。

缺失场景迁移到 `executable_capability_test.cpp`，直接测试 internal policy 或
真实 `Compiler::Compile`。不要复制 `CapabilityResult` DTO。

**Step 2: 删除 facade**

删除：

- `CapabilityBoundary`
- `CapabilityStatus`
- `CapabilityIssue`
- `CapabilityRequest`
- `CapabilityResult`
- public `CapabilityVerifier`
- `ProbeCompilerExecution`

完整 executable proof 只由 `Compiler::Compile` 提供；调用方需要诊断时读取编译
异常。

**Step 3: 删除构建和测试注册**

从 `CMakeLists.txt` 删除：

- `include/kxc/compiler/capability.h`
- `src/compiler/capability.cc`
- `compiler_capability_test` target 和 CTest 注册。

**Step 4: 更新 NLP capability matrix**

把仅引用 `compiler_capability_test.cpp` 的证据迁移到真实
`executable_capability_test.cpp` 或 operator/compiler end-to-end test。

不保留指向已删除测试的路径。

**Step 5: 搜索验证**

Run:

```powershell
rg -n "CapabilityBoundary|CapabilityStatus|CapabilityRequest|CapabilityResult|api::CapabilityVerifier|ProbeCompilerExecution|compiler_capability_test" `
  CMakeLists.txt include src test docs
```

Expected: 当前源码、构建和 active 文档零引用；历史 plans 可保留。

**Step 6: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  executable_capability_test compiler_contract_test operator_compilation_test `
  check_public_headers check_include_layers -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(executable_capability_test|compiler_contract_test|operator_compilation_test)$"
python python/tools/check_relay_op_contract.py `
  --root . --matrix contracts/relay_op_contract.json
```

Expected: tests PASS；Relay contract 保持当前全通过数量。

**Step 7: Commit**

```bash
git add CMakeLists.txt test/executable_capability_test.cpp test/nlp_validation/transformer_capability_matrix.json docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md docs/ARCHITECTURE_STATUS.md docs/REPO_IMPLEMENTATION_OVERVIEW.md
git add -u include/kxc/compiler/capability.h src/compiler/capability.cc test/compiler_capability_test.cpp
git commit -m "refactor: remove duplicate compiler capability facade"
```

---

### Task 6: 让 topology builder 成为唯一 executable Relay 验证者

当前 static/control path 都会先做 whole-function capability traversal，然后在
构图时再次调用 `ResolveRelayCall`。删除前置重复遍历，但保留 trust-boundary
校验。

**Files:**

- Modify: `src/compiler/analysis/relay_program.cc`
- Modify: `src/compiler/graph/value_graph.cc`
- Modify: `src/compiler/control_flow/relay_control_plan.cc`
- Modify: `src/compiler/control_flow/executable_capability.cc`
- Modify: `src/compiler/internal/executable_capability.h`
- Modify: `test/executable_capability_test.cpp`
- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/control_runtime_integration_test.cpp`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`

**Step 1: 锁定 end-to-end failure behavior**

为 static 和 control 两条真实入口增加 table-driven regression cases：

```cpp
struct RejectedProgramCase {
    const char* name;
    Function function;
    const char* expected_fragment;
};
```

至少覆盖：

- undefined/free Relay value；
- untyped tensor；
- invalid tuple index；
- bad operator attrs；
- wrong operator arity；
- missing lowering binding；
- stateful/effectful operator；
- unsupported device placement；
- static path residual `If`；
- control path unsupported unbounded `While`。

测试只断言错误包含稳定 capability fragment，不绑定完整错误文本。

**Step 2: 删除 PrepareRelayProgram 的 ordinary-op preflight**

`PrepareRelayProgram` 只负责：

- config validation；
- input/residual control capability analysis；
- pipeline execution；
- policy 对 residual control capabilities 的检查；
- immutable prepared result。

删除对整张 typed ANF 的 `VerifyExecutableCapability` 调用。

**Step 3: 删除 ValueGraphBuilder 的 whole-function preflight**

从 `ValueGraphBuilder::Build()` 删除
`VerifyExecutableCapability(function, ...)`。

让现有 deterministic traversal 成为 static dataflow 权威：

- 每个 Call 只调用一次 `ResolveRelayCall` 并保存结果；
- Var/Constant/Tuple/TupleGetItem/Let 在各自分支就地校验；
- 遇到 If/While/Function value 或未知节点立即 fail closed；
- logical tensor leaf 只展开一次。

不要新增 `BuildValueGraphUnchecked`。

**Step 4: 删除 ControlPlanBuilder 的 whole-function preflight**

从 control plan builder 删除先遍历、后重新解析的
`VerifyExecutableCapability`。

ControlPlan lowering 自己负责：

- branch/loop topology；
- predicate、phi、loop-carried values；
- 每个 ordinary Call 的单次 `ResolveRelayCall`；
- production effect/alias subset；
- unsupported node fail closed。

**Step 5: 缩小 executable_capability helper**

如果 `CollectExecutableCapabilityIssues` 和
`VerifyExecutableCapability` 已无生产调用者：

- 将其保留为 focused test helper 会继续制造第二套权威；
- 删除 whole-function collector；
- 保留真正被 builder 共享的局部 predicate/diagnostic helper；
- 若没有共享 helper，删除 `executable_capability.h/.cc`，让
  `ResolveRelayCall`、ValueGraph 和 ControlPlan 分别报告自己的边界错误。

不要为了保留文件名制造空 wrapper。

**Step 6: 搜索生产调用点**

Run:

```powershell
rg -n "VerifyExecutableCapability|CollectExecutableCapabilityIssues|ResolveRelayCall\\(" `
  src/compiler
```

Expected:

- `ResolveRelayCall` 只存在于 static ValueGraph builder 和 control-plan
  builder；
- 没有“先 collector、后 builder”的同图双重解析；
- focused test 可以直接调用 `ResolveRelayCall`。

**Step 7: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  executable_capability_test compiler_contract_test operator_compilation_test `
  control_plan_test control_runtime_integration_test `
  shape_production_exact_test -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(executable_capability_test|compiler_contract_test|operator_compilation_test|control_plan_test|control_runtime_integration_test|shape_production_exact_test)$"
```

逐个运行，Expected: 全部 PASS。

**Step 8: Commit**

```bash
git add src/compiler/analysis/relay_program.cc src/compiler/graph/value_graph.cc src/compiler/control_flow/relay_control_plan.cc src/compiler/control_flow/executable_capability.cc src/compiler/internal/executable_capability.h test/executable_capability_test.cpp test/compiler_contract_test.cpp test/control_runtime_integration_test.cpp docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md
git commit -m "refactor: resolve relay calls once per topology build"
```

---

### Task 7: 删除未闭环的 multi-device compiler surface

只删除 compiler lowering surface；保留独立 distributed runtime、CPU CCL、
worker/session 和 execution-plan 数据结构。

**Files:**

- Delete: `include/kxc/compiler/distributed/multi_device.h`
- Delete: `src/compiler/distributed/multi_device.cc`
- Modify: `CMakeLists.txt`
- Modify: `src/ffi/builtin_registry.cc`
- Modify: `test/compiler_contract_test.cpp`
- Modify: `docs/ARCHITECTURE_STATUS.md`
- Modify: `docs/REPO_IMPLEMENTATION_OVERVIEW.md`

**Step 1: 证明当前路径未闭环**

Run:

```powershell
rg -n "InsertDeviceCommunicationPass|BuildDiscoPlacementPass|LowerRelayToExecPlanPass|CompilerDistributed" `
  CMakeLists.txt include src test
```

Expected:

- production Compiler/RuntimeSession 没有调用者；
- 调用只来自该 source 的 FFI 注册和 focused test。

**Step 2: 删除 compiler surface**

删除 public header 和 implementation，并从 CMake public headers/compiler
sources 移除。

**Step 3: 删除 FFI anchor**

从 `src/ffi/builtin_registry.cc` 删除 `CompilerDistributed` 声明和调用。

不修改其他 distributed anchors：

- `DistributedCclCpu`
- `DistributedSession`
- `DistributedWorker`

**Step 4: 删除结构-only compiler test**

从 `test/compiler_contract_test.cpp` 删除
`TestMultiDeviceLoweringEntry` 及其 include/注册。

保留 distributed subsystem 自己的 placement、CCL 和 session tests。

**Step 5: 更新文档**

文档明确：

- 当前 Compiler 只发布单 target static/control plans；
- multi-device compiler lowering 已删除；
- 重新引入条件是 ExecutionPlan 能查找并启动真实 `CompiledModule`，且至少有
  一个 numerical integration test；
- 不提前保留 public compatibility type。

**Step 6: 验证**

Run:

```powershell
rg -n "kxc/compiler/distributed/multi_device.h|LowerRelayToExecPlanPass|CompilerDistributed" `
  CMakeLists.txt include src test
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test type_registration_test `
  check_public_headers check_include_layers -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(compiler_contract_test|type_registration_test)$"
```

Expected: 搜索零命中；tests PASS。

**Step 7: Commit**

```bash
git add CMakeLists.txt src/ffi/builtin_registry.cc test/compiler_contract_test.cpp docs/ARCHITECTURE_STATUS.md docs/REPO_IMPLEMENTATION_OVERVIEW.md
git add -u include/kxc/compiler/distributed/multi_device.h src/compiler/distributed/multi_device.cc
git commit -m "refactor: remove unfinished multi device compiler path"
```

---

### Task 8: 让 CompileConfig 创建后不可变

当前可写 `operator->` 迫使 Compiler、Shape 和 Adaptive 在每个边界重新
Validate、clone 和比较 config/target。配置应在 `Create` 时一次冻结。

**Files:**

- Modify: `include/kxc/compiler/compile_config.h`
- Modify: `src/compiler/compile_config.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/analysis/relay_program.cc`
- Modify: `src/compiler/shape_exact.cc`
- Modify: `src/compiler/adaptive/production_path_experimental.cc`
- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/profile_bundle_test.cpp`
- Modify: `test/shape_production_exact_test.cpp`
- Modify: `test/adaptive_preparation_v2_test.cpp`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`

**Step 1: 增加 immutable construction tests**

测试以下行为：

```cpp
profiling::ProfileOptions profile;
profile.enabled = true;
profile.bundle_dir = "bundle";

const CompileConfig config = CompileConfig::Create(
    BuildTarget(Device::CPU()), 2, profile);

CHECK(config->opt_level == 2);
CHECK(config->profile_options.enabled);
CHECK(config->profile_options.bundle_dir == "bundle");
```

同时保留 invalid target/opt-level tests，但删除“创建后故意修改 config 再
Validate”的测试。

**Step 2: 扩展唯一 constructor**

将 factory 改为：

```cpp
static CompileConfig Create(
    Target target,
    int opt_level = 2,
    profiling::ProfileOptions profile_options = {});
```

`Create` 内完成：

1. target snapshot 复制；
2. opt level 设置；
3. profile environment override；
4. 一次完整 validation；
5. 返回只读配置。

不新增 builder class。

**Step 3: 删除 mutable operator**

从 public `CompileConfig` 删除：

```cpp
CompileConfigNode* operator->();
```

只保留：

```cpp
const CompileConfigNode* operator->() const;
```

**Step 4: 更新 profiling 配置调用方**

所有先 `Create()` 再写 `config->profile_options` 的调用改为提前构造
`ProfileOptions` 并传给 `Create`。

**Step 5: 删除 defensive config clone**

- `shape_exact.cc` 不再手工 Create + copy profile options；
- `production_path_experimental.cc` 删除 `CloneCompileConfig`；
- immutable `CompileConfig` 句柄可以安全复制；
- `PrepareRelayProgram` 不再重复 clone target；
- 已有 canonical identity 校验保留，但不再保存同一 target 的三份独立副本。

**Step 6: 缩减重复 Validate**

`Create` 之后配置不可变，因此：

- public compiler entry 保留一次 trust-boundary `Validate`；
- 内部同一调用链的 `PrepareCompilerGraph`、`CompilePrimitiveUnits`、
  `Finish/Assemble` 不重复 Validate；
- 从 ObjectRef 恢复的配置仍在构造时检查节点类型。

**Step 7: 搜索 mutation**

Run:

```powershell
rg -n "config->(opt_level|target|profile_options)\\s*=|CloneCompileConfig|CompileConfigNode\\* operator->" `
  include src test
```

Expected: 零命中。

**Step 8: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  compiler_contract_test profile_bundle_test shape_production_exact_test `
  adaptive_hot_swap_v2_test pipeline_resolver_test `
  check_public_headers -j 4
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^(compiler_contract_test|profile_bundle_test|shape_production_exact_test|adaptive_hot_swap_v2_test|pipeline_resolver_test)$"
```

逐个运行，Expected: 全部 PASS。

**Step 9: Commit**

```bash
git add include/kxc/compiler/compile_config.h src/compiler/compile_config.cc src/compiler/compiler.cc src/compiler/analysis/relay_program.cc src/compiler/shape_exact.cc src/compiler/adaptive/production_path_experimental.cc test/compiler_contract_test.cpp test/profile_bundle_test.cpp test/shape_production_exact_test.cpp test/adaptive_preparation_v2_test.cpp docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md
git commit -m "refactor: freeze compiler configuration at creation"
```

---

### Task 9: 最终架构、合同和远端验证

**Files:**

- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`
- Modify: `docs/ARCHITECTURE_STATUS.md`
- Modify: `docs/REPO_IMPLEMENTATION_OVERVIEW.md`
- Modify: `docs/handoffs/compiler-foundation/core.md`
- Modify: `docs/handoffs/compiler-foundation/control-flow.md`
- Modify: `docs/handoffs/compiler-foundation/shape.md`

**Step 1: 更新最终 Compiler 流程**

文档必须只描述实际存在的类型和路径：

```text
PrepareRelayProgram
  -> BuildValueGraph | LowerPreparedRelayToControlPlanWithSidecar
  -> PrimitiveUnit[]
  -> CompilePrimitiveUnits
  -> AssembleCompiledGraph | BindControlPlanForRuntime
```

删除已不存在的：

- `CompileResult` compiler state machine；
- public `CapabilityVerifier`；
- `PreparedProgramPlan` variant；
- whole-graph `LoweredGraph` production seam；
- multi-device compiler lowering。

**Step 2: 全仓禁止残留搜索**

Run:

```powershell
rg -n "CompileResultNode|PrimitiveCompileState|PreparedProgramPlan|PreparedStaticPlan|PreparedControlPlan|api::CapabilityVerifier|CapabilityBoundary|LoweredGraph|LowerRelayToExecPlanPass|CompilerDistributed" `
  CMakeLists.txt include src test docs
```

Expected: active code/docs 零引用；历史 plan 中的旧设计说明可以保留。

**Step 3: 本地 CPU 启用态测试矩阵**

Run:

```powershell
cmake -S . -B out/build/dev-mingw-adaptive -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug `
  -DKXC_ENABLE_LLVM=OFF `
  -DKXC_ENABLE_CUDA=OFF `
  -DKXC_ENABLE_CONTROL_RUNTIME=ON `
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON `
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON `
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON `
  -DKXC_BUILD_PASS_TESTS=ON `
  -DKXC_BUILD_CODEGEN_TESTS=OFF

cmake --build out/build/dev-mingw-adaptive --target `
  object_test `
  pass_pipeline_test `
  pipeline_resolver_test `
  compiler_contract_test `
  compiler_identity_test `
  operator_compilation_test `
  primitive_cache_test `
  executable_capability_test `
  shape_system_test `
  shape_specialization_test `
  shape_production_exact_test `
  restricted_symbolic_shape_test `
  adaptive_hot_swap_v2_test `
  control_plan_test `
  control_runtime_integration_test `
  runtime_session_test `
  profile_bundle_test `
  check_include_layers `
  check_public_headers `
  -j 4

ctest --test-dir out/build/dev-mingw-adaptive `
  --output-on-failure --no-tests=error
```

逐个运行二进制，Expected: 全部 PASS。

**Step 4: 合同检查**

Run:

```powershell
python python/tools/check_relay_op_contract.py `
  --root . --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py `
  --root . --matrix contracts/pass_contract.json
git diff --check
```

Expected: contract 和 diff check 全部 PASS。

**Step 5: Omen CUDA 验证**

在确认 Omen worktree 干净、分支与本地 commit 相同后，按根目录
`AGENTS.md` 的 CUDA 12.9 配置构建：

```text
object_test
pass_pipeline_test
device_info_test
infer_type_test
profile_bundle_test
cupti_smoke_test
check_relay_op_contract
```

如果 primitive cache、ObjectRef 或 lifetime ownership 在实现中发生变化，再运行
ASan/UBSan `object_test_asan`。本计划不允许以“只是重构”为理由跳过。

**Step 6: 文档 commit**

```bash
git add docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md docs/ARCHITECTURE_STATUS.md docs/REPO_IMPLEMENTATION_OVERVIEW.md docs/handoffs/compiler-foundation/core.md docs/handoffs/compiler-foundation/control-flow.md docs/handoffs/compiler-foundation/shape.md
git commit -m "docs: record minimal compiler authority chain"
```

**Step 7: 推送前检查**

Run:

```bash
git fetch origin
git status --short
git diff --check origin/compiler-foundation-acceptance-cleanup...HEAD
git log --oneline origin/compiler-foundation-acceptance-cleanup..HEAD
```

Expected:

- worktree clean；
- 只有本计划定义的 commits；
- 无未解释 generated outputs；
- 无 unrelated files。

---

## 2. 完成定义

- [ ] 前置 primitive artifact assembler 计划已完成。
- [ ] 普通静态编译不再构造或重放 `CompileResult`。
- [ ] `CompiledPrimitiveBatch` 是 backend compilation 的唯一结果。
- [ ] whole-graph lowering aggregate 只存在于 test support。
- [ ] topology 直接从 residual capability set 派生。
- [ ] public `CapabilityVerifier` 和 dead boundary vocabulary 已删除。
- [ ] 每个 Relay Call 在选定 topology 中只解析一次。
- [ ] multi-device compiler lowering 在真实 numerical execution 前不占 public surface。
- [ ] `CompileConfig` 创建后不可变。
- [ ] primitive cache、artifact pin 和 runtime ABI 安全边界未削弱。
- [ ] static/control runtime hot paths 仍然分离。
- [ ] local CPU、contract、header/layer 和 Omen 验证完成。

## 3. 停止条件

遇到以下情况停止当前任务，不用新抽象绕过：

- 删除 public API 会破坏已承诺的外部 ABI；
- topology builder 无法在一次 traversal 中保持现有 fail-closed 行为；
- immutable config 需要修改 Target 的外部所有权模型；
- multi-device 已存在仓库外 production consumer；
- 前置 primitive assembler 计划尚未完成或分支未同步。

这些情况需要单独、证据驱动的计划；不得在本轮新增 compatibility hierarchy。
