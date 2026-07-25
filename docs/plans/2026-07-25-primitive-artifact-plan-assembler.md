# Primitive Artifact 与 Plan Assembler Implementation Plan

> **For implementer:** REQUIRED SKILL: `executing-plans`，按任务顺序实现并在每个提交后验证。

**Goal:** 让普通编译、Shape exact 和 Adaptive Hot Swap 共用唯一的 primitive artifact 编译与 plan 组装模型；Adaptive 只重编译发生变化的 primitive，仍以完整不可变 plan generation 为发布单位。

**Architecture:** 将当前 `Compiler::Compile` 拆成 `PrepareProgram → CompilePrimitiveArtifacts → PlanAssembler::Assemble`。`PrimitiveArtifactPin` 继续是缓存和 executable 生命周期权威；`PreparedPlanTemplate` 保存不随 kernel 版本变化的拓扑、value routing 和执行合同；`PlanAssembler` 只验证并组合已选 artifact，不运行 Relay/TE/TIR/backend 编译。普通编译选择全部初始 artifact，Adaptive 合并“旧选择 + 新替换”，二者通过同一个 assembler 生成不可变 `CompiledGraph`。

**Tech Stack:** C++17、CMake/Ninja、现有 Relay/TE/TIR compiler pipeline、`PrimitiveArtifactKey`、primitive cache、`CompiledModule`、`ExecutablePlan`、`RuntimeSession`。

---

## 0. 执行前提

- 从最新 `origin/compiler-foundation-acceptance-cleanup` 创建干净、独立的 worktree。
- 不在当前含有其他人 header/documentation 修改的工作树中执行本计划。
- 每个任务只提交任务列出的文件；禁止 `git add .`、`git add include` 或
  `git add src` 这类宽范围暂存。
- 若目标分支在执行期间前进，只允许 fetch 后 rebase/merge 经检查的任务提交，
  不覆盖其他人的工作树。

---

## 1. 目标模型

### 1.1 编译和发布流程

```text
Relay Function + CompileConfig
             │
             ▼
PrepareCompilerProgram
             │
             ▼
PreparedPlanTemplate
  - GraphSemanticKey
  - frozen PrimitiveUnit[]
  - LogicalValueContract[]
  - immutable topology / value routing
  - target + CompilerExecutionContract
  - prevalidated memory plan
             │
             ├──────────────────────────────────────┐
             ▼                                      ▼
CompilePrimitiveArtifacts                  Reuse existing artifacts
  - lower / TIR / schedule                 - unchanged ArtifactPin
  - cache / backend                        - same immutable owner
  - returns PrimitiveArtifactSelection[]
             │                                      │
             └──────────────────┬───────────────────┘
                                ▼
                    PlanAssembler::Assemble
                      - complete slot coverage
                      - UnitSemanticKey match
                      - target / Kernel ABI match
                      - link-symbol binding
                      - module + plan + pins
                                │
                                ▼
                     immutable CompiledGraph
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             RuntimeSession          GenerationLease
                                      route publication
```

### 1.2 权威类型

```cpp
namespace kxc::api::internal {

struct PreparedPlanTemplate final {
    GraphSemanticKey graph_semantic_key;
    PreparedStaticGraph static_graph;
    runtime::ExecutablePlan executable_plan;
    Target target;
    CompilerExecutionContract execution_contract;
};

struct PrimitiveArtifactSelection final {
    PrimitiveUnitId unit_id{-1};
    PrimitiveArtifactPin artifact;
};

class PlanAssembler final {
public:
    static CompiledGraph Assemble(
        const PreparedPlanTemplate& plan_template,
        std::vector<PrimitiveArtifactSelection> selections,
        std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);
};

}  // namespace kxc::api::internal
```

约束：

- `PrimitiveArtifactPin` 是 ready backend artifact、Kernel ABI 和 executable 生命周期的唯一权威。
- `PrimitiveArtifactSelection` 只表达 `unit_id → artifact`；link symbol 和 unit semantic contract 均从 template 获取，不保存第二份权威。
- `PreparedPlanTemplate` 不保存 mutable cache lease、backend flight、generation 或 adaptive policy。
- `PlanAssembler` 不读取 Relay registry，不运行 pass，不调用 backend，不写 primitive cache。
- `CompiledGraph` 继续是不可变发布快照；没有 in-place kernel pointer mutation。
- unchanged primitive 直接复用原 `PrimitiveArtifactPin`；只为 replacement 编译新 artifact。
- publication 前必须拥有完整选择表；不能发布“半旧半新但尚未验证”的可见状态。
- `PlanAbiFingerprint` 不包含 artifact selection identity；`PlanVariantKey` 包含实际选择。

### 1.3 正常编译与 Adaptive 的共同路径

```text
Normal:
  template = PrepareCompilerProgram(graph, config)
  selections = CompilePrimitiveArtifacts(template, AllUnitIds)
  return PlanAssembler::Assemble(template, selections)

Adaptive:
  baseline = current GenerationLease
  replacements = CompilePrimitiveArtifacts(template, ChangedUnitIds)
  selections = Merge(baseline.selections, replacements)
  candidate = PlanAssembler::Assemble(template, selections)
  ValidateSamePlanAbi(baseline, candidate)
  Publish(candidate)
```

### 1.4 重编译意图的唯一权威

Runtime、Shape guard 和 profiler 只上报事实，不得直接调用
`PrimitiveVariantCompiler`、`PlanAssembler` 或 publication API。只有
`AdaptiveController` 能把 observation 变成重编译请求：

```text
Runtime / Shape Guard / Profiler / Explicit Warmup API
                       │
                       └─ AdaptiveObservation
                                  │
                                  ▼
                         RecompilePolicy::Evaluate
                                  │
                         optional RecompileIntent
                                  │
                                  ▼
                         AdaptiveController
                                  │
             ResolveAffectedPrimitiveUnits(template, intent)
                                  │
                                  └─ PlanCompileRequest
                                           │
                                           ▼
                                PrimitiveVariantCompiler
```

内部意图至少区分：

```cpp
enum class RecompileReason : std::uint8_t {
    kShapeCoverageMiss,
    kHotShapeSpecialization,
    kPerformanceRegression,
    kAutotuningCandidate,
    kExplicitWarmup,
};

struct RecompileIntent final {
    RecompileReason reason;
    GraphSemanticKey graph_semantic_key;
    PlanVariantKey base_variant_key;
    std::optional<ShapeProfileKey> shape_profile_key;
    std::string optimization_objective;
};
```

规则：

- 新 shape 已被当前 guard/variant 覆盖时，不产生 correctness 型重编译。
- 新 shape 没有任何可执行 variant 覆盖时，可以产生高优先级
  `kShapeCoverageMiss`。
- 已覆盖 shape 只有达到 hotness/收益阈值后，才可以产生
  `kHotShapeSpecialization`。
- profiler 只提交结构化测量；性能回退、调优候选是否值得编译由 policy
  决定。
- explicit warmup 也必须经过 controller 的 admission、singleflight、预算和
  publication 检查，不能绕过 controller。
- intent 不接受 caller 指定的可变 kernel/artifact 集合；受影响的
  `PrimitiveUnitId` 由 compiler 根据 `PreparedPlanTemplate` 中的
  shape/value/unit 依赖关系计算。
- target、device 或 Plan ABI 改变不属于同一 program 的 hot-swap；必须建立
  新 template/program。

本轮实现这条 authority boundary、observation/intent 类型和可注入的
deterministic policy seam。真实 profiler 自动触发、收益预测模型和 schedule
搜索仍属于后续工作。

### 1.5 本轮非目标

- 不实现 mutable `KernelSlot` 或运行中 plan 原地改指针。
- 不把 AdaptiveController 放进 `RuntimeSession`。
- 不实现真实 profiler 自动触发策略或收益预测算法。
- 不实现 bucket/polymorphic/dynamic-output Shape。
- 不实现新的 schedule 搜索算法；本轮只建立可接收不同 artifact selection 的正确架构。
- 不改变“一 ordinary Relay Call 一 PrimitiveUnit”的当前 partition 策略。

---

## 2. 验收不变量

1. 普通 `Compiler::Compile` 与 Adaptive candidate 使用同一个 `PlanAssembler`。
2. `PlanAssembler` 输入顺序变化不影响结果；输出按 `PrimitiveUnitId` 规范排序。
3. 缺失、重复、未知或语义不匹配的 selection 在 module 构造前失败。
4. replacement 可以改变 `PrimitiveArtifactKey` 和 backend executable，但不得改变 Plan ABI。
5. 未替换 unit 的 `PrimitiveArtifactPin` owner 必须与上一 generation 相同。
6. 新 generation 发布后，旧 lease 仍能执行旧 executable。
7. 新请求只在请求边界选择 generation；kernel launch loop 不增加 slot lookup。
8. 相同 `PlanVariantKey` 的重复请求不得制造无意义的新 generation。
9. Adaptive worker 不再调用完整 `Compiler::Compile`。
10. Runtime public API 不依赖 `PreparedPlanTemplate`、primitive cache 或 Adaptive。
11. 只有 `AdaptiveController` 能将 observation 转换为 `PlanCompileRequest`；
    Runtime、Shape 和 profiler 不依赖 primitive compiler 或 publication API。
12. 已被当前 variant 覆盖的新 shape 不产生 correctness 型重编译；未覆盖
    shape 和性能优化意图必须具有不同 reason。

---

### Task 1: 用回归测试锁定当前普通编译与 hot-swap 行为

**Files:**

- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/adaptive_preparation_v2_test.cpp`
- Modify: `CMakeLists.txt`

**Step 1: 为普通编译增加 artifact/plan 对齐测试**

增加 `TestCompiledGraphHasOnePinPerOrderedCall()`：

```cpp
const CompiledGraph graph = Compiler::Compile(MakeTwoPrimitiveGraph(), config);
CHECK(graph.plan().calls().size() == graph.artifact_pins().size());
for (size_t i = 0; i < graph.plan().calls().size(); ++i) {
    CHECK(graph.artifact_pins()[i].defined());
}
```

**Step 2: 为 Adaptive 增加单 primitive 替换行为测试**

先用现有 fixture 表达以下预期：

```cpp
const auto old_generation = controller.CompileAndPublish({baseline_request});
const auto new_generation = controller.CompileAndPublish({replace_unit_one});

CHECK(old_generation->generation() < new_generation->generation());
CHECK(old_generation->plan_abi() == new_generation->plan_abi());
CHECK(old_generation->selection_plan_key() !=
      new_generation->selection_plan_key());
```

额外断言旧、新 launcher 都能独立执行。

**Step 3: 运行测试确认基线**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target compiler_contract_test adaptive_hot_swap_v2_test -j 4
out/build/dev-mingw-adaptive/compiler_contract_test.exe
out/build/dev-mingw-adaptive/adaptive_hot_swap_v2_test.exe
```

Expected: 当前行为测试 PASS；尚未引用新 assembler 类型。

**Step 4: Commit**

```bash
git add test/compiler_contract_test.cpp test/adaptive_preparation_v2_test.cpp CMakeLists.txt
git commit -m "test: lock plan artifact publication behavior"
```

---

### Task 2: 引入不可变 `PreparedPlanTemplate`

**Files:**

- Create: `src/compiler/internal/prepared_plan_template.h`
- Modify: `src/compiler/internal/execution_contract.h`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/shape_exact.cc`
- Test: `test/compiler_contract_test.cpp`
- Test: `test/shape_production_exact_test.cpp`

**Step 1: 写一个失败的 template snapshot 测试**

测试准备完成后修改调用方 Relay、Target 和 `CompileConfig`，再证明 template 的：

- `GraphSemanticKey`
- target snapshot
- `PrimitiveUnit`
- `ExecutablePlan`

均不变化。

**Step 2: 运行测试确认失败**

Expected: FAIL，因为当前 `PreparedCompilerGraph` 尚未提供独立 plan template。

**Step 3: 实现 `PreparedPlanTemplate`**

要求：

- 构造时调用 `BuildStaticExecutablePlan` 和 `runtime::internal::PlanMemory`。
- 保存完整 target/execution-contract snapshot。
- `ValidatePreparedPlanTemplate()` 检查 unit id、call id、value id 和 target 一致性。
- 不保存 `CompileResult` stage machine 或 profile span。

**Step 4: 将 `PrepareCompilerGraph` 改为返回 preparation bundle**

临时 bundle 可以包含：

```cpp
struct PreparedCompilerProgram final {
    PreparedPlanTemplate plan_template;
    std::shared_ptr<profiling::ProfileContext> profile_context;
    std::string profile_run_id;
};
```

诊断状态与可复用模板分开，避免 template 成为 profiling owner；Shape
adapter 的 preparation counters 继续作为 adapter-local 诊断快照。

**Step 5: 迁移 Shape exact preparation**

`PreparedGraphTemplate::Impl` 只持有冻结的 `PreparedCompilerProgram`，不得重新遍历调用方 Relay。

**Step 6: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target compiler_contract_test shape_production_exact_test -j 4
out/build/dev-mingw-adaptive/compiler_contract_test.exe
out/build/dev-mingw-adaptive/shape_production_exact_test.exe
```

Expected: PASS。

**Step 7: Commit**

```bash
git add src/compiler/internal/prepared_plan_template.h src/compiler/internal/execution_contract.h src/compiler/compiler.cc src/compiler/shape_exact.cc test/compiler_contract_test.cpp test/shape_production_exact_test.cpp
git commit -m "compiler: freeze reusable plan templates"
```

---

### Task 3: 将 primitive 编译结果收口为 artifact selection

**Files:**

- Modify: `src/compiler/internal/primitive_compiler.h`
- Modify: `src/compiler/primitive_compiler.cc`
- Modify: `src/compiler/internal/primitive_cache.h`
- Modify: `src/compiler/cache/primitive_cache.cc`
- Test: `test/operator_compilation_test.cpp`
- Test: `test/primitive_cache_test.cpp`

**Step 1: 写 selection 权威测试**

断言 selection 的 key、signature、launch metadata 和 executable 都来自同一个 `PrimitiveArtifactPin`，不存在可单独修改的副本。

**Step 2: 写 subset 编译测试**

准备三个 unit，只请求 `{2, 0}`：

```cpp
const auto selected = CompilePrimitiveArtifacts(
    plan_template, {PrimitiveUnitId{2}, PrimitiveUnitId{0}});
CHECK(selected.size() == 2);
CHECK(selected[0].unit_id == 0);
CHECK(selected[1].unit_id == 2);
```

同时验证重复和越界 id fail closed。

**Step 3: 运行测试确认失败**

Expected: FAIL，因为当前 `CompilePrimitiveUnits` 要求 dense 全量 unit。

**Step 4: 实现**

将接口调整为：

```cpp
std::vector<PrimitiveArtifactSelection> CompilePrimitiveArtifacts(
    const PreparedPlanTemplate& plan_template,
    std::vector<PrimitiveUnitId> requested_units);
```

行为：

- 输入先排序和去重检查。
- 只 lower/compile requested unit。
- cache hit 直接返回 pin。
- TIR 和 cache-hit 等诊断放入独立 `PrimitiveCompileDiagnostics`，不进入 selection 权威。
- plan-local link symbol 由 template 提供，不进入 artifact identity。

**Step 5: 删除 `CompiledPrimitive` 中重复 ABI 字段**

删除可由 pin 派生的：

- `artifact_key`
- `signature`
- `launch_metadata`
- `kernel`
- public `ArtifactPin`

需要的内部访问统一通过 `PrimitiveArtifactPin`。

**Step 6: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target operator_compilation_test primitive_cache_test -j 4
out/build/dev-mingw-adaptive/operator_compilation_test.exe
out/build/dev-mingw-adaptive/primitive_cache_test.exe
```

Expected: PASS。

**Step 7: Commit**

```bash
git add src/compiler/internal/primitive_compiler.h src/compiler/primitive_compiler.cc src/compiler/internal/primitive_cache.h src/compiler/cache/primitive_cache.cc test/operator_compilation_test.cpp test/primitive_cache_test.cpp
git commit -m "compiler: expose immutable primitive artifact selections"
```

---

### Task 4: 建立唯一 `PlanAssembler`

**Files:**

- Create: `src/compiler/internal/plan_assembler.h`
- Create: `src/compiler/plan_assembler.cc`
- Modify: `CMakeLists.txt`
- Modify: `src/compiler/internal/compiled_graph_access.h`
- Test: `test/plan_assembler_test.cpp`

**Step 1: 新增失败测试 target**

覆盖：

- 完整 selection 成功。
- selection 输入乱序后结果确定。
- missing、duplicate、unknown unit 拒绝。
- `UnitSemanticKey` 不匹配拒绝。
- artifact target、signature 或 launch metadata 不兼容时拒绝。
- artifact key 可不同但 callable Plan ABI 相同的 candidate 成功。

**Step 2: 运行测试确认无法链接**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target plan_assembler_test -j 4
```

Expected: FAIL，`PlanAssembler` 尚不存在。

**Step 3: 实现 assembler**

顺序必须固定：

1. `ValidatePreparedPlanTemplate`。
2. 将 selections 规范排序。
3. 完整 slot coverage 检查。
4. unit semantic、target 和 physical Kernel ABI 检查。
5. 将 cached executable relocation 到 plan-local link symbol。
6. 构造 `CompiledModule`。
7. 复用 template 中已验证的 `ExecutablePlan`。
8. 以 plan call 顺序生成 public `ArtifactPin[]`。
9. 通过 `CompiledGraphAccess::Create` 一次性发布对象。

在第 9 步之前不得产生对外可见的部分结果。

**Step 4: 运行测试**

Expected: `plan_assembler_test` PASS。

**Step 5: Commit**

```bash
git add src/compiler/internal/plan_assembler.h src/compiler/plan_assembler.cc src/compiler/internal/compiled_graph_access.h test/plan_assembler_test.cpp CMakeLists.txt
git commit -m "compiler: add deterministic plan assembler"
```

---

### Task 5: 让普通 Compiler 只通过 assembler 发布

**Files:**

- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/internal/compile_state.h`
- Modify: `src/compiler/compile_state.cc`
- Modify: `src/compiler/internal/primitive_compiler.h`
- Modify: `src/compiler/primitive_compiler.cc`
- Test: `test/compiler_contract_test.cpp`
- Test: `test/compiler_identity_test.cpp`
- Test: `test/runtime_session_test.cpp`

**Step 1: 增加普通编译与直接 assembler 的等价测试**

用同一 prepared template 和 artifact selections 分别走新的内部 assembly seam
与 public `Compiler::Compile`，比较 graph identity、ordered artifact keys、Plan
ABI 和可执行数值结果。不要为此向 production API 增加 test-only counter。

**Step 2: 迁移 `Compiler::Compile`**

改成：

```cpp
PreparedCompilerProgram prepared =
    PrepareCompilerProgram(function, config, contract);
auto selections = CompilePrimitiveArtifacts(
    prepared.plan_template, AllPrimitiveUnitIds(prepared.plan_template));
return PlanAssembler::Assemble(
    prepared.plan_template, std::move(selections),
    prepared.profile_context);
```

**Step 3: 删除旧组装双轨**

删除或内联后移除：

- `CompilePreparedPrimitiveUnits` 中 module/pin 复制。
- `AssembleModule`。
- `CompileResult::AfterBackends` 作为最终 module authority 的职责。
- `FinishCompilerGraph` 中重复 module/plan/pins 拼装。

`CompileResult` 若仍用于 profiling stage machine，只能保留诊断数据，不能拥有第二份最终 artifact selection。

**Step 4: 运行回归**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target compiler_contract_test compiler_identity_test operator_compilation_test runtime_session_test plan_assembler_test -j 4
out/build/dev-mingw-adaptive/compiler_contract_test.exe
out/build/dev-mingw-adaptive/compiler_identity_test.exe
out/build/dev-mingw-adaptive/operator_compilation_test.exe
out/build/dev-mingw-adaptive/runtime_session_test.exe
out/build/dev-mingw-adaptive/plan_assembler_test.exe
```

Expected: 全部 PASS。

**Step 5: Commit**

```bash
git add src/compiler/compiler.cc src/compiler/internal/compile_state.h src/compiler/compile_state.cc src/compiler/internal/primitive_compiler.h src/compiler/primitive_compiler.cc test/compiler_contract_test.cpp test/compiler_identity_test.cpp test/runtime_session_test.cpp
git commit -m "compiler: publish static programs through plan assembler"
```

---

### Task 6: 迁移 Shape exact 到共同 assembler

**Files:**

- Modify: `src/compiler/shape_exact.cc`
- Modify: `include/kxc/compiler/shape_exact.h`
- Test: `test/shape_production_exact_test.cpp`
- Test: `test/shape_specialization_test.cpp`

**Step 1: 写跨路径相等测试**

同一 concrete graph：

```cpp
const CompiledGraph normal = Compiler::Compile(graph, config);
const ExactPlanVariant exact = AssembleExact(graph, config);

CHECK(normal.graph_semantic_key() ==
      exact.plan_variant_key().graph_semantic_key());
CHECK(SameOrderedArtifactKeys(normal, exact));
CHECK(SamePlanAbi(normal, exact));
```

**Step 2: 运行测试确认当前路径仍有独立 finish seam**

Expected: 测试应暴露 `FinishCompilerGraph` 双轨或缺少共同 assembler 证据。

**Step 3: 迁移实现**

`ProductionExactShapeAdapter::AssembleExactPlan`：

- 从 `PreparedPlanTemplate` 编译全部 exact artifact。
- 调用同一个 `PlanAssembler`。
- `ExactPlanVariant` 继续只包装 `CompiledGraph + ShapeProfileKey + PlanVariantKey`。

**Step 4: 删除 `FinishCompilerGraph`**

仓库内无调用后，从 `execution_contract.h` 和 `compiler.cc` 删除。

**Step 5: 运行测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target shape_system_test shape_specialization_test shape_production_exact_test -j 4
out/build/dev-mingw-adaptive/shape_system_test.exe
out/build/dev-mingw-adaptive/shape_specialization_test.exe
out/build/dev-mingw-adaptive/shape_production_exact_test.exe
```

Expected: 全部 PASS。

**Step 6: Commit**

```bash
git add src/compiler/shape_exact.cc include/kxc/compiler/shape_exact.h test/shape_production_exact_test.cpp test/shape_specialization_test.cpp src/compiler/internal/execution_contract.h src/compiler/compiler.cc
git commit -m "compiler: assemble exact shape plans from shared artifacts"
```

---

### Task 7: 将 Adaptive request 改为 template + replacement 编译

**Files:**

- Modify: `include/kxc/compiler/adaptive_production_experimental.h`
- Modify: `src/compiler/adaptive/production_path_experimental.cc`
- Modify: `include/kxc/compiler/adaptive_hot_swap_v2.h`
- Modify: `src/compiler/adaptive/adaptive_hot_swap_v2.cc`
- Create: `src/compiler/internal/recompile_policy.h`
- Create: `src/compiler/internal/primitive_variant_compiler.h`
- Create: `src/compiler/internal/adaptive_controller_access.h`
- Test: `test/adaptive_preparation_v2_test.cpp`

**Step 1: 写失败测试**

证明 Adaptive worker：

- 只请求指定 unit。
- 未指定 unit 沿用 baseline pin。
- candidate 由 assembler 生成。
- 不调用完整 `Compiler::Compile`。
- Shape guard、Runtime observation 和 profiler sample 不能直接取得
  primitive compiler 或 publication authority。
- 已覆盖的新 shape 不产生 correctness 型 request；coverage miss 与
  performance intent 使用不同 reason。
- explicit warmup 也经过 controller admission 和 singleflight。

fixture adapter 记录收到的 unit ids：

```cpp
CHECK(adapter->compiled_unit_ids() ==
      std::vector<PrimitiveUnitId>{changed_unit});
```

**Step 2: 将 whole-graph adapter 移出 API**

删除 experimental header 中返回整图的：

```cpp
virtual CompiledGraph Compile(
    const ProductionCompileRequest& request) = 0;
```

在 compiler internal 建立最小接口：

```cpp
class PrimitiveVariantCompiler {
public:
    virtual ~PrimitiveVariantCompiler() = default;
    virtual std::vector<PrimitiveArtifactSelection> Compile(
        const PreparedPlanTemplate& plan_template,
        const std::vector<PrimitiveUnitId>& requested_units) = 0;
};
```

默认实现调用 `CompilePrimitiveArtifacts`。测试通过
`internal::AdaptiveControllerAccess` 注入 fixture；public/experimental header
不得出现 `ForTesting()`、fake compiler 或 compiler-internal类型。

**Step 3: 重塑 request**

public experimental 层使用 opaque Pimpl `PreparedAdaptiveProgram`，内部持有
`PreparedPlanTemplate` 和 baseline selections。`PlanCompileRequest` 至少包含：

- 不可变 `PreparedAdaptiveProgram` owner。
- baseline generation 或 baseline selections。
- requested replacement unit ids。
- 已注册的 `variant_policy_id`；该 policy 参与
  `PrimitiveArtifactKey` 的 schedule/pipeline identity。
- expected `DispatchKey`。
- expected `PlanAbiFingerprint`。
- deadline/cancellation 留在 controller-level `CompileRequest`。

删除 request 对 caller-mutable Relay 和可重复 `Compiler::Compile` 的依赖。

**Step 4: 建立 observation → intent → request 的唯一入口**

在 `src/compiler/internal/recompile_policy.h` 定义：

- `AdaptiveObservation`：Shape coverage、hotness、performance sample、
  autotuning candidate 或 explicit warmup 的只读事实。
- `RecompileIntent`：policy 接受 observation 后产生的内部决策。
- `RecompilePolicy`：deterministic、可注入、无 publication authority 的策略
  接口。

只有 `AdaptiveController` 可以：

1. 接收 observation。
2. 调用 policy。
3. 根据 template 解析受影响的 primitive units。
4. 建立 `PlanCompileRequest`。
5. 将 request 送入既有 admission/singleflight 队列。

测试 policy 可以确定性地产生 intent，但不得直接注入 artifact、generation
或 publication transaction。

**Step 5: 改写 worker**

Worker 顺序：

1. admission/singleflight。
2. 编译 replacement selections。
3. 与 baseline selections 合并。
4. `PlanAssembler::Assemble`。
5. candidate validation。
6. 计算 `PlanVariantKey`。
7. 若 selection key 未变化，返回当前 lease，不分配 generation。
8. 否则进入既有 publication transaction。

**Step 6: 保留 publication 协议**

不得改变：

- final cancellation/deadline observation。
- generation authority。
- route/ABI key。
- quarantine tombstone。
- old lease retention。
- transactional `routes.swap`。

**Step 7: 运行 Adaptive 测试**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target adaptive_hot_swap_v2_test plan_assembler_test -j 4
out/build/dev-mingw-adaptive/adaptive_hot_swap_v2_test.exe
out/build/dev-mingw-adaptive/plan_assembler_test.exe
```

Expected: 全部 PASS；测试日志证明 worker 没有 whole-graph compile。

**Step 8: Commit**

```bash
git add include/kxc/compiler/adaptive_production_experimental.h src/compiler/adaptive/production_path_experimental.cc include/kxc/compiler/adaptive_hot_swap_v2.h src/compiler/adaptive/adaptive_hot_swap_v2.cc src/compiler/internal/recompile_policy.h src/compiler/internal/primitive_variant_compiler.h src/compiler/internal/adaptive_controller_access.h test/adaptive_preparation_v2_test.cpp
git commit -m "compiler: assemble adaptive plans from primitive replacements"
```

---

### Task 8: 收口命名、删除兼容双轨

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `tools/architecture/check_public_headers.py`
- Modify: `docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md`
- Modify: `docs/DYNAMIC_COMPILED_MODULE_ABI.md`
- Delete or rename after zero-reference proof:
  - legacy whole-graph adaptive preparation names
  - obsolete `CompiledPrimitiveBatch` assembly fields
  - obsolete `FinishCompilerGraph`

**Step 1: 全仓搜索旧入口**

Run:

```powershell
rg -n "FinishCompilerGraph|AssembleModule|ProductionPathCompilerAdapter|CompiledPrimitiveBatch|compiler->Compile\\(f->request\\)" CMakeLists.txt include src test docs
```

Expected: 只出现迁移说明；生产代码零引用。

**Step 2: 更新文档**

文档必须明确：

- replacement unit 是编译粒度。
- immutable plan generation 是发布粒度。
- `PlanAssembler` 是普通/Shape/Adaptive 唯一组装器。
- Runtime 不持有 compiler/cache/controller。

**Step 3: 运行架构门禁**

Run:

```powershell
cmake --build out/build/dev-mingw-adaptive --target check_include_layers check_public_headers -j 4
```

Expected: include-layer 与 header-manifest 全部通过。

**Step 4: Commit**

```bash
git add CMakeLists.txt tools/architecture/check_public_headers.py docs/COMPILER_LOWERING_TARGET_ARCHITECTURE.md docs/DYNAMIC_COMPILED_MODULE_ABI.md
git add -u -- src/compiler/compiler.cc src/compiler/internal/execution_contract.h src/compiler/internal/primitive_compiler.h src/compiler/primitive_compiler.cc include/kxc/compiler/adaptive_production_experimental.h src/compiler/adaptive/production_path_experimental.cc
git commit -m "docs: record primitive artifact assembly architecture"
```

---

### Task 9: 本地完整回归

**Files:** No source changes unless a failure requires a scoped fix.

**Step 1: 构建启用态矩阵**

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
  compiler_contract_test `
  compiler_identity_test `
  operator_compilation_test `
  primitive_cache_test `
  plan_assembler_test `
  shape_system_test `
  shape_specialization_test `
  shape_production_exact_test `
  restricted_symbolic_shape_test `
  adaptive_hot_swap_v2_test `
  runtime_session_test `
  check_include_layers `
  check_public_headers `
  -j 4
```

Expected: 全部构建成功。

**Step 2: 逐个执行**

逐个运行上述测试二进制，Expected: 全部 PASS。

**Step 3: 合同检查**

Run:

```powershell
python python/tools/check_relay_op_contract.py --root . --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . --matrix contracts/pass_contract.json
git diff --check
```

Expected:

- Relay operator contract 24/24。
- Pass contract 20/20。
- `git diff --check` 无错误。

**Step 4: Commit fixes if needed**

每个独立失败使用独立修复提交，不把无关生成物加入提交。

---

### Task 10: Omen 验收

**Files:** No source changes unless validation finds a real defect.

**Step 1: 推送并安全快进 Omen**

确认本地与 Omen 无 task-unrelated tracked 修改后：

```bash
git push origin compiler-foundation-acceptance-cleanup
ssh omen
cd /home/oops/repo/hdkx-aicompiler
git fetch origin
git switch compiler-foundation-acceptance-cleanup
git pull --ff-only origin compiler-foundation-acceptance-cleanup
```

**Step 2: 配置启用态 CUDA build**

```bash
cmake -S . -B out/build/omen-cuda-adaptive -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON \
  -DKXC_BUILD_PASS_TESTS=ON \
  -DKXC_BUILD_CODEGEN_TESTS=OFF \
  -DKXC_BUILD_RESNET18_IR_DUMP=OFF \
  -DCMAKE_CXX_FLAGS="-I/usr/local/cuda/extras/CUPTI/include -DCUpti_ActivityKernel10=CUpti_ActivityKernel9"
```

**Step 3: 构建并执行**

至少构建和执行：

```text
compiler_contract_test
compiler_identity_test
operator_compilation_test
primitive_cache_test
plan_assembler_test
shape_system_test
shape_specialization_test
shape_production_exact_test
restricted_symbolic_shape_test
adaptive_hot_swap_v2_test
runtime_session_test
```

Expected: CPU/fixture 路径全部 PASS。CUDA runtime smoke 若因 NVML driver/library mismatch 跳过，必须明确记录为环境缺口。

**Step 4: Sanitizer**

使用独立 CMake ASan/UBSan build 构建并运行：

```text
object_test
primitive_cache_test
plan_assembler_test
adaptive_hot_swap_v2_test
```

Expected: 无 ASan/UBSan 报告。

**Step 5: 最终工作树核对**

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/compiler-foundation-acceptance-cleanup
```

只清理本轮测试生成目录；保留所有用户或 Omen 原有未跟踪文件。

---

## 3. 完成定义

- [ ] 普通、Shape exact、Adaptive 三条路径共用唯一 `PlanAssembler`。
- [ ] Adaptive worker 不再调用完整 `Compiler::Compile`。
- [ ] 只有 `AdaptiveController` 能把 observation 转换成
  `PlanCompileRequest`。
- [ ] Runtime、Shape guard 和 profiler 不依赖 primitive compiler 或
  publication API。
- [ ] Shape coverage miss 与 hot-shape/performance intent 有不同 reason 和
  admission policy。
- [ ] replacement 编译只访问指定 `PrimitiveUnitId`。
- [ ] unchanged artifact pins 跨 generation 复用。
- [ ] candidate ABI 不兼容时 publication 前拒绝。
- [ ] 相同 selection 不增加 generation。
- [ ] 新请求使用新 generation，旧 lease 保持旧 executable 可运行。
- [ ] Runtime kernel launch loop 无 mutable slot lookup。
- [ ] 旧 whole-graph adaptive compilation seam 零引用并删除。
- [ ] 启用态本地测试、架构门禁、合同检查全部通过。
- [ ] Omen 启用态构建、测试和 sanitizer 结果有完整记录。
