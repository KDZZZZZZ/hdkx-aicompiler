# Shape Exact Dispatch Closure Implementation Plan

> **For implementer:** REQUIRED SKILL: `executing-plans`，按任务顺序实现并在每个提交后验证。

> **修订记录（2026-07-26，Task 1 执行中）：** 停止条件 §4.1 第一条已触发——
> oracle 派生（`shape_specialization.cc` `InstantiateExactProfile`：模板全文
> canonical + bindings + policy `"exact"`）与 plan 派生
> （`experimental_identity.cc` `BuildStaticExactShapeProfileKey`：仅输入边界
> + 固定 bindings 串 + policy `"static-exact-plan-v2"`）的 `ShapeProfileKey`
> 在四个 canonical 字段上分叉。经用户决定：本计划范围收窄为 **oracle 键空间
> 内的 dispatch 闭环**（Tasks 2-3 全部 identity 来自同一 prepared 模板，
> 自洽）；验收不变量 2 改为**锁定分叉事实**的不等断言；oracle/plan 键空间
> 对齐（含 adaptive route identity 迁移决策）是独立的后续计划，证据回填
> issue #46。

**Goal:** 闭合 shape 子系统的消费侧（issue #46）：让 `ExactPlanVariant` 铸造与
adaptive 一致的 `DispatchKey`，让受限符号模板可以按 binding 物化出可编译的
concrete Relay Function，并用一个确定性测试锁定“输入 shape → profile →
dispatch key → variant 查找 → 静态执行”的完整请求边界闭环。

**Architecture:** 不新增 VariantTable、Router、Dispatcher 或 policy class。
路由表是调用方自己持有的普通 map；编译意图只能来自用户显式调用
`Compiler::Compile()`。shape 层只新增四个 free/static function：物化、route
identity、编译后验证和输入 shape 到 binding 的换算。identity 全部复用
`experimental_identity.h` 已有的 `BuildStaticExactDispatchKey` /
`BuildStaticExactShapeProfileKey` / `BuildPlanVariantKey`，不引入第二套 key。

**Tech Stack:** C++17、CMake/Ninja、现有 Relay/TE/TIR pipeline、
`shape_exact` / `shape_specialization` / `restricted_symbolic_shape`、
`Compiler::Compile`、`RuntimeSession`、`experimental_identity`。

---

## 0. 执行前提

- 从最新 `origin/compiler-foundation-acceptance-cleanup` 创建干净、独立的
  worktree。当前主工作树含 30 个未提交 header 修改，禁止在其中执行。
- 前置计划 `2026-07-25-primitive-artifact-plan-assembler.md` 和
  `2026-07-25-compiler-redundancy-cleanup.md` 已在该分支完成；本计划直接
  依赖 `AssembleCompiledGraph()` 和 immutable `CompileConfig`。
- 每个任务只暂存任务列出的文件；禁止 `git add .`。
- 不引入新依赖、新 class hierarchy、新 gate。
- 全程使用启用态配置构建（`KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON`、
  `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON`、
  `KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON`）。

---

## 1. 目标模型

### 1.1 现状与缺口

生产侧已完整：`AssembleExactPlan()` 产出携带 module/plan/pins 的
`ExactPlanVariant`（`src/compiler/shape_exact.cc:981` 起，经共享的
`AssembleCompiledGraph`）。消费侧为空：

1. `shape_exact` 铸 `ShapeProfileKey` 和 `PlanVariantKey`，但不铸
   `DispatchKey`——而 adaptive 路由表恰好以
   `(DispatchKey, PlanAbiFingerprint)` 为 key
   （`src/compiler/adaptive/adaptive_hot_swap_v2.cc:50`）。
2. `BuildPlanVariantKey` 有两个独立生产者（`shape_exact.cc:998` 与
   `production_path_experimental.cc:126`），无一致性测试。
3. `RestrictedSymbolicShapeAdapter::MintExact` 能为非 representative 的
   binding 铸出决策，但没有任何路径把决策变成可编译的 concrete Function；
   非 representative profile 的 variant 无法产生。
4. 请求边界没有从实际输入 shape 求 binding 的入口。

### 1.2 闭环后的完整链

```text
用户显式 control plane（本计划中即测试代码；不进 src/）
┌──────────────────────────────────────────────────────────┐
│ 编译期（每个 profile 一次，编译意图显式来自用户）：       │
│   prep = RestrictedSymbolicShapeAdapter::Prepare(...)    │
│   d    = MintExact(prep, bindings)                       │
│   fn   = MaterializeExactFunction(prep, d)               │
│   g    = Compiler::Compile(fn, config)      // 用户显式  │
│   VerifyCompiledExactVariant(prep, d, g)                 │
│   route[ExactDispatchKey(prep, d)] = g                   │
│                                                          │
│ 请求边界（每次执行请求，永不隐式编译）：                  │
│   b  = BindingsFromInputShapes(prep, ShapesOf(inputs))   │
│   d  = MintExact(prep, b)              // 越界 fail closed│
│   it = route.find(ExactDispatchKey(prep, d))             │
│   miss → 明确失败，不触发编译                             │
│   hit  → RuntimeSession(g.module(), g.plan()).Run(inputs)│
└──────────────────────────────────────────────────────────┘
```

route family identity = representative 的 `GraphSemanticKey`（restricted
模板由 representative 的 frozen exact 模板构成，
`src/compiler/restricted_symbolic_shape.cc:283-284`），profile 携带差异。
这与 adaptive 的 `BuildStaticExactDispatchKey(graph_semantic_key,
shape_profile_key)` 完全同构。

### 1.3 唯一新增 API

`ExactPlanVariant` 增加一个 accessor：

```cpp
// include/kxc/compiler/shape_exact.h
const DispatchKey& dispatch_key() const;
```

`RestrictedSymbolicShapeAdapter` 增加四个 static function：

```cpp
// include/kxc/compiler/restricted_symbolic_shape.h
static Function MaterializeExactFunction(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision);

static DispatchKey ExactDispatchKey(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision);

static void VerifyCompiledExactVariant(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const RestrictedDispatchDecision& decision,
    const CompiledGraph& compiled);

static specialization::BindingSet BindingsFromInputShapes(
    const PreparedRestrictedSymbolicTemplate& prepared,
    const std::vector<std::vector<int64_t>>& input_shapes);
```

没有别的。不新增 class、不新增 struct 返回包、不改 adaptive。

### 1.4 identity 语义约定

- `ExactDispatchKey` = `BuildStaticExactDispatchKey(
  prepared.graph_template().key(), decision.exact_oracle().profile().key())`。
- 物化出的 concrete Function 有它自己的 `GraphSemanticKey`（包含 concrete
  shape），与 family key **有意不同**。`VerifyCompiledExactVariant` 验证的
  是 plan 边界 dtype/shape 与决策求值后的 contract 相等，**不是** key 相等。
- 决策是 route identity 的唯一授权来源；编译产物只提供 artifact。两者由
  验证函数绑定，不靠命名约定。

### 1.5 非目标

- 把 shape variant 发布进 `AdaptiveController` 路由表。重新审视条件：出现
  对单个 shape variant 做热替换/generation 治理的真实需求，且仍只经用户
  显式 `Submit()`。
- bucket、polymorphic、dynamic-output、ragged shape。
- 扩大受限算子子集（当前固定 rank `relu`/`sqrt`、等形 `add`/`mul`）。
- MintExact 结果缓存或任何请求边界性能优化——每次请求跑一遍
  solver/profile 实例化是已知成本，只有测量证明成为瓶颈后才处理。
- 生产 `VariantTable` / `ShapeDispatcher` / `VariantRouter` class。
- FFI / Python 暴露。
- 修改 `shape_exact` 的 gate 默认值。

---

## 2. 验收不变量

1. `ExactPlanVariant::dispatch_key()` 与
   `BuildStaticExactDispatchKey(模板 key, profile key)` 逐字节一致。
2. 对同一 concrete graph，`shape_exact` 的 oracle 派生 profile key 与
   `BuildStaticExactShapeProfileKey(semantic key, 编译后 plan)` 相等。
3. 对同一 concrete graph，`shape_exact` 的 `PlanVariantKey` 可由公开
   builder 从 `Compiler::Compile` 结果重建并相等。
4. `PrepareGraphTemplate` 产出的模板 key ==
   `Compiler::BuildGraphSemanticKey(同一 function)`。
5. 同一 restricted 模板下不同 binding 的决策产生不同 `DispatchKey`；
   相同 binding 产生相同 `DispatchKey`。
6. 物化 + `Compiler::Compile` + 验证后的 variant 可被 `RuntimeSession`
   执行并产出正确数值。
7. `MintExact` 对越界/不整除/未绑定 symbol 全部 fail closed。
8. route 查找 miss 时明确失败，primitive cache miss 与 in-flight 计数
   不变（无隐式编译）。
9. 请求边界代码路径中没有 `Compiler::Compile`、`CompilePrimitiveUnits`
   或任何编译入口调用。
10. 不新增 speculative class/interface/policy；全仓 rg 无
    `VariantTable|ShapeDispatcher|VariantRouter`。
11. adaptive 子系统零修改。
12. 决策与编译产物边界不匹配时 `VerifyCompiledExactVariant` 抛出。

---

### Task 1: ExactPlanVariant 铸造 DispatchKey 并锁定跨生产者 identity

即 issue #46 选项 2。

**Files:**

- Modify: `include/kxc/compiler/shape_exact.h`
- Modify: `src/compiler/shape_exact.cc`
- Test: `test/shape_production_exact_test.cpp`

**Step 1: 写失败测试**

在 `test/shape_production_exact_test.cpp` 对已有的两 unit 测试图：

```cpp
const auto prepared =
    ProductionExactShapeAdapter::PrepareGraphTemplate(fn, config);
const auto oracle =
    ProductionExactShapeAdapter::InstantiateExactProfile(prepared, {});
const auto variant =
    ProductionExactShapeAdapter::AssembleExactPlan(prepared, oracle);

// 不变量 4：模板 key 就是 compiler 的 graph semantic key。
const GraphSemanticKey semantic = Compiler::BuildGraphSemanticKey(fn);
CHECK(prepared.graph_template().key() == semantic);

// 不变量 1：dispatch key 用公开 builder 可重建。
CHECK(variant.dispatch_key() ==
      BuildStaticExactDispatchKey(semantic, variant.shape_profile_key()));

// 不变量 2：oracle 派生 profile == plan 派生 profile。
const CompiledGraph normal = Compiler::Compile(fn, config);
CHECK(variant.shape_profile_key() ==
      BuildStaticExactShapeProfileKey(semantic, normal.plan()));

// 不变量 3：PlanVariantKey 可由 normal 编译结果重建。
CHECK(variant.plan_variant_key() ==
      BuildPlanVariantKey(semantic, variant.shape_profile_key(),
                          SelectionsOf(normal),
                          runtime::internal::kStaticMemoryPlanVersion));
```

`SelectionsOf` 是测试内局部 helper，从 `normal.plan().calls()` 和
`normal.artifact_pins()` 构造 `OrderedArtifactSelectionIdentity`，与
`shape_exact.cc:988-996` 相同的方式。

**Step 2: 运行确认失败**

```powershell
cmake --build out/build/dev-mingw-adaptive --target shape_production_exact_test -j 4
```

Expected: FAIL，`dispatch_key()` 不存在。

**Step 3: 实现**

- `ExactPlanVariant::Impl` 增加 `DispatchKey dispatch` 字段。
- `AssembleExactPlan` 构造 Impl 时调用
  `BuildStaticExactDispatchKey(prepared.impl_->graph.key(),
  oracle.profile().key())`。
- header 增加 accessor。不加 `compiled_graph()` accessor——本计划没有它的
  消费者。

**Step 4: 运行测试**

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  shape_production_exact_test compiler_identity_test -j 4
out/build/dev-mingw-adaptive/shape_production_exact_test.exe
out/build/dev-mingw-adaptive/compiler_identity_test.exe
```

Expected: 全部 PASS。若不变量 2 或 4 失败，触发停止条件 §4.1，不得用
适配代码掩盖。

**Step 5: Commit**

```bash
git add include/kxc/compiler/shape_exact.h src/compiler/shape_exact.cc test/shape_production_exact_test.cpp
git commit -m "shape: mint dispatch keys on exact plan variants"
```

---

### Task 2: 受限决策物化为可编译 concrete Function

**Files:**

- Modify: `include/kxc/compiler/restricted_symbolic_shape.h`
- Modify: `src/compiler/restricted_symbolic_shape.cc`
- Test: `test/restricted_symbolic_shape_test.cpp`

**Step 1: 写失败测试**

representative 用 N=4，symbol 约束 `{0, 0, "N", 2, 8, 2}`：

```cpp
const auto prep = RestrictedSymbolicShapeAdapter::Prepare(
    fn_n4, config, {InputAxisSymbol{0, 0, "N", 2, 8, 2}});
const auto d6 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bind("N", 6));

const Function fn6 =
    RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, d6);
const CompiledGraph g6 = Compiler::Compile(fn6, config);
RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(prep, d6, g6);

kxc::runtime::RuntimeSession session(g6.module(), g6.plan());
CHECK(session.Run(inputs_n6) == expected_n6);   // 逐元素数值校验

// identity：不同 binding 不同 route，相同 binding 相同 route。
const auto d4 = MintExact(prep, Bind("N", 4));
CHECK(ExactDispatchKey(prep, d6) != ExactDispatchKey(prep, d4));
CHECK(ExactDispatchKey(prep, d6) ==
      ExactDispatchKey(prep, MintExact(prep, Bind("N", 6))));

// 验证函数真的在验证：拿错 variant 必须抛。
const Function fn4 = MaterializeExactFunction(prep, d4);
const CompiledGraph g4 = Compiler::Compile(fn4, config);
Throws([&] { VerifyCompiledExactVariant(prep, d6, g4); });
```

**Step 2: 运行确认失败**

```powershell
cmake --build out/build/dev-mingw-adaptive --target restricted_symbolic_shape_test -j 4
```

Expected: FAIL，四个函数都不存在。

**Step 3: 实现 MaterializeExactFunction**

对 representative Function 做带类型覆盖的深拷贝：参数 `Var` 重建为决策
求值后 contract 的 `TensorType`，body 中的引用重映射。实现优先复用
`shape_exact.cc` 匿名命名空间中已有的 `RelaySnapshotCloner`（提升到
src-private header 并增加参数类型覆盖钩子）；若钩子使 cloner 复杂化，则
在 `restricted_symbolic_shape.cc` 本地实现受限子集（Var/Call/Tuple/
TupleGetItem/Let，`ValidateSyntax` 已保证只有这些节点）的重建，并注释
说明这是有意的边界内重复。两条路线都不得把 cloner 变成 public API。

物化前重新验证决策属于该 prepared 模板（模板 key 与决策
`graph_template().key()` 相等），否则抛出。

**Step 4: 实现 ExactDispatchKey / VerifyCompiledExactVariant / BindingsFromInputShapes**

- `ExactDispatchKey`：按 §1.4，一行委托给
  `BuildStaticExactDispatchKey`，外加模板/决策归属校验。
- `VerifyCompiledExactVariant`：
  1. `compiled.plan()` 的输入/输出边界 dtype 与 shape 逐一等于决策
     `exact_oracle().profile()` 求值后的 contract；
  2. plan call 数量等于模板 unit skeleton 数量；
  3. 任一不匹配抛 `std::runtime_error`，消息含 "restricted exact variant"。
  复用 `restricted_symbolic_shape.cc` 已有的 `ValidContract` /
  `RequireSameShape` helper，不新写第二套比较。
- `BindingsFromInputShapes`：
  1. 输入个数与 representative 参数个数一致；
  2. 对每个 `InputAxisSymbol` 读 `input_shapes[param][axis]` 绑定 symbol，
     重复 symbol 值冲突时抛出；
  3. 非 overlay 轴必须等于 representative 的静态维度，否则抛出；
  4. 返回 canonical `BindingSet`。

**Step 5: 运行测试**

```powershell
cmake --build out/build/dev-mingw-adaptive --target `
  restricted_symbolic_shape_test shape_production_exact_test -j 4
out/build/dev-mingw-adaptive/restricted_symbolic_shape_test.exe
out/build/dev-mingw-adaptive/shape_production_exact_test.exe
```

Expected: 全部 PASS。

**Step 6: Commit**

```bash
git add include/kxc/compiler/restricted_symbolic_shape.h src/compiler/restricted_symbolic_shape.cc test/restricted_symbolic_shape_test.cpp
git commit -m "shape: materialize compilable variants from restricted decisions"
```

（若 Step 3 选择提升 cloner，本次 commit 额外包含对应的
`src/compiler/internal/` 新 header 与 `shape_exact.cc` 的搬迁 diff。）

---

### Task 3: 请求边界 dispatch 闭环锁定测试

**Files:**

- Create: `test/shape_exact_dispatch_test.cpp`
- Modify: `CMakeLists.txt`

**Step 1: 写闭环测试**

新测试 target `shape_exact_dispatch_test`，与
`restricted_symbolic_shape_test` 相同的 gate 条件注册。测试完整实现
§1.2 的两段：

编译期——对 `N ∈ {2, 4, 6}` 各铸决策、物化、`Compiler::Compile`、验证、
放进 `std::map<std::string, CompiledGraph>`（key 为
`ExactDispatchKey(...).canonical_bytes()`）。route map 就是测试局部变量，
证明控制面可以完全活在用户代码里。

请求边界——

```cpp
// 命中：N=6 的输入路由到 N=6 的 variant 并算出正确数值。
const auto b = BindingsFromInputShapes(prep, ShapesOf(inputs_n6));
const auto d = MintExact(prep, b);
const auto it = route.find(ExactDispatchKey(prep, d).canonical_bytes());
CHECK(it != route.end());
kxc::runtime::RuntimeSession session(it->second.module(), it->second.plan());
CHECK(session.Run(inputs_n6) == expected_n6);

// fail closed 一组：
Throws([&] { MintExact(prep, Bind("N", 3)); });    // 不整除
Throws([&] { MintExact(prep, Bind("N", 10)); });   // 越上界
Throws([&] { BindingsFromInputShapes(prep, bad_static_axis_shapes); });

// N=8 合法但未编译 → 查找 miss，且无隐式编译：
const auto stats_before = GetPrimitiveCacheStats();
const auto d8 = MintExact(prep, Bind("N", 8));
CHECK(route.find(ExactDispatchKey(prep, d8).canonical_bytes()) == route.end());
const auto stats_after = GetPrimitiveCacheStats();
CHECK(stats_after.misses == stats_before.misses);
CHECK(stats_after.in_flight == stats_before.in_flight);
```

`GetPrimitiveCacheStats` 复用 `adaptive_preparation_v2_test.cpp` 中已有的
统计读取方式。

**Step 2: 注册构建**

`CMakeLists.txt` 在 restricted gate 块内添加 target 与 CTest 注册，链接
对象与 `restricted_symbolic_shape_test` 相同。

**Step 3: 运行测试**

```powershell
cmake --build out/build/dev-mingw-adaptive --target shape_exact_dispatch_test -j 4
out/build/dev-mingw-adaptive/shape_exact_dispatch_test.exe
ctest --test-dir out/build/dev-mingw-adaptive --output-on-failure `
  -R "^shape_exact_dispatch_test$"
```

Expected: PASS。

**Step 4: 禁止词汇搜索**

```powershell
rg -n "VariantTable|ShapeDispatcher|VariantRouter|class .*Dispatch" `
  include src
git diff origin/compiler-foundation-acceptance-cleanup -- src/compiler/adaptive include/kxc/compiler/adaptive_hot_swap_v2.h include/kxc/compiler/adaptive_production_experimental.h
```

Expected: 第一条零命中；第二条空 diff（不变量 11）。

**Step 5: Commit**

```bash
git add CMakeLists.txt test/shape_exact_dispatch_test.cpp
git commit -m "shape: lock request boundary exact dispatch loop"
```

---

### Task 4: 文档与最终验收

**Files:**

- Modify: `docs/handoffs/compiler-foundation/shape.md`
- Modify: `docs/ARCHITECTURE_STATUS.md`

**Step 1: 更新 handoff**

`shape.md` 改为如实描述：

- exact variant 携带 `DispatchKey`，与 adaptive 的 static-exact 路由
  identity 同构；
- restricted 模板可按 binding 物化 concrete Function，编译意图显式来自
  用户 `Compiler::Compile`；
- 请求边界闭环存在且 fail closed，控制面是用户持有的普通 map；
- `AdaptiveController` 路由发布仍是非目标，及其重新审视条件（§1.5）；
- 受限子集与 unsupported 列表原文保留。

**Step 2: 全矩阵回归**

按 `2026-07-25-compiler-redundancy-cleanup.md` Task 9 Step 3 的完整启用态
配置构建并运行全部测试 target（含新 `shape_exact_dispatch_test`），随后：

```powershell
python python/tools/check_relay_op_contract.py --root . --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . --matrix contracts/pass_contract.json
git diff --check
```

Expected: 全部 PASS。

**Step 3: Omen 验收**

按 `AGENTS.md` CUDA 12.9 配置构建 shape 三测试 + 新 dispatch 测试；ASan/
UBSan 构建运行 `restricted_symbolic_shape_test` 和
`shape_exact_dispatch_test`（Task 2 引入了新的 Relay 节点构造路径，不得以
"只是加测试"为由跳过 sanitizer）。

**Step 4: Commit 并回填 issue**

```bash
git add docs/handoffs/compiler-foundation/shape.md docs/ARCHITECTURE_STATUS.md
git commit -m "docs: record exact shape dispatch closure"
```

在 issue #46 评论记录：选项 2 已完成（Task 1），选项 3 以用户持有 route
的最小形式完成（Task 2-3），controller 集成维持非目标及其条件。

---

## 3. 完成定义

- [ ] `ExactPlanVariant::dispatch_key()` 存在且可由公开 builder 重建。
- [ ] oracle 派生与 plan 派生的 `ShapeProfileKey` 有一致性测试。
- [ ] 两个 `PlanVariantKey` 生产者有一致性测试。
- [ ] 非 representative binding 可物化、编译、验证并正确执行。
- [ ] 请求边界闭环测试覆盖命中、越界、静态轴不匹配、未编译 miss 四类。
- [ ] miss 与 fail-closed 路径经 cache 统计证明无隐式编译。
- [ ] 未新增 router/table/dispatcher class；adaptive 零修改。
- [ ] handoff 文档与实际能力一致；issue #46 已回填。
- [ ] 本地全矩阵、contract、Omen 与 sanitizer 验收有记录。

## 4. 停止条件

1. Task 1 中 `prepared.graph_template().key() !=
   Compiler::BuildGraphSemanticKey(fn)`，或 oracle/plan 两条 profile 派生
   不相等：说明 identity 语义已分叉，停止本计划，单独制定 identity 对齐
   计划；禁止在 shape 层加转换函数掩盖。
2. 物化需要受限子集之外的 Relay 节点或算子：停止扩展，记录缺口。
3. `VerifyCompiledExactVariant` 需要读取 compiler internal 才能完成边界
   比较：停止，先在 public plan 表面补齐所需只读信息的独立计划。
4. 请求边界闭环无法在不触碰 adaptive 的前提下表达：停止，重新评估
   §1.5 第一条非目标，不得顺手改 adaptive。
