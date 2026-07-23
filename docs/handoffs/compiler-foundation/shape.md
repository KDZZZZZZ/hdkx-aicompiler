# Compiler Foundation / Shape 轨道交接

> **分支：** `feature/compiler-foundation-shape`
>
> **基线：** `e295a73`（compiler-foundation roadmap）
>
> **状态：** W2 增加 default-OFF 的 production exact Relay/compiler adapter；它只支持 concrete Relay 的单一 empty profile 和现有 static RuntimeSession。隔离 Shape contract 仍是 **experimental review candidate**，没有稳定 public v1；bucket、polymorphic、symbolic/dynamic output 以及稳定 Shape ABI 均未完成。
>
> **重要边界：** 本页不宣称生产主链已经支持 dynamic shape、bucket kernel、polymorphic launch、dynamic output allocation 或稳定 Shape ABI。bucket/polymorphic 仍只有 guarded deterministic fake。

## 1. 已交付范围

本轨先建立了不依赖 Relay、Compiler、RuntimeSession 或 backend 的 `kxc::shape` 模块，并保持现有静态执行路径不变。

### 1.1 纯 Shape 基础

实验头：`include/kxc/shape/shape.h`

- ABI 暴露边界：
  - 四个 Shape 头由 `kxc_shape_api` 的 public header file set 安装/export；“experimental”表示可供隔离 contract 消费和测试，不表示未安装。
  - 每个 installed header 都显式标记 experimental v1；全部隔离 API 位于 `kxc::shape::experimental::v1`，稳定的 `kxc::shape` public v1 尚不存在。
  - `kShapeContractVersion = 1`、`kShapeAbiVersion = 1` 只描述 experimental canonical format；不承诺跨版本 source compatibility、binary C++ ABI 或稳定 public API。
  - guarded fake 类型只从 `include/kxc/shape/fakes/compiler_foundation_v1.h` 暴露；该头另行声明它不是 Compiler/Runtime/cache API。
- `DimExpr`
  - `Const(n >= 0)`、`Symbol(name)`、`Add`、`Mul`、`FloorDiv(positive_const)`、`Min`、`Max`。
  - `Add`/`Mul`/`Min`/`Max` 做确定性的 flatten、排序、常量折叠和去重。
  - 加法、乘法、stride 推导检查 `int64_t` overflow。
  - `Const(0)` 是合法零长度维；负值和 `-1` 不进入新 Shape IR。
- `BindingSet`
  - symbol 唯一、值非负、按 symbol 排序、确定性 canonical form。
- `Constraint`
  - exact correctness subset：`Eq`、`Range`、`DivisibleBy`、`BroadcastCompatible`。
  - 对称约束按 canonical expression 排序。
  - `SameRank`、`LayoutCompatible` 尚未实现；`SupportsConstraint` 明确返回 false，`RequireConstraintSupport` fail closed，不能被静默忽略。
- `ExactConstraintSolver`
  - 只推导 `Symbol = 已完全可求值表达式`。
  - range/divisibility/broadcast 不用于猜值。
  - 未绑定、额外 binding、矛盾、负值和 overflow 均 fail closed。
- 三层 tensor contract：
  - `LogicalShape`：数学维度和可选 axis name。
  - `PhysicalShape`：capacity、stride、layout、alignment、memory scope。
  - `ValidExtent`：本次调用的有效范围。
  - symbolic 与 concrete contract 都按值携带强类型 `DataType`、`DeviceDescriptor(kind,id)` 和 `TargetBackendAbiDescriptor(target,backend,abi_version)`，不从 fingerprint 文本猜 `f32`/`cuda`。
  - experimental exact v1 只接受 `contiguous.row_major`；显式 stride 必须等于 canonical row-major stride。未知 layout、zero/overlap/noncanonical writable stride、element/byte extent overflow 均 fail closed；只读 broadcast stride 尚未建模，因此不放行。
  - concrete evaluation 强制 `valid <= logical <= physical`；exact specialization 进一步强制三者相等。
- identity DTO：
  - `GraphTemplateKey`：graph semantic、pipeline、capability、partition fingerprint、contract version 和强类型 target/backend ABI descriptor。
  - `ShapeProfileKey`：template key、不可由调用者直接构造的完整 `GraphTemplateContentKey`、完整 concrete bindings、显式 policy id 和 shape-ABI version。
  - canonical bytes 使用长度前缀字段；相等判断比较完整结构，不以 hash 单独决定等价。
- `ShapeProgram`
  - 声明 symbols、named input/output tensor contracts 和 constraints。
  - 构造/`Verify` 检查名称、rank、symbol 引用及确定性排序。
  - `Evaluate` 只求 shape/extent/stride；不分配、不查 cache、不编译、不 launch。

### 1.2 GraphTemplate / exact profile 拆分

实验头：`include/kxc/shape/specialization.h`

- `GraphLocalCallLocator` 与 `UnitSemanticKey` 明确分离。
  - call locator 只服务 plan routing/诊断。
  - artifact identity 不包含 call locator、graph value name/id、storage id、entry symbol 或对象地址。
- `GraphTemplate`
  - 持有 `GraphTemplateKey`、`ShapeProgram` 和有序 `UnitSkeleton`。
  - `GraphTemplateContentKey` 从 key、`ShapeProgram::CanonicalString()`、ordered unit semantic key、call locator 和完整 input/output routing 的 canonical bytes 内部 mint；同 key 不代表同 template。
  - verifier 检查 producer/consumer 顺序、重复 producer、缺失 producer、显式 unit boundary，以及每个 tensor contract 的 target/backend ABI 与 template 一致。
  - 重复 `UnitSemanticKey` 合法，允许等价 unit 共享 artifact。
- `InstantiateExactProfile`
  - 返回只能由该函数 mint 的 `ExactOracle`；oracle/profile 绑定完整 `GraphTemplateContentKey`，所有消费点重新比较完整 template 内容。
  - same caller-supplied key/different ShapeProgram 或 ordered routing 会在 artifact request 前拒绝。
  - exact profile 强制所有 value 的 logical/physical/valid 完全相等。
  - profile instantiation 不暴露 graph-pass/partition hook，因此不会在该层重跑 graph preparation。
- `KernelArtifactKey`
  - 包含 unit semantic key、shape ABI、pipeline/capability fingerprint、强类型 target/backend ABI descriptor 和有序 boundary contracts。
  - boundary canonical identity 结构化包含 dtype、device kind/id、target/backend/ABI；f32/f16、CPU/CUDA、device id、target、backend 或 ABI version 任一变化都 miss。
  - 不包含 `ShapeProfileKey` 整体，因此 graph-local/template identity 或与该 unit 无关的 binding 不会污染 artifact identity。
- `MakeExactSpecializationRequests`
  - 生成有序 exact requests；request 保留 graph-local routing，artifact key 不保留。
- `PlanVariantKey`
  - plan identity 可记录 call/entry/generation；与 artifact semantic identity 分开。

### 1.3 版本化 deterministic fake

实验 fake 头：`include/kxc/shape/fakes/compiler_foundation_v1.h`

namespace：`kxc::shape::experimental::v1::fakes::compiler_foundation_v1`

- `kContractVersion = 1`。
- `DeterministicMockCoordinator`
  - 同步解析 exact requests。
  - 比较完整 canonical artifact key；相同 key 得到同一 deterministic fake entry。
  - generation 固定为 `0`。
  - 仅用于跨轨 contract test；不实现 queue、singleflight、cache、failure TTL 或后台编译。
- `DeterministicMockPlanAssembler`
  - 校验 template/profile/artifact/signature/ordered call。
  - 生成不可变 fake frozen plan 并按值保留 selected artifacts。
  - 不执行 kernel，不依赖 `RuntimeSession`。

### 1.4 exact 之后的显式 guarded profile

实验 contract 头：`include/kxc/shape/guarded_specialization.h`

- `ApplicabilityGuard`
  - bucket 和 polymorphic policy 均必须提供非空、规范化约束域。
  - `Matches` 使用 exact solver 验证完整 binding；未绑定、域外或未声明 symbol 返回 false。
- exact-first 门禁
  - `BuildBucketProfile` 和 `BuildPolymorphicProfile` 都必须接收同一 `GraphTemplate` mint 的 `ExactOracle`。
  - builder 会重新求 exact differential contract；不能直接从 capacity 或 policy 构造优化 profile。
- `BucketPolicy`
  - 显式记录 bucket id/version、guard、每个 named value 的 physical capacity/stride/layout/alignment/scope、workspace 上界和每个 ordered unit 的 tail contract。
  - 初始 contract fake 对 padded bucket 保守要求 staging pad、mask、predicate 和 output crop 全部声明。
  - `valid` 保持 exact request logical extent；physical 使用 bucket contract。
  - 大 capacity 本身不构成 applicability；guard miss、缺 tail/pad/crop、较小 capacity、错误 stride/layout 均拒绝。
- `PolymorphicPolicy`
  - 显式记录 guard、逐 unit allowlist/proof、强类型 target/backend ABI descriptor、workspace 上界、带 dtype/device/ABI 的 symbolic boundaries 和版本化 runtime-extent scalar ABI。
  - scalar ordinal/name/symbol/range/divisibility 必须完整且有序；所有声明 symbol 必须有 guard 和 runtime binding。
  - 当前仅冻结 contract/fake；没有把现有 `-1` input ABI 当作 runtime extent ABI。
- `GuardedArtifactKey`
  - bucket identity 包含显式 bucket/guard/physical/tail/workspace/environment contract，但排除 exact request binding。
  - polymorphic identity 包含 symbolic boundary/guard/proof/runtime extent ABI/environment，但排除 concrete request binding。
  - 因此多个请求只能在同一显式 guard 和完整证明相同时共享；不存在 `cached_dims >= query_dims` 路由。
- guarded deterministic fake
  - bucket/polymorphic 没有 production resolver、compiler 或 runtime 路径；相关 fake 类型仅在 `kxc::shape::experimental::v1::fakes::compiler_foundation_v1`。
  - `GuardedDeterministicMockCoordinator::Resolve` 在边界接收可信 `const GraphTemplate` 和由 private policy 状态 mint 的 `const GuardedShapeProfile`，调用 `MakeGuardedSpecializationRequests` 重建完整 expected sequence；调用者提供的 mutable request 只作为待验证输入。
  - resolver 在改变去重状态前逐 ordinal 比较 call index/locator、profile/exact identity、kind、guard canonical、有序 input/output concrete contract（rank、logical/physical/valid、canonical stride、axis/layout/alignment/scope、dtype/device/target/backend ABI）和完整 `GuardedArtifactKey` kind/semantic/payload；数量或任一字段不符即拒绝。
  - 去重、entry symbol 和 selected artifact 一律使用重建的 expected request，不保存调用者对象；`GuardedDeterministicMockPlanAssembler` 继续保持 generation `0`、完整 key 比较和有序 assembly。
  - fake frozen plan 同时保留 concrete logical/valid、guarded physical、tail/runtime-extent metadata；不执行任何数据面操作。

## 2. 文件与依赖边界

新增 Shape 模块文件：

- `include/kxc/shape/shape.h`
- `include/kxc/shape/specialization.h`
- `include/kxc/shape/guarded_specialization.h`
- `include/kxc/shape/fakes/compiler_foundation_v1.h`
- `src/shape/shape.cc`
- `src/shape/specialization.cc`
- `src/shape/guarded_specialization.cc`

新增测试：

- `test/shape_system_test.cpp`
- `test/shape_specialization_test.cpp`
- `test/shape_guarded_specialization_test.cpp`

构建/架构：

- `CMakeLists.txt` 新增独立 `kxc_shape_api`、`kxc_shape_obj` 和三个 focused test target。
- `tools/architecture/check_include_layers.py` 保持 Shape 只允许依赖自身的底层模块；Compiler 被明确允许依赖 Shape，Runtime/Shape 不反向依赖 Compiler。
- Shape experimental 头只 include Shape/stdlib；没有 include `kxc/compiler/*`、`kxc/relay/*`、`kxc/runtime/*`、frontend 或 private header。
- 安装 smoke 已确认四个头落在 `${prefix}/include/kxc/shape[/fakes]`，且安装副本均保留 experimental-v1/no-source-or-binary-ABI-compatibility 标记。
- `.github/workflows/ci.yml` 的 CPU job 明确执行三个 `run_shape_*` target，而不再只编译 test executable。
- W2 在 Compiler/lowering 内增加可复用 prepared graph 和 production exact adapter，统一 Compiler 与 production primitive cache 的稳定 target-capability canonicalizer，并把 public `CompiledModule::constants()` 收紧为 payload deep snapshots；RuntimeSession 仅改用 private immutable constant borrow，职责和静态数据面不变。未修改 Relay/type inference、`KernelSignature`、`ValueSpec`、memory-planning 算法或 frontend。

## 3. W2 production exact（default-OFF）

### 3.1 API、流和边界

新增 installed experimental-v1 adapter：`include/kxc/compiler/shape_exact.h`，实现：`src/compiler/shape_exact.cc`，namespace 为 `kxc::api::experimental::shape_exact::v1`。实现调用真实 production compiler/cache，但 public surface 直接消费 experimental Shape 类型，故头文件明确不承诺 source/binary ABI compatibility；没有伪装成稳定 `kxc::api` ABI。CMake gate 为 `KXC_ENABLE_SHAPE_PRODUCTION_EXACT`，默认 `OFF`；实现始终编译，但 API 在 gate 关闭时明确拒绝。该 API 没有 `CompileConfig` opt-in 字段，也没有 fake backend。

- `ProductionExactShapeAdapter::PrepareGraphTemplate(Function, CompileConfig)` 仅解析一次真实 `CompilerExecutionContract`，在真实 `PassContext::MergeTarget`/Scope 下执行 validate、Relay normalized pipeline、entry/post-pass/pre-partition 三个 capability boundary、`BuildValueGraph` 和 `PartitionValueGraph`。
- `src/compiler/internal/prepared_static_graph.h` 与 `PrepareStaticGraph`/`LowerPreparedStaticGraph` 分开图准备和 per-unit lowering；旧 `LowerGraph` 仍由两步组合，兼容现有静态路径。prepared object 绑定 Device、Target 和 pipeline identity；adapter 的 immutable pimpl 保存 deep-copied target/config snapshot、execution contract、prepared partition 和 `GraphTemplate`。constant NDArray 在 preparation 时复制为独立 payload snapshot，调用方后续修改原始 NDArray 不会改变 prepared graph/module；public `CompiledModule::constants()` 也返回独立 payload snapshots，无法反向修改 variant 或 prepared template，RuntimeSession 则通过 private const borrow 避免执行面复制。prepare/finish 共用同一 profiling context/run id。实际操作边界递增的 counters 为 contract=1、Relay pipeline=1、capability=3、ValueGraph=1、partition=1。
- `GraphTemplate` 直接从 prepared `ValueGraph`/`PartitionedGraph` 构造，不重新遍历用户 Relay 或重跑 pass。graph semantic identity 不再借用 debug Relay printer，而是长度分隔的 prepared-graph canonical bytes：包含 ordered graph/value/unit routing、value type/origin/output role 和 constant dtype/shape/full payload bytes。constant payload 因此绑定 graph/profile/plan，但不进入可跨图复用的 `UnitSemanticKey`/primitive `ArtifactKey`。shape unit semantic key 包装完整 production core `UnitSemanticKey::canonical_bytes()`；`value.<id>` 仅用于 routing。
- target identity 使用 Compiler/production cache 共用的一个 canonicalizer，包含所有稳定 codegen capability（含 `max_shared_memory_per_block`）；volatile `available_global_memory` 明确不进入 reusable identity。调用方 Target 在 adapter 入口 deep-copy，此后变化不影响 prepared template。
- 当前 Relay `TensorTypeNode::shape` 仍为 concrete `Array<int64_t>`、TE lowering 仍要求 `IntImm`，故 `ShapeProgram` 声明零 symbols，只有 empty `BindingSet` 可以 instantiate；non-empty binding/multi-profile 明确 hard-gate。`-1`、负维、非 f16/f32、非 unit lane、arch/device/target/backend 不匹配均 fail closed；没有 larger/fuzzy reuse。
- exact contract 是 contiguous row-major、显式 canonical stride、`global` memory scope、natural dtype alignment，并强制 logical=physical=valid。adapter 在接受 variant 前逐一检查 exact profile、value-name↔unit-boundary contracts、shape artifact full fields/signature digest、`ValueSpec`、ordered `KernelCall` ids、compiled signature dtype/shape/device/alignment、launch metadata、target/backend ABI、full reconstructed production `ArtifactKey`、public strong `ArtifactPin` 及 pin signature/launch digests。shape digest 永不等同 compiler artifact digest。
- `AssembleExactPlan` 使用真实 compiler TIR/codegen 和 production cache singleflight/publish/wait，保留 pins；返回 `ExactPlanVariant(module, plan, shape profile key, shape plan key, pins)`。shape plan identity 以长度分隔字段绑定 graph/profile、每个 shape+production artifact、pin signature/launch digest、link symbol、static generation `0`，以及完整 static `ExecutablePlan` value/storage/call routing 和 `last-use-sequential-single-stream-v1` memory-plan version。`RuntimeSession` 只接收 variant 的 module+plan，不接触 Compiler、Shape 或 cache。

### 3.2 Tests and current evidence

新增 `test/shape_production_exact_test.cpp`，CTest labels：`shape-production;cpu`，并有 `run_shape_production_exact_test` custom target。

- Gate OFF: adapter fails closed.
- Gate ON + LLVM OFF: true preparation/counter immutability, prepared constant payload deep-freeze、public `CompiledModule` constant accessor deep-copy regression, empty exact profile, non-empty binding hard gate, `-1`/unsupported dtype/no-compute/foreign-profile negatives, stable target snapshot与 production artifact target identity（并验证 volatile available-memory exclusion），same-shape/dtype different-constant graph/profile split，以及已知 backend unavailable 在 cache acquire 前拒绝；这些负例/prepare-only cases 不改变 entry/miss/in-flight/failure 计数。
- `#if KXC_USE_LLVM`: clears production cache; verifies two-unit add→mul miss then hit/pins, all preparation counters unchanged, graph-local unit renumbering reuses/relocates semantic artifacts, a different concrete extent produces new exact misses（无 larger/smaller fuzzy reuse），不同 constant payload 复用同一 primitive artifact 但产生不同 frozen PlanVariant/module numeric result（并验证 prepare 后原 payload mutation 不泄漏）, module/plan cardinality and cache lookup, static `RuntimeSession` numeric execution, runtime shape/dtype negatives, and session lifetime after all variant/prepared/oracle objects are destroyed plus cache clear.

本机已执行（CPU, CUDA/LLVM OFF, gate ON）：

```bash
cmake -S . -B out/build/shape-production -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_BUILD_PASS_TESTS=ON -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/build/shape-production --target shape_production_exact_test --parallel 2
ctest --test-dir out/build/shape-production --output-on-failure -R shape_production_exact_test
ctest --test-dir out/build/shape-production --output-on-failure --no-tests=error \
  --label-regex '(^|;)cpu(;|$)'
cmake --build out/build/shape-production --target check_include_layers check_public_headers --parallel 2
cmake -S . -B out/build/shape-production-off -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=OFF \
  -DKXC_BUILD_PASS_TESTS=ON -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/build/shape-production-off --target shape_production_exact_test --parallel 2
ctest --test-dir out/build/shape-production-off --output-on-failure -R shape_production_exact_test
```

结果：gate-ON `shape_production_exact_test` passed；随后完整 CPU label CTest 为 **39/39 passed**；Relay op contract **19/19**、pass contract **20/20**；include-layer 244 files passed；96 public headers self-contained/compiled passed。另以同样 CPU/LLVM-OFF 配置、`-DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=OFF` 构建并运行 `shape_production_exact_test`，gate-OFF branch passed。LLVM package 本机不可用，因此上列 conditional LLVM E2E numeric/cache/runtime coverage **未在本机运行**。CI CPU smoke 和 LLVM job 均启用 gate；LLVM job builds/runs this test in its explicit regex。本机配置明确关闭 CUDA，当前也没有 W2 CUDA exact E2E 证据，故不作 CUDA production-exact 完成声明。

当前硬 blocker 仍是 Relay `Array<int64_t>`/TE `IntImm` concrete-only shape representation；没有 production bucket/polymorphic claim，也没有 dynamic shape/output claim。

## 4. W1 Shape contract 测试证据

### 4.1 分阶段 focused tests

CPU-only、LLVM/CUDA disabled：

```bash
cmake -S . -B out/shape-phase1 -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_PASS_TESTS=OFF -DKXC_BUILD_CODEGEN_TESTS=OFF

cmake --build out/shape-phase1 --target \
  run_shape_system_test \
  run_shape_specialization_test \
  run_shape_guarded_specialization_test \
  check_include_layers check_public_headers -j2
```

结果：

- `shape_system_test`：4/4 groups passed。
- `shape_specialization_test`：3/3 groups passed。
- `shape_guarded_specialization_test`：2/2 groups passed。
- include-layer check passed。
- 85 个 public headers self-contained compile passed。
- Relay operator contract：19/19 passed。
- Pass contract：19/19 passed。
- `git diff --check` passed。

覆盖的关键负例包括：`-1`/负维、常量和 binding-time overflow、derived stride/physical byte overflow、未知 layout、zero/overlap/noncanonical stride、未绑定/矛盾 constraint、显式 gated SameRank/LayoutCompatible、非法 broadcast/range/divisibility、logical/physical/valid 越界、same key/different full template oracle、f32/f16、device kind/id、target/backend/ABI miss、非 exact profile、larger exact artifact 误复用、consumer-before-producer、bucket 无 guard/tail/pad/crop、capacity 过小、错误 physical stride、polymorphic 域外/整除失败/缺 proof/runtime scalar/错误 ordinal。guarded resolver 边界另覆盖 forged call order/locator、input/output sequence、rank、noncanonical stride、physical byte overflow、dtype、device kind/id、target、backend、backend ABI、guard canonical、完整 artifact payload 和缺失 request；全部拒绝且不污染 coordinator 状态。

### 4.2 ASan + UBSan

```bash
cmake -S . -B out/shape-sanitize -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_PASS_TESTS=OFF -DKXC_BUILD_CODEGEN_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -O1 -g' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'

cmake --build out/shape-sanitize --target \
  shape_system_test shape_specialization_test shape_guarded_specialization_test -j2
```

三个 Shape test 在 `ASAN_OPTIONS=abort_on_error=1:detect_leaks=1`、`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` 下全部通过。

另以 `-Wall -Wextra -Wpedantic -Werror` 构建并运行三个 Shape tests，全部通过。

### 4.3 现有静态路径回归

在 `out/shape-regression`（CPU-only、LLVM/CUDA disabled）构建并运行：

- `graph_partition_test`
- `infer_type_test`
- `compiler_contract_test`
- `compiler_extension_contract_test`
- `kernel_signature_test`
- `executable_plan_test`
- `runtime_session_test`
- `operator_compilation_test`
- `pass_pipeline_test`
- 全部三个 Shape tests

以上全部通过。`runtime_session_test::dynamic_input` 仍作为 legacy `-1` input sentinel 回归存在；本轨没有把它接入或宣传为新 Shape 系统。

本机 Python 缺少 ONNX/Numpy，因此 CMake 按既有逻辑跳过 `onnx_importer_test`；遵守约束，未安装或下载依赖。

## 5. 历史 W1 提交与 W2 工作树状态

W1 按 exact-first 顺序形成原子提交：

1. `7667750 feat(shape): add symbolic shape foundation`
2. `079f7b9 feat(shape): add exact specialization contracts`
3. `0911f1b feat(shape): add guarded optimized profiles`
4. `0cc1f1a fix(shape): disambiguate guarded request check`
5. `2755501 docs(shape): add compiler foundation handoff`
6. `fb96107 fix(shape): harden experimental exact contracts`
7. guarded resolver trusted reconstruction、forged-request negatives、installed-header ABI 标记与本 handoff 更新。

本节 W2 production exact 以单一 atomic feature commit 交付；最终 commit SHA 由本次交接报告记录。未 push、未 merge，也未修改其他 worktree。

## 6. 跨轨硬阻塞

### 6.1 Core / Track 01

W2 exact 已消费下列 Core facilities；symbolic/multi-profile 扩展仍必须维持这些边界：

1. capability verifier 在 compiler 入口、graph Pass 后、partition 前 fail closed；
2. `UnitSemanticKey` 从当前 partition structural hash 中移除 graph-local value id、symbol、span 等；
3. 唯一 `PipelineResolver`/pipeline fingerprint；
4. target/backend/schedule/ABI 的正式 artifact key 字段；
5. immutable artifact pin，避免 cache hit 后二次 peek 被淘汰；
6. dtype、argument role、alias、workspace 等非 Shape signature contract。

W2 adapter 不写 fake artifact：它只通过 production primitive cache 交易并保留真实 pin。

### 6.2 Relay / Frontend

生产 symbolic path 仍被以下事实阻塞：

- `TensorTypeNode::shape` 仍是 `Array<int64_t>`；
- 当前 type rules 仍传播 legacy `-1`；
- lowering 仍把 shape 固化为 `IntImm`；
- ONNX v1 路径仍可能将 unknown dim 具体化。

集成前必须提供 preserve/bind/reject 的版本化 frontend/type adapter，并对每个 allowlisted op 建立 type relation、ShapeProgram 和 lowering 的同源规则或 concrete differential test。不能把 unknown non-batch dim 填 `1` 后写入新 key。

### 6.3 Adaptive / Track 03

当前 coordinator 只是 deterministic synchronous fake。生产 shape-aware dispatch 需要 Track 03 提供：

- full-key singleflight；
- immutable artifact/pin；
- failure/cancel/backoff/budget；
- `KernelSlot` generation 与 lease；
- publish validation 和 in-flight 保活。

Track 03 接入不得改变本轨 exact applicability，也不得将 cache miss 或 larger capacity 变成 fallback。

### 6.4 Runtime / Codegen / Track 05

bucket 真正执行之前必须具备：

- runtime-owned logical/physical/valid-extent `ValueSpec` contract；
- physical stride/layout/alignment-aware allocation 和 storage reuse；
- 显式 pad/mask/crop plan steps；
- tail-safe kernel 数值和越界证据。

polymorphic 真正执行之前必须具备：

- 版本化 runtime extent scalar calling convention；
- `KernelSignature`/launcher/codegen 对 scalar ordinal 和 guard 的一致支持；
- backend tail predicate 与 workspace 上界验证。

可确定 dynamic output 真正分配之前必须具备 Track 05 的：

```text
ShapeEvalTask -> AllocateTask -> KernelTask
```

当前 `ShapeProgram::Evaluate` 已能纯计算可确定 output contract，但本轨没有越权修改 runtime task/allocator。data-dependent/ragged output 继续拒绝。

## 7. 建议集成顺序

1. **先接 Core M1 DTO。** 用正式 capability/pipeline/unit/artifact/dispatch key 替换 `compiler_foundation_v1` fake 字段；保留 full canonical equality 测试。
2. **接 frontend/Relay preserve-bind-reject adapter。** 新 symbolic 路径不得产生 `-1`；旧静态构造保持兼容。
3. **只接 exact。** `PrepareGraphTemplate` 一次完成 graph Pass/capability/partition；多次调用 `InstantiateExactProfile`，再映射到现有 static `PrimFunc`/`KernelSignature`/`ExecutablePlan`。exact 与独立 concrete baseline 做数值差分。
4. **冻结 runtime tensor contract v2。** 先让 exact 的 logical=physical=valid 走通且静态回归不变。
5. **接 Adaptive production resolver。** 以 exact request 验证 pin/singleflight/generation；`RuntimeSession` 仍只消费 frozen module+plan。
6. **再开放 bucket。** 仅消费 `BucketPolicy` 明确 guard/physical/tail/pad/crop 的 allowlisted unit；域外走 exact/其他显式 profile/拒绝。
7. **最后开放 polymorphic。** 等 runtime extent ABI 与 codegen guard 同时冻结后，映射 `PolymorphicPolicy`；域外 dispatch 必须在 launch 前拒绝。
8. **dynamic output 最后。** 仅将可确定 ShapeProgram 结果交给正式 ShapeEval/Allocate task；ragged/data-dependent 另立协议。

## 8. 交接判定

本工作树提供可复审的 experimental-v1 Shape contract，并提供 **default-OFF、concrete-only W2 production exact adapter**。adapter 是 production compiler/cache/module/plan/pin 的窄桥，但 installed experimental headers 仍不承诺 source/binary ABI compatibility，也不构成 stable public v1 或完整 dynamic Shape 完成声明。

未完成项仍需要其他轨道的生产 contract 或数据面：Relay/frontend symbolic representation、Adaptive multi-profile lifecycle、Runtime physical bucket plan、Codegen extent ABI、Region task vocabulary。状态为 **W2 exact gated / stable public ABI and dynamic production integration not done**；不通过侵入现有 Relay/Runtime、`-1` 或 `cached_dims >= query_dims` 绕过依赖。
