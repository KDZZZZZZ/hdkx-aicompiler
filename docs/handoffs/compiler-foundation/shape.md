# Compiler Foundation / Shape 轨道交接

> **分支：** `feature/compiler-foundation-shape`
>
> **基线：** `e295a73`（compiler-foundation roadmap）
>
> **状态：** 隔离 Shape contract、exact oracle、guarded bucket/polymorphic contract fake **已完成**；生产 Relay/Frontend/Runtime/Codegen 接入只剩下文列出的 Core/Adaptive/Runtime/Region 硬依赖。
>
> **重要边界：** 本页不宣称生产主链已经支持 dynamic shape、bucket kernel、polymorphic launch 或 dynamic output allocation。

## 1. 已交付范围

本轨先建立了不依赖 Relay、Compiler、RuntimeSession 或 backend 的 `kxc::shape` 模块，并保持现有静态执行路径不变。

### 1.1 纯 Shape 基础

公共头：`include/kxc/shape/shape.h`

- 版本：
  - `kShapeContractVersion = 1`
  - `kShapeAbiVersion = 1`
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
- `ExactConstraintSolver`
  - 只推导 `Symbol = 已完全可求值表达式`。
  - range/divisibility/broadcast 不用于猜值。
  - 未绑定、额外 binding、矛盾、负值和 overflow 均 fail closed。
- 三层 tensor contract：
  - `LogicalShape`：数学维度和可选 axis name。
  - `PhysicalShape`：capacity、stride、layout、alignment、memory scope。
  - `ValidExtent`：本次调用的有效范围。
  - concrete evaluation 强制 `valid <= logical <= physical`；exact specialization 进一步强制三者相等。
- identity DTO：
  - `GraphTemplateKey`：graph semantic、pipeline、capability、partition fingerprint 和 contract version。
  - `ShapeProfileKey`：template key、完整 concrete bindings、显式 policy id 和 shape-ABI version。
  - canonical bytes 使用长度前缀字段；相等判断比较完整结构，不以 hash 单独决定等价。
- `ShapeProgram`
  - 声明 symbols、named input/output tensor contracts 和 constraints。
  - 构造/`Verify` 检查名称、rank、symbol 引用及确定性排序。
  - `Evaluate` 只求 shape/extent/stride；不分配、不查 cache、不编译、不 launch。

### 1.2 GraphTemplate / exact profile 拆分

公共头：`include/kxc/shape/specialization.h`

- `GraphLocalCallLocator` 与 `UnitSemanticKey` 明确分离。
  - call locator 只服务 plan routing/诊断。
  - artifact identity 不包含 call locator、graph value name/id、storage id、entry symbol 或对象地址。
- `GraphTemplate`
  - 持有 `GraphTemplateKey`、`ShapeProgram` 和有序 `UnitSkeleton`。
  - verifier 检查 producer/consumer 顺序、重复 producer、缺失 producer 和显式 unit boundary。
  - 重复 `UnitSemanticKey` 合法，允许等价 unit 共享 artifact。
- `InstantiateExactProfile`
  - 返回只能由该函数 mint 的 `ExactOracle`。
  - exact profile 强制所有 value 的 logical/physical/valid 完全相等。
  - profile instantiation 不暴露 graph-pass/partition hook，因此不会在该层重跑 graph preparation。
- `KernelArtifactKey`
  - 包含 unit semantic key、shape ABI、pipeline/capability fingerprint 和有序 boundary contracts。
  - 不包含 `ShapeProfileKey` 整体，因此 graph-local/template identity 或与该 unit 无关的 binding 不会污染 artifact identity。
- `MakeExactSpecializationRequests`
  - 生成有序 exact requests；request 保留 graph-local routing，artifact key 不保留。
- `PlanVariantKey`
  - plan identity 可记录 call/entry/generation；与 artifact semantic identity 分开。

### 1.3 版本化 deterministic fake

公共头：`include/kxc/shape/fakes/compiler_foundation_v1.h`

namespace：`kxc::shape::fakes::compiler_foundation_v1`

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

公共头：`include/kxc/shape/guarded_specialization.h`

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
  - 显式记录 guard、逐 unit allowlist/proof、target、workspace 上界、symbolic boundaries 和版本化 runtime-extent scalar ABI。
  - scalar ordinal/name/symbol/range/divisibility 必须完整且有序；所有声明 symbol 必须有 guard 和 runtime binding。
  - 当前仅冻结 contract/fake；没有把现有 `-1` input ABI 当作 runtime extent ABI。
- `GuardedArtifactKey`
  - bucket identity 包含显式 bucket/guard/physical/tail/workspace/environment contract，但排除 exact request binding。
  - polymorphic identity 包含 symbolic boundary/guard/proof/runtime extent ABI/environment，但排除 concrete request binding。
  - 因此多个请求只能在同一显式 guard 和完整证明相同时共享；不存在 `cached_dims >= query_dims` 路由。
- guarded deterministic fake
  - `GuardedDeterministicMockCoordinator` 和 `GuardedDeterministicMockPlanAssembler` 保持 generation `0`、完整 key 比较和有序 assembly。
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
- `tools/architecture/check_include_layers.py` 将 Shape 设为只允许依赖自身的底层模块。
- Shape 公共头只 include Shape/stdlib；没有 include `kxc/compiler/*`、`kxc/relay/*`、`kxc/runtime/*`、frontend 或 private header。
- 未修改生产 Relay、type inference、lowering、`KernelSignature`、`ValueSpec`、memory planner、`RuntimeSession`、cache 或 frontend。

## 3. 测试证据

### 3.1 分阶段 focused tests

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

覆盖的关键负例包括：`-1`/负维、overflow、未绑定/矛盾 constraint、非法 broadcast/range/divisibility、logical/physical/valid 越界、非 exact profile、larger exact artifact 误复用、consumer-before-producer、bucket 无 guard/tail/pad/crop、capacity 过小、错误 physical stride、polymorphic 域外/整除失败/缺 proof/runtime scalar/错误 ordinal。

### 3.2 ASan + UBSan

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

### 3.3 现有静态路径回归

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

## 4. 提交

按 exact-first 顺序形成原子提交：

1. `7667750 feat(shape): add symbolic shape foundation`
2. `079f7b9 feat(shape): add exact specialization contracts`
3. `0911f1b feat(shape): add guarded optimized profiles`
4. `0cc1f1a fix(shape): disambiguate guarded request check`
5. handoff：当前 `docs(shape): add compiler foundation handoff` 提交（见本分支 HEAD/log）

未 push、未 merge，也未修改其他 worktree。

## 5. 跨轨硬阻塞

### 5.1 Core / Track 01

生产 GraphTemplate/artifact 接入必须等待：

1. capability verifier 在 compiler 入口、graph Pass 后、partition 前 fail closed；
2. `UnitSemanticKey` 从当前 partition structural hash 中移除 graph-local value id、symbol、span 等；
3. 唯一 `PipelineResolver`/pipeline fingerprint；
4. target/backend/schedule/ABI 的正式 artifact key 字段；
5. immutable artifact pin，避免 cache hit 后二次 peek 被淘汰；
6. dtype、argument role、alias、workspace 等非 Shape signature contract。

在这些字段冻结前，本轨只提供 `compiler_foundation_v1` fake，不能把 fake artifact 写入共享生产 cache。

### 5.2 Relay / Frontend

生产 symbolic path 仍被以下事实阻塞：

- `TensorTypeNode::shape` 仍是 `Array<int64_t>`；
- 当前 type rules 仍传播 legacy `-1`；
- lowering 仍把 shape 固化为 `IntImm`；
- ONNX v1 路径仍可能将 unknown dim 具体化。

集成前必须提供 preserve/bind/reject 的版本化 frontend/type adapter，并对每个 allowlisted op 建立 type relation、ShapeProgram 和 lowering 的同源规则或 concrete differential test。不能把 unknown non-batch dim 填 `1` 后写入新 key。

### 5.3 Adaptive / Track 03

当前 coordinator 只是 deterministic synchronous fake。生产 shape-aware dispatch 需要 Track 03 提供：

- full-key singleflight；
- immutable artifact/pin；
- failure/cancel/backoff/budget；
- `KernelSlot` generation 与 lease；
- publish validation 和 in-flight 保活。

Track 03 接入不得改变本轨 exact applicability，也不得将 cache miss 或 larger capacity 变成 fallback。

### 5.4 Runtime / Codegen / Track 05

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

## 6. 建议集成顺序

1. **先接 Core M1 DTO。** 用正式 capability/pipeline/unit/artifact/dispatch key 替换 `compiler_foundation_v1` fake 字段；保留 full canonical equality 测试。
2. **接 frontend/Relay preserve-bind-reject adapter。** 新 symbolic 路径不得产生 `-1`；旧静态构造保持兼容。
3. **只接 exact。** `PrepareGraphTemplate` 一次完成 graph Pass/capability/partition；多次调用 `InstantiateExactProfile`，再映射到现有 static `PrimFunc`/`KernelSignature`/`ExecutablePlan`。exact 与独立 concrete baseline 做数值差分。
4. **冻结 runtime tensor contract v2。** 先让 exact 的 logical=physical=valid 走通且静态回归不变。
5. **接 Adaptive production resolver。** 以 exact request 验证 pin/singleflight/generation；`RuntimeSession` 仍只消费 frozen module+plan。
6. **再开放 bucket。** 仅消费 `BucketPolicy` 明确 guard/physical/tail/pad/crop 的 allowlisted unit；域外走 exact/其他显式 profile/拒绝。
7. **最后开放 polymorphic。** 等 runtime extent ABI 与 codegen guard 同时冻结后，映射 `PolymorphicPolicy`；域外 dispatch 必须在 launch 前拒绝。
8. **dynamic output 最后。** 仅将可确定 ShapeProgram 结果交给正式 ShapeEval/Allocate task；ragged/data-dependent 另立协议。

## 7. 交接判定

本工作树内可独立完成的 Shape 轨工作已闭环：Shape IR、exact solver、三层 tensor contract、ShapeProgram、template/profile split、exact oracle、versioned deterministic fake、显式 guarded bucket/polymorphic DTO 与正反例均已实现并验证。

未完成项均需要改变其他轨道拥有的生产 contract 或数据面：Core identity/capability/cache、Relay/frontend symbolic representation、Adaptive lifecycle、Runtime physical plan、Codegen extent ABI、Region task vocabulary。故本轨状态为 **isolated contract Done / production integration hard-blocked**，不通过侵入现有 Relay/Runtime 或恢复 fuzzy cache 来绕过依赖。
