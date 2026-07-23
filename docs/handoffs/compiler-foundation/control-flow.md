# Compiler Foundation / Dynamic Graph & Control Flow 交接

> **W2 更新：** 本页记录 W1 preparation baseline；default-OFF resolved CPU executor、
> typed `CompiledModule` adapter 与新限制见
> [`control-runtime.md`](control-runtime.md)。真实 Relay/TE artifact resolver 是
> `CompileControlFlowExact` 的 default-OFF LLVM-only 路径；本页记录其受限边界。
>
> 分支：`feature/compiler-foundation-control-flow`
>
> 范围：Track 04，static-exact Shape 控制流语义与 Track 05 frozen mock 边界
>
> 状态：Track 04 的 ANF、capability gate、structured `If`/loop DTO、verifier、Relay `If` 和 bounded `While` preparation lowering、deterministic reference executor 已完成；preparation kernel 显式保持 `UnresolvedRelayKernel`。`CompileControlFlowExact` 仅在显式 gate 和 LLVM 下将每个 region Call 解析为真实 artifact；runtime 仍只消费已解析 schema。

## 1. 能力边界：三个“动态”不是同一件事

| 概念 | 本分支状态 | 精确定义 |
|---|---|---|
| 动态控制流 | **已完成 static-exact 语义层** | Relay 数据谓词 `If` 可降为经过验证的 `ControlPlan v2`；手写 frozen DTO 支持有界 loop、Phi 和 loop-carried values，并可由 reference executor 执行。当前 `RuntimeSession` 尚不执行这些控制任务。 |
| 动态图 topology | **未实现** | 不支持 eager 每次执行构图、tracing、按已走路径泛化、运行期增删 task 或任意 data-dependent topology。`ControlPlan` 是预先冻结的 structured region/task 图。 |
| dynamic Shape | **显式拒绝** | `ControlPlan::ValidateStaticExact()` 拒绝任何负维度，包括 `-1`。Phi、loop initial/body argument/backedge/result 必须具有完全相同的 dtype、rank、shape 和 device。没有 symbolic `DimExpr`、bucket、polymorphic、dynamic allocation 或动态输出。 |

仓库其他静态 runtime ABI 中的 `kDynamicDimension = -1` 仍只是 legacy input validation sentinel；本轨没有把它当作 dynamic Shape、动态图或控制流能力。

本分支也**没有**实现或声称支持 eager/tracing。`LowerRelayToControlPlan` 只准备冻结的编译 IR，不创建 session、不启动 backend、不做 tracing。

## Relay `While` restricted path

Relay now has `While(initial_state, loop_var, condition, body, max_trip_count)`.
It is deliberately one lexical carried-state binder, not a general function/recursion
mechanism.  State is a static `TensorType` or recursively nested `TupleType` of
Tensor leaves; the bound is mandatory and non-negative.  Evaluation is
condition-before-body: zero trips are valid, and a true condition after the bound
raises `loop max_trip_count exhausted` rather than unrolling on the host.

`InferType`, deterministic ANF, virtual-device collection, printer, cloning and
manual Relay walkers either traverse this scope or reject it explicitly.  The
restricted control lowering flattens the carried leaves into `LoopSpec` bindings
`{result, initial, body_argument, backedge}`, creates separate condition/body
regions, and captures outer values explicitly.  It accepts only static exact,
pure deterministic non-aliasing registered Calls and CPU scalar-bool conditions;
control tasks are CPU:0/default-stream.  `Compiler::Compile` remains static
dataflow and rejects it with `control_flow.loop` before `ValueGraph`.

`Compiler::CompileControlFlowExact` is still default OFF.  When enabled it accepts
validated bounded loops and resolves every condition/body Call through the normal
`Compiler::Compile` path, retaining the resulting pins in its typed lease.  LLVM
is required for its CPU production subset; a non-LLVM build fails closed.  The
runtime execution plan/session remains Relay-, Compiler-, and cache-free.

## 2. 已冻结语义

### 2.1 Relay executable gate 与 Let/ANF

新增 API：

- `kxc::relay::NormalizeToANF`
- `kxc::relay::IsANF`
- `kxc::relay::VerifyANF`
- internal `kxc::api::internal::VerifyExecutableCapability`
- internal `StaticDataflowExecutableCapabilities()`

`Compiler` 在 Relay graph passes 和最终 `InferType` 后执行 ANF normalization。归一化保证：

- executable `Call`、`If`、`TupleGetItem` 被显式 Let 绑定；
- Call 参数和 If predicate 为原子值；
- 求值顺序确定，shared producer 不被复制；
- 两个分支分别归一化，分支工作不会被错误提升到父 region；
- checked type、VirtualDevice 和 Span/source metadata 被保留；
- 已归一化输入再次执行保持 idempotent。

`BuildValueGraph` 入口首先执行 static-dataflow capability verifier，然后才遍历 Relay：

- `Let` 通过词法环境解析；绑定 value 只解析一次，Let 本身不产生 value 或 kernel；
- free/unbound Var 仍 fail closed；
- `If` 在进入 ValueGraph 前以 `required capability=if` 拒绝；
- 所有 TensorType 必须是 static exact；
- function values、closures、recursion 和非 ordinary registered Op 被拒绝。

因此没有 Relay `If` 未经 verifier 穿过 ValueGraph 的路径。现有无控制流 DAG 仍走原来的 per-Call `ValueGraph -> CompilationUnit -> ExecutablePlan -> RuntimeSession` fallback。

### 2.2 `ControlPlan v2`：Track 04/05 frozen mock

公共 runtime-neutral DTO 位于 `include/kxc/runtime/control_plan.h`，不依赖 Relay、TE、TIR、Compiler、cache 或 backend module。

核心结构：

- `ControlValueSpec`：stable value id、static logical shape、dtype、完整 `Device(type,id)`、source locator；v2 中 logical/physical/valid extent 视为完全相同，尚未引入 capacity/padding。
- `ControlTask`：全局唯一 task id、`kKernel`/`kBranch`/`kLoop`、`KernelBindingState`、unique boundary inputs、duplicate-preserving `argument_values`、outputs、dependencies、完整 `Device(type,id)`、single `default` stream、effect/alias、source locator；v2 kernel 只能是 `kUnresolvedRelayKernel`，branch/loop 只能是 `kNotApplicable`。
- `ControlRegion`：全局唯一 region id、live-ins/live-outs、ordered tasks、effect/alias summary 和 source locator。
- `BranchSpec` / `PhiBinding`：CPU scalar bool predicate、两个不同 child regions、每个 result 对应 then/else source。
- `LoopSpec` / `LoopCarriedBinding`：独立 condition/body regions，显式 `{result, initial, body_argument, backedge}`，CPU scalar bool condition，mandatory `max_trip_count`。
- `graph_inputs`、`constant_values`、`graph_outputs`：输入与常量均是显式 source；常量 payload 由提交 reference/frozen plan 的调用方绑定，不藏在 DTO 中。

ID、source locator、`kernel_ref` 只用于 plan routing、诊断和 mock artifact 定位，不是跨图 artifact semantic key。`kernel_ref + kUnresolvedRelayKernel` 不是 executable symbol/handle；compiler-side artifact resolver 完成真实 TE output 与 artifact ABI 验证并产出独立 frozen executable task 前，任何 runtime adapter/executor 都不得把它当作可执行 kernel。新增必填 binding 语义使 schema 从 v1 提升为 v2；validator 对旧 v1 fail closed，不在同一版本号下重解释旧 plan。

### 2.3 Structured verifier

`ControlPlan::ValidateStaticExact()` / `VerifyControlPlan()` 检查：

- schema 必须是 v2；旧 v1 明确拒绝；value/region/task id 唯一且非负；所有引用存在；
- region 必须形成从 entry 可达的 structured tree，不允许递归 region/unstructured jump；
- task dependency 只能指向同 region 的前序 task；读取本 region producer 的 task 必须声明 producer dependency；
- 每个非 source value 只有一个 producer，source/body argument 不得被 task 重定义；
- region live-in/live-out、已消费 value 的 producer 和 task dependency closure 完整；entry boundary 精确匹配 graph inputs、constants 和 outputs；纯 task 的未选 tuple leaf/dead output 可保留为有 producer但无 consumer 的值；
- kernel 的 boundary inputs 唯一，但 `argument_values` 保留逻辑顺序与重复实参；其 unique set 必须等于 inputs；
- kernel inputs/outputs 与 task 的完整物理 device identity 一致；v2 只允许显式 CPU/CUDA data placement 和单 `default` stream；CUDA ordinal 不得被合并；
- branch predicate 必须是 CPU scalar bool，不做隐式 device-to-host copy；
- Phi result、then source、else source exact-contract 相等，且 source 必须是对应 child live-out；
- loop 在 body 前执行 condition，允许 zero trip；true condition 超过 `max_trip_count` 明确失败，不静默展开；
- loop initial、body argument、backedge、result exact-contract 相等；shape-changing loop 明确失败；
- v2 只接受 pure/fresh-output task：effect reads 必须与 inputs 相符，writes/allocates/host callback/device sync 被拒绝；`must_alias`/`may_alias` 被拒绝，`no_alias` pair 需合法且无重复。

当前 alias/effect 约束是保守正确性基线。Track 05 在引入 in-place、workspace、event 或 storage reuse 时必须升级 schema/validator，而不能绕过这些检查。

### 2.4 Relay `If` preparation lowering

公共 API：

```cpp
kxc::runtime::ControlPlan
kxc::api::LowerRelayToControlPlan(kxc::Function function);
```

该 API 执行：

```text
InferType -> NormalizeToANF -> VerifyANF
          -> static-exact capability verifier (allow_if=true)
          -> ControlPlan v2 lowering -> ValidateStaticExact
```

规则：

- 参数先分配 deterministic graph-input ids；常量进入 `constant_values`；tuple 递归 flatten 为 tensor leaves；
- Let 只建立词法别名，不创建 task；
- ordinary Call 只有在 OperatorSpec 为 pure、deterministic、`alias_contract == "none"`，输入/输出 arity、attrs、类型关系、single/multi 输出结构及 TE lowering binding 类型正确且 `std::function` 非空时，才成为 unresolved kernel task；static ValueGraph 使用同一 fail-closed capability gate；
- kernel task 保存 unique inputs 和 duplicate-preserving logical arguments，并按同 region producer 生成依赖；
- Call 的所有 inputs/outputs 必须位于同一完整 `Device(type,id)`，structural Tuple/TupleGetItem/Let/Function alias 的显式 placement 也必须与 leaf 一致，禁止隐式 copy 或丢失 CUDA ordinal；
- If 生成父 branch task、两个 child regions 和每个 flattened result 对应的 Phi；nested If 保持 structured；
- branch capture（包括 constant）成为 child live-in 和 branch task input；
- equivalent Relay objects 生成相同 canonical plan text；canonical values 按 id、regions 按显式 `region_order` 输出；
- duplicate graph output id 在 v2 明确拒绝。

本轮采用 preparation 边界 **B**：`LowerRelayToControlPlan` 不调用 TE hook，也不声称已解析真实 TE outputs；它只验证 callable/binding/schema/arity 后，将每个 Relay kernel 标成 `KernelBindingState::kUnresolvedRelayKernel`。因此 single hook 的 undefined output、multi hook 的空/undefined/count/dtype/rank/static-shape 错误不会被 preparation 假装成 executable artifact。

真实无控制流生产路径 `Compiler::Compile -> LowerGraph -> LowerCompilationUnit` 仍逐 unit 调用 TE hook，并在进入 `LowerTensorGraphToTIR`/artifact 构建前，按 `unit.output_value_ids -> ValueInfo` 精确验证 output count、defined、dtype、rank、每一维为 static `IntImm` 且 shape 完全相等。生产 static ValueGraph 继续 fail closed 拒绝 nested Tuple Call output；nested leaves 目前只可存在于 unresolved ControlPlan preparation，不能绕过 flat production ABI gate。

这条 API 与生产 `Compiler::Compile` 分离。生产静态路径仍拒绝 If，避免在 Track 05 runtime verifier/executor 完成前把控制流塞入 ordered `ExecutablePlan`。

### 2.5 Deterministic reference executor

测试 oracle 位于 `test/support/control_plan_reference_executor.h`，消费相同 frozen DTO，并以 test-only fake callback 显式解析 `kUnresolvedRelayKernel`。该行为不属于 runtime artifact 语义；它：

- 执行前调用 plan verifier；
- 校验 graph input/constant/output 的 static contracts；
- 按已验证 region/task 顺序执行，并记录 deterministic read/task/write trace；
- branch 只执行被选 child，未选分支无 task/effect trace；
- Phi 只复制被选 source；
- loop 每轮绑定 body arguments，先执行 condition region，再按 backedge 更新 carried values；
- max-trip exhaustion、缺值、错误 arity、错误 dtype/shape/device 均明确失败。

该 executor 是语义 oracle 和 Track 05 mock integration fixture，不是生产 RuntimeSession，也不代表 backend control-flow execution 已接通。

## 3. 测试证据

### 3.1 新增测试

| 测试 | 主要覆盖 |
|---|---|
| `relay_anf_test` | nested/shared/tuple ANF、分支局部性、determinism、idempotence、Span 保留、ANF 负诊断 |
| `executable_capability_test` | static-exact/If/free Var/function value gate、Let ValueGraph 等价性、effect/alias/placement、input/output arity、missing/wrong/empty binding 早期拒绝；真实 per-unit single undefined、multi empty hook/output、undefined/count/dtype/rank/static-shape 负例；nested Call production flatten gate、nested TupleGetItem leaf routing、tuple parameter 不过度声明 |
| `control_plan_test` | schema/value/region/task/dependency closure、`KernelBindingState` unresolved/non-kernel 负例、完整 Device identity/stream、Branch/Phi、loop-carried、effect/alias、dynamic dim 和 shape-changing loop 负例、canonical determinism |
| `control_plan_reference_executor_test` | constants、重复逻辑实参、true/false、未选分支不执行、nested If、多 Phi、zero/one/multi-trip、多 carried、max exhaustion、deterministic trace |
| `relay_control_plan_test` | Relay Let/shared Call、Relay true/false If reference execution、nested If、tuple/multi-Phi、nested tuple projection/dead pure leaf、nested output flatten 为 unresolved task、constant capture、empty TE hook/effect/alias gate、CPU/CUDA ordinal/structural placement/dynamic shape/duplicate output 负例、ValueGraph 仍拒绝 If |

### 3.2 分阶段验证

最终使用现有依赖、CPU-only 配置验证；没有下载、安装或联网：

```bash
cmake -S . -B /tmp/kxc-control-flow-phase1 \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_CODEGEN_TESTS=OFF \
  -DKXC_BUILD_PASS_TESTS=ON
cmake --build /tmp/kxc-control-flow-phase1 -j2
cmake --build /tmp/kxc-control-flow-phase1 \
  --target run_control_flow_tests -j2
ctest --test-dir /tmp/kxc-control-flow-phase1 \
  --output-on-failure --no-tests=error -L control-flow
```

`run_control_flow_tests` 依赖并通过 CTest 实际执行以下 5 个 CPU control tests；`.github/workflows/ci.yml` 的 CPU-only job 使用同一 target，禁止只构建不执行：

- `relay_anf_test`
- `executable_capability_test`
- `control_plan_test`
- `control_plan_reference_executor_test`
- `relay_control_plan_test`

focused suite 为 **5/5 passed**。随后逐个执行 23 个已构建测试程序，全部通过，包括全部新增测试以及：

- `graph_partition_test`
- `executable_plan_test`
- `compiler_contract_test`
- `operator_compilation_test`
- `compiler_extension_contract_test`
- `kernel_signature_test`
- `compiled_module_test`
- `runtime_session_test`
- 其他现有 object/FFI/pass/device/type/profile/CUDA-schedule CPU contract tests

静态契约检查：

- Relay operator contract：**19/19 passed**
- Relay/TIR pass contract：**19/19 passed**
- include-layer check：**217 files scanned, passed**
- public-header compile check：**84 headers compiled, passed**
- `git diff --check`：passed

未运行 LLVM/CUDA backend 数值测试，因为本次验证配置显式关闭 LLVM/CUDA；本交接不据此声称真实 backend control-flow 支持。

## 4. 提交

| Commit | 内容 |
|---|---|
| `1dd2acf` | `feat(compiler): verify and normalize static Relay dataflow` |
| `2e53bdc` | `feat(runtime): add static-exact structured control plan` |
| `5b1ec7b` | `fix(runtime): complete control plan value contracts` |
| `a4630a2` | `fix(runtime): freeze control task placement` |
| `8288acf` | `feat(compiler): lower static Relay If to control plans` |
| `237385a` | `docs(compiler): hand off control flow foundation` |
| `24fae17` | `fix(compiler): close executable capability gaps` |
| `0ff3690` | `fix(compiler): harden control placement and output gates` |
| `efd42e4` | `fix(compiler): preserve structural control contracts` |
| `726abbc` | `fix(compiler): validate Let result placement` |
| `f145a0c` | `docs(compiler): record control flow hardening` |

本表不自引用当前 baseline closure 提交；以本文件所在 HEAD 为准。

## 5. 保留的 fallback 与明确未接线能力

### 保留

- `Compiler::Compile` 的无控制流 DAG 继续生成 per-Call units、multi-entry module 和 ordered `ExecutablePlan`。
- `RuntimeSession(module, plan)` 未增加 Relay、Compiler、cache、shape predictor 或 control policy 依赖。
- Let 归一化后通过已验证词法映射进入 ValueGraph，不改变 one ordinary Call = one fallback unit 的当前 policy。
- exact static path 继续是正确性 oracle。

### 未接线/未声称

- `RuntimeSession` 不执行 `ControlPlan`；`kUnresolvedRelayKernel` 不得作为 executable artifact 解释。
- 没有把 unresolved ControlTask 解析为 selected symbol/generation/signature 的 production artifact adapter。
- 没有真实 backend branch/loop task、event、copy、allocation 或 memory planning。
- 没有 Relay loop AST/frontend syntax；loop v2 当前由 frozen DTO fixtures 验证。
- 没有 eager/tracing、graph break、trace cache 或 path guards。
- 没有 dynamic Shape Phi、shape-changing loop、dynamic output、bucket、polymorphic 或 fuzzy fallback。
- 没有 device predicate 的隐式 host copy；非 CPU predicate 直接拒绝。
- 没有恢复旧 Adaptive Runtime，也没有把 RuntimeSession 变成编译控制面。

## 6. 跨轨硬阻塞

1. **Track 05：真实 frozen task-DAG/runtime contract。** 需要生产 `Task`/`ValueContract`/artifact generation、plan validator、single-stream control executor、branch/loop-aware liveness 和 event retirement。当前 ordered `ExecutablePlan` 的 call-index memory model不能证明 branch/backedge lifetime。
2. **Track 01：统一 capability/pipeline 门禁。** 本轨已有局部 executable verifier，并消费现有 OperatorSpec effect/alias 字段；进入生产主链前仍需 Track 01 的统一 capability API、pipeline phase/invariant 和版本化 contract ownership，避免形成第二套生产门禁。
3. **Track 02 + Track 05：动态形状/动态分配。** dynamic Shape Phi、shape-changing loop、dynamic output 必须等待 `DimExpr`/constraint/ShapeProgram 以及 logical/physical/valid-extent、AllocateTask 和 dependency-aware memory contract。
4. **Relay loop frontend contract。** 当前 Relay 没有 LoopNode、函数值调用或递归执行模型。不能用 op-name、host unroll 或 recursion 假装 loop；需要单独冻结 AST、visitor、type inference、pass 和 source locator 语义后再接到现有 `LoopSpec`。
5. **设备谓词与多 stream。** CUDA/device predicate 需要 Track 05 显式 Copy/Event/Sync task；多 stream 需要 completion event 参与 liveness。v2 因此只接受 CPU predicate 和 `default` stream。
6. **artifact/module 绑定。** `kernel_ref` 目前是 deterministic mock locator，`binding_state` 明确为 `kUnresolvedRelayKernel`；二者都不是 artifact semantic key/executable handle，也未绑定 `CompiledModule` generation。compiler-side resolver 必须持有 Relay/TE/CompilationUnit 上下文，重新调用真实 per-unit TE lowering，并在发布 selected symbol/generation/signature 前按 ControlValueSpec/ValueInfo 验证 flatten count、defined、dtype、rank/static shape；Track 05 runtime adapter 只消费 resolver 产出的独立 executable contract，并拒绝 unresolved task。生产接线仍依赖 Track 01/03/05 的 semantic key、immutable generation 和保活协议。

这些阻塞均不能通过让 Relay If 穿过 ValueGraph、恢复 fuzzy cache、把 capacity 当 logical shape，或让 RuntimeSession 调用 Compiler 来绕过。

## 7. 生产接线步骤

按以下顺序集成，任一步失败都保持现有 static DAG fallback：

1. **在 Track 01 注册统一 capability。** 将 `static_dataflow` 与 `static_exact_control_if` 变为版本化 capability；固定 post-graph-pass/pre-partition 调用点，并让 OperatorSpec effect/alias、target/layout 和 pipeline invariant 成为同一 verifier 输入。
2. **冻结 Track 05 runtime adapter。** Track 05 只消费 compiler-side resolver 产出的 runtime-only executable `FrozenPlanVariant`（保留 source ControlPlan schema/version provenance），不直接读取 unresolved `ControlPlan`；adapter 不得 include Relay/TE/TIR、不得按 kernel/op 名推断依赖，并必须拒绝任何仍为 `kUnresolvedRelayKernel` 的 task。
3. **在 compiler side 解析并绑定 kernel artifacts。** resolver 消费 `ControlPlan::kSchemaVersion == 2`，并持有 Relay Call、attrs、CompilationUnit/ValueInfo 和 TE registry 上下文；对每个 unresolved `kKernel` 建立真实 per-unit lowering inputs、调用 TE hook，并精确复核 flatten output count、defined、dtype、rank/static shape 与 ControlValueSpec/ValueInfo。通过后才使用 unit semantic key + target/ABI/pipeline fingerprint 查找或编译 immutable artifact；`kernel_ref` 只保留 locator/provenance，resolver 将 selected symbol/generation/signature 写入独立 executable frozen task contract，交给第 2 步的 runtime adapter。
4. **实现 control executor。** 首先只做单 device/single `default` stream：CPU predicate branch、condition-before-body loop、max-trip guard、Phi/loop-carried copy/bind。compiler-side resolver 输入必须通过 ControlPlan validator；runtime executor 只接收并复核 Track 05 frozen executable-plan validator 的产物。
5. **实现 dependency-aware memory。** live range 由 task dependencies、selected branch、loop backedge 和 completion event决定；Phi live-out 不得引用已退役 child storage，loop-carried storage 需跨 backedge 保活。未证明时禁用复用。
6. **建立双 oracle contract suite。** 对本分支 fixtures，让 Track 04 reference executor 与 Track 05 fake/single-stream executor比较 outputs、executed task ids、branch decisions、loop iteration counts 和拒绝类别。
7. **只在门禁通过后增加 Compiler feature gate。** `Compiler::Compile` 默认仍拒绝 If；显式 control-flow compile mode 才调用 `LowerRelayToControlPlan -> compiler-side resolver -> executable FrozenPlanVariant -> Track 05 runtime adapter/executor`。任一步失败立即回到“明确拒绝”，不能把 If 放回 ordered ValueGraph plan。
8. **保持 RuntimeSession 单向依赖。** 可新增消费纯 runtime frozen plan 的 executor/session 类型，或为 RuntimeSession 增加纯 runtime overload；不得让它持有 Compiler、Relay registry、cache policy 或 ShapePredictor。
9. **最后接动态 Shape。** 只有 Track 02/05 提供 ShapeProgram、exact/bucket applicability、physical capacity、valid extent、dynamic allocation 和差分测试后，才升级 schema 开放 dynamic Phi/output/shape-changing loop；v2 继续作为 exact unresolved preparation fallback。
10. **另立 eager/tracing frontend 工作。** 若未来需要 tracing，adapter 必须输出完整 guards/effect/alias/shape bindings；一次已走路径不得推广为本轨 structured control-flow 支持。

完成上述 1–7 且集成 contract suite 通过之前，只能声明“static-exact ControlPlan preparation/reference semantics”，不能声明生产动态控制流 runtime 已支持。

## 8. W3 restricted real-artifact Relay `If` path

W3 adds an explicit, **default-OFF** bridge for the deliberately narrow case
where the existing real compiler can produce CPU artifacts for every frozen
branch Call:

```text
Relay Function with If or bounded While
  -> InferType / ANF / static-exact ControlPlan v2
  -> compiler-private { task id -> frozen Relay Function(Call) } sidecar
  -> Compiler::Compile(one control-free Call Function) for every region kernel task
  -> resolved ControlExecutionPlan v1
```

Public surface and build registration:

- `KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION=OFF` is propagated to library and
  runtime-test translation units.  `Compiler::CompileControlFlowExact` is
  declared in the already-installed `kxc/compiler/compiler.h`; its supporting
  binding contract remains in the already-installed
  `kxc/compiler/control_flow.h` and resolved runtime contract in the
  already-installed `kxc/runtime/control_execution_plan.h`.  All three headers
  are present in `KXC_PUBLIC_HEADERS`.
- `Compiler::Compile` is unchanged and continues to reject Relay `If` and
  `While` through the static-dataflow capability gate.  The new API is the only
  opt-in entry.
- The lowerer sidecar is `src/compiler/control_flow/internal_lowering.h`, is
  not installed, and preserves the original Relay Call/Op/attrs/constants and
  lexical operands.  The resolver never parses `kernel_ref`; it uses task id
  only to find that private payload and calls the real `Compiler::Compile` for
  each condition, body, or branch kernel.  A missing sidecar entry fails
  closed.
- The resolver accepts only CPU:0/default-stream, static-exact plans.  A
  production binding has `binding_revision == 0` and an opaque compiler-minted
  lease; a fixture binding instead has a nonzero revision and no lease.  The
  two forms are mutually exclusive.  `CompileControlFlowExact` mints lease
  generations monotonically: it issues `uint64_t` maximum once, retains that
  terminal value, and then fails closed instead of wrapping or reusing a
  generation.  `BoundControlKernel` stores the opaque keepalive in its
  immutable state, so copies of the resolved plan retain selected artifacts
  through cache clear and asynchronous completion ownership.  The
  generation/lease remain observability/control metadata; runtime does not
  authenticate or use them to select an artifact.
- The terminal-generation test seam is the source-private
  `src/compiler/control_flow/production_control_flow_test.h`, included by its
  test with a relative source path.  It is outside `include/`, absent from
  `KXC_PUBLIC_HEADERS`, and is therefore neither installed nor public API.
- Binding remains exact: task-id mapping, selected module entry, ABI
  non-output order (live inputs followed by constants), signature roles and
  output order are checked by `BindControlPlanForRuntime`.  Wrong entry/ABI,
  branch task mismatch, malformed constants/artifacts, or a malformed private
  sidecar fail before execution.  Existing control-runtime tests also retain
  private constant snapshots and completion state across owner destruction.

Supported Relay control syntax is restricted `If` (including nested `If`) and
bounded `While`, when every selected Call is in the restricted subset.  A
`While` has exactly one lexical carried-state binder, a non-negative static
bound, CPU:0 scalar-bool condition, and one exact placement shared by its
initial state, binder, body, and result.  It is not general recursion, host
unrolling, or a JIT loop.  Captured live-ins and constants are represented by
the existing lowerer/ABI contract.

### W3/W4 local evidence and limit

On this machine, the follow-up passed `run_control_flow_tests` (all six tests:
`relay_anf_test`, `executable_capability_test`, `control_plan_test`,
`control_plan_reference_executor_test`, `relay_control_plan_test`, and
`control_runtime_integration_test`) in both CPU-only configurations:

```bash
cmake --build /tmp/kxc-relay-while-off --target run_control_flow_tests -j2
# KXC_ENABLE_CUDA=OFF, KXC_ENABLE_LLVM=OFF,
# KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION=OFF, KXC_ENABLE_CONTROL_RUNTIME=ON

cmake --build /tmp/kxc-relay-while-on --target run_control_flow_tests -j2
# KXC_ENABLE_CUDA=OFF, KXC_ENABLE_LLVM=OFF,
# KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION=ON, KXC_ENABLE_CONTROL_RUNTIME=ON
```

This covers default-off rejection, enabled-without-LLVM fail-closed behavior,
default `Compiler::Compile` rejection for `If` and `While`, placement-gate
negatives, the source-private terminal-generation seam, and existing resolved
plan/runtime lifetime coverage.  The public-header compile check also passed
for all 104 installed headers; the private seam was not added to that surface.

An LLVM-requested configuration was also generated and its six control-flow
tests passed:

```bash
cmake -S . -B /tmp/kxc-relay-while-llvm \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION=ON \
  -DKXC_ENABLE_CONTROL_RUNTIME=ON \
  -DKXC_BUILD_CODEGEN_TESTS=OFF -DKXC_BUILD_PASS_TESTS=ON
cmake --build /tmp/kxc-relay-while-llvm --target run_control_flow_tests -j2
```

CMake reported that LLVM was not found and disabled code generation
(`LLVM_DIR=LLVM_DIR-NOTFOUND`, hence `KXC_USE_LLVM=0`).  Therefore this third
run is additional fail-closed evidence only, not LLVM JIT evidence; the numeric
Relay-`While` E2E is compiled only when the production, LLVM, and control-runtime
gates are all true and did not execute locally.  A real LLVM result must still
cover zero/one/multiple and exhausted `While` trips, true/false and nested
`If`, constants/live-in capture, malformed sidecar/artifact data, cache clear,
and async completion before claiming backend support.
