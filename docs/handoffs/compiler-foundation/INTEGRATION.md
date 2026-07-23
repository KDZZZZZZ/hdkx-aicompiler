# Compiler Foundation 集成交接

> **集成分支：** `integration/compiler-foundation`
>
> **上游源码基线：** `dev@3b95aca188122ff52ebdb2f43d390d21273ff3e2`
>
> **W1 固定基线：** `baseline/compiler-foundation-w1@525950a`
>
> **W2 operator 集成头：** `a72a406`
>
> **状态：** W2 exact-static/default-OFF adapters 已完成本地集成和 CPU 合同验证；
> 本页记录的是 `implemented/local-evidence` checkpoint，不是 dynamic Shape、自动生产
> hot swap、production Relay control flow、完整 Transformer/KV cache 或已验证 CUDA
> 能力声明。

## 1. 集成边界

W1 六条基础轨道均保留独立 feature branch 和 merge 边界：

1. `feature/compiler-foundation-core`
2. `feature/compiler-foundation-shape`
3. `feature/compiler-foundation-adaptive`
4. `feature/compiler-foundation-control-flow`
5. `feature/compiler-foundation-runtime-plan`
6. `feature/compiler-foundation-nlp-gpu`

W2 在固定 W1 基线上按以下边界集成：

| 增量 | 集成提交 | 当前能力边界 |
|---|---|---|
| Runtime artifact manifest / observability | `f649592` | trusted declaration、结构复验、fallback/observer events 和 retention lease；不提供 provenance/authentication |
| Resolved control runtime | `01523e8` | default-OFF、CPU:0、static-exact fixture executor；不是 Relay control backend |
| Exact Shape production adapter | `c0a669e` | frozen Relay/constants、exact concrete profile 和 immutable pins；不支持 symbolic/bucket/fuzzy reuse |
| Module constant integration fix | `458922d` | module-owned deep copy 与 alignment contract |
| Adaptive production experiment | `97982ff` | default-OFF compiler/plan adapter、generation slot/lease 和 callback fail-fast；没有自动健康/回滚 authority |
| Transformer operator slices | `a72a406` | Gather、Where、LayerNorm、Concat、Slice exact-static vertical slices；不是完整 Transformer/decode runtime |

## 2. 保持不变的生产边界

- `RuntimeSession` 仍是静态、强类型 data-plane executor，不 include Compiler、Relay、
  primitive cache、Shape predictor 或 adaptive policy。
- production `Compiler::Compile` 仍拒绝 unresolved Relay `If`；resolved control executor
  不改变该 capability boundary。
- exact concrete Shape reuse 是当前唯一 production adapter oracle；不存在
  `cached_dims >= query_dims` 一类 fuzzy compatibility。
- physical Shape/layout/workspace 改变仍要求新 `PlanVariant`；same-ABI generation 才能
  使用 slot replacement。
- Runtime manifest identity/generation/lease 是可信上层声明与可观测性合同，不是
  provenance、认证或安全凭据。
- 新增公共 C++ surface 要求源码重新编译，不声明跨版本 precompiled C++ ABI。

## 3. 主要实现结果

### 3.1 Core contracts / identity / cache

- operator/pass metadata、normalized pipeline、capability verification 和 canonical identity
  使用统一合同。
- semantic kernel identity 不包含 graph-local `value_id`。
- production primitive cache 使用完整 canonical key、immutable artifact pins 和 same-key
  transaction/singleflight；不存在 fuzzy Shape key reuse。
- `NormalizeToANF` 是正式 Pass contract；lexical `Let` 可验证，`If` 继续 fail closed。

### 3.2 Shape / adaptive / control

- exact Shape adapter 将 preparation 与 assembly 分离，并冻结 lowering-relevant
  Function、attrs、types、inputs 和 constants，避免旧 semantic key 对应新代码。
- adaptive experiment 冻结完整 config/Target snapshot，校验 ordered call-to-artifact
  binding、pin signature、metadata、target、launcher 和 ordering；cache clear/eviction 不使
  已发布 immutable pins 失效。
- control runtime 执行 resolved Branch/Loop/Phi/backedge，使用 private constant snapshots
  和 typed module entry adapter；`binding_revision` 仅是 caller fixture label。

### 3.3 Runtime / NLP

- default-OFF Region/Task DAG 提供 validator、memory planner、task executor 和结构化
  observability；`kTaskStart` 在动作前，`kTaskLaunch` 只表示成功提交。
- Relay operator contract 从 19 项扩展到 24 项。
- ONNX Gather 仅映射 initializer-backed、静态 range-validated subset；Relay zero-fill
  Gather 明确保留为 KXC extension。
- LayerNorm 对 float32 输入使用 float64 中间 accumulation/mean/variance/sqrt/affine。
- Shape products、iteration extents、flatten indices 和 byte sizes 使用 checked arithmetic。
- ONNX protobuf → Python serializer → C++ reifier → LLVM `RuntimeSession` fixture 已注册，
  但本机因依赖缺失未执行该 E2E。

## 4. 本地统一验证

### 4.1 所有 production/experimental gates OFF

配置摘要：

```text
KXC_ENABLE_CUDA=OFF
KXC_ENABLE_LLVM=OFF
KXC_ENABLE_REGION_TASK_DAG=OFF
KXC_ENABLE_CONTROL_RUNTIME=OFF
KXC_ENABLE_SHAPE_PRODUCTION_EXACT=OFF
KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=OFF
```

完整 build 和 `ctest -L cpu` 结果：

```text
40/40 passed
```

### 4.2 W2 adapters 全部 ON

配置摘要：

```text
KXC_ENABLE_CUDA=OFF
KXC_ENABLE_LLVM=OFF
KXC_ENABLE_REGION_TASK_DAG=ON
KXC_ENABLE_CONTROL_RUNTIME=ON
KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON
KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON
```

完整 build 和 `ctest -L cpu` 结果：

```text
41/41 passed
```

其中包括：

- `shape_production_exact_test`
- `adaptive_production_experimental_test`
- `control_runtime_integration_test`
- `runtime_session_test`
- `infer_type_test`
- `operator_compilation_test`
- `onnx_import_spec_contract_test`
- Relay/Pass contracts、include layers 和 public headers

独立机器可读检查：

```text
Relay operators: 24/24 passed
Passes:          20/20 passed
NLP/GPU checker: PASS
Python py_compile: PASS
git diff --check: PASS
ASan/UBSan runtime-plan + adaptive: 5/5 passed
```

ASan/UBSan 配置构建了 `task_plan_test`、`task_executor_test`、
`runtime_session_test`、`control_runtime_integration_test` 和
`adaptive_production_experimental_test`；本次本地执行未发现 sanitizer failure。

## 5. 证据等级

| 能力 | 状态 | 说明 |
|---|---|---|
| CPU exact-static production path | `validated` | 本地 OFF/ON 两套完整 CPU CTest |
| Shape exact production adapter | `implemented/local-evidence` | CPU exact profile；无 symbolic/bucket claim |
| Adaptive production experiment | `implemented/local-evidence` | default-OFF；无 cancellation/health/rollback authority |
| Resolved control runtime | `implemented/local-evidence` | fixture-backed CPU:0；production Compiler 仍拒绝 `If` |
| Runtime manifest/observability | `implemented/local-evidence` | 结构一致性与事件验证；声明不是 provenance |
| ONNX protobuf → LLVM Runtime E2E | `implemented/source-and-CI-registration` | 本机缺 `onnx`/`numpy` 和 LLVM，等待 CI |
| CUDA Where/Slice/Concat bounded rank-1 slices | `implemented/local-evidence` | 没有本机 GPU numeric validation |
| Symbolic/dynamic Shape 与 dynamic output allocation | `unsupported` | W3/W4 工作 |
| Production automatic hot swap/rollback | `unsupported` | W3/W4 工作 |
| Production Relay control lowering | `unsupported` | W3/W4 工作 |
| 完整 Transformer/KV-cache/sampling | `unsupported` | W3/W4 工作 |

## 6. CI 门禁

CI 配置覆盖：

- default-OFF CPU checkpoint；
- Region Task DAG `OFF/ON` 与 Control Runtime `OFF/ON` 的组合；
- exact Shape 与 adaptive experiment ON checkpoint；
- dependency-free NLP capability checker；
- ASan/UBSan Runtime-plan 与 adaptive experiment；
- LLVM codegen/numeric、production Compiler→Task-DAG、Shape/adaptive tests；
- 安装 `onnx`/`numpy` 后的 protobuf importer E2E；
- Python ONNX importer tests。

CI 注册不等于验证完成；只有远端 workflow 绿色后才能把对应 LLVM/ONNX 行升级为
`validated`。CUDA 仍需要独立硬件/toolchain gate。

## 7. 环境限制与后续工作

本机未安装 LLVM、Python `onnx`/`numpy`，也没有可用 CUDA toolchain/device，因此：

- 未运行 LLVM JIT/numeric 和 production Compiler→Task-DAG E2E；
- 未运行真实 ONNX protobuf fixture；
- 未运行 CUDA numeric、pending-retention、Compute Sanitizer 或 CUPTI；
- 未下载、安装或修改任何系统依赖。

W2 checkpoint 不完成总 W0–W4 计划。W3/W4 仍包括：restricted symbolic Shape、runtime
Shape propagation、dynamic outputs/allocation、authoritative adaptive generation leases、
cancellation/negative cache/health/rollback、真实 Relay control lowering、KV cache/decode/
sampling，以及广 rank CUDA production evidence。
