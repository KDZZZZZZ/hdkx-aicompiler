# Compiler Foundation Adaptive / Hot-Swap 轨道交接

> 分支：`feature/compiler-foundation-adaptive-production`
>
> 状态：**default-off experimental；不是 production-ready**。
>
> APIs：隔离 fake 为 `kxc::api::experimental::adaptive::v1`；W2 production-path adapter 为
> `kxc::api::adaptive::experimental::production_path`。

## 1. 已交付

### 1.1 隔离 fake 控制面

`include/kxc/compiler/adaptive.h` 仍提供独立的 `CompileCoordinator`、`KernelSlot`、
generation lease、fake validation/health authority 和 fake plan assembler。它用于验证调度、
cancellation/deadline/token/slot 语义，不是 RuntimeSession production integration。

### 1.2 W2 production-path experimental adapter

Gate：

```text
KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=OFF
```

开启后，`include/kxc/compiler/adaptive_production_experimental.h` 与
`src/compiler/adaptive/production_path_experimental.cc` 提供同步 static-exact adapter：

- request 深冻结 `CompileConfig`、`Target`、opt/profile 全字段；返回 adapter 的 config 也是
  独立 deep copy；
- whole-graph identity 只接受 exact Relay Expr node whitelist；undefined/unknown/derived node
  在 key、adapter 与 publish 前 fail closed；
- verified baseline 冻结 ordered `(call index, link symbol, primitive ArtifactKey)`，作为
  request 的 structural/typed baseline；
- candidate 的每个 production pin 都核对内部 `CachedPrimitive` signature、launch
  metadata、target/backend、launcher、provenance、bytes、validation record 及 module entry；
  candidate 可在完整 typed contract 仍成立时选择不同的 immutable primitive ArtifactKey；
- callable/runtime `PlanAbiFingerprint v4` 不含 selected artifact、generation 或 receipt；
  ordered selected artifact mapping 派生单独的 whole-plan selection identity，并进入
  `PlanVariantKey`、lease、quarantine 和 observability。
- whole-graph key 不能作为 primitive fixture key；
- same-key singleflight、different-key 并行和 active-flight/slot backpressure；
- frozen whole-plan generation、exact acquire、completion-held variant/artifact lease；
- trusted control-plane `RollbackAdministrative`，只影响未来 routing；
- 同步 observer 在 lock 外执行，throw 被隔离；callback 窗口内任何线程进入同 controller
  API 都 fail-fast（包括无关外部调用），不同 controller 不受影响；
- generation 与 primitive cache ticket/stamp overflow fail closed。

`RuntimeSession` 没有 compiler/cache/controller/generation 选择职责，仍是静态数据面。

## 2. 必须保持的 authority 边界

当前 executable payload 没有统一可序列化格式。candidate selection 的现有 proof 是：

```text
immutable process-local primitive ArtifactKeys + exact ordered call mapping
+ full typed CachedPrimitive/module equality + module-entry launcher identity
```

不同的 selected primitive identities 可以共享同一 callable/runtime ABI；derived whole-plan
selection identity 因而不是 launcher identity，也不是 graph request identity。它仍**不是**
cryptographic code provenance、remote attestation 或数值健康证明。不得升级措辞。

`AdministrativeQuarantineRequest::ForTrustedControlPlane` 是部署层已经授权后的行政入口；API
自身没有 authentication、one-shot consumption、replay protection 或 automatic health
judgement。没有这些外部能力前，`RollbackAdministrative` 不能称作健康 authority。

Plan ABI 只覆盖代码当前表示的 target、plan value/storage、dtype/device/static shape/flags、
call topology、signature、launch metadata、constants 和 ordered artifacts。layout/stride、
workspace、effect/error contract 等未建模字段不能宣称已证明。

## 3. state-machine 降级说明

W2 adapter 没有 cancellation、deadline、negative cache、failure TTL/backoff、priority queue 或
retry authority。adapter failure 会向同 flight 的全部 waiter 传播，并释放 flight；之后调用方
可以再次请求，但 retry 时机/次数不由 controller 决策。

因此 API、namespace、option、CTest label 和文档都保留 `experimental`，feature 默认 OFF。
原 fake coordinator 虽有更完整实验状态机，也不能被解释为 W2 production runtime authority。

## 4. lifetime/accounting

- cache clear/eviction 只删除 discoverability；已 pin variant 继续 launch；
- completion 追加保活 RuntimeSession context 和 adaptive variant/lease；
- `max_discoverable_generations` 与 `discoverable_history_variants` 只描述 controller history，
  不限制外部 lease/completion；
- cache `active_pins` 只统计仍在 cache map 的 entry 外部引用，evicted/cleared pin 无法计数；
- cache/TIR accounted bytes 不是 native executable/device resident bytes。

本机 CPU event 同步完成，无法提供可控 pending fake event；pending CUDA completion 是外部
硬件 gate。

## 5. 验证入口

Gate OFF/ON 必须使用独立 build directory；ON focused suite：

```bash
cmake -S . -B out/adaptive-production-on -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON
cmake --build out/adaptive-production-on \
  --target adaptive_production_experimental_test check_include_layers check_public_headers
ctest --test-dir out/adaptive-production-on --output-on-failure \
  -L adaptive-production-experimental -L cpu
```

`adaptive_production_experimental_test` 包含：

- same launcher wrong primitive key/signature/launch metadata、corrupt binding、wrong order；
- request config/Target/profile mutation 与并发读取；
- undefined/unknown/derived Relay semantic identity 在生成 key、调用 adapter 或 publish 前
  fail-closed；
- `CompileAndPublish`、`Acquire`、`RunAsync`、`RollbackAdministrative` observer reentry fail-fast，
  observer throw 不改变 routing；有界 timeout 覆盖 callback spawn+join same-key compile 与
  跨线程 `Snapshot`，并验证不同 controller 互不干扰；
- same-flight adapter throw fanout 与受控 retry；
- different-key parallel 和 global active-flight backpressure；
- trusted administrative rollback/quarantine；
- cache clear 后已 pin variant launch、completion retention；
- exact dispatch/runtime/slot boundary。

本提交本机证据（GCC 13.3，CPU-only LLVM/CUDA codegen 均关闭）：

- gate OFF：CPU CTest **38/38**；
- gate ON：CPU CTest **39/39**；
- adaptive scoped：**4/4**；production-path focused：**1/1**；
- ASan+UBSan+LeakSanitizer focused：**1/1**，无诊断；
- include-layer：**243 files**；enabled-gate public headers：**96 headers**。

`llvm-config` 本机不可用，所以条件式真实 LLVM test 未执行。CUDA 12.9 `nvcc` 和 NVIDIA
设备节点存在，但本轮是 CPU gate，未执行 CUDA publication/pending-event 验证，不能报告为
CUDA 通过。

## 7. W3 v2 whole-plan lifecycle control plane

Gate：`KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=OFF`；打开它会显式打开其唯一依赖
`KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION`。公共 API 是
`kxc::api::adaptive::hot_swap::v2`（`adaptive_hot_swap_v2.h`），实现为
`src/compiler/adaptive/adaptive_hot_swap_v2.cc`。W3 直接使用 W2 的 `ProductionPathCompilerAdapter`、`PrepareCandidate` 完整
static-exact validation，以及 Core primitive-cache pin transaction；它不调用会自行发布路由的
W2 `AdaptiveController`，也不复制这些 proof。

W3 独占 whole-plan queue/retry/generation/routing authority：有界 worker、queue、flight 和
waiter；same-key singleflight；每个 ticket 自己的 cooperative cancellation/deadline；分类
negative cache（permanent/unsupported 不过期，transient/timeout 用配置 backoff）；以及不回绕的
`GenerationLease`。取消或 deadline 只结束该 waiter，不能取消或污染共享 flight；backend hard
cancel/timeout 仍不是本 API 声称的能力。

发布在 v2 route lock 下执行，只有 W2 candidate 的 exact `DispatchKey` 和
`PlanAbiFingerprint` 都与 request 相等才可以替换 future routing。该 ABI 包含当前模型表示的
signature alignment/layout-relevant metadata；因此不同的已表示 physical/layout/workspace ABI
不能替换。未建模的 stride/workspace/effect 字段没有新增 proof，不能据此宣称全物理布局或
numeric truth 已被验证。

discoverability eviction 同时按 generation 数与 producer-reported byte 数工作；它只删除
controller routing/history，外部 `GenerationLease` 和 completion retained variant 继续存活。
这些 bytes 不是 native/device resident measurement。Health authority 是可选 process-local
injection seam，`VerifyAndConsume` 是 one-shot authority；接受的 quarantine 原子地将 future
routing 回退到一个仍可发现的 predecessor。health evidence、resident bytes 和 numeric truth
均非 W3 自行认证；external authentication/attestation 和 CUDA pending completion 仍不支持。
Observer callbacks 在锁外且异常隔离，并在 callback 窗口拒绝本 controller 的 reentry。
`Evaluate` 可在锁外执行；`VerifyAndConsume` 在 route/health locks 下串行执行、是 `noexcept`，
对同一 controller 的 re-entry 会 fail-fast，避免重取这些锁导致 deadlock。negative cache 由已验证的 `max_negative_cache_entries` 和累计
`max_negative_diagnostic_bytes` 精确约束：按插入顺序确定性地优先驱逐 expired/retryable
(transient/timeout) record，绝不驱逐 permanent/unsupported record；若 permanent/unsupported
record 已填满 entry bound，controller 全局 compile/publish fail-closed，直到显式
`ClearNegativeCacheForTesting`。snapshot 提供 entry/diagnostic-byte、eviction/drop 和 blocked
counters，observer 发出 negative eviction/saturation events。每个 route 的 artifact quarantine
由已验证的 `max_quarantine_tombstones_per_route` 约束且 tombstone 永不驱逐；饱和时 route
compile/publish fail-closed（已存在的 healthy predecessor 仍可 Acquire），直到显式
`ClearQuarantinesForTesting`。snapshot 提供 tombstone/blocked-route/saturation counters，observer
发出 quarantine-saturation event。取消和 backpressure 结果不进入 negative cache，且
queue/in-flight legacy slot 加法在构造时检查 overflow。

W3 的 `PreparedCandidate` 是 prepare-only：validation receipt、selected artifacts 和
`RuntimeSession` 在 authority issuance 前冻结，不能直接路由或构造 generation。opaque
`GenerationLease` 只能由 `GenerationAuthority::MakeLease` 创建；它绑定 immutable prepared
candidate、derived selection identity、exact route/ABI、receipt 和 producer byte declaration。`Issue` 和
`NextGenerationForTesting` 在 route lock 下调用；对同一 controller 的 re-entry 会 fail-fast。
`EvaluateHealth` 允许不同 callers concurrent `Evaluate`，但仅在 lease 仍为 route head 时、在
serialized `VerifyAndConsume` 中消耗 evidence；过时 health 不会消耗 token。observer callback
仍在所有 controller APIs 上维持跨线程 fail-fast window；health authority evaluation/verification
不占用该窗口，因此并发 health consumers 不会互相被误拒绝。`VerifyAndConsume` 对同一
controller 的 re-entry 会 fail-fast（它在 route/health locks 下执行）。

W3 focused gate：

```bash
cmake -S . -B out/adaptive-v2-on -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON
cmake --build out/adaptive-v2-on --target run_adaptive_hot_swap_v2_tests \
  check_include_layers check_public_headers
ctest --test-dir out/adaptive-v2-on --output-on-failure \
  -L adaptive-hot-swap-v2 -L cpu
```

The deterministic v2 additions cover cancellation/deadline isolation (including no-`Wait`
all-cancelled suppression), repeated same-key bounded stress, negative-cache clear/retry,
producer-byte/history eviction with an external lease, non-wrapping generation exhaustion,
transactional injected pre-commit stages, route/tombstone bounds, one-shot concurrent health
consumption, observer isolation, and exact ABI non-routing. It also proves a candidate with a
real different selected primitive key and the same ABI gets a different selection identity while
a recording authority binds the opaque lease. The v2 test reuses the W2 fixture so every candidate
continues through real production-path validation.

### Public C++ interface compatibility

No cross-version C++ ABI guarantee is made. This branch changes public headers (including the
prepare/authority interfaces and the `PlanAbiFingerprint v4` canonical byte contract). Consumers
must rebuild source against matching headers and library; previously compiled objects/binaries
must not be mixed with this library, and old source that relied on selected artifacts being part of
Plan ABI must be updated. Schema/contract version constants version data formats, not the C++ ABI.

## 6. 外部门禁/未完成项

1. supported host TSan；
2. LLVM-enabled real compiler integration（仅 `KXC_USE_LLVM` 条件执行）；
3. CUDA target/launcher/stream 与真实 pending event retention；
4. native executable/device resident byte accounting；
5. 部署层 authentication、one-shot/replay-safe health authority 和数值证据；
6. 若未来扩展 shape，先提供 versioned applicability proof，禁止 fuzzy fallback。
