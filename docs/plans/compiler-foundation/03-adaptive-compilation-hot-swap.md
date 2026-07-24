# 03：自适应编译与安全热替换

> **状态：** 两条默认隔离的实验路径；均不生产就绪。
> **Feature gate：** `KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=OFF`（默认）。
> **API：** 原 fake 为 `kxc::api::experimental::adaptive::v1`；W2 adapter 为
> `kxc::api::adaptive::experimental::production_path`。
> **所属路线：** [编译器基础路线图](README.md)
> **共同基础：** [01：契约、identity 与 cache](01-core-contracts-identity-cache.md)

## 1. 当前边界

原有 Track03 是隔离的 fake 控制面，用于验证 coordinator、slot、generation、lease、
canary 和 token 语义。W2 新增的是一个 **production-path experimental adapter**：它把
真实 `CompiledModule`、`ExecutablePlan`、production-cache pin 和 `RuntimeSession` 接到
一个同步 static-exact controller，但不把 controller 职责放入 `RuntimeSession`。

```text
caller-immutable Function + deep-frozen CompileConfig + verified typed baseline
  -> graph ArtifactKey + exact DispatchKey
  -> ordered primitive ArtifactKey/call mapping + callable/runtime Plan ABI
  -> real Compiler::Compile or injected ProductionPathCompilerAdapter
  -> structural validation against immutable production-cache pins
  -> derived selected-whole-plan identity + opaque authority lease
  -> frozen whole-plan generation + RuntimeSession + completion retention
```

该 adapter 明确不是 production-ready state machine。它当前没有：

- cancellation 或 backend hard timeout；
- deadline/queue scheduling；
- controller negative cache、failure TTL、backoff 或 retry authority；
- 自动数值健康判定、one-shot health token 或不可伪造 attestation；
- slot eviction、resident executable/device byte accounting；
- dynamic/bucket/fuzzy shape applicability。

因此 feature 默认关闭，API/option/namespace 都保留 `experimental`。adapter throw 会结束
本次 same-key flight，并把同一异常交给所有 waiter；之后调用方可以显式重试，但 controller
不决定 retry policy。

## 2. identity 与 artifact authority

### 2.1 请求快照

`ProductionCompileRequest` 在构造时深拷贝 `CompileConfigNode`、`TargetNode`、全部
`DeviceAttributes`、`opt_level` 和 `ProfileOptions`。后续 `config()` 每次返回独立深拷贝，
因此原 config/target 或 adapter 所持副本的修改不会改变请求，也不会与请求并发读取共享
可写节点。实际 compiler invocation 使用该冻结快照。

`Function` 仍是共享的 Relay IR handle，不做通用 deep clone；调用方必须在 request 构造前
停止修改，并在 request 生命周期内保持 immutable/synchronized。该限制是 experimental API
边界，不能把 Function 描述为 request-owned frozen graph。

graph semantic canonicalization 只接受 exact runtime type whitelist 中的 Relay Expr 节点；
任何 undefined Expr（包括 Function body、Call op/arg、Let/If/Tuple 子表达式）以及从受支持
节点派生但可能携带额外字段的未知类型都直接拒绝。不得生成 `undefined`/`unsupported`
identity 后继续，也不得进入 compiler adapter 或 publish。

### 2.2 ordered selected primitive artifacts

编译器输出和 verified baseline 都必须满足：

1. 每个 ordered `ExecutablePlan::KernelCall` 恰有一个 `ArtifactPlanBinding` 和一个
   production-backed `ArtifactPin`；
2. `call_index`、link symbol 和 ordered position 完整一致；
3. whole-graph request key 不能冒充 primitive key；candidate 可在完整 typed contract
   成立时选择不同的 immutable primitive `ArtifactKey`，不要求与 baseline 逐项相等；
4. public `ArtifactRecord` 的 key、signature/launch digest、provenance、accounted bytes 和
   validation record 与内部 immutable `CachedPrimitive` 全值一致；
5. pinned signature、launch metadata、target/backend、compiled-kernel contract 和 launcher
   与 module entry 全值一致；
6. 错序、同 key/不同 signature 或 metadata 都拒绝；同 launcher/不同合法 selected key
   不是 selection proof，也不应被拒绝。

`PlanAbiFingerprint v4` 只覆盖 callable/runtime compatibility：target、value contract、dtype/
device、static shape/flags、ordered calls、signature、launch metadata 与 constants。它不含
selected artifacts、generation、receipt、graph-local value/storage locator。不同 selected
artifacts 可以具有相同 ABI；ordered `OrderedArtifactIdentity` 改为派生 whole-plan selection
identity，并进入 `PlanVariantKey`、lease 和 quarantine。相等性始终比较 canonical bytes；digest
只用于观测/索引。

实际 executable payload（例如 JIT/native module bytes）目前没有统一可序列化表示。因此
当前 proof 仅是 **immutable process-local primitive keys + exact ordered mapping + launcher
object identity + complete typed contract equality**。它不是 cryptographic code provenance、remote
attestation，也不证明数值正确性。

未建模的 physical layout/stride/workspace/effect/error contract 不得宣称已被证明。

## 3. controller/observer 语义

- 完整同 key flight singleflight；不同 key 在 `max_in_flight_compiles` 内并行。
- `max_slots` 和 active-flight budget 饱和时在调用 adapter 前拒绝。
- publish 后才对未来 `Acquire` 可见；运行中的强引用始终使用入口取得的 frozen variant。
- observer 同步、controller lock 外、best-effort 执行。callback 异常和 event payload 构造
  异常不改变 validation/publication/routing。
- callback 活跃窗口内，任何线程进入同一 controller 的 `CompileAndPublish`、`Acquire`、
  `RunAsync`、`RollbackAdministrative` 或 `Snapshot` 都立即 fail-fast，不能等待 flight。
  callback 创建线程后执行 same-key compile 并 join 也不会形成 future 自等待。该保守契约也会
  拒绝窗口内与 callback 无关的外部调用；工作必须延后到 callback 返回之后。不同 controller
  实例互不影响。

`RollbackAdministrative` 只接受显式命名的 `AdministrativeQuarantineRequest`。它是
**trusted control-plane administrative action**，不是自动 runtime health proof。调用者必须
在 API 外完成 authentication、authorization、replay control 和 evidence provenance。
当前没有 one-shot health authority，因而不能将此 API 描述为健康验证闭环。

Generation counter 在耗尽前 fail-closed。primitive cache ticket/stamp 也在 `uint64_t` 空间
耗尽时拒绝新工作，不允许 wrap 后重用 identity/order；若 failure stamp 已耗尽，只完成
当前 waiter failure，不创建歧义 negative record。

## 4. eviction 与异步生命周期

`ArtifactPin`、`FrozenPlanVariant`、`RuntimeSession` 和 completion 都是强所有权。cache
clear/LRU eviction 只移除 discoverability；已发布 variant 仍可 launch。`RunAsync` 把
variant 和 artifact lease 追加到 completion retention，直到 completion handle 销毁。
`RuntimeSession` 保持静态数据面 executor，不编译、不查 cache、不选择 generation。

边界名称按实际含义定义：

- `max_discoverable_generations` 只限制 controller history；外部 lease/completion 可继续
  保活更旧 generation；
- snapshot 的 `discoverable_history_variants` 不是进程 live variant 或 resident byte 数；
- primitive cache `active_pins` 只统计仍可发现 entry 的外部引用，无法统计已 eviction/clear
  的 pin；
- accounted bytes 是 producer-reported cache accounting，不是 native code/device resident
  bytes。

本机 CPU backend 的 event 是同步完成空句柄，没有可控 pending fake event。本轮覆盖同步
completion 的追加 owner 和销毁保活；真实 pending CUDA event 生命周期仍是外部门禁，不能
用 CPU completed event 冒充。

## 5. build/test gate

Gate ON 时注册 `adaptive_production_experimental_test`，labels 为
`adaptive;adaptive-production-experimental;cpu`，并提供：

```bash
cmake -S . -B out/adaptive-production-on -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON
cmake --build out/adaptive-production-on \
  --target run_adaptive_production_experimental_tests
```

测试覆盖 ordered artifact 攻击、same launcher wrong key/signature/metadata、wrong order、
config/Target mutation 与并发读取、undefined/unknown/derived Relay fail-closed 且不 publish、
observer 五个 API 同线程 reentry/throw isolation，以及有界等待下 callback spawn+join 的
same-key compile 与跨线程 `Snapshot` fail-fast（并验证不同 controller 不受影响）、same-flight
failure fanout/retry、different-key parallel/backpressure、trusted administrative rollback、
cache clear 后 launch、completion retention、static-exact runtime/slot 边界。CTest 另有 60 秒
进程级 timeout；public-header checker 在 Gate ON 时显式定义 feature macro，确保 enabled API
分支也可独立编译。

LLVM integration test 仅在 `KXC_USE_LLVM` 时编译执行；CUDA/pending-event/TSan/真实 resident
bytes/部署层 health authority 必须在具备对应工具和硬件的外部 gate 验证。本机不可用时不得
报告为通过。

### Public C++ compatibility

This is not a binary-compatible public API evolution. `PreparedCandidate`/authority declarations
and the `PlanAbiFingerprint v4` byte contract changed; all consumers must source-recompile with
matching headers and library. Do not link pre-change C++ objects/binaries with this library. The
contract version is a data-format marker only, not an ABI promise.
