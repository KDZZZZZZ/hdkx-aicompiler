# Compiler Foundation Adaptive / Hot-Swap 轨道交接

> 分支：`feature/compiler-foundation-adaptive`
>
> 范围：隔离的 static-exact 控制面实验；不是生产编译或运行时集成。
>
> API：`kxc::api::experimental::adaptive::v1`
>
> 状态：**experimental、fake、static-exact、nonproduction**。控制面已实现为隔离实验；生产 Core/runtime 集成仍被阻塞。

## 1. 已交付的隔离控制面

公共合同在 `include/kxc/compiler/adaptive.h`，实现位于
`src/compiler/adaptive/`。它定义了强类型 key、`CompileCoordinator`、
`KernelSlot`、generation-bound `ArtifactLease`、冻结的
`FrozenPlanVariant`，以及 `ArtifactCompiler`、
`AdaptiveValidationAuthority`、`ExactPlanAssembler` 三个注入 seam。

当前链路仅是测试/控制面链路：

```text
caller-supplied visible strings
  -> CompileCoordinator
  -> fake ArtifactCompiler / fake validation authority
  -> immutable KernelArtifact
  -> KernelSlot generation + ArtifactLease
  -> fake ExactPlanAssembler / FrozenPlanVariant
```

这不是 `RuntimeSession` 链路。此轨没有生产 `ArtifactCompiler`、
assembler、controller 或 runtime adapter，也没有 `RuntimeSession` 集成或修改。

### 1.1 API 与 identity 的实际含义

- 该 API 的唯一名称空间是
  `kxc::api::experimental::adaptive::v1`；它不承诺 source 或 ABI 稳定性。
- `KernelSlotKey`、`KernelArtifactKey`、`DispatchKey`、
  `PlanAbiFingerprint` 与 `PlanVariantKey` 是不同类型。协调器以完整
  `(artifact key, dispatch key, required ABI)` 比较 singleflight identity；
  hash（若用于 canary 分流）不定义相等性。
- 当前 `CanonicalKey` 与 `DispatchKey::Exact` 接受的是调用者提供、可见的
  `std::string`。它们只做非空检查和完整字符串比较；**它们不是 opaque bytes，
  不是 Core canonical bytes，也不验证 canonicalization**。
- Core 必须以后提供真正 opaque、版本化的 canonical identity 和 ABI
  canonicalizer；届时才可将 semantic key、artifact key 和 ABI fingerprint
  作为生产身份合同。当前字符串不得跨图、跨版本或跨进程声称语义等价。
- `DispatchKey` 只有 exact 构造路径。没有 shape 偏序、`cached_dim >= query_dim`
  或 fuzzy fallback。

### 1.2 生产信任边界尚不存在

`ArtifactExecutable::IsReady()` 和 `KernelArtifact::byte_size()` 都是 producer
reported 值。当前控制面可检查它们与请求/记录的一致性及配置预算，但不能独立证明
backend readiness、链接、launch、数值正确性、实际分配大小或运行时健康。

`AdaptiveValidationAuthority`、`ArtifactValidationToken` 与
`GenerationHealthToken` 是**注入的、确定性的 fake trust seam**。token 的消费和
记录绑定是实验 API 约束，不是生产 attestation，也不是 cryptographic validation。
真实 Core/runtime adapter 必须提供 one-shot、线程安全、不可伪造的验证/健康证据，
并把它们绑定到真实 artifact、backend、ABI 和运行结果。

## 2. 控制面语义与限制

### 2.1 CompileCoordinator

- 相同完整 key 合并；不同 dispatch 或 ABI 不合并。ready/failure terminal record
  有数量和 producer-reported artifact-byte 上限。
- worker queue 有界，排序为 demand、canary、prewarm，再按 priority、deadline 与
  FIFO。**同 key demand 到达时会提升已排队 prewarm 的调度请求**；队列满时 demand
  可逐出 queued prewarm。
- deadline 是 **queue-start expiry**：仍在队列中且到 deadline 的 waiter 会过期；
  它不是编译器或 backend 的硬超时，也不抢占已经开始的工作。
- cancellation 是 cooperative：token 只要求注入 compiler 观察取消；它不是硬 backend
  timeout。无法协作取消或永久阻塞的生产 backend 仍需进程隔离/有界生命周期方案。
- coordinator 用 owned operational state 与独立 `ThreadGroup` 管理 worker。worker
  可安全地只请求 stop；management caller 负责同步 join。若 worker callback 析构最后一个
  coordinator owner，owned deferred join service 会在非 worker 上接管 thread handles 并
  完成 join；不 detach worker，也不承诺中断 backend。

### 2.2 KernelSlot、记录和 lease

- publish 只创建不可变 generation；`Acquire` 返回持有 artifact 的
  generation-bound `ArtifactLease`。新 publish 只影响未来 acquire。
- slot 的 generation 与 dispatch 上限以及 `retained_artifact_bytes` 是**可发现的
  retained-record bytes**。它不是 lease、frozen variant 或异步操作仍持有的全部
  resident bytes，也不能作为进程总内存/设备内存账本。
- record eviction 只移除 slot 的发现记录；已经发出的 lease 继续强持有 artifact，
  因而可在 record eviction 和 slot 析构后存活。
- canary 需要 stable predecessor，并仅在显式 `RoutingContext` 允许时按稳定请求 key
  分流。promote 需要 authority 接受的 healthy record。
- rollback 写入 quarantine record。quarantine 对 slot 的生命周期 durable：坏 stable
  之后不能 acquire、不能成为 rollback target、也不能再次 publish；它不会因 record
  eviction 而解除。withdrawn canary 也不具备 rollback eligibility。

## 3. static-exact ABI 边界

同 slot 热替换只在完整 `PlanAbiFingerprint` 精确相等时才可讨论。生产 canonicalizer
至少必须覆盖参数 role/顺序、dtype、device、rank、logical/physical layout、stride、
alignment、valid extent、constant/alias/effect/error contract、workspace、launch metadata、
target/backend capability、ABI/pipeline/schedule/backend version、entry/link/rebinding，以及
不改变 plan topology/storage contract 的证明。

layout、capacity、workspace、topology 或 physical memory plan 变化必须产生独立
`PlanVariantKey`/plan variant；不能发布到现有 exact slot。Shape 轨道接入前，不存在
bucket 或 polymorphic reuse。

## 4. CMake/CTest 与验证证据

`CMakeLists.txt` 现在 `include(CTest)`，并在 `BUILD_TESTING` 下注册：

| CTest name | labels |
|---|---|
| `adaptive_contract_test` | `adaptive;cpu` |
| `adaptive_coordinator_test` | `adaptive;cpu` |
| `adaptive_kernel_slot_test` | `adaptive;cpu` |

`run_adaptive_tests` 是实际 aggregate target，执行：

```bash
ctest --output-on-failure -L adaptive -L cpu
```

CPU preset 为 `dev-ninja-cpu-adaptive`，复用 `dev-ninja-cpu` 的 configure preset 和
`out/build/dev-ninja-cpu` 路径：

```bash
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu --target run_adaptive_tests
# 或：ctest --preset dev-ninja-cpu-adaptive
```

CPU smoke CI 也执行同一 CTest label 交集。

本轮实际验证结果：

- Debug CPU `run_adaptive_tests`：3/3 通过；
- focused concurrency：`adaptive_coordinator_test` 与
  `adaptive_kernel_slot_test` 各重复 100 次，全部通过；
- ASan + UBSan + LeakSanitizer：3/3 通过，无诊断；
- `-Wall -Wextra -Wpedantic -Werror`：adaptive header/source/tests 语法检查通过；
- `check_public_headers`（82 headers）与 `check_include_layers`（213 files）通过；
- TSan binary 编译/链接成功。一次可启动的 coordinator 运行发现并修复了 test
  `EventLog` notify/destruction 同步竞态；修复后重跑以及多数尝试都在进入测试前报告
  `FATAL: ThreadSanitizer: unexpected memory mapping ...`，slot 亦如此。因此没有最终有效的
  TSan pass，需在受支持 host 重跑。

可复现的普通 CPU 命令：

```bash
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu --target run_adaptive_tests
ctest --preset dev-ninja-cpu-adaptive
```

## 5. 生产集成阻塞项

1. Core 提供 opaque、版本化 canonical identities 与完整 static-exact ABI
   canonicalizer，不能复用 caller-visible strings。
2. Core/backend 提供真实 `ArtifactCompiler` 和 artifact ownership/byte accounting。
3. runtime 提供真实 validation/health authority、不可伪造证据和真实 token lifecycle；
   fake authority 不能升级为生产安全边界。
4. runtime 提供真实 `ExactPlanAssembler`/controller/runtime adapter，并定义 module、plan、
   artifact lease 与 completion 的所有权交接。
5. 集成设计必须保持 `RuntimeSession` 为数据面 executor；本轨没有、也不声称有
   `RuntimeSession` 改动。
6. Shape 轨道提供版本化 applicability proof 后才可扩展 `DispatchKey`；无 proof 时
   只能 exact compile/wait/reject。

## 6. 必守不变量

- immutable artifact、validation record、generation record 与 frozen variant 不原地修改；
  generation 单调。
- full-key equality fail-closed；无 hash-only identity、无 raw `void*` artifact ABI、无
  map element 裸指针生命周期。
- producer-reported readiness/bytes 和 fake tokens 不得被宣传为 production validation。
- failed candidate 不覆盖 healthy generation；quarantined identity 不得 acquire、rollback
  或 republish。
- lease survives discoverability eviction；slot byte budget 不得被宣传为 lease 持有的总
  resident bytes。
- cooperative cancellation/deadline 不得被宣传为 backend 硬超时。
