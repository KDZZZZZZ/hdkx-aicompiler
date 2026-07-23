# 03：自适应编译与安全热替换

> **状态：** experimental isolated control-plane 已实现；**生产 Core/runtime 集成被阻塞**。这不表示已有 production adaptive runtime。
> **API：** `kxc::api::experimental::adaptive::v1`（fake、static-exact、nonproduction）。
> **所属路线：** [编译器基础路线图](README.md)
> **共同基础：** [01：契约、identity 与 cache](01-core-contracts-identity-cache.md)
> **可选集成：** [02：Shape 系统与特化](02-shape-system-and-specialization.md)
> **范围：** 编译协调、版本发布、回退、canary 与观测；不解决 shape 基数。

## 1. 目标与当前实现边界

目标仍是：对一个确定的编译请求去重、验证、发布不可变 generation，并让未来请求
安全采用它；已执行的工作通过 lease 保持旧 generation。

已实现的是隔离的 static-exact 控制面：

```text
caller-visible strings
  -> CompileCoordinator -> injected fake compiler/authority
  -> immutable KernelArtifact -> KernelSlot -> ArtifactLease
  -> injected fake ExactPlanAssembler -> FrozenPlanVariant
```

这不是 production compiler/runtime 路径。没有 production `ArtifactCompiler`、
validation authority、assembler、controller 或 runtime adapter；没有
`RuntimeSession` 集成，也不声称或计划在本轨直接修改 `RuntimeSession`。

02 未接入时只允许 exact `DispatchKey`。若请求各有唯一 shape，singleflight 仍不能
解决 high cardinality；本计划不制定 bucket 或 ShapePredictor 策略。

## 2. identity 与可信输入

`KernelSlotKey`、`KernelArtifactKey`、`DispatchKey`、`PlanAbiFingerprint` 和
`PlanVariantKey` 必须保持分型。singleflight 和 publication gate 比较完整
`(artifact key, dispatch key, required ABI)`；hash 仅可用于索引或 canary 分流，不能
定义相等性。

当前 API 所接收的 canonical 字段是调用者提供、可见的非空字符串。它们只做完整值比较：

- 不是 opaque byte representation；
- 不是 Core canonical identity；
- 没有格式、版本或 semantic-equivalence 验证；
- 不得据此声称跨图、跨进程、跨版本 identity。

生产接入的硬前置条件是 Core 提供真正 opaque、版本化 canonical identity 和
`PlanAbiFingerprint` canonicalizer。至少它们要覆盖 semantic unit、target、pipeline、
backend/schedule、参数 role/顺序、dtype/device/rank、logical/physical layout、stride、
alignment、valid extent、constant/alias/effect/error contract、workspace、launch metadata、
backend capability/version、entry/link/rebinding 和 plan topology/storage proof。

layout、capacity、workspace、topology 或 physical memory plan 任一变化都必须创建新的
`PlanVariantKey`/variant，不能替换现有 static-exact slot。禁止
`cached_dim >= query_dim`、`-1` 推断和任何 fuzzy fallback。

## 3. 已实现的控制面语义

### 3.1 Coordinator

- 完整同 key 请求 singleflight；不同 dispatch/ABI 永不合并。
- 有界 worker、queue、waiter、terminal record 和 cached artifact budget；ready、失败、
  cancellation、backpressure 和 retry 彼此区分。
- 调度顺序为 demand、canary、prewarm，再按 priority、deadline、FIFO。same-key demand
  会提升 queued prewarm；满队列时 demand 可逐出 queued prewarm。
- deadline 是 **queue-start expiry**，仅使仍在队列里的 waiter 过期；它不是 backend
  compile 的硬 timeout，也不取消已开始的 backend 操作。
- cancellation 只有 cooperative 语义：注入 compiler 必须观察 token。永久阻塞或不可
  中断的 production backend 仍需进程隔离或上层有界 worker 生命周期，不能靠 detach。
- lifecycle 由 owned operational state 与独立 `ThreadGroup` 管理。worker 只请求 stop，
  management caller 同步 join；仅在 worker callback 析构最后 owner 时，owned deferred
  join service 才在非 worker 上接管 thread handles。没有 detached worker。

### 3.2 Artifact、authority 与 byte accounting

`KernelArtifact` 不可变且强持有 typed executable，但 `IsReady()` 和 `byte_size()` 都是
producer-reported。控制面只校验它们与 request/record/budget 的一致性；它不证明真实
backend readiness、link、launch、数值健康、设备分配或总 resident bytes。

`AdaptiveValidationAuthority`、validation/health records 和 one-shot tokens 是注入的
确定性 **fake trust seam**。它们不是 cryptographic validation、production attestation
或 runtime health validation。生产 adapter 必须提供真正的、线程安全 one-shot 消费和
不可伪造、artifact-bound evidence。

### 3.3 Slot、generation 与 lease

- publish 创建不可变、单调递增 generation；未来 `Acquire` 才看到新 head。
- `ArtifactLease` 强持有 artifact。删除 retained record 只移除 slot 可发现性；lease、
  frozen variant 或 operation 仍可保活 artifact，且可以在 record eviction 后继续存在。
- `KernelSlot` 的 retained byte budget 是**可发现 retained records**的
  producer-reported artifact bytes；不是 lease/variant/operation 持有的总 resident bytes，
  更不是进程或设备总内存 accounting。
- canary 必须有 stable predecessor，并且只有显式 routing context 才能分流；promote
  要求 authority 接受的 healthy record。

## 4. 发布、rollback 与 quarantine 不变量

以下是 production integration 也不得破坏的 immutable-record/token/quarantine 不变量：

1. artifact、validation record、health record、slot record 和 frozen variant 一经创建即
   不可变；更新只创建 generation/record，不原地修改 in-flight executable。
2. validation 和 health token 必须绑定精确 artifact/slot/dispatch/ABI/generation，且由
   authority one-shot 消费。构造 record 或 token 本身不授予信任。
3. validation failure 不得覆盖 healthy stable generation。
4. rollback 的 regression record 必须将当前 stable 标记 quarantined；quarantine 对该
   `KernelSlot` 的生命周期 durable，不因 generation record eviction 而消失。
5. quarantined artifact identity 不得被 `Acquire`，不得成为 rollback target，也不得再
   publish。withdrawn canary 同样不具备 rollback eligibility。
6. rollback 和 withdraw 只影响未来 routing；已有 lease/frozen variant 不失效。

## 5. 生产集成阻塞项与非目标

### 5.1 必须由 Core/runtime 补齐

1. opaque、版本化 semantic/artifact/ABI canonicalization，消除 caller-visible string
   seam。
2. 真实 artifact compiler、backend ownership/pin 和实际 byte accounting。
3. 后端 ready、symbol/link、signature、launch、target、ABI、数值/健康的真实验证，及
   不可伪造 validation/health authority。
4. 真实 exact plan assembler/controller/runtime adapter，定义 artifact lease、module、
   plan 与 completion 的 ownership transfer。
5. 受支持环境中的 end-to-end runtime integration verification；不能将 fake tests 宣称
   为 `RuntimeSession` coverage。

### 5.2 非目标

- 不把 `RuntimeSession` 变为 compiler/cache/shape/controller 或后台线程所有者。
- 不改变运行中的 module、plan 或 function pointer。
- 不在没有 verified matching record 时承诺 fallback；只能 wait、reject 或 unavailable。
- 不以 cache miss 表达 absent、compiling、failed、cancelled 的全部状态。
- 不实施 shape-cardinality、bucket、polymorphic routing 或生产流量控制策略。

## 6. CTest、验证与剩余工作

当 `BUILD_TESTING` 开启时，三个 adaptive executable 注册为 `adaptive;cpu` CTest：
`adaptive_contract_test`、`adaptive_coordinator_test`、`adaptive_kernel_slot_test`。
`run_adaptive_tests` 用相同 label 交集实际运行 CTest；
`dev-ninja-cpu-adaptive` preset 复用 `dev-ninja-cpu` 配置路径。CPU smoke CI 运行：

```bash
ctest --test-dir out/build/ci-cpu --output-on-failure -L adaptive -L cpu
```

本轮 Debug CPU adaptive CTest 3/3 通过；coordinator/slot focused concurrent suite
各重复 100 次通过；ASan+UBSan+LeakSanitizer 3/3 通过；adaptive warning-as-error
语法检查、public-header 与 include-layer 检查通过。TSan 可编译链接；一次可启动运行发现
并修复了 test observer log 的 notify/destruction 同步竞态，但修复后重跑及多数尝试都在
进入测试前报 `ThreadSanitizer: unexpected memory mapping`，因此没有最终有效的 TSan
pass。这些 fake seam 结果不是 production integration 证据；supported-host TSan 和真实
Core/runtime adapter end-to-end 仍待完成。

完成 production integration 前的门禁：

- Core canonical bytes/version 和真实 trust authority 已冻结；
- static-exact adapter 证明真实 ABI、artifact byte accounting、lease retain 和 completion
  ownership；
- quarantine/token/immutable-record 不变量有真实 backend 覆盖；
- `RuntimeSession` 仍只作为数据面 executor，且其边界未因 adaptive control-plane 反转；
- 02 如扩展 dispatch，必须先提供 versioned applicability proof。
