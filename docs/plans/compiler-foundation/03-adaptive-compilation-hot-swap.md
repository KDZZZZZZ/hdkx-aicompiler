# 03：自适应编译与安全热替换

> **状态：** 规划中；不表示当前已有 adaptive runtime
> **所属路线：** [编译器基础路线图](README.md)
> **共同基础：** [01：契约、identity 与 cache](01-core-contracts-identity-cache.md)
> **可选集成：** [02：Shape 系统与特化](02-shape-system-and-specialization.md)；static exact 热替换不等待 02
> **权威输入：** [编译器基础架构审查](../../COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)
> **范围：** 编译协调、版本发布、回退、canary、观测；**不解决 shape 基数**。

## 1. 目标与边界

本计划解决“一个确定的编译请求如何去重、验证、发布并安全替换”。

它不决定应产生多少种 shape artifact；那是 02 的 exact/bucket/polymorphic profile 策略。

若请求各有唯一 shape，singleflight 仍无法避免高 cardinality。

本计划可先基于 **static exact Plan ABI** 实现，不等待完整 Shape 系统。

最小闭环是：为同一静态 ABI 的 unit 编译第二个验证过的 artifact，供后续请求采用。

此时 logical/physical/valid extent 均可为 exact 静态合同，无 ShapeProgram/dispatcher 也可工作。

02 接入后只扩展 `DispatchKey` 与 request 来源，不改变发布安全协议。

```text
request -> CompileCoordinator -> compile/validate KernelArtifact
        -> KernelSlot publish generation -> frozen PlanVariant
        -> static RuntimeSession launch with retained lease
```

`RuntimeSession` 仍是数据面 executor：不编译、不预测、不查 cache、不作 variant policy。

热替换是发布新不可变 generation，绝不原地覆盖运行中的 module/function pointer。

## 2. 当前事实与约束

当前 session 验证 module/plan 的 symbols、signature、device、constants，并顺序 launch。

completion 已保活 module、plan、value table 和 prior operations；这是 lease/RCU 的基础。

primitive cache 当前为 256-entry 进程内 LRU，保存 immutable executable entry。

其 structural hash 含 graph-local value id，接入协调器前必须拆为真正 semantic/artifact key。

cache hit 后再 `Peek` 会受并发淘汰影响，须由编译事务持有 immutable artifact/pin 修正。

这些是 identity/ownership 缺陷，不是恢复 fuzzy cache 的理由。

当前单设备、单 stream、静态 output 的 session 边界保持不变。

## 3. 依赖、非目标、并行

### 3.1 硬依赖

- `PlanAbiFingerprint`：可比较的 signature、layout、workspace、alias 和 target ABI 合同。
- `KernelArtifactKey`：unit semantic key + target + pipeline + ABI + backend/schedule；不含 value id/symbol。
- immutable、引用计数的 module/executable handle；不向调用者泄露 cache 容器裸指针。
- publish 前 backend-ready、link/symbol、signature、launch、target、ABI 与健康验证。
- static `RuntimeSession` 能执行冻结 module+plan，completion 能 retain artifact generation。

### 3.2 软依赖

- 02 的 GraphTemplate、ShapeProfile、bucket dispatcher；不存在时以 exact `DispatchKey` 工作。
- persistent cache、distributed registry、GPU worker、预测优先级、真实 canary 流量标签。
- task DAG、多 stream、region fusion、动态输出和在线 frontend。

### 3.3 可并行工作包

| 工作包 | 可并行对象 | 汇合条件 |
|---|---|---|
| ABI fingerprint/verifier | 02 DTO | 02 增字段不改变 exact ABI 含义 |
| coordinator state/singleflight | KernelArtifact DTO | 只依赖 request/result 接口 |
| KernelSlot/lease/RCU | static session adapter | completion retain lease |
| queue/failure/backoff | profiling schema | event/key 字段一致 |
| canary/rollback | backend validation tests | policy 不碰 kernel 参数 |

03 的 exact 阶段不等待 02；02 可用 fake/synchronous coordinator 独立开发。

### 3.4 非目标

- 不降低 shape cardinality，不制定 bucket，不做 ShapePredictor。
- 不恢复 `cached dims >= query dims` fuzzy 选择。
- 不让 RuntimeSession 持有 Compiler、Relay、ShapePredictor、KernelCache 或后台线程。
- 不修改正在执行的 module/plan/function pointer。
- 没有已验证 variant 时，不承诺 fallback；只能等待、拒绝或报 unavailable。
- 不用 cache miss 同时表达 absent、compiling、failed、cancelled。

## 4. identity、artifact 与 Plan ABI

### 4.1 identity 分层

```text
UnitSemanticKey   = normalized unit computation + boundary semantics
KernelArtifactKey = semantic key + target + pipeline + ABI + backend/schedule
DispatchKey       = artifact family + exact shape/profile/layout applicability
PlanVariantKey    = template/revision + selected call generations + physical plan
GraphValueLocator = graph revision + value_id + output index
```

03 首版可将 `DispatchKey` 设为 exact ABI/profile fingerprint。

locator、storage id、entry symbol 只服务 plan/diagnostic/linking，不能作跨图语义 key。

hash 仅用于索引；等价必须比较完整 canonical key/bytes。

### 4.2 KernelArtifact 草案

```cpp
struct KernelArtifact {
  KernelArtifactKey key;
  CompiledEntry executable;       // immutable, reference-counted
  KernelSignature signature;
  LaunchMetadata launch;
  Applicability applicability;    // exact first; 02 later adds guards
  PlanAbiFingerprint compatible_abi;
  CompileProvenance provenance;
  HealthStatus health;
};
```

只有 executable ready 且完整验证通过的 artifact 才能成为 Ready。

artifact 不携带 graph-local call id、不拥有 slot、不发布半成品。

lease、PlanVariant、编译事务或 AsyncOperation 引用期间必须保活 executable/module。

### 4.3 同 Plan ABI 条件

可替换的 slot candidate 至少完全兼容：

- 参数数量、顺序、role，以及 `[input][constant][output]` ABI；
- dtype、device、rank、logical/physical layout、alignment、mutability、constant key；
- shape/extent applicability、workspace、alias 与错误模型；
- target/backend capability、ABI version、entry rebinding/link 规则。

static exact 首版要求全部 shape/layout/capacity 数值精确一致。

02 的 bucket/polymorphic 仅在 extent、tail safety、workspace 和 guard 已证实时放宽。

physical capacity/layout/workspace、call topology、storage plan 任一变化，都必须切换 `PlanVariant`，不能替换 slot。

## 5. 接口与状态草案

### 5.1 CompileCoordinator

```cpp
class CompileCoordinator {
 public:
  Future<CompileResult> Request(CompileRequest request);
  CancelResult Cancel(RequestId id);
  CoordinatorSnapshot Snapshot() const;
};
struct CompileRequest {
  KernelArtifactKey artifact_key;
  DispatchKey dispatch_key;
  PlanAbiFingerprint required_abi;
  Priority priority;
  RequestKind kind;  // demand, prewarm, canary
  Deadline deadline;
};
```

相同完整 `(artifact_key, dispatch_key, required_abi)` 的 in-flight 请求共享 future。

返回 future/result，不返回 map/vector 元素地址；合并前比较完整 key，不只比较 hash。

### 5.2 请求状态机

```text
Absent -> Queued -> Compiling -> Validating -> Ready
             |          |             +-> Failed(retry_after)
             |          +-> Cancelled
             +-> Cancelled
Failed(retry_after) -> Queued       // only retryable, after backoff
```

`Absent` 不是 cache miss；artifact cache 仅保存成功 immutable artifact，request state 独立管理。

Queued prewarm 可取消；demand 有明确 wait/timeout 语义。

Compiling 的取消协作式；不可中断 backend 完成物只能验证后丢弃或受控发布。

Validating 对 dispatch 不可见；unsupported target/shape/ABI 为稳定失败，不能重试风暴。

Failed 记录分类、attempt、first/last seen、retry_after、retryability 与诊断。

### 5.3 backpressure

预算至少覆盖全局、per-model、per-target/backend 的并发和临时 module/artifact bytes。

排队顺序：demand 优先 prewarm，然后按 deadline、预计收益、编译成本。

饱和时先丢弃低收益 prewarm；不能按每个 miss 无界创建 LLVM/NVRTC 工作。

记录 waiter 数、merge rate、queue/wait/compile time、取消、预算拒绝和拒绝原因。

## 6. KernelSlot、generation 与 PlanVariant

### 6.1 KernelSlot 草案

```cpp
class KernelSlot {
 public:
  ArtifactLease Acquire(DispatchKey, PlanAbiFingerprint) const;
  PublishResult Publish(std::shared_ptr<const KernelArtifact> candidate);
  RollbackResult Rollback(Generation generation);
};
struct VariantRecord {
  Generation generation;          // strictly monotonic
  std::shared_ptr<const KernelArtifact> artifact;
  Applicability applicability;
  HealthStatus health;
  PublishMetadata published;
};
```

slot identity 是 `(UnitSemanticKey, target)`，不是某次 graph call。

record publish 后不可变；更新产生单调递增 generation。

`Acquire` 仅返回健康、已验证、ABI/applicability 匹配的 snapshot。

实现可用 atomic `shared_ptr` 或等价 RCU，但 API 语义必须保证 snapshot + 强引用。

slot 不解释模型语义，不猜测 shape 兼容性，不持有 compiler 私有 IR。

### 6.2 PlanVariant

```text
PlanVariant {
  frozen plan/value routing;
  PlanAbiFingerprint + exact/profile contract;
  per-call { slot, selected generation, entry binding };
  physical memory/workspace plan;
}
```

请求边界选择/组装 immutable variant；`RuntimeSession::Run` 内禁止重新查 cache。

同 ABI artifact 可在后续请求采用新 generation；已冻结 variant 要么 pin generation，要么下次请求重新解析。

ABI/physical plan 改变必须建立独立 PlanVariant，旧 variant 保持可运行。

## 7. 发布、RCU 与执行协议

### 7.1 publish 前

1. coordinator 合并/记录 request，持有完整 key、deadline、provenance。
2. backend compile 后验证 ready、symbol/link、signature、launch、target、required ABI、applicability。
3. 运行最小 ABI/数值 health check；失败进入 Failed/quarantine。
4. `compatible_abi` 相等才创建 immutable generation 并 publish。
5. 记录 predecessor、验证证据、publish reason、slot generation 和 profile event。

验证失败绝不能覆盖 healthy generation。

### 7.2 Acquire 到 launch

1. controller 为新请求选择明确 frozen PlanVariant。
2. 每 call `Acquire` 得到 generation-bound `ArtifactLease`。
3. session 再验证 lease 的 signature/ABI 与 plan value contract。
4. session launch lease entry，并将 lease/artifact 纳入 completion retain。
5. 新 publish 只影响未来 Acquire；已开始的 run 持有旧 snapshot 到异步完成。

这提供 RCU 读侧语义：读者不阻塞发布，但以强引用保活旧版本。

cache eviction 只能移除发现入口，不能释放 lease/variant/operation 正在使用的 executable。

### 7.3 variant 切换

| 情况 | 行为 |
|---|---|
| 同 ABI、同 physical plan、healthy | 新请求可采用新 generation |
| 同 ABI、canary | 仅 canary routing 采用候选 |
| ABI/layout/capacity/workspace 变化 | 组装或选择独立 PlanVariant |
| run 已提交 | 不切换 generation/variant |
| 无 healthy matching record | wait、显式验证 variant 或 unavailable |

“显式验证 variant”必须有 applicability/ABI 证明，不能是 fuzzy fallback。

## 8. failure、rollback、canary

| failure | 处理 |
|---|---|
| unsupported | stable negative cache，直接报告 |
| deterministic compile | 限次/版本变化后重试，保留诊断 |
| transient OOM/driver | budget + exponential backoff + jitter |
| validation | quarantine，绝不 publish |
| cancellation | 丢弃低价值 waiter/完成物，审计原因 |
| runtime canary health | withdraw candidate，rollback |

failure 永不替换已有 healthy artifact。

每次 publish 记录 predecessor healthy generation；rollback 仅改变未来 dispatch head，不修改 in-flight artifact。

无 predecessor 时 slot 进入 unavailable，不能使用未验证候选。

canary 前提是同 Plan ABI 和基础验证通过；按稳定 request/model/profile hash 分流，单请求不混 generation。

比较数值、launch error、错误率、延迟、内存和后端诊断；hard correctness gate 失败立即 withdraw。

没有安全对照/镜像输出时，只做 shadow compile/profile，禁止自动 promote。

## 9. profiling 与 Issue #14

事件至少关联 request id、model/template revision、unit semantic key、artifact key；shape/profile 是可选标签。

事件：`request_merged`、`queued`、`budget_rejected`、`compile_started`、`validation_failed`、`published`、`acquired`、`launch_retained`、`promoted`、`withdrawn`、`rolled_back`。

记录 queue/wait/compile/validate/publish 时间、waiter、priority、failure kind、generation、lease lifetime、ABI、PlanVariantKey、workspace/memory contract。

衡量 compile-storm、P95、artifact bytes、merge rate、命中和实际 kernel 性能；cache hit 不是性能结论。

### 9.1 Issue #14 保留

- 同 key 请求去重，避免重复 backend compile。
- 已验证 artifact 可在后续请求边界发布和采用。
- in-flight execution 不因替换/淘汰失效。
- 失败、预算、队列、收益可观测。
- 静态 executor 与编译控制面分层。

### 9.2 Issue #14 废弃

- `RuntimeSession` 同时持有 Relay、Compiler、ShapePredictor、KernelCache、BackgroundCompiler、KernelRunner。
- `cached dimension >= query dimension` fuzzy cache。
- map 元素裸指针、`void*`/裸函数指针长期执行 ABI。
- cache miss 混同 absent/compiling/failed/cancelled。
- 原地覆盖 in-flight kernel/module，或未保活即淘汰。
- runtime 按 op 名、frontend 格式、registry 顺序决定执行/缓存策略。

“自适应”保留为显式 control-plane，不保留历史耦合实现。

## 10. 实施、测试、Done

### 10.1 实施步骤

1. 分离 semantic/artifact key，冻结 exact `PlanAbiFingerprint` 与 full-key equality。
2. 定义 immutable artifact、request/result、failure record、lease，并做纯状态机测试。
3. 实现 synchronous coordinator singleflight，接入 static exact compile；先不启后台线程。
4. 加入 queue、priority、budget、cancel、negative cache、retry/backoff。
5. 实现 slot record、generation publish/Acquire、lease retain，接入 static session completion。
6. 实现 exact PlanVariant generation selection；ABI/physical 变化强制新 variant。
7. 加入 validation、quarantine、rollback、审计；最后启用 canary。
8. 02 接入后仅提供 profile/dispatch 请求，先证明 applicability 再提交 coordinator。

集成门禁 A：没有 02 时步骤 1--7 必须在 static exact ABI 闭环。

集成门禁 B：bucket/polymorphic 只能由 02 dispatcher 证明后进入 coordinator。

集成门禁 C：`RuntimeSession` 的 module+plan API 与依赖边界不变。

### 10.2 测试

| 层次 | 正例 | 反例 |
|---|---|---|
| key/ABI | 等价 unit 跨 value id 复用 | attr/layout/ABI 误命中 |
| singleflight | N 同 key demand 只编译一次 | hash collision/不同 ABI 合并 |
| state | retryable 转换正确 | unsupported 无限重试/failed 覆盖 healthy |
| backpressure | demand 优先、prewarm 丢弃 | queue/module bytes 无界 |
| slot/RCU | generation snapshot、旧 lease 可完成 | ABI 不同 publish/in-flight 失效 |
| PlanVariant | physical 改变走新 variant | Run 内切换 generation |
| rollback/canary | bad candidate withdraw/恢复 healthy | 无对照自动 promote |
| integration | static exact 数值不变 | session 依赖 Compiler |

并发 publish/Acquire/lease 交错需压力测试；实现阶段用 ASan/TSan 验证所有权/竞态。

### 10.3 Done 条件

- static exact ABI 完成 request -> singleflight -> validate -> publish -> acquire -> launch retain 闭环。
- 同 full key 只 backend compile 一次；不同 key 永不错误合并。
- coordinator 有完整状态、预算、backpressure、结构化失败与退避。
- artifact、record、variant 不可变；generation 单调、可审计。
- 仅同 Plan ABI 可替换；capacity/layout/workspace/topology 改变必走新 PlanVariant。
- lease/variant/AsyncOperation 释放前旧 generation 可执行，且没有裸指针生命周期漏洞。
- validation/canary 失败不替换 healthy；rollback 只影响未来请求且可观测。
- 03 不含 shape-cardinality 策略；02 接入不改变状态/发布安全协议。
- `RuntimeSession` 仍不依赖 Compiler/Relay/ShapePredictor，static API/测试保持通过。
