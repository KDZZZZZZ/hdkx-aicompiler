# Compiler Foundation Adaptive / Hot-Swap 轨道交接

> 分支：`feature/compiler-foundation-adaptive`
>
> 范围：当前 static exact Plan ABI 下的独立控制面
>
> 状态：**轨道内 fake/static-exact 合同已闭环；生产 Compiler/Plan 接入等待 01，shape-aware routing 等待 02**

## 1. 本轨交付

新增公共控制面合同 `include/kxc/compiler/adaptive.h`，实现位于
`src/compiler/adaptive/`。该层属于 compiler/control-plane，不在
`RuntimeSession` 内部：

```text
CompileRequest(full exact key)
  -> bounded CompileCoordinator
  -> immutable KernelArtifact
  -> KernelSlot generation publication
  -> ArtifactLease (shared ownership / RCU read snapshot)
  -> ExactPlanAssembler
  -> FrozenPlanVariant
  -> future runtime adapter -> RuntimeSession(module, plan)
```

本轨没有修改 `include/kxc/runtime/session.h`、`src/runtime/session.cc`、
primitive cache、Relay/TIR lowering 或现有 plan 执行语义。

### 1.1 强类型 identity 与 artifact

- `KernelSlotKey`、`KernelArtifactKey`、`DispatchKey`、
  `PlanAbiFingerprint`、`PlanVariantKey` 是不同 C++ 类型，不能互换。
- `KernelArtifactKey` 显式包含 slot identity；singleflight 使用完整
  `(artifact key, dispatch key, required ABI)` 比较。
- 内部协调表使用完整值排序/相等，不以 hash 判等；canary 的 FNV 仅用于流量
  分流，不参与 key equality。
- `DispatchKey` 当前只能通过 `DispatchKey::Exact(...)` 创建；不存在维度偏序、
  “更大 shape”命中或 fuzzy reuse。
- `KernelArtifact` 构造后不可变，强持有
  `std::shared_ptr<const ArtifactExecutable>`；没有 `void*`、裸函数指针或裸 map
  元素指针。
- canonical bytes 当前由 fake/未来 01 canonicalizer 提供。本轨只验证非空和完整
  bytes equality，不擅自复制 01 的 canonicalization policy。

### 1.2 CompileCoordinator

实现状态流：

```text
Absent -> Queued -> Compiling -> Validating -> Ready
                    |              |
                    +-> Cancelled  +-> Failed(retry_after)
Failed(retryable, retry_after elapsed) -> Queued
```

行为：

- same-full-key singleflight；不同 dispatch/ABI 永不合并；
- 有界 worker、queue、单 flight waiter、单 artifact bytes、terminal record 和
  aggregate cached artifact bytes；
- demand/canary 优先于 prewarm，队列饱和时 demand 可先逐出 queued prewarm；
- caller-specific cancellation：取消一个 waiter 不影响其他 waiter；最后一个 waiter
  取消后，废弃 flight 不再接受新请求；
- unsupported/deterministic/validation failure 负缓存；transient failure 按有界指数
  backoff 和最大 attempt，由后续 request 在 `retry_after` 后触发重试；
- compiler exception 被结构化为 deterministic failure；candidate 必须再次通过
  artifact key、exact dispatch、Plan ABI、ready 状态和 byte budget 验证；
- shutdown 先停止 admission、完成 queued/active tickets、发送 cooperative
  cancellation，再由非 worker 管理线程 join 全部 worker；没有 detached thread；
- compiler/observer 在 worker 上请求 shutdown 不会 self-join，observer shutdown
  re-entry 不会锁递归；
- snapshot/event 暴露 queue、waiter、attempt、failure、bytes、request identity、
  queue/compile/validation 时间和生命周期计数。

### 1.3 KernelSlot / generation / canary / rollback

- slot 绑定一个 `KernelSlotKey + static exact PlanAbiFingerprint`；
- publish 产生严格单调 generation，不原地修改 artifact；
- `Acquire` 在 mutex 下选定完整 exact dispatch head，并返回 generation-bound
  `ArtifactLease`；lease 以共享所有权保活旧 artifact；
- 新 publish 只影响未来 acquire，已有 lease/FrozenPlanVariant 继续使用旧 generation；
- generation history 和 exact dispatch 数量均有界；回收只删除 slot 可发现记录，
  已发出的 lease 不失效；
- canary 必须已有 stable predecessor，且只有显式 routing context 才可进入；同一
  stable request key 的分流确定；
- promote 必须提供显式 health evidence；withdraw 只影响未来路由；
- withdrawn canary 不进入 rollback eligible 集合，不能被回滚为 stable；
- rollback 只能选择仍保留、曾为 healthy stable 的同 exact dispatch generation；
- `ExactPlanBinding` 将每个 plan selection 的 slot、dispatch、ABI 与 lease 再比较；
  `FrozenPlanVariant` 强持有全部 lease 和 typed `PlanExecutable`。

## 2. Static exact ABI 约束

本轨允许同一 slot hot-swap 的前提是 `PlanAbiFingerprint` **完整、规范且逐 byte
相等**。01/真实 assembler 接入时 fingerprint 至少必须覆盖：

1. 参数数量、顺序、`[input][constant][output]` role 和 mutability；
2. dtype、device、rank、每个 exact logical/physical dimension；
3. layout/stride/padding、alignment、valid extent（exact 阶段等于 logical extent）；
4. constant key/绑定合同、alias/effect/error model；
5. workspace、dynamic shared memory、launch metadata；
6. target/backend capability、ABI version、pipeline/schedule/backend version；
7. entry symbol/rebinding 规则，以及不会改变 static plan topology/storage contract 的证明。

任一 layout、capacity、workspace、topology 或 physical memory plan 变化都必须产生新
`PlanVariantKey`/plan variant，不能发布到现有 exact slot。`kDynamicDimension = -1`
不得进入新 `DispatchKey` 或作为 shape compatibility 证明。

## 3. 测试证据

### 3.1 构建配置

```bash
cmake -S . -B out/build/adaptive-control-plane -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_CODEGEN_TESTS=OFF
```

### 3.2 本轨测试

下列测试全部通过：

```bash
./out/build/adaptive-control-plane/adaptive_contract_test
./out/build/adaptive-control-plane/adaptive_coordinator_test
./out/build/adaptive-control-plane/adaptive_kernel_slot_test
```

覆盖：

- strong key/full equality、exact dispatch、immutable typed artifact；
- 64 个并发 same-key waiter 只编译一次；
- full-key 分离、ready reuse、aggregate terminal byte budget；
- queue/backpressure、prewarm displacement、priority；
- 单 waiter/queued/last-active cancellation 和 replacement flight；
- waiter budget、negative cache、transient retry、unsupported stable failure；
- invalid/oversized artifact validation；
- worker/observer shutdown re-entry 和 deterministic shutdown；
- generation/ABI gate、旧 lease 与 frozen plan 保活；
- incompatible plan binding 拒绝；
- canary withdraw/promote/rollback、orphan canary 拒绝；
- 16 reader + 500 publish 并发 slot 压力；
- fake compiler -> coordinator -> slot -> fake assembler 完整闭环。

并发测试重复运行 30 轮，30/30 通过。

### 3.3 Sanitizer / warnings

ASan + UBSan + LeakSanitizer 构建下三个 adaptive 测试全部通过，无诊断：

```bash
for test in adaptive_contract_test adaptive_coordinator_test adaptive_kernel_slot_test; do
  ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
    ./out/build/adaptive-check-asan-ubsan/$test || exit 1
done
```

`-Wall -Wextra -Wpedantic` 构建没有来自 adaptive header/source/tests 的 warning。

TSan binary 可成功编译链接，但当前执行环境在进入测试前报：

```text
FATAL: ThreadSanitizer: unexpected memory mapping ...
```

因此本机没有有效 TSan 运行结果；这不是 TSan race report。集成 CI/受支持环境仍应补跑。

### 3.4 架构与静态回归

通过：

- `check_include_layers`
- `check_public_headers`
- `check_relay_op_contract`（19/19）
- `check_pass_contract`（19/19）
- `graph_partition_test`
- `executable_plan_test`
- `kernel_signature_test`
- `compiled_module_test`
- `runtime_session_test`
- `compiler_contract_test`
- `compiler_extension_contract_test`
- `operator_compilation_test`

## 4. 原子提交

| Commit | 内容 |
|---|---|
| `863ffb7` | `feat(compiler): add exact adaptive artifact contracts` |
| `eb748cb` | `feat(compiler): coordinate bounded exact compilation` |
| `eee18dd` | `feat(compiler): publish adaptive kernel generations` |
| `9930a50` | `fix(compiler): harden adaptive lifecycle bounds` |
| 本交接提交 | `docs(compiler): hand off adaptive control plane` |

## 5. 明确未做与跨轨集成项

### 5.1 依赖 01 Core Contracts

生产接入前需要 01 提供/冻结：

1. 不含 graph-local value id、symbol、storage id、object address 的
   `UnitSemanticKey`/`KernelArtifactKey` canonicalizer；
2. 完整 `PlanAbiFingerprint` canonicalizer 和 versioning；
3. ready artifact store/pin 与真实 byte accounting；
4. backend-ready、symbol/link、signature、launch、target、ABI、数值 health 的
   不可伪造 validation record；
5. 真实 `ArtifactCompiler` adapter；它把当前 per-unit compile 产物包装为
   `ArtifactExecutable`，并遵守 cooperative cancellation；
6. 真实 `ExactPlanAssembler` adapter；它用 exact bindings 生成 runtime-only
   module + plan，并让 `PlanExecutable` 保活 selected artifacts。

当前 fake 合同故意不读取 private Compiler/Relay/TIR header，也没有修复 01 所属的
primitive cache 二次 `Peek`、structural hash value-id 污染或 artifact symbol/graph
call identity 分离。

### 5.2 依赖 02 Shape

shape-aware routing 只允许通过版本化接口扩展：

- 02 提供 `ShapeProfile`、logical/physical/valid extent 和 applicability proof；
- 这些字段进入新版本 `DispatchKey`/`ExactPlanBinding`；
- coordinator state/singleflight/negative cache 和 slot generation/lease 协议保持不变；
- 未有 proof 时只能 exact compile/wait/reject；禁止 `cached_dim >= query_dim`、
  `-1` 推断或 fuzzy fallback。

### 5.3 生产策略上限

当前实现为单进程全局 worker/queue budget。真实服务策略仍需在控制面扩展
per-model/per-target quota、deadline/aging、收益/成本估计和稳定 failure 持久化；这些
字段不得进入 artifact equality，也不得下沉到 `RuntimeSession`。

shutdown 的确定性依赖真实 compiler 遵守 cooperative cancellation。无法中断且可能永久
阻塞的 backend compiler 需要进程隔离或上层有界 worker 生命周期；不能靠 detached thread
规避。`CompileCoordinator` 必须由非 worker lifecycle owner 最终析构/join。

## 6. 不变量检查

- `RuntimeSession` 无 Compiler/Relay/ShapePredictor/cache/background-thread 依赖。
- runtime Run 内不查 cache、不选 generation、不触发 compile。
- 无 raw `void*` artifact ABI、raw map pointer、hash-only equality。
- 无 fuzzy shape reuse；static exact equality fail closed。
- publish/rollback 不修改 in-flight artifact；lease/frozen variant 共享所有权保活。
- cancellation/shutdown 不 detach worker，失败 candidate 不覆盖 healthy generation。
