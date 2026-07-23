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
- whole-graph identity 遇未知 Relay node fail closed；
- verified baseline 冻结 ordered `(call index, link symbol, primitive ArtifactKey)`；
- candidate 每个 production pin 与 baseline 完整 key 相等，并核对内部
  `CachedPrimitive` signature、launch metadata、target/backend、launcher、provenance、bytes、
  validation record 及 module entry；
- ordered selected artifact mapping 进入 Plan ABI 和 PlanVariant identity；
- whole-graph key 不能作为 primitive fixture key；
- same-key singleflight、different-key 并行和 active-flight/slot backpressure；
- frozen whole-plan generation、exact acquire、completion-held variant/artifact lease；
- trusted control-plane `RollbackAdministrative`，只影响未来 routing；
- 同步 observer 在 lock 外执行，throw 被隔离；同 controller API reentry 全部 fail-fast；
- generation 与 primitive cache ticket/stamp overflow fail closed。

`RuntimeSession` 没有 compiler/cache/controller/generation 选择职责，仍是静态数据面。

## 2. 必须保持的 authority 边界

当前 executable payload 没有统一可序列化格式。现有 artifact proof 是：

```text
immutable process-local cache ArtifactKey
+ exact ordered call mapping
+ full typed CachedPrimitive/module equality
+ launcher shared-object identity
```

它可以阻止同 launcher/不同 artifact 换壳、signature/metadata 换壳和错序，但**不是**
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
- unknown Relay semantic identity fail-closed；
- `CompileAndPublish`、`Acquire`、`RunAsync`、`RollbackAdministrative` observer reentry fail-fast，
  observer throw 不改变 routing；
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

## 6. 外部门禁/未完成项

1. supported host TSan；
2. LLVM-enabled real compiler integration（仅 `KXC_USE_LLVM` 条件执行）；
3. CUDA target/launcher/stream 与真实 pending event retention；
4. native executable/device resident byte accounting；
5. 部署层 authentication、one-shot/replay-safe health authority 和数值证据；
6. 若未来扩展 shape，先提供 versioned applicability proof，禁止 fuzzy fallback。
