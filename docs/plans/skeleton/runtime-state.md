# Runtime / ExecutablePlan 状态骨架

## 目标与参考契约

借鉴 TVM `NDArray` 的引用计数 Storage 所有权、XLA/JAX 的显式 donation
（未声明不得原地写）、以及 ONNX Runtime `OrtValue` 的 tensor/storage 分离：
`ExecutablePlan` 声明稳定 value identity 和物理 storage contract，
`RuntimeSession` 独占持久 state Storage 并只执行已冻结的 module/plan。

本变更只支持静态、单设备、顺序 plan。一个 state 是一个 session-owned、零初始化
的 source `ValueSpec`；它跨 `Run`/`RunAsync` 保留。kernel 要修改它时，必须产生一个
显式 `kInPlace` alias value，该 value 指向 source value id。普通输入也可作为
显式 donation source；常量、图输出和写后仍被读取的 source 被拒绝。

术语：**value** 是逻辑 ABI id；**storage** 是物理 buffer id；**state** 是
session-owned source value；**alias** 是与 source 同 Storage 的 produced value；
**valid bytes** 是该静态 value 可读/写的有效字节上限（`-1` 表示完整 tensor）。

## 不变量

- `state_value_ids` 与 `ValueSpec::is_state` 一一对应；state 不能同时是 input、constant
  或 graph output，且没有 producer，但必须被至少一个 call 消费（拒绝死 state）。
- alias 必须有不同的 source id、同 shape/dtype/device/storage id，且 producer call 必须把
  source 列为输入；不能写常量或 graph output，其 source 在 alias producer 之后不得再使用。
  只有 `kInPlace` 可带 alias source。
- `valid_bytes` 必须在静态 tensor 容量内，并在 runtime storage 绑定时再次核对；它不改变
  physical shape 或 KernelSignature ABI。
- state Storage 的 alignment 覆盖同 slot 的全部 kernel 参数（包括 alias output）；普通输入
  donation（含 alias chain）不满足任一 alias output alignment 时必须在任何 launch 前失败。
- 未声明 alias 的 value 永不复用 source Storage；既有 `is_alias` 仍是保守的 memory-plan
  禁止复用标记，旧 plan 语义不变。
- session 在创建时一次性分配并清零 state；每次执行把同一 `NDArray` 绑定进局部
  `ValueTable`。completion 保活该局部 table/Storage。

## API / ABI

`ValueSpec` 追加 `is_state`、`alias_source_value_id`、`ValueWriteMode` 和
`valid_bytes`；`ExecutablePlan` 追加可选的 `state_value_ids` 构造参数和 accessor。
旧五参数 `ExecutablePlan` 调用和旧 `ValueSpec` 默认参数保持 source 兼容。plan ABI
fingerprint 纳入 state list、alias source ordinal、write mode 和 valid bytes。该项目没有 plan
序列化 ABI；若未来导出/FFI plan，必须先为这些字段增加 schema version，不能猜默认值。

`RuntimeSession` 保持 `Run`/`RunAsync` API；不暴露裸 state pointer 或模型名。state 初始化
和销毁由 session/ObjectRef 生命周期管理，未来参数加载/训练更新应通过另一个明确验证的 API，
而不是复用 input 或修改 module constants。

## 线程和异步

无 state plan 仍可并发执行，保持现有局部 `ValueTable` 行为。含 state plan 在 session
内部串行化：提交前若上一次 state completion 未 ready 则 fail closed，不在不同 stream 上
推测依赖。新 completion 被 session 保存，同时返回给调用方；因此丢弃外部 result/session/input
不会提前释放已提交的 state Storage。`Run` 等待其 completion。CPU completion 同步完成；
CUDA/pending backend 使用同一 readiness gate。

## 兼容和非目标

旧无状态 plan 不分配 state、不加锁、不改变 output/constant ABI。没有动态 shape、state
rebind、snapshot/COW、paged KV cache、cache position 更新、distributed coherence、训练 optimizer
或模型专用字段；`valid_bytes` 仅是静态 contract，非动态 sequence-length protocol。

## 文件与测试计划

- `executable_plan.{h,cc}`：声明并验证 state/alias/write/extent contract。
- `memory_plan.cc`：复制新增 metadata，state/alias 不参与 reuse。
- `session.{cc,internal/session_node.h}` 与 `ValueTable`：分配、绑定、alias、alignment 和
  completion gate。
- `executable_plan_test.cpp`：合法及非法 alias、state、extent 合同及 memory-plan 保真。
- `runtime_session_test.cpp`：CPU state 跨执行保存、alias storage、dtype/shape/device 失败、
  连续 `RunAsync` 的顺序和 completion 保活。
- `codegen_llvm_test.cpp`：真实 CPU/LLVM kernel 的 state alias 重复执行正例和构造期负例。
- `compiler_identity_test.cpp`：state/alias/extent 必须改变 plan ABI fingerprint；旧 control
  runtime 的 pure-fresh effect model 对这些新语义 fail closed。

## Ponytail full QA 记录

1. **A — YAGNI/复用：**复用 `ValueSpec`、`NDArray`、`Storage`、每-run `ValueTable` 与
   `AsyncOperation`；只增加 state map、按 storage 的 alignment map 和 `ValueTable::Alias`，
   没有新 cache/state hierarchy、factory 或依赖。删除了“state registry/handle”设想。
2. **B — 架构/identity：**state 仍是 plan-local stable value id，Storage 仍是物理 id；
   RuntimeSession 不 include Compiler/Relay，不做 variant/cache 选择。state list 让 plan identity
   显式，旧 source/output ordering 未变。
3. **C — 死代码/并发/fail-closed：**新增字段在 plan validation、memory-plan、session
   allocation/binding 和 tests 全部消费；不保留第二张 state table。未 ready state completion
   拒绝重叠，非法 alias/extent/shape/dtype/device 在 launch 前拒绝；已提交 work 由 completion
   和 session-held state 保活。完整 KV/paged/distributed 留为非目标。
4. **D — 端到端复核修订：**前三轮未明确覆盖 alias source 必须进入 producer ABI、memory
   planner 必须保留 alias slot，以及 state/donation 的最大 alignment。补上这三项后才足够；
   用 fake CPU 契约测试覆盖失败前无 launch，并用真实 LLVM 重复执行覆盖实际同址写入。
