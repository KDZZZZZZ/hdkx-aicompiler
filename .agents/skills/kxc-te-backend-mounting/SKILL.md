---
name: kxc-te-backend-mounting
description: Extend KXC TE compute, Schedule/Stage primitives, TE-to-TIR lowering, target-aware default scheduling, TIR proof, LLVM or CUDA codegen. Use for split/reorder/vectorize/unroll/parallel extensions, reductions, indirect loads, CUDA kernels, launch metadata, or backend numerical support.
compatibility: KXC repository with explicit Schedule + Target TE-to-TIR lowering.
---

# KXC TE / TIR / Backend 挂载

先完整读取 [总则](../kxc-capability-mounting/SKILL.md)。如果任务同时新增 Relay op，再读取 `kxc-relay-operator-mounting`。

## 边界与 owner

```text
include/kxc/te/te.h, src/te/te.cc                    TE DAG 与 Schedule relation
src/compiler/internal/te_to_tir.h
src/compiler/lowering/te_to_tir.cc                   唯一 schedule materializer
src/compiler/lowering/lowered_graph.cc               Target-aware default schedule caller
src/compiler/primitive/primitive_compiler.cc         pipeline、artifact key、backend compile
src/tir/transforms/bind_cuda_threads.cc               CUDA thread/launch 唯一 authority
src/codegen/llvm/**, src/codegen/cuda/**              backend emission
KernelSignature / KernelLaunchMetadata                callable/launch ABI
```

`te::Schedule` 必须显式传入 TE→TIR。`kxc.te.schedule_contract` 是实际 schedule identity，并进入 `PrimitiveArtifactKey`；不能退回常量版本字符串。

## 三类改动

### 1. 新 TE compute

- TE body 只表达纯 tensor 计算和既有 reduction。
- shape/dtype 与 Relay InferType 完全一致。
- Producer DAG 必须能被 `create_schedule` 收集。
- 不在 compute 中读 Target、绑定 thread 或分配 runtime storage。

### 2. 新 schedule primitive

只有同时完成以下内容才可公开 API：

1. `Stage` 中最小、可验证的 relation/annotation。
2. 调用时校验 current leaf、domain、重复应用和 reduction 安全性。
3. `BuildStageAxisPlan`/lowering 真正物化为 TIR。
4. canonical schedule 编码 relation、leaf order 和执行类型。
5. TIR 结构变化、identity 区分、LLVM 数值、非法输入负例。

没有 lowering consumer 的 primitive 直接删除/不添加。不要恢复假的 `fuse/tile/bind/thread_axis`；若真实需求出现，再按上面五项实现。

Canonical schedule 使用 stage/axis ordinal 和结构，不使用 tensor/op/axis 的 graph-local 名字，否则会破坏跨图 artifact reuse。

### 3. 新 backend 能力

- 从现有 TIR/`KernelSignature` 构建 ABI，不从 op 名猜参数。
- LLVM 与 CUDA 对同一语义保持数值一致；backend 差异进入 target/backend identity。
- launch metadata 只由 typed backend/TIR accessor 产生。
- unsupported dtype、rank、reduction、indirect access 在 codegen/launch 前拒绝。

## Target 策略

- CPU default policy 可安全选择 split/vectorize/unroll/parallel，但必须保持静态精确语义。
- CUDA TE 默认保持 serial。
- `BindCudaThreads(PrimFunc, Target)` 是唯一 CUDA independence proof、`ThreadBinding` 和 grid/block owner。
- CUDA reduction/indirect load 需要扩展既有 TIR proof/变换或明确 library/custom-call 路径；不要在 TE 新造绑定机制。
- Target 决策使用显式 immutable `Target`，不只依赖 ambient `PassContext`。

## Reduction / tail 安全

- split 采用静态正 factor；非整除必须有 tail predicate。
- data leaf 必须保持在 reduction leaf 之前，除非 lowering 结构也被正确扩展。
- parallel/vectorize 只能用于可证明独立的数据轴。
- reduction init 在 data domain 一次，update 在 reduction domain 执行。
- 新 indirect load 必须证明 index dtype/range及 backend lowering，不能把越界责任推给用户。

## 不被 CUDA 环境阻塞

```text
无 nvcc       -> TE/TIR + synthetic CUDA Target + fail-closed tests
有 nvcc无 GPU -> source emission/compile + launch metadata tests
有 GPU        -> 再增加真实 launch 数值测试
```

任何层级都不得 CPU fallback 冒充 CUDA 成功。NVIDIA 专有 API 确实超出现有本地指导，且当前环境提供 `nvidia-live-skill-lookup` 时才读取它；未经批准不安装外部 skill/dependency。

## 测试

- TIR：loop 顺序、ForType、predicate、ThreadBinding/launch attr。
- Identity：不同真实 schedule 不得相同；仅 graph-local 名字不同应相同。
- LLVM：默认 schedule 与手工/tail schedule 数值。
- CUDA：synthetic policy/authority 正负例；硬件可用时真实 launch。
- Production：经 `Compiler::Compile` 的 artifact key 包含 schedule contract。

```bash
ctest --test-dir <build> --output-on-failure \
  -R 'te_schedule_test|cuda_schedule_test|codegen_llvm_test|op_numeric_llvm_test|operator_compilation_test'
```

完成标准：每个公开 schedule/backend 能力都改变真实生产 TIR 或 executable，identity 可区分，CPU/LLVM 正确，CUDA 不支持路径明确拒绝。
