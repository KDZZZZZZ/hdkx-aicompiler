---
name: kxc-runtime-state-mounting
description: Attach persistent or in-place runtime capabilities to KXC ExecutablePlan and RuntimeSession. Use for KV cache, parameter or optimizer state, input donation, aliasing, valid extents, repeated Run/RunAsync, asynchronous lifetime, storage planning, or stateful training/decode execution.
compatibility: KXC repository with ValueSpec state/alias/write-mode contracts.
---

# KXC Runtime State 挂载

先完整读取 [总则](../kxc-capability-mounting/SKILL.md)。并发/atomic 语义复杂时同时读取 `memory-model`。

## 现有 owner

```text
include/kxc/runtime/executable_plan.h
src/runtime/executable_plan.cc                 value/state/alias 合同与顺序证明
src/runtime/memory_plan.cc                     storage slot/lifetime
src/runtime/session.cc                         唯一生产 consumer
src/runtime/internal/value_table.h             每次执行的 binding/alias table
src/compiler/identity/experimental_identity.cc plan ABI/variant identity
AsyncOperation / Storage / NDArray             异步保活与物理内存
```

`RuntimeSession` 独占 persistent state storage；每次 run 仍使用局部 `ValueTable`。不要建立模型名到 state 的第二 registry。

## 表达现有状态能力

- Persistent source：`is_state=true` 并列入 `state_value_ids`；无 producer，但必须被 call 消费。
- 原地写：producer 输出声明 `alias_source_value_id`、`is_alias=true`、`ValueWriteMode::kInPlace`。
- Donation：普通 input 可作为显式 alias source；未声明就不能复用。
- Alias source/target 必须同 shape、dtype、device、storage contract，producer 必须消费 source，source 写后不能再活跃。
- Constant 和 graph output 不能被 donation。
- `valid_bytes` 当前是静态容量内 contract，不是动态 sequence-length protocol。

KV cache、参数更新和 optimizer state 应组合这些通用语义，而不是增加 `kv_cache_id`、`parameter_name` 或 training-only session。

## 扩展步骤

1. **先分类**：新需求是 persistent ownership、in-place write、动态有效区、初始化/加载，还是跨-run ordering？不要用一个 bool 混合多种语义。
2. **扩展 `ValueSpec/ExecutablePlan` owner**：字段必须有严格默认值，旧无状态 plan 保持兼容。
3. **在 plan validation 证明**：角色互斥、producer/use order、storage equality、lifetime、capacity。
4. **更新 memory planner**：state/alias 不得错误 reuse；改变 slot 结果时更新 `kStaticMemoryPlanVersion`。
5. **让 `RuntimeSession` 消费**：构造期分配/验证，run 时绑定，launch 前 preflight，completion 保活。
6. **更新 identity**：可调用 ABI 变化进入 `PlanAbiFingerprint`；variant storage 变化进入 memory-plan version。
7. **让其他 runtime 明确选择**：pure-fresh control runtime 不支持时继续 fail closed，不要偷偷忽略字段。

## 异步与并发

- 无 state plan 保持既有并发。
- 有 state plan 必须在一个 session 内定义提交顺序。当前 pending completion 会 fail closed；不要假定不同 stream 自动依赖。
- `RunAsync` 返回的 completion 和 session-held completion 都必须保活所有参与 Storage。
- 下一次 run 只有在前一次 state 写可见后才能读取。
- 需要真正排队而非 fail-closed 时，应显式设计依赖链/stream event，并增加 data-race 与析构后保活测试；不要只扩大 mutex 临界区掩盖问题。

## 新动态有效区

若 KV cursor/valid extent 需要跨 run 动态变化：

- 不要把动态值塞进当前静态 `valid_bytes`。
- 添加通用、显式、可版本化的 runtime metadata/value，并定义谁写、谁读、如何进入 kernel ABI 和 plan identity。
- 必须验证不超过 physical capacity，且失败发生在 launch 前。
- shape/profile 路由仍由 shape control owner；RuntimeSession 不选择或编译 profile。

## 测试

- Plan：合法 state/alias chain；非法角色、storage、write-after-live、capacity。
- Memory：metadata 保真、slot 不被错误复用、version identity。
- Session：首次初始化、重复 run 持久性、独立 session 隔离、input donation alignment、pending async、析构后 completion。
- LLVM：真实 kernel 同址更新与数值累积。
- Identity：state/alias/extent 改动必须改变 plan ABI。
- Negative：所有 contract mismatch 在 backend launch 前，launcher call count 为 0。

```bash
ctest --test-dir <build> --output-on-failure \
  -R 'executable_plan_test|runtime_session_test|codegen_llvm_test|compiler_identity_test|control_runtime_integration_test'
```

## 禁止

- 第二张 session state table 或裸 state pointer API。
- 未声明 alias 的隐式 in-place。
- 把 module constant 当可变参数。
- 用 CPU 同步行为推断 CUDA completion 安全。
- 在本能力中顺带实现 distributed coherence；除非用户明确重启该范围。

完成标准：plan 声明、验证、memory plan、session consumer、completion、identity 和真实 LLVM 重复执行全部闭合。
