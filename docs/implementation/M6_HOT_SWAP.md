# M6：证明热替换真的发生且安全

第一波已经完成执行侧 profiling，但还没有把一次真实 MiniMind 运行与 candidate generation 关联起来。当前 adaptive 控制面能显式准备、发布、获取候选并维持 generation lease；它不等于 RuntimeSession 已在发布前后安全换代。MiniMind-L1 的 KV cache 又属于 session state，不能在请求中间凭 route head 直接迁移；MiniMind-O 的 Thinker/Talker 多流和实时服务更不能由无状态 fixture 代替。

本模块先用无状态的 MiniMind L1a 静态 prefill 或同等 Where 图证明两个离线编译候选的真实切换，再把 M1 bundle 的导出 receipt、stage、generation 和 plan ABI 关联起来。等 M2 明确 KV 所有权后，再决定带 state 的替换边界；没有迁移合同就明确拒绝中途替换。

> 状态：待实施，依赖 M1 已完成的执行证据、M9 的 L1a receipt；带 KV 的部分依赖 M2。当前安排见 [WAVE_2](WAVE_2.md)，控制面见 adaptive_hot_swap.h。

## 当前已有和缺少什么

- `Submit/CompileAndPublish` 可以显式准备并发布 candidate；`Acquire` 只查询已发布 route，不在运行时编译。
- `RunAsync` 已经通过 lease 保持 candidate/session，并把 lease 放进 completion 的 retention。
- generation、route、health、quarantine、rollback 事件已有控制面表示。
- 缺少真实执行顺序、输出等价、旧 generation 的生命周期和错误/回滚行为的端到端测试。
- 第一波 profiling 已能记录某次运行和 kernel/plan 事件，但还没有把 MiniMind 的 export receipt、stage、generation/plan ABI 稳定关联；在此之前 health 决策只能保持显式、一次性的外部输入。

## S1：两个静态候选的真实切换

选一个现有 CPU/LLVM elementwise 或 `Where` 小图，显式生成 baseline 和 candidate。若当前 compiler 没有两个实际不同但都受支持的实现，第一版使用两个不同的已声明 pipeline/优化配置，并确认它们会产生不同 artifact identity；不能伪造一个字符串差异。

步骤：

1. 在准备阶段完成各自的 Relay/shape/backend validator，得到匹配的 DispatchKey、PlanVariantKey 和 PlanAbiFingerprint。
2. 通过现有 `CompileAndPublish` 显式发布 baseline；运行请求使用 `Acquire` 后的 lease 执行一次并保存输出、generation 和 profile 关联。
3. 显式编译并发布 candidate，验证 route head 改变；新请求只能得到 candidate，旧 lease 仍可完成。
4. 让一个延迟 completion 请求跨过发布点，检查它仍然使用旧 session，旧 generation 在 completion 销毁前不能被 eviction/释放。
5. 对相同输入比较 baseline/candidate 输出、shape、dtype、调用次数和 ABI；对不匹配的 route/ABI 检查 Acquire 在 launch 前报错。
6. 运行 health evaluation。健康结果只在 lease 仍是 route head、receipt/route/ABI 都匹配时被消费；quarantine/rollback 后新请求回到有效 generation，旧请求仍按 lease 完成。

这里的替换是控制面的显式动作，不是 RuntimeSession 根据热度自动编译。失败、取消、backpressure 和 tombstone 都保留现有分类，不得变成静默回退。

## S2：profiling 参与决策但不改变执行

M1 产生的 bundle 需要包含 generation、DispatchKey/ABI 摘要、candidate route 和执行结果状态。HealthAuthority 读取这些证据后只能作明确的健康/隔离结论；它不读裸对象地址、不猜 kernel 名，也不在回调中重新进入 controller。

先做 one-shot health：给定一个已完成 run 和固定阈值，重复评价得到同一结论。再测试健康回调异常、证据缺失、generation 已非 route head 的处理，均应 fail closed。自动候选搜索和自动调优不在本模块范围。

## S3：带状态的换代边界

等 M2 的状态生命周期稳定后，决定“一个 generation 是否拥有一份独立 KV cache”。首选策略是：进行中的请求继续使用旧 lease/session，新请求只使用新 generation；不在请求中间迁移 state。若产品确实需要迁移，必须在 ExecutablePlan/RuntimeSession owner 中定义容量、布局、同步、失败恢复和 identity，不能在 hot-swap controller 创建 state registry。

带 state 的 route 不能用静态无状态测试代替。没有迁移合同时，对该请求类型明确拒绝替换，保留 fail-closed。

## 代码与测试

| 位置 | 工作 |
|---|---|
| [adaptive_hot_swap.cc](../../src/compiler/adaptive/adaptive_hot_swap.cc)、[preparation](../../src/compiler/adaptive/adaptive_hot_swap_preparation.cc) | 复用现有 publish/acquire/lease/health；只补行为缺口 |
| [adaptive_hot_swap.h](../../include/kxc/compiler/adaptive_hot_swap.h) | 只有真实新字段才扩公共合同；事件版本与 identity 一致 |
| [runtime/session.cc](../../src/runtime/session.cc)、[device_stream.h](../../include/kxc/runtime/device_stream.h) | 确认 completion 保活和旧 session 生命周期，不让 controller 直接操纵内存 |
| `test/adaptive_preparation_test.cpp`及拟新增 `test/adaptive_runtime_test.cpp` | 准备与执行证据分别保留；新增目标 `adaptive_runtime_test` 验证发布前后执行、lease、健康、回滚、取消和无效 route |
| M1 bundle/diagnostic 测试 | generation/run/span 关联和错误证据 |

测试需要在 `KXC_ENABLE_ADAPTIVE_HOT_SWAP=ON`、CPU/LLVM 可用的配置中运行；默认关闭构建仍应保持原编译路径。不要把只检查 EventKind 的测试称为真实替换证据。

```bash
cmake -S . -B out/build/adaptive-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON
cmake --build out/build/adaptive-llvm -j2
ctest --test-dir out/build/adaptive-llvm --output-on-failure --no-tests=error \
  -R 'adaptive_hot_swap_test|adaptive_runtime_test|profile_bundle_test|runtime_session_test|codegen_llvm_test|compiler_identity_test'
```

## 完成条件

- [ ] 至少两个已编译静态候选真实执行，route head 和 generation 与输出对应。
- [ ] 发布期间旧 lease、module、storage 和 completion 安全保活；新请求不会取到已 quarantine/tombstone 的候选。
- [ ] route、ABI、validation receipt 不匹配时零 launch；不存在默认 candidate 或 CPU fallback。
- [ ] health/rollback 只消费一次，过时或缺失证据不改变 route。
- [ ] profiling 能把运行事件连到 generation，且 profiling 开关不影响数值和调度。
- [ ] 带 state 的替换边界有明确拒绝或 M2 定义的所有权，不能用无状态结果代替。

identity 变化沿用现有 builders。替换代际号是生命周期/观测信息，不代替 PlanAbiFingerprint，也不因每次 route 切换生成新的数学语义 key。
