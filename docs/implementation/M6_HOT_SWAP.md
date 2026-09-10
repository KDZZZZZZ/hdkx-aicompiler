# M6：证明热替换真的发生且安全

M6 已把实际 MiniMind 静态 prefill 与 candidate generation 关联：两代各执行 650 个 LLVM kernel，17 个输出逐位一致，执行 Bundle 能被一次性健康检查消费。小图验证了旧请求跨发布继续执行、completion 保活、隔离和回滚。方法、数据和限制见 [技术报告](M6_RUNTIME_REPORT.md)。

后续已接通静态容量状态热替换：每个请求持有独立 RuntimeSession，代际切换只更换本步执行模块，KV 不迁移；完整八层模型跨代际 1→2→1→1 保持 16 份缓存地址与有效长度连续，见 [状态热替换报告](M6_STATEFUL_REPORT.md)。有界 CPU 无状态/会话状态随后也已通过小图与完整八层模型验证，使用 adapter 授权的相同输入 profile 与状态 ABI，见 [有界报告](M6_BOUNDED_REPORT.md)。请求批处理热替换也已有完整八层 CPU 模型证据：原队列和 KV 跨五个批次保留，代际 1→2→2→1→1，含排队后发布、退出复用及排队后回滚，见 [请求报告](M6_REQUEST_BATCHING_REPORT.md)。静态 CUDA 热替换已在真实 RTX 4070 Ti SUPER 上通过 CUPTI、health/rollback 和错误设备拒绝，见 [CUDA 报告](M6_CUDA_REPORT.md)；bounded KV/请求 CUDA 换代与 MiniMind-O 多流仍待完成。

> 状态：S1/S2 静态 CPU/LLVM 执行证据已完成；S3 已扩展静态容量、有界状态和请求批处理的真实模型换代（2026-09-10）。最新默认、adaptive、bounded 与组合配置全量回归分别通过 55/55、56/56、70/70、75/75，公开状态绑定入口及集成审计均通过。方法与效果记录在 [静态报告](M6_RUNTIME_REPORT.md)、[状态报告](M6_STATEFUL_REPORT.md)、[有界报告](M6_BOUNDED_REPORT.md)和[请求报告](M6_REQUEST_BATCHING_REPORT.md)。原始安排见 [WAVE_2](WAVE_2.md)。

## 已验证的行为

- `Submit/CompileAndPublish` 可以显式准备并发布 candidate；`Acquire` 只查询已发布 route，不在运行时编译。
- `RunAsync` 已经通过 lease 保持 candidate/session，并把 lease 放进 completion 的 retention。
- generation、route、health、quarantine、rollback 事件已有控制面表示。
- `adaptive_runtime_test` 验证真实执行顺序、输出等价、旧 generation 生命周期和错误/回滚；模型 opt 0/1 保持拓扑，小图 opt 1/2 产生不同的实际 artifact。
- `RunAsync` 从取得的 lease 附加保留 metadata，调用方传模型 receipt/stage；真实完成 Bundle 驱动显式一次性 health。没有默认自动健康策略。

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

采用请求拥有状态、每步拥有 lease 的合同：`CreateStatefulSession` 在取得已验证路由后分配请求会话；`RunAsync` 按该会话的固定 DispatchKey/Plan ABI 获取本步代际。已经提交的步骤持有旧 lease 完成，下一步使用新代际或回滚代际。容量、绑定、填充值、状态复制与长度提交仍由同一 ExecutablePlan/RuntimeSession 负责，controller 不创建 state registry。

静态容量和相同有界 profile 路径已有独立小图与完整模型证据。不同容量、布局或 Plan ABI 不做迁移；提交后失败的会话不能靠切回旧模块恢复。有界请求由 adapter 铸造，范围与逐个调用连接在编译前核验。请求批处理沿同一 producer/profile 合同接入：`StatefulSession` 可拥有整组请求，`RunNextBatch` 为本批固定 lease，经原 RuntimeSession 完成合批和状态提交。已经执行的批次保持旧代际，排队请求在批次开始时选择当前代际。

公开 `CompiledGraph::BindStateOutputs/BindBoundedStateOutputs/BindRequestBatching` 复用原 ExecutablePlan 校验并保留编译产物的模块、pin 和图语义；完整模型 consumer 已通过这些入口准备状态，无需内部重组 API。

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

- [x] 至少两个已编译静态候选真实执行，route head 和 generation 与输出对应。
- [x] 发布期间旧 lease、module、storage 和 completion 安全保活；新请求不会取到已 quarantine/tombstone 的候选。延迟测试使用真实 LLVM 加 host gate，未声明 GPU 完成证据。
- [x] route、ABI、validation receipt 不匹配时零 launch；不存在默认 candidate 或 CPU fallback。
- [x] health/rollback 只消费一次，过时或缺失证据不改变 route。
- [x] profiling 能把运行事件连到 generation，且 profiling 开关不影响数值和调度。
- [x] 静态、有界与请求批处理状态由 M2 的同一 RuntimeSession 持有，匹配 profile/状态 ABI 时可换代；不同容量、布局或 ABI 的迁移明确拒绝。

identity 变化沿用现有 builders。替换代际号是生命周期/观测信息，不代替 PlanAbiFingerprint，也不因每次 route 切换生成新的数学语义 key。
