# M6 技术报告：真实 kernel 换代、执行证据与回滚

日期：2026-09-09。范围：显式准备的静态无状态 CPU/LLVM 图，`KXC_ENABLE_ADAPTIVE_HOT_SWAP=ON`。状态：本模块切片已完成，默认、adaptive 与 bounded 全量回归通过。

## 达到的效果

完整八层 MiniMind 静态 prefill（B=1、S=16）已通过两代真实执行：每代 **650 次 LLVM kernel 调用**，显式替换第 649 个 matmul primitive，使用保持该图拓扑的 opt 0/1 编译配置。logits 与 16 个 K/V 输出在两代之间逐位一致，与独立 ONNX 参考的最大绝对误差为 **8.82149e-6**。模型 export receipt、prefill stage、generation、route、ABI 与 validation receipt 在 1,300 对 submit/exec 事件中关联完整，真实 Bundle 可被一次性健康检查消费。

小图同时证明发布期间旧请求继续执行、completion 保活、坏候选隔离与回滚。运行入口拒绝错误身份和伪造 metadata；带会话状态的图在替换准备前明确拒绝。完整模型与小图的每次 Run 均检查 primitive cache 的全部统计保持不变。

## 使用的方法

### 从实际执行的 lease 记录代际

复用 `CompileAndPublish → Acquire → GenerationLease → RuntimeSession`。候选准备仍调用生产 `CompilePrimitiveUnits` 和 `AssembleCompiledGraph`，只重编请求列出的 primitive。执行入口不编译、不查询 primitive cache。

`AdaptiveHotSwapController::RunAsync` 新增接收 `ExecutionMetadata` 的重载，调用方传模型 `export_receipt` 和 `stage`。controller 取得 lease 后填写 `adaptive_contract`、`generation`、`dispatch_key`、`plan_abi`、`plan_variant` 和 `validation_receipt`，然后通过 M1 的既有通道传给 session。六个字段来自实际执行的 lease；调用方尝试覆盖会在启动前报错。旧的三参数入口委托到同一路径。

这些字段随同一 `run_id` 进入 run、kernel submit/exec、分配和复制事件。它们是观测信息，不参与数学语义、调度或 kernel ABI。controller 合同从 v3 升至 **v4**。

### 用完成后的 Bundle 做一次性健康检查

测试的 `BundleHealth` 实现已有 `HealthAuthority`，使用项目已依赖的 LLVM JSON parser 读取实际 `events.jsonl`。它核对成功完成的 run、每个 kernel 的 submit/exec、调用数，以及 lease 对应的 route、ABI、variant 和 receipt；以 `trace_id:run_id` 作为证据与 replay token。

测试显式配置固定时长阈值：最大值允许候选，零阈值隔离一次实际耗时大于零的候选。这个策略用于证明证据能够驱动控制面，不是默认健康策略，也不证明候选更快。controller 在消费之前检查 generation、非空证据/token、合法 disposition，以及 lease 仍是当前 route head。验证与消费仍在原有 route/health 锁内完成。

缺失、重复、过时和回调异常均不能切换 route。隔离后回滚到仍有效的前代，保留 tombstone，禁止同一 selection 重新发布。health、quarantine 和 rollback 的 observer 事件现在均带 route/ABI 摘要。

### 证明发布期间旧请求继续执行

小图为 `(a+b)*b`，两代使用实际声明的 LLVM 优化配置 opt 1/2，替换第二个 primitive。测试检查 artifact key 确实变化，而 route 和物理 ABI 保持一致。

另一个测试在第一个真实 LLVM primitive 执行后，用测试专用 host gate 暂停旧请求。此时发布新代并执行新请求，再释放旧请求。discoverable generation 上限为 1，强制旧代从路由历史淘汰；弱引用证明运行期间仍保活，清除显式 lease 和 outputs 后 completion 仍保活，最后释放 completion 后旧代才销毁。

gate 的 launcher 委托真实 LLVM，保留原 unit semantic key，使用明确的测试 instrumentation 身份；实际替换的是另一个真实 opt 1/2 primitive。这是 CPU 主机请求跨发布点的证据。CPU 路径在 `RunAsync` 返回前完成，不据此宣称 CUDA 异步设备完成已验证。

并发检查还发现并修正了 observer 限制过宽的问题：原实现用全局 callback 计数拒绝其他线程的正常 `Acquire`。现在用线程局部调用栈检测同一 controller 的递归进入。测试让发布 observer 暂停，确认其他线程仍可 Acquire，而 callback 内递归调用被拒绝。

### 沿用状态 owner，明确替换边界

准备阶段现在只接收 `kStatic` 且没有 state value 的图，并先检查 ordered calls 与 artifact pins 一一对应。带状态或动态模式在 candidate/session 准备之前拒绝；preparation 合同从 v1 升至 **v2**。测试把真实编译的 past/present 图通过 M2 的 `BindStateOutputs` 变成状态计划，验证 adaptive admission 拒绝且不编译新 primitive。

M2 的状态仍归 `ExecutablePlan/RuntimeSession` 所有。controller 没有 KV registry、状态迁移或跨会话共享缓存接口。输入长度变化需要选择匹配的已准备 route，或者使用原产物已验证的 bounded 合同；不能把不同物理 ABI 的 kernel 强行发布进同一 route。

### 控制模型 identity 的复制成本

完整 MiniMind 的 route/ABI canonical 数据合计 **828,297,404 字节**，包含原有身份合同中的模型常量。默认小图路由预算会明确拒绝；测试为这一条锁定模型 route 配置实际字节数加 framing 空间，并设 2 GiB 的夹具上限。

模型测试还暴露了 request/lease 复制这些大字符串的成本。`DispatchKey`、`PlanAbiFingerprint` 和 `PlanVariantKey` 改为共享不可变 canonical 字节，复用 `GraphSemanticKey/ShapeProfileKey` 已有做法。独立构造的 key 仍逐字节比较，digest 仍只是摘要；canonical 格式、artifact identity 与计划 ABI 版本不变。公共 C++ handle 的内部布局有变化，使用方需要重新构建。

共享复制的检查通过，完整两代模型也能完成。但控制面仍会拼接和复制 canonical route/flight 字符串。此次完整专项测试耗时 **69.54 秒**，GNU time 记录的最大 RSS 为 **12,496,812 KiB**；这包含导入、准备、编译、两代执行、JSON 校验和小图测试，不能解释为模型推理延迟或性能提升。大模型控制面的内存和准备成本仍是明确限制。

## 验证与达到的效果

执行测试位于 [adaptive_runtime_test.cpp](../../test/adaptive_runtime_test.cpp)，原准备、singleflight、取消与 primitive replacement 证据继续保留在 [adaptive_preparation_test.cpp](../../test/adaptive_preparation_test.cpp)。共享 identity 的复制、移动、独立构造比较与生命周期检查加入 [compiler_identity_test.cpp](../../test/compiler_identity_test.cpp)。

| 验证 | 已取得的结果 |
|---|---|
| 两个实际 LLVM 静态候选 | 小图 opt 1/2 的第二个 primitive 不同，输出均为 `[12,24]`，route/ABI 相同 |
| 完整 MiniMind | opt 0/1，第 649 个 matmul 替换；两代共 1,300 次调用、17 输出逐位一致，ONNX 最大误差 8.82149e-6 |
| 生命周期 | 旧请求在 host gate 暂停期间完成新代发布与运行；eviction 后旧 lease 由 completion 保活，最终引用释放后销毁 |
| 健康与回滚 | 成功 run 的证据仅消费一次；缺失、空证据、错误 generation、旧 head 和异常均拒绝；quarantine 后回滚并拒绝同 selection 重新发布 |
| 身份拒绝 | 实际不同长度图产生错误 route/ABI，Acquire/Run 均拒绝；空/错配 validation receipt 不产生可执行 lease；六个保留 metadata 字段不可覆盖 |
| observer | 同 callback 线程递归进入拒绝，其他线程可并发 Acquire |
| 状态边界 | 真实 LLVM past/present 图绑定 M2 state 后在 adaptive admission 拒绝，不编译替换 kernel |
| profile 关闭 | 小图数值与 artifact identity 不变；开启时 run/kernel 的真实 lease 和模型字段完整 |
| 共享 identity | 128 份 route/variant 复制共享字节，移动与原 handle 释放后仍有效，独立构造相同键仍相等，不同键仍不等 |

| 回归与审计 | 结果 |
|---|---|
| 默认 CPU/LLVM | 48/48 CTest 通过，42.18 秒；adaptive、bounded、control gates 关闭 |
| adaptive CPU/LLVM | 48/48 CTest 通过，42.42 秒；包含原准备/singleflight/取消测试和新运行测试；完整 MiniMind 另行显式启用 fixture，通过 69.54 秒专项 |
| bounded CPU/LLVM | 62/62 CTest 通过，250.33 秒；显式启用 projection、heads、RoPE、GQA、attention、完整 prefill/decode 七组模型 fixture，以及原静态容量 decode-loop fixture |
| Python | 280/280 通过，1.18 秒 |
| 契约与架构 | Relay 37/37、pass 20/20 与生成物 freshness 通过；include 279、public headers 89+10 编译、docs 55 篇、NLP checker 与 diff check 通过 |
| 静态库 | 默认/adaptive/bounded 分别 1,567/1,679/1,847 个强全局符号，无重复定义 |

模型两代的 route digest 均为 `b42dc30a8e7ee40`，plan ABI digest 均为 `4a3d8b9ecfa673d9`；generation 为 1/2，selection digest 分别为 `774c042e564313ec` / `679dc70d8838ca0e`。导出 receipt 为 `sha256:minimind_prefill_static.onnx:eb1c60c73980b0e56693876906e4e6d1a861b2609bd40469e2637d09b0eef455`。这些摘要用于关联本次证据，controller 的身份相等性仍依靠完整 canonical bytes。

## 三轮 Ponytail QA

1. **复用**：复用 production compiler、lease/completion、runtime metadata、Profile Bundle 和 LLVM JSON parser；没有新执行器、JSON parser 或默认调优器。
2. **权威与版本**：route、ABI 和 selection 均来自既有 builders；metadata 不生成第二套 identity；状态拒绝由现有 preparation 执行。只升级 adaptive/preparation 合同，共享存储保持 canonical 语义。
3. **实际消费与反例**：实际 LLVM 输出、跨发布运行、完成证据、健康消费与回滚贯通；反例覆盖错误 route/ABI/receipt、metadata 覆盖、过时代际、证据重放、callback 异常、tombstone 和状态计划。无新增生产策略死接口。

## 复现

```bash
cmake -S . -B out/build/adaptive-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON -DLLVM_DIR=/usr/lib/llvm-20/cmake
cmake --build out/build/adaptive-llvm -j2
KXC_MINIMIND_ADAPTIVE_PREFILL_DIR="$PWD/out/fx_decode_stateful" \
  ctest --test-dir out/build/adaptive-llvm --output-on-failure --no-tests=error
```

模型夹具复用 [M2 静态模型报告](M2_MINIMIND_STATE_REPORT.md) 的 decode-loop 生成物，包含 `prefill.json/.params`、input ids、ONNX logits/KV 参考和 export receipt。未设置环境变量时只跳过完整模型，小图与生命周期测试仍必须执行。模型是固定随机权重的八层 MiniMind，验证编译和执行等价性，不验证语言质量。

本模块不宣称自动候选搜索、性能改善、KV 状态迁移、任意动态 shape 热替换、GPU 完成事件或多机服务已经支持。

2026-09-10 后续：静态容量状态的请求会话与步间热替换已有完整模型证据，控制面身份容器也已减少 payload 复制，见 [状态热替换报告](M6_STATEFUL_REPORT.md)。本文保留原静态无状态切片的测量与边界。
