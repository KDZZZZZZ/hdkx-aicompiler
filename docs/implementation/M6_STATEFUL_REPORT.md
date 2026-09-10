# M6：保留会话 KV 的热替换技术报告

2026-09-10。静态容量 CPU/LLVM 路径已完成真实小图、完整八层 MiniMind、三套 CPU 回归及集成审计。bounded、请求批处理及 CUDA 的热替换尚未验收。

## 达到的效果

同一请求现在能在 decode 步之间切换已验证的 kernel 代际，继续使用原 RuntimeSession 的 KV 存储与有效长度。完整 MiniMind 从真实 LLVM prefill 接收 16 份 K/V，四步 decode 使用代际 **1 → 2 → 1 → 1**；第二代由完成后的执行证据触发隔离并回滚。所有 KV 地址不变，有效长度 **16 → 20**。

完整模型共执行 **650 次 prefill + 2,732 次 decode LLVM kernel**，其中每步 683 次，替换 unit 682。全部 prefill logits/16 KV、每步 logits/16 份完整容量缓存与独立 ONNX 参考共比较 **1,012,736 个值**。最大绝对误差：prefill/KV **1.0252e-5**，decode logits **5.54323e-6**，低于 `1e-4` 容差。生成 token 由前一步实际 logits 的 greedy argmax 决定，依次为 `5756, 5756, 1576, 1576`。

这是完整架构、随机 float32 权重和固定 B=1/C=32 输入合同下的正确性证据，不是预训练质量或性能加速结论。

独立事件审计核对了全部 2,732 对 decode kernel 提交/完成事件，以及 **64 次 KV 追加复制、98,304 字节**。每次追加都发生在本步全部 kernel 完成之后、run 完成之前，并带有正确的模型、代际、路由、Plan ABI 与导出 receipt。初始候选上下文记录 16 次 KV 分配，替换候选上下文没有额外 KV 分配；篡改代际、复制数量、事件顺序、extent、kernel 数量、ABI 或初始分配的 7 项反例全部拒绝。

## 使用的方法

采用 [IREE invocation execution model](https://iree.dev/developers/design-docs/invocation-execution-model/)中“可复用代码与每个 context 的可变状态分离”的不变量。KXC 继续让 `ExecutablePlan/RuntimeSession` 唯一持有状态，controller 只选择已发布代际，没有建立另一张 KV 表。

### 每个请求保留自己的会话

`CreateStatefulSession` 先按 DispatchKey/Plan ABI 获取 lease，再在路由锁外创建 RuntimeSession。返回的不透明 `StatefulSession` 同时保存不可变执行身份与会话；复制句柄共享同一会话，分别创建的句柄具有独立状态。初始化、长度与诊断接口直接委托给 RuntimeSession。

每个 `RunAsync(stateful_session, ...)` 固定本步取得的 lease，调用 `RuntimeSession::RunAsyncWithModule`。原会话的计划、缓存、前缀与长度表继续参与执行；执行模块、观测器及 completion 保活使用本步候选。发布或回滚不复制、清空或迁移 KV。普通 `RunAsync(execution_request, ...)` 对状态计划明确拒绝，防止多个请求误用同一份代际会话。

### 准备阶段保留状态合同

静态候选仍由 `PrepareCompilerGraph → CompilePrimitiveUnits → AssembleCompiledGraph` 生成。组装后，使用现有 `CompiledGraphAccess` 挂回 baseline 的状态计划，严格验证原有调用边界、容量、填充值与 append 绑定。完整 Plan ABI 必须与请求一致。

新增 `RuntimeSession::Validate` 复用原有 module/plan 校验，不分配状态张量；CompiledGraph 构造与 candidate preparation 都消费该接口。有状态 candidate 的 `session()` 为空，实际请求创建时才分配 KV；无状态候选仍保留原共享会话行为。

`RunAsyncWithModule` 在分配或启动前复核模块与原计划的匹配，以及原会话的常量键和 alignment 映射；签名、launch metadata、invocation contract 的完整 canonical 内容必须相等。运行时不推断数学等价，也不选择或编译候选；语义身份及候选验证继续属于控制面。错误输入、未定义 stream、错误状态身份和容量超限均拒绝。

### 生命周期与失败

真实 LLVM 小图在第一条 kernel 后暂停，期间发布新代际并让另一个请求完成运行；暂停的步骤仍以旧 lease 完成，下一步才使用新代际。返回 completion 与会话持有的最后 completion 保活旧代际，下一步提交并释放外部句柄后旧 lease 可释放。

替换入口保留原默认模块，原模块仍随 RuntimeSession 存活。每个请求由调用方顺序编排；底层沿用既有 state mutex、pending-completion 校验和提交规则，没有新增状态执行队列。

故障注入在第一条真实 LLVM kernel 完成后令第二条启动失败：长度未提交，测试中的缓存保持原值，会话被标记为不可继续执行。再次选择原模块也不能清除该失败状态，重新初始化同样拒绝。代码回滚不会宣称修复已经失败的状态；需要显式创建新会话。

## 完整模型暴露的内存问题

第一次完整状态模型测试在 53.45 秒时被系统 OOM killer 终止。内核日志记录 `anon-rss=12689520 KiB`，这是被杀时的驻留量，不是已完成运行的峰值。失败记录保留在 `out/windows-gpu/logs/state-hot-swap-model-initial*.log`。

控制面的 flight、route、负缓存与 quarantine 容器原来会复制包含模型权重的大块 canonical 字节。此次修复使 flight key 共享不可变字节，route 保存已有 DispatchKey/PlanAbiFingerprint，隔离记录保存已有 PlanVariantKey。容器复制不再复制整份 payload；哈希只选择 bucket，相等性仍核对完整字节。原逻辑 metadata 字节预算保持不变。

修复后同一完整测试通过，GNU time 测得 **58.89 秒、峰值 RSS 10,429,824 KiB**。这些时间包含导入、编译、发布、推理、参考比较、健康检查和回滚，不能视为推理延迟。完整模型 route/ABI canonical 内容仍为 **832,516,104 字节**；首次构建 flight key 与身份生成仍有较高成本。

## 身份与三轮实现审查

1. A：复用 RuntimeSession、状态绑定、完整模块合同校验、generation lease、健康检查和现有编译组装链。`StatefulSession` 只持有已有对象，不拥有第二份状态索引。
2. B：adaptive 合同升至 **v5**、preparation 升至 **v3**，显式记录有状态候选不共享会话的新行为。状态候选的 selection 使用既有 static-external-stateful memory-plan version；现有静态状态 Plan ABI **v10** 已覆盖容量、绑定、填充值、设备和调用边界，未改变其格式。容器存储优化不改变身份字节或相等规则。
3. C：真实 LLVM 小图和完整模型消费新入口；验证两代/回滚、独立请求、旧步骤跨发布、完成保活、错误元数据、输入/stream/容量拒绝、状态 ABI 不匹配与提交后失败。Runtime 单测另验证替换模块的选择、默认模块保留、常量映射拒绝及最终执行句柄释放。

## 验证与复现

| 检查 | 结果 |
|---|---|
| 原 adaptive preparation/lifecycle、运行时与状态小图 | 定向 3/3，通过；最新新增状态 ABI 反例由全量回归再次执行 |
| 完整 MiniMind 状态换代 | 通过，1,012,736 个参考值；实际执行与健康证据均按代际关联 |
| 默认 CPU 全量与文档复测 | 55 项最终通过；首轮 54/55，修复新报告索引后 docs 1/1 通过 |
| adaptive / bounded CPU 全量 | 56/56（260.09 秒）/ 70/70（232.47 秒）；指定模型 fixture 均实际执行 |
| 生成契约与依赖层 | Relay 40/40、Pass 20/20、两份生成物 freshness、284 个文件的 include 检查通过 |
| 公共头文件、文档及诊断引擎 | 三套 CPU 的 public_headers 均通过；79 份 Markdown 检查通过；诊断 4/4 |
| 静态库 strong symbol | 默认/adaptive/bounded LLVM/bounded CUDA 分别 1,606/1,725/1,886/1,897 个，重复定义均为 0；新增两个运行时接口各有唯一实现 |
| CUDA 构建兼容性 | 五个相关测试目标及静态库重建通过；runtime_session_test 通过，5 项设备门禁明确 skip |
| CUDA 状态热替换 | 未执行，不计为通过 |

默认首轮耗时 146.79 秒，docs 复测 4.70 秒；不能把两次运行写成一次全量 55/55。CUDA 兼容构建的 adaptive 开关关闭，证明共享 runtime/compiler 改动可以构建，不证明 CUDA 热替换。静态库审计先显式重建交付库，再检查新增符号，避免只重链测试程序而读取旧 archive。

```bash
cmake --build out/build/adaptive-llvm -j2
KXC_MINIMIND_ADAPTIVE_STATE_DIR="$PWD/out/fx_minimind_cuda_state" \
OPENBLAS_NUM_THREADS=1 \
ctest --test-dir out/build/adaptive-llvm --output-on-failure --no-tests=error \
  -R 'adaptive_hot_swap_test|adaptive_runtime_test|adaptive_state_model_test|runtime_session_test'
```

`adaptive_state_model_test` 使用现有完整 CUDA-state fixture 的独立 ONNX 参考文件，但实际执行设备是 CPU/LLVM；fixture 未提供时返回 77，不替代模型验证。fixture 的 decode 导出身份为 `sha256:minimind_decode_capacity.onnx:d9be9abcb40946add9838c0da2d2e4c827abfe2b713413b747646f1d4545895a`。

本地证据以 `out/windows-gpu/state-hot-swap-source-manifest.json` 与 `state-hot-swap-source.tar.gz` 固定源码；`state-hot-swap-local-evidence.tar.gz` 保存三套 CPU 日志、三份模型 Profile Bundle、独立事件审计、初次 OOM 及修复后 GNU time 记录、CUDA 构建与集成检查。文件哈希和实际执行边界见 `out/windows-gpu/logs/state-hot-swap-progress-audit.json`。这份源码与证据没有上传 Windows 或推送 GitHub。

后续仍需把这条状态保留机制接到 bounded/profile 路由与请求批处理，并取得 CUDA 热替换的硬件证据。不同容量、布局或 Plan ABI 不允许通过本接口迁移；新 IR、跨机器 KV 一致性和模型性能优化没有被本报告声明完成。
