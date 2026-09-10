# M6：请求批处理热替换技术报告

2026-09-10，相同有界 profile/请求状态 ABI 下的 CPU/LLVM 请求批处理热替换已完成。真实小图、完整八层模型、四配置回归、公开状态绑定入口及集成审计均通过。CUDA 热替换设备执行仍待验收，总项目目标继续进行。

## 问题与采用的方法

已有请求批处理由 RuntimeSession 持有队列、物理槽位、KV 和每个请求的有效长度，但只能执行创建会话时的模块。已有有界热替换支持单个状态会话，preparation 仍拒绝请求批处理计划。因此发布新代际无法供正在排队的请求使用。

参考 [Triton 有状态批处理](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) 对序列状态归属的要求，以及 [IREE 执行模型](https://iree.dev/developers/design-docs/invocation-execution-model/) 对共享模块和 context 状态的区分，保留状态 owner，只在每个批次开始时固定一个已经发布的模块代际。这里采用的是状态与代码生命周期分离这个不变量，没有引入外部服务或调度器。

## 实现与语义边界

- `CompiledGraph::BindStateOutputs`、`BindBoundedStateOutputs`、`BindRequestBatching` 通过原 ExecutablePlan 绑定方法派生编译产物，保留模块、artifact pin 和图语义。完整模型的状态/批处理准备改用这些公开入口，调用方不再依赖内部重组 API。三个方法不编译、不查缓存、不分配会话状态。
- `RuntimeSession::RunNextBatchWithModule` 先复用 `ValidateExecutionModule`，核对完整签名、launch、invocation、常量映射与 storage 对齐，再进入原批处理循环。普通 `RunNextBatch` 仍使用初始模块；二者共用 `RunNextBatchImpl`。
- 输入打包和 graph run 的观测器来自本批实际模块。FIFO、同 P/同非 batch 形状分组、前缀收集、追加散写与长度提交不另造实现；状态、队列和请求编号仍由同一个 RuntimeSession 持有。
- adaptive `StatefulSession` 直接转发原有接纳、入队、释放、长度和 KV 副本接口；没有第二张请求或状态表。`AdaptiveHotSwapController::RunNextBatch` 查询已有 route/ABI，固定 lease，等待该批状态提交后返回请求结果和实际 lease。
- 排队时不绑定代际；开始执行的批次固定代际。因此发布或回滚影响尚未开始的批次，已经执行的批次继续使用原 lease。返回的输出行独立保活 storage，离开和槽位复用不影响旧输出。
- preparation 接受经过原计划验证的请求批处理声明，继续核验 adapter 权限、输入范围和逐个调用连接。selection 使用既有 `kRequestBatchingMemoryPlanVersion`，完整 Plan ABI 继续包含最大批量和物理状态合同。没有修改旧 Plan ABI 字节或引入新路由身份。
- adaptive 合同更新为 v7、preparation 为 v5。请求编号、队列和当前有效长度不进入编译缓存身份。不同最大批量、容量、布局或 profile 不作为同一个状态 owner 的可替换候选。
- 批次仍按既有 CPU/CUDA 同步提交合同返回；本轮数值验收目标是 CPU/LLVM。模块一旦在 kernel 提交后失败，会话沿原 poison 合同拒绝后续操作，不能靠代码回滚恢复状态。

## 定向验证

小图使用真实 concatenate、relu 与辅助输出，状态 extent 为第 2 轴，每步追加 2 个 token，3 个槽位、最大合批 2。指定的单个 kernel 换代，其他 artifact 保持相等；跨代际的所有结果与原始 LLVM 会话精确一致。

用例覆盖不同 P 的请求留队、发布后再执行、第二代健康隔离和回滚、退出取消排队步骤、新编号复用物理槽位、独立会话隔离、KV 副本和旧输出保活。最大批量、输入容量、调用入口、伪造代际 metadata 和不兼容模块均有拒绝检查；模块预检拒绝后，原队列仍能正确执行，且没有执行分配、复制、kernel 或 cache 活动。

另用真实 LLVM kernel 后的显式线程握手暂停旧批次，再发布新代际。旧批次保持原代际，独立会话可用新代际推进，同会话的并发操作明确拒绝；旧代际在执行及结果 lease 生命周期结束前保活，下一批沿同一 KV 继续执行。故障注入使替换模块的第二个 kernel 失败，第一个真实 kernel 已提交；随后即使传回旧模块，会话仍拒绝运行。

初次构建因测试中一个多变量 `auto` 声明混用了整数和 cache 统计类型而失败，拆分声明后重新构建及小图通过。首轮默认回归另有一项文档索引检查失败：新报告还未接入文档索引；链接补齐后重新执行回归。两次初始失败日志均保留，不计为成功结果。

## 完整模型与集成验收

完整八层 MiniMind 用例复用锁定 bounded decode 导出与固定随机 float32 权重，保留全部 774 个普通编译单元和 16 份 KV；每份 KV 的物理形状为 `[3,9,4,96]`，每批最多 2 个请求。实际执行 4 个请求、7 个请求步、5 个批次，代际为 1/2/2/1/1。初始 token/KV 来自既有 fixture；本项不重复执行 prefill，实际 prefill 到状态的交接已有 [有界状态热替换证据](M6_BOUNDED_REPORT.md)。这项数值证据不代表预训练模型质量。

| 批次 | 代际 | 请求 | 执行前 P | 执行后长度 |
|---|---:|---|---:|---|
| 1 | 1 | A、C；B 留队 | 4 | A、C 为 5 |
| 2 | 2 | 发布前已排队的 B | 0 | B 为 1 |
| 3 | 2 | A 退出后接纳的 D | 0 | D 为 1 |
| 4 | 1 | 回滚前已排队的 B、D；C 留队 | 1 | B、D 为 2 |
| 5 | 1 | C | 5 | C 为 6 |

初始四个请求步的全部 logits/KV 直接与独立 ONNX 参考比较，共 **99,328 个值**，最大绝对误差 **1.12504e-5**，低于 `5e-5`。全部七个请求步另与原始 LLVM 产物逐请求执行的结果比较，共 **179,968 个值**，逐位相等。后三个请求步使用独立 LLVM 对照，本项没有声称它们额外执行了 ONNX。下一步 token 来自实际批处理 logits 的 argmax，原始参考状态不参与被测会话的更新。

换代只替换 unit 773，其余 773 个 artifact key 保持相等。当前合批仍执行 **3,870 次 kernel**，独立逐请求执行为 **5,418 次**；这是保留已有合批效果的证据，不能归因为本次热替换带来的额外加速。

首次完整定向检查在新增公开绑定接口前通过，CTest **178.94 秒**、峰值 RSS **11,041,016 KiB**；补齐公开接口后的组合回归再次通过同一整模型用例，单项 **179.84 秒**。时间包含导入、形状证明、显式编译、身份准备、执行、数值对照和回滚，不是推理延迟。route/Plan ABI 合计仍有 **828,619,922 字节**，本次使用显式有限的测试预算，没有调大 controller 的全局默认值。

独立事件审计核对 **80 次前缀复制（368,640 字节）**均在 kernel 前完成，**112 次追加复制（172,032 字节）**均在全部 kernel 后发生；第二代没有新的状态/前缀容量分配。13 项篡改检查涵盖代际、请求编号、批量、P、提交后长度、复制顺序、缺失 kernel、ABI、初始分配和参考请求归属，全部被拒绝。输入快照、合批打包和诊断副本仍有 graph run span 外的分配/复制；不能只用 graph span 推断端到端批处理延迟。

| 检查 | 当前结果 |
|---|---|
| 组合配置构建、真实小图、并发与故障注入 | 通过 |
| 原 RuntimeSession 与请求批处理定向回归 | 2/2 通过 |
| 完整八层请求批处理换代与回滚 | 通过；99,328 个 ONNX 值、179,968 个独立 LLVM 值 |
| 四配置 CPU/LLVM 全量回归 | 默认 55/55、adaptive 56/56、bounded 70/70、组合 75/75；均在最终公开接口改动后重建并完整执行 |
| 生成契约、公共头文件、依赖、文档与强符号审计 | 两份生成契约新鲜；Relay 40/40、Pass 20/20；四配置公共头文件通过；依赖层 284 文件、文档 81 文件、NLP 检查与诊断 4/4 通过；五个库无重复强符号 |
| CUDA 编译兼容性 | 当前 runtime 静态库及三个 consumer 目标重建通过；RuntimeSession 单测通过，三项硬件门禁因设备不可用返回 77，不计作 GPU 通过 |
| CUDA 热替换设备数值 | 未执行 |

四配置 CTest 实际用时依次为 **146.82、260.61、231.84、743.97 秒**。最终回归中，静态容量、有界状态和请求批处理三个完整模型 consumer 均通过公开 CompiledGraph 绑定接口准备产物；模型 fixture 已逐项审计，没有用缺 fixture 的跳过记录替代这些执行证据。

五个库的外部强符号数依次为 **1,612 / 1,742 / 1,896 / 1,982 / 1,907**，对应默认、adaptive、bounded、组合与 bounded-CUDA，重复定义均为 0。`RunAsyncWithModule`、`RunNextBatchWithModule`、纯校验 `Validate` 及三个 CompiledGraph 绑定方法在每个库中各有且仅有一个定义；静态库已显式重建，未以旧归档代替当前对象。

三轮 Ponytail QA：A，复用原 RuntimeSession 队列、state、copy/commit 和 controller lease；B，Plan ABI 保持最大批量与槽位合同，selection 使用已有 memory version，路由不选择或编译新 profile；C，真实小图、三个公开状态绑定的完整模型消费者、四配置回归和事件审计均通过，拒绝检查覆盖预检无副作用及提交后 poison。

复现入口见 [构建说明](../BUILDING.md)。本地 `out/windows-gpu/request-hot-swap-*` 保存回归驱动、CPU/事件/兼容性/符号审计、源码及证据归档；`logs/cpu/kxc-request-hot-swap-*` 保留四配置完整输出和初次文档索引失败。最终归档固定本次代码与三份实际执行 Bundle，并保留失败日志；本轮没有执行 Git commit、push 或上传新包到 Windows。

本模块不新增后台调度器、ragged/padded 混合 P、状态迁移或隐式编译，也不代表 CUDA 热替换、跨机器状态或吞吐提升已验证。总项目目标继续保持未完成状态。
