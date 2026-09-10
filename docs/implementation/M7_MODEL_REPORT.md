# M7：静态整图分发与完整 MiniMind prefill 技术报告

2026-09-10，**已完成（限定范围）**。公开整图绑定、多输出、共享小图及完整八层 prefill 的 CPU/LLVM 证据、四配置回归、四份 profile 审计和跨配置兼容性检查均已通过。本报告的完成范围是进程内、静态、无状态、CPU:0 的整图分发；分布式 decode/KV、CUDA、跨机器传输和自动分区仍是后续目标。

## 问题与方法

此前 Disco 可执行调用方逐节点构造的小图，但缺少把既有完整编译图绑定到 worker 的公开入口，最终输出也只有一个。完整 MiniMind prefill 需要保留共享数据依赖、编译模块中的权重、logits 及全部 16 份 K/V 输出。

参考 [TVM Disco 的 DRef/worker 归属](https://tvm.apache.org/docs/reference/api/python/runtime/disco.html)和 [PyTorch PipelineStage 的形状与通信缓冲区合同](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)，采用显式放置和静态边界验证。复用仓库自己的编译产物、运行时计划、Disco 引用与通信，不接入外部框架或新的 IR。

## 接入方式

- `ExecutionPlan::FromExecutablePlan` 接收已编译模块、原 ExecutablePlan、完整 placement、逐输入/逐调用的 worker 列表和输出 worker。复用纯 `RuntimeSession::Validate` 检查原模块与图的签名、常量和连接；只接受静态、无状态、完整值与独立 storage 的 CPU/LLVM 计划。
- 输入顺序、调用顺序和完整输出顺序来自原计划。调用方显式选择每个原语的 worker；绑定器不重编译、不查 cache、不重新选择算子或推断分区策略。
- 权重常量继续由模块注入，不复制进 Disco 初始值表。实际跨 worker 的数据值生成现有 `device.copy` 节点；同一个来源到同一个 worker 的副本供多个消费者复用。所有返回值显式归集到输出 worker。
- 原 ExecutionPlan 增加有序 `output_value_ids`；旧单输出构造和 `output_value` 保留为受校验的首输出视图。`ExecuteForOutputs` 返回全部 DRef，预检在任何通信/launch 前验证每个最终输出的 worker 可用性。
- JSON 写出 v3 的有序输出列表；读取 v2 时把旧单输出转换为一个元素的列表，仍执行完整原合同检查。不同 schema 的未知/冲突字段继续拒绝。单目标 Kernel ABI、primitive artifact 和图语义身份不变。
- NDArray 的原 dtype 名称表由解析与新增 `DataTypeToString` 共用，用于把原 ValueSpec 的 DLPack 类型挂到分布式计划；未扩大原有命名类型范围。

## 实际消费者

小图是带共享生产者和模块常量的 `a=x+bias`、`relu(a)`、`a+y`，三个输出返回给同一 worker。验证计划、JSON、复制复用、跨两 worker 的实际 LLVM、独立 RuntimeSession 对照、次要输出不可用时零执行、缓存清空后的保活和通信失败后的无状态重试。

完整模型复用锁定 B1/S16 八层 MiniMind 导出和固定随机 float32 权重。两次执行分别交换前后两半原语的 worker，并改变最终输出 worker；全部 logits 和 K/V 同时与单机 RuntimeSession 及独立 ONNX 参考比较。该测试不执行分布式 decode，不建立另一套 KV owner，也不代表预训练质量或吞吐提升。

## 验证结果

共享小图通过：3 个实际 kernel、3 次复制，全部三个输出与独立 LLVM 会话一致；共享中间值只向另一个 worker 传一次。追加检查覆盖错误 worker/Target、次要输出不在声明 worker、空输出列表、状态计划和直接暴露模块常量的拒绝。图、调用方模块及 cache 释放后，executor 仍能完成执行；复制在第一个真实 kernel 后失败时，后续节点停止，旧输出保持有效，无状态重试成功。

完整八层模型使用同一编译产物、相同 650 个 kernel 和两套显式放置：前 325 个与后 325 个 kernel 交换 worker，输出也随之换到另一个 worker。每次返回 logits 和 16 份 KV，均与独立单机 RuntimeSession **逐位相等**。

| 放置 | worker 0 / worker 1 kernel | 实际复制次数 | 从计划形状/dtype 计算的复制字节 | 输出 |
|---|---:|---:|---:|---:|
| 0：前半在 0，后半在 1，输出到 0 | 325 / 325 | 11 | 655,424 | 17 |
| 1：前半在 1，后半在 0，输出到 1 | 325 / 325 | 12 | 655,552 | 17 |

两次共执行 **1,300 次 LLVM kernel、23 次 CPU CCL 复制**。全部 logits/16 KV 同时对照独立 ONNX 参考，共 **401,408 个值**，最大绝对误差 **8.82149e-6**，低于 `5e-5`；另有同样数量的逐位 LLVM 对照值。第二次运行后第一组输出仍有效，两次输出不共享 storage。准备、JSON 往返、运行和拒绝均无运行时编译或 primitive cache 查询。

修复测试设置后的定向整模型 CTest 为 **13.19 秒**，GNU time 峰值 RSS **1,245,588 KiB**。该时间包括导入、编译、单机参考、两次分布式运行、序列化和数值对照，不是推理延迟或加速比。

四套 CPU 配置（default、adaptive、bounded、adaptive-bounded）分别通过 **56/56、57/57、71/71、76/76**；总 CTest 时间分别为 **159.86、273.23、244.56、755.17 秒**。每套 Bundle 都包含 **2,650 条完成事件**。审计将两份模型 receipt、执行计划、每个 kernel/copy 节点和 worker 调用逐层关联；核对所有节点顺序、数据可用性、复制来源/目标、650 个 kernel 的 ABI、全部输出顺序与归属。每套的 14 种事件/计划篡改均被拒绝。复制字节来自实际计划中的形状和 dtype，`comm_exec` 记录同步 CCL 调用区间；没有声称 DMA 硬件计时或裸 kernel 耗时。

初次构建和共享小图通过后，完整模型首轮因测试目录尚未创建而无法写出计划，发生于分布式执行前。补充状态负例时又先后修正了错误的属性构造和未保留普通输出的测试图，随后小图及完整模型重新通过。原失败日志均保留，缺 fixture 的 77 不计作模型执行通过。

最终兼容性检查重建了四套 LLVM 配置和一套 bounded CUDA 配置；公共头文件测试、`distributed_runtime_test` 与 `runtime_session_test` 均通过，三个需要 GPU 的 CUDA 用例按设备门禁跳过。五套 `libkxc_runtime.a` 均无重复强符号，必需的整图绑定、输出执行、dtype 转换、状态/批处理入口各有一个定义；强符号数依配置为 **1,617、1,747、1,901、1,987、1,912**。文档链接、NLP 能力矩阵校验和 `git diff --check` 均通过（82 篇 Markdown）。

## 三轮设计审查与边界

A：复用原 ExecutablePlan、CompiledModule、Disco placement/DRef/CCL 和 dtype 名称 owner。B：有序多输出扩展同一分布式合同，JSON 版本覆盖新行为，单目标编译身份保持不变。C：公开绑定与输出接口由真实共享小图和完整模型消费，并通过数值、回归、profile 和兼容性审计。

本模块针对进程内 CPU 静态整图的显式跨 worker 分发。每个节点仍等待自己的 worker 作业和复制完成；未验证 micro-batch 重叠、长期 worker pool、CUDA、跨机器传输、动态 shape 或分布式持久 KV。
