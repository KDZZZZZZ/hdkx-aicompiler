# 文档

本目录只保留当前维护中的文档。源码、机器可读契约和可重复测试仍是最终依据。

## 从这里开始

- [项目目标](PROJECT_GOAL.md) — 唯一目标权威：五根支柱、非目标、依赖关系与优先级。定义要成为什么。
- [架构总览](ARCHITECTURE.md) — 系统边界、编译与运行流程、身份、功能开关和已知限制。定义当前是什么。
- [TE Program 设计与实现边界](TE_PROGRAM_IR.md) — 静态 Program、确定性 lowering、缓存身份和首个融合切片。
- [构建说明](BUILDING.md) — 支持的本地配置和测试命令。
- [开发流程](DEVELOPMENT_WORKFLOW.md) — 分支、评审和验证规则。

## 契约和能力

- [编译器扩展契约](COMPILER_EXTENSION_CONTRACT.md) — 添加算子和 Pass 时必须遵守的集成流程。
- [Pass 契约](PASS_CONTRACT.md) — `PassSpec`、流水线、不变量和目标谓词语义。
- [Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md) — 已声明算子及各目标后端的能力边界。
- [ONNX 导入器](ONNX_IMPORTER.md) — 支持的静态导入子集和序列化格式。
- [Transformer 模型算子清单](OP_TODO.md) — 用于跟踪模型需求，不代表算子已经受支持。

## 现状快照

- [代码库清单：基座层与外挂层](CODEBASE_INVENTORY.md) — 按目标划分基座与外挂、基座缺口、算子归属与证据现状。

## 实施计划

当前生效的分派计划例外地维护在 [implementation/](implementation/README.md) 下，随实施推进逐步退出：

- [模块实施计划总览](implementation/README.md) — 第一波收尾状态、MiniMind-L1 目标、M0–M10 模块表与模块间衔接。
- [第一波并行计划](implementation/WAVE_1.md) — 已完成的 G0 共同基线、A/B/C 三条并行线、所有权与集成验收记录。
- [第二波 MiniMind-L1 并行计划](implementation/WAVE_2.md) — 导出门禁、L1a prefill、KV/形状/前端/控制流五条并行线和 G2 验收。
- [G0 基线记录](implementation/G0_BASELINE.md) — 已完成的共同基点：commit、测试清单与口径。
- [G1 第一波验收记录](implementation/G1_RECORD.md) — 第一波组合验收证据与已知缺陷。
- [G2 第二波验收记录](implementation/G2_RECORD.md) — 第二波逐项结果、L1a 的确切阻塞点与跨线冲突处置。
- [G2 greedy 技术报告](implementation/G2_GREEDY_REPORT.md) — 真实 LLVM prefill/decode 的 host 采样和模型 bundle。
- [M1 模型关联技术报告](implementation/M1_MODEL_ASSOCIATION_REPORT.md) — 运行 metadata 的方法、验证和效果。
- [M1 复制完成观测与诊断技术报告](implementation/M1_COPY_EVENT_REPORT.md) — CPU 实测复制耗时、异步完成配对、上下文保活与诊断计时边界。
- [M1 CUDA 模型关联与时钟对齐报告](implementation/M1_CUDA_CORRELATION_REPORT.md) — 完整 prefill 的 1,300 条设备活动、并发 profile、调用序号与主机/设备时间域。
- [M1 CUDA 复制关联与异步保活报告](implementation/M1_CUDA_COPY_REPORT.md) — DMA 到 copy_id 的逐条关联、偏移视图、真实 pending 与跨线程完成。
- [M9 MiniMind-V 完整视觉链技术报告](implementation/M9_MINIMIND_V_VISION_REPORT.md) — 12 层 SigLIP2、GELU 投影、固定 Shape 证明、独立参考和 LLVM 数值。
- [M9 MiniMind-V 图文联合推理技术报告](implementation/M9_MINIMIND_V_JOINT_REPORT.md) — 固定单图的视觉 token 注入、完整八层 prefill、会话 KV 和四步 decode，三组输入与独立参考对齐。
- [M9 MiniMind-V 固定双图联合推理技术报告](implementation/M9_MINIMIND_V_MULTI_IMAGE_REPORT.md) — 固定双图 profile、5D 像素输入、完整八层 prefill、会话 KV 和四步 LLVM decode。
- [GPU 驱动一致性修复报告](implementation/GPU_DRIVER_REPAIR_REPORT.md) — 驱动版本、安装核验、启动文件刷新与重启后的硬件门禁。
- [Windows GPU 与 CUPTI 实测报告](implementation/GPU_WINDOWS_VALIDATION_REPORT.md) — Tailscale/SSH 接入、CUDA 12.9 原生构建、实际 kernel/复制/设备活动，以及内存插桩的未完成项。
- [CUDA 多维输出与线程内归约报告](implementation/GPU_OWNED_REDUCTION_REPORT.md) — MatMul/Dense、批次广播、多维算子和 sum/max 的真实 GPU 数值，所有权证明与剩余边界。
- [CUDA 有界形状基础执行报告](implementation/GPU_BOUNDED_CORE_REPORT.md) — 同一产物的实际形状、MatMul、设备 extent、缓存身份与剩余模型边界。
- [CUDA 有界归约与注意力报告](implementation/GPU_BOUNDED_REDUCTION_REPORT.md) — 变长 Softmax、masked Softmax、RMS 和三头注意力的行内存储证明、数值与设备活动。
- [CUDA 完整变长 MiniMind 接入报告](implementation/GPU_BOUNDED_PREFILL_REPORT.md) — 静态表索引、形状重排、742 个原语、完整 prefill 数值和 CUPTI 设备验收。
- [CUDA 完整变长 decode 编译报告](implementation/GPU_BOUNDED_DECODE_REPORT.md) — 动态前缀拼接的分支证明、774 个原语、完整 decode/state 数值和设备活动验收。
- [CUDA 有界 KV 状态接入报告](implementation/GPU_BOUNDED_STATE_REPORT.md) — 同一 RuntimeSession 的前缀打包、状态提交、两条 stream 和设备复制顺序验收。
- [CUDA 请求批处理接入报告](implementation/GPU_REQUEST_BATCHING_REPORT.md) — 同设备准入、显式 stream、完整模型请求合批数值和 CUPTI 复制审计。
- [CUDA 多阶段归一化报告](implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md)
- [CUDA 同步内存完成语义报告](implementation/GPU_SYNC_MEMORY_REPORT.md)
- [CUDA Gather 与 Pow 报告](implementation/GPU_GATHER_POW_REPORT.md) — 索引范围与溢出证明、空张量和运行时索引数值。
- [完整 MiniMind 静态 CUDA Prefill 报告](implementation/GPU_MINIMIND_PREFILL_REPORT.md) — 八层 650 kernel、全部 logits/16 KV、Windows 栈和验收门禁。
- [M2 MiniMind 状态技术报告](implementation/M2_MINIMIND_STATE_REPORT.md) — 会话持有 K/V、显式初始化交接、容量与数值验证。
- [M2/M3 变长 KV 状态技术报告](implementation/M2_BOUNDED_STATE_REPORT.md) — 有效前缀布局、固定容量、真实 LLVM prefill 交接与连续 greedy。
- [M2/M3 请求批处理技术报告](implementation/M2_REQUEST_BATCHING_REPORT.md) — 请求队列、等长合批、KV 槽位与真实八层 MiniMind 数值证据。
- [M5/M3 全 mask 注意力技术报告](implementation/M5_MASKED_SOFTMAX_REPORT.md) — 显式 bool mask、全屏蔽行零输出、同产物变长 attention 与缓存/数值证据。
- [M3 int64 索引技术报告](implementation/M3_INT64_INDEX_REPORT.md) — 索引 Add 导入边界及已验证效果。
- [M3 有界注意力技术报告](implementation/M3_BOUNDED_ATTENTION_REPORT.md) — 同一 LLVM 产物执行不同 B/Q/T、动态归约、内存上界和数值证据。
- [M3 共享图与多输出技术报告](implementation/M3_GRAPH_STRUCTURE_REPORT.md) — 保留共享调用、重复实参和明确的 tuple 输出，附真实动态导出结构审计。
- [M3 加权投影技术报告](implementation/M3_WEIGHTED_PROJECTION_REPORT.md) — 静态权重、可证明广播、变长 RMSNorm/QKV 的真实 MiniMind 子图与 LLVM 数值证据。
- [M3/M4 ONNX 拆头技术报告](implementation/M3_ONNX_HEADS_REPORT.md) — 保留标量索引和形状控制链，一份 LLVM 产物完成实际 MiniMind 四组 B/S 的拆头与 Q/K 归一化。
- [M3/M4 GQA 技术报告](implementation/M3_GQA_REPORT.md) — 在整个 B/S 范围内证明控制条件，执行真实 K/V 重复链及动态 Squeeze/Unsqueeze。
- [M3/M4 完整变长 decode 技术报告](implementation/M3_FULL_DECODE_REPORT.md) — 八层动态 past/present、位置窗口、形状算术和连续 greedy LLVM 执行。
- [M3 KV 长度追加技术报告](implementation/M3_KV_APPEND_REPORT.md) — P+C 派生形状、连续追加、空缓存和多 batch 的 LLVM 执行证据。
- [M3 单计划多 token 技术报告](implementation/M3_MULTITOKEN_REPORT.md) — 同一 LLVM 计划的 C=1/2/3 当前长度、动态位置窗口和 prefill→decode 外部 K/V 交接。
- [M3/M4 完整变长 prefill 技术报告](implementation/M3_FULL_PREFILL_REPORT.md) — 八层 token→logits/16 KV 同产物执行，位置前缀证明、FFN/残差及大图身份共享存储。
- [M3/M4 因果 attention 技术报告](implementation/M3_CAUSAL_ATTENTION_REPORT.md) — 同一产物执行真实第一层 attention，含动态 mask、固定 -1 推导及未来 token 因果性验证。
- [M3/M4 RoPE 技术报告](implementation/M3_ROPE_REPORT.md) — 一份 LLVM 产物执行真实 RMSNorm/QKV/分头/位置旋转组合链，含 Slice 控制证明与参数/常量排序回归。
- [M4-D Split 多输出技术报告](implementation/M4_SPLIT_REPORT.md) — 静态 Split 的基础 ONNX → Relay → LLVM → RuntimeSession 链路、输出顺序和负例。
- [M4-D Split 多路扩展报告](implementation/M4_SPLIT_VARIADIC_REPORT.md) — 任意两路以上静态常量分段、变长输出 tuple 和 fail-closed 边界。
- [第二波并行计划](implementation/WAVE_2.md) — MiniMind-L1 的 A–E 五条并行线与 G2 组合验收。
- [M9：MiniMind 模型目标与导出验收](implementation/M9_MINIMIND_TARGET.md) — 导出门禁 E0 与 L1a/L1b 证据阶梯。
- [M0：共同基线与证据口径](implementation/M0_BASELINE.md)
- [M1：执行侧观测](implementation/M1_RUNTIME_PROFILING.md)
- [M2：KV cache 与动态运行状态](implementation/M2_KV_STATE.md)
- [GPU 技术报告：会话 KV 状态与完整四步 decode](implementation/GPU_KV_STATE_REPORT.md)
- [M3：形状值与有界形状计算](implementation/M3_SHAPE_VALUES.md)
- [M4：ONNX 导入与多输出](implementation/M4_ONNX_IMPORT.md)
- [M5：新逐元素算子](implementation/M5_ELEMENTWISE_OPS.md)
- [M6 状态热替换技术报告](implementation/M6_STATEFUL_REPORT.md) — 请求持有 KV、步间换代与回滚、完整八层模型数值和控制面内存修复。
- [M6 有界形状热替换技术报告](implementation/M6_BOUNDED_REPORT.md) — 有界编译权限、范围与连接校验、完整 CPU 模型状态换代和回滚。
- [M6 请求批处理热替换技术报告](implementation/M6_REQUEST_BATCHING_REPORT.md) — 排队请求、合批与槽位跨代际保留；CPU 完整模型、四配置回归及集成审计通过。
- [M6 CUDA 热替换技术报告](implementation/M6_CUDA_REPORT.md) — RTX 4070 Ti SUPER 上的真实 CUDA kernel 换代、CUPTI health/rollback 与 fail-closed 审计。
- [MiniMind-O 完成边界报告](implementation/M9_MINIMIND_O_BOUNDARY_REPORT.md) — L3 的模型、状态、流式运行、软实时预算与热替换退出条件。
- [M6：热替换执行与决策证据](implementation/M6_HOT_SWAP.md)
- [M6 技术报告：真实换代、Profile Bundle 健康检查与回滚](implementation/M6_RUNTIME_REPORT.md)
- [M7：分布式证据与执行桥接](implementation/M7_DISTRIBUTED.md)
- [M7 技术报告：进程内多 worker 的已编译 CPU 执行](implementation/M7_DISTRIBUTED_REPORT.md)
- [M7 静态整图分发与 MiniMind prefill 技术报告](implementation/M7_MODEL_REPORT.md) — 多输出、显式 worker 放置与完整模型执行；限定范围验收完成。
- [M8：TE Program 与首个跨算子融合](implementation/M8_TE_PROGRAM.md)
- [M8 技术报告：静态 Program 与 LLVM 融合执行](implementation/M8_TE_PROGRAM_REPORT.md)
- [M9：MiniMind 模型目标与导出验收](implementation/M9_MINIMIND_TARGET.md)
- [M9 E0 导出 receipt](implementation/M9_E0_RECEIPT.md) — 锁定的导出合同、SHA 与动态导出门禁结论。
- [M9 E1 receipt](implementation/M9_E1_RECEIPT.md) — L1a 静态 prefill 的端到端验收：固定 ABI、数值证据与复现方式。
- [M9 E2 decode 签名审计](implementation/M9_E2_SIGNATURE.md) — past/present 签名与 M2 绑定要求。
- [M10：结构化控制流（If / 有界 While）](implementation/M10_STRUCTURED_CONTROL.md)
- [M10 控制流 receipt](implementation/M10_CONTROL_RECEIPT.md) — C0 审计、C1 gate-on 证据与 L1b 循环选择。

## 工具文档

- [性能工作台](../tools/workbench/README.md) — Profile Bundle 查看、比较以及 UI/CLI 入口。

## 文档策略

- `PROJECT_GOAL.md` 是唯一目标权威文档；任何工作都必须能归入其五根支柱之一。
- `ARCHITECTURE.md` 是唯一架构权威文档。
- 聚焦文档只描述当前行为或机器契约；引入边界时必须链接回架构文档。
- 实施计划、Issue 设计、迁移交接和测试会话报告通常放在 GitHub Issue 或 Pull Request 中；当前分派计划维护在 `docs/implementation/`。按 2026-09-08 的项目目标交付要求，每个模块还在该目录保留面向人类的技术报告，说明方法、验证与效果。
- 生成报告和二进制导出应放在 `out/` 下，不提交入库。
- 任何调整公共边界的代码变更，都必须在同一个 Pull Request 中更新 `ARCHITECTURE.md`。
- 评审前运行 `python tools/architecture/check_docs.py --root .`。
