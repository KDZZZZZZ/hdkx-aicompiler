# 文档

本目录只保留当前维护中的文档。源码、机器可读契约和可重复测试仍是最终依据。

## 从这里开始

- [项目目标](PROJECT_GOAL.md) — 唯一目标权威：五根支柱、非目标、依赖关系与优先级。定义要成为什么。
- [架构总览](ARCHITECTURE.md) — 系统边界、编译与运行流程、身份、功能开关和已知限制。定义当前是什么。
- [TE Program IR 设计提案](TE_PROGRAM_IR.md) — 尚未实现的 `te::Program` 边界、确定性 Lowering、融合接入与验收条件。
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

- [模块实施计划总览](implementation/README.md) — 当前基线事实、M0–M8 模块表与模块间衔接。
- [第一波并行计划](implementation/WAVE_1.md) — G0 共同基线、A/B/C 三条并行线、所有权与集成验收。
- [G0 基线记录](implementation/G0_BASELINE.md) — 已完成的共同基点：commit、测试清单与口径。
- [G1 第一波验收记录](implementation/G1_RECORD.md) — 第一波组合验收证据与已知缺陷。
- [M0：共同基线与证据口径](implementation/M0_BASELINE.md)
- [M1：执行侧观测](implementation/M1_RUNTIME_PROFILING.md)
- [M2：KV cache 与动态运行状态](implementation/M2_KV_STATE.md)
- [M3：形状值与有界形状计算](implementation/M3_SHAPE_VALUES.md)
- [M4：ONNX 导入与多输出](implementation/M4_ONNX_IMPORT.md)
- [M5：新逐元素算子](implementation/M5_ELEMENTWISE_OPS.md)
- [M6：热替换执行与决策证据](implementation/M6_HOT_SWAP.md)
- [M7：分布式证据与执行桥接](implementation/M7_DISTRIBUTED.md)
- [M8：TE Program 与首个跨算子融合](implementation/M8_TE_PROGRAM.md)

## 工具文档

- [性能工作台](../tools/workbench/README.md) — Profile Bundle 查看、比较以及 UI/CLI 入口。

## 文档策略

- `PROJECT_GOAL.md` 是唯一目标权威文档；任何工作都必须能归入其五根支柱之一。
- `ARCHITECTURE.md` 是唯一架构权威文档。
- 聚焦文档只描述当前行为或机器契约；引入边界时必须链接回架构文档。
- 实施计划、Issue 设计、迁移交接和测试会话报告应放在 GitHub Issue 或 Pull Request 中；当前生效的分派计划例外地维护在 `docs/implementation/`（见上文"实施计划"），完成后移出。
- 生成报告和二进制导出应放在 `out/` 下，不提交入库。
- 任何调整公共边界的代码变更，都必须在同一个 Pull Request 中更新 `ARCHITECTURE.md`。
- 评审前运行 `python tools/architecture/check_docs.py --root .`。
