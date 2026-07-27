# 文档

本目录只保留当前维护中的文档。源码、机器可读契约和可重复测试仍是最终依据。

## 从这里开始

- [架构总览](ARCHITECTURE.md) — 系统边界、编译与运行流程、身份、功能开关和已知限制。
- [构建说明](BUILDING.md) — 支持的本地配置和测试命令。
- [开发流程](DEVELOPMENT_WORKFLOW.md) — 分支、评审和验证规则。

## 契约和能力

- [编译器扩展契约](COMPILER_EXTENSION_CONTRACT.md) — 添加算子和 Pass 时必须遵守的集成流程。
- [Pass 契约](PASS_CONTRACT.md) — `PassSpec`、流水线、不变量和目标谓词语义。
- [Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md) — 已声明算子及各目标后端的能力边界。
- [ONNX 导入器](ONNX_IMPORTER.md) — 支持的静态导入子集和序列化格式。
- [Transformer 模型算子清单](OP_TODO.md) — 用于跟踪模型需求，不代表算子已经受支持。

## 工具文档

- [性能工作台](../tools/workbench/README.md) — Profile Bundle 查看、比较以及 UI/CLI 入口。

## 文档策略

- `ARCHITECTURE.md` 是唯一架构权威文档。
- 聚焦文档只描述当前行为或机器契约；引入边界时必须链接回架构文档。
- 实施计划、Issue 设计、迁移交接和测试会话报告应放在 GitHub Issue 或 Pull Request 中，不放入 `docs/`。
- 生成报告和二进制导出应放在 `out/` 下，不提交入库。
- 任何调整公共边界的代码变更，都必须在同一个 Pull Request 中更新 `ARCHITECTURE.md`。
- 评审前运行 `python tools/architecture/check_docs.py --root .`。
