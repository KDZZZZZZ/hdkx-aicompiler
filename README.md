# hdkx-aicompiler

`hdkx-aicompiler` 是一个受 TVM 启发的精简 C++17 AI 编译器/运行时。它实现了基于契约的
Relay-TE/TIR 后端编译流水线，并为运行时发布不可变模块与执行计划。

该仓库是工程与研究代码库，不是 TVM 的直接替代品。未支持的算子、形状、目标后端与运行时
模式会被明确拒绝。

## 快速开始

最小化 Windows 构建可使用 MinGW，无需 CUDA 或 LLVM：

```powershell
cmake --preset dev-mingw-cpu
cmake --build --preset dev-mingw-cpu
ctest --test-dir out/build/dev-mingw-cpu --output-on-failure --no-tests=error
```

Linux 与可选 LLVM/CUDA 配置见
[构建说明](docs/BUILDING.md)。

## 架构

仓库的权威系统说明在
[架构总览](docs/ARCHITECTURE.md)。主静态路径为：

```text
Relay Function + CompileConfig
  -> 类型完备的 ANF 与规范化 Pass 契约
  -> 值图与 PrimitiveUnit 分区
  -> TE 调度与 TIR
  -> LLVM 或 CUDA 产物
  -> CompiledModule + ExecutablePlan
  -> RuntimeSession
```

运行时只执行已编译计划：不导入模型，不运行 Relay Pass，不选择形状，也不会在缓存未命中时触发编译。

## 文档

- [文档索引](docs/README.md)
- [项目目标（唯一目标权威）](docs/PROJECT_GOAL.md)
- [模块实施总览与第一波并行计划](docs/implementation/README.md)
- [架构总览](docs/ARCHITECTURE.md)
- [TE Program IR 设计提案（尚未实现）](docs/TE_PROGRAM_IR.md)
- [构建说明](docs/BUILDING.md)
- [编译器扩展契约](docs/COMPILER_EXTENSION_CONTRACT.md)
- [Relay 算子支持矩阵](docs/OP_SUPPORT_MATRIX.md)
- [ONNX 导入器](docs/ONNX_IMPORTER.md)
- [Pass 契约](docs/PASS_CONTRACT.md)
- [开发流程](docs/DEVELOPMENT_WORKFLOW.md)
- [性能工作台](tools/workbench/README.md)

机器可读的算子与 Pass 元数据位于
`contracts/relay_op_contract.json` 与 `contracts/pass_contract.json`。
