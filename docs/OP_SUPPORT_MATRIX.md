# Relay 算子支持矩阵

> **状态：** 当前契约清单。
> 已声明算子不代表它已在所有目标后端上通过验证。

`contracts/relay_op_contract.json` 是 Relay 算子规范名称与必要集成阶段的机器可读来源。检查器会验证已声明的 Schema、类型推导、Lowering 绑定、FFI 接口、后端引用与测试是否同步。

生产支持是更强的声明：

```text
已声明契约
  + 有效的 Relay 类型与形状行为
  + 生产级逐 PrimitiveUnit Lowering
  + 目标调度通过校验
  + 后端编译
  + 数值执行证据
  = 在该目标上通过验证的算子
```

仅有整图 `relay::LowerToTIR` 测试或生成源码，不能作为生产能力证据。

## 已声明算子

| 类别 | 规范算子 |
|---|---|
| 张量数学运算 | `add`, `subtract`, `mul`, `divide`, `where`, `sqrt`, `matmul`, `equal`, `neg`, `sigmoid`, `pow` |
| 张量变换 | `nn_flatten`, `reshape`, `transpose`, `gather`, `cast`, `concatenate`, `slice`, `expand` |
| 张量归约 | `reduce_mean`（已知缺陷：keepdims=1 的 size-1 保留轴后跟非单例轴时 LLVM 数值错误，见 [G1 记录](implementation/G1_RECORD.md)；Transformer fixture 使用的尾轴归约不受影响） |
| 神经网络 | `nn_dense`, `nn_gemm`, `nn_layer_norm`, `nn_relu`, `nn_conv2d`, `nn_max_pool2d`, `nn_avg_pool2d`, `nn_global_avg_pool2d`, `softmax` |

以上列表包含当前 29 个契约条目。不在机器契约中的算子，必须在对应的注册表、前端或编译器边界明确报错。

M4/M5 补充边界：`neg`/`sigmoid`/`pow` 是 float32-only 子集（`pow` 为同 dtype 二元广播，经 `llvm.pow` 调用分派执行）；`expand` 的目标 shape 是导入期已解析进 `ExpandAttrs` 的常量控制输入（numpy `broadcast_to` 规则），动态 shape 输入未开放；ONNX `Unsqueeze` 在导入期规范化为既有 `reshape`，没有独立算子条目。

## 注册自动化状态

声明驱动的注册路径仍在渐进迁移。`add`、`equal`、`neg`、`sigmoid`、`pow` 当前使用生成式注册翻译单元与静态链接锚点（生成的 InferType/TE 符号按契约绑定，静态库内单一注册来源）。其他已声明算子（含 `expand` 的 `ExpandAttrs` 路径）在迁移前仍保留唯一的手工注册权威。同一个算子禁止同时使用生成式注册和手工注册。

迁移状态不改变运行时语义，也不改变目标后端的验证结论。

## 后端边界

### LLVM

LLVM 是主要的可执行 CPU 后端，要求 LLVM 20 或更新版本。算子要在 LLVM 上通过验证，必须在启用 LLVM 的构建中通过适用的逐单元编译、LLVM 代码生成、数值与端到端测试。禁用 LLVM 的 CPU 构建只能证明核心契约成立。

### CUDA

CUDA 支持范围刻意窄于 Relay 契约：

- 当前 `BindCudaThreads` 路径要求一个外层串行数据并行循环，并且各写入互不冲突；
- 嵌套归约和写冲突会被拒绝；
- 间接 `Load` 索引，包括通用 Gather 调度，会被拒绝；
- 找到 CUDA Toolkit 不代表本地设备能够启动内核；
- 不支持的 CUDA 图不会回退到 LLVM 或 CPU。

因此，即使某些 Relay 算子已有 Relay 与 LLVM 实现，只要涉及归约、间接加载或嵌套循环结构，就仍未在通用 CUDA 路径上通过验证。

### C

C 后端输出可读源码，也可作为 AOT 构建组件。生成 C 源码可以用于诊断；只有当结果完成编译、加载、通过内核 ABI 启动并通过数值检查后，才能作为可执行证据。

## 明确的契约限制

- `nn_gemm` 在 `transA != 0` 时不能通过生产级 TE 路径执行。
- 当前值图 ABI 不支持函数元组参数。扁平多输出 Lowering 不等于支持元组参数。
- 默认 ONNX `Gather` 映射比通用 Relay 算子更窄：索引必须来自初始化器，而且每个值都能静态证明在有效范围内。
- `concatenate` 当前只支持二元、静态精确子集。
- `slice` 只支持 [ONNX 导入器](ONNX_IMPORTER.md) 中定义的静态精确、正步长子集。
- `nn_layer_norm` 只支持文档定义的静态精确仿射子集，不暴露 ONNX 的 Mean/InvStdDev 输出。
- 不支持的数据类型、秩、形状、属性、目标能力、调度或 ABI 组合必须给出带上下文的错误，不能返回空张量、空操作内核或静默替代结果。

## 验证

```powershell
python python/tools/check_relay_op_contract.py --root . `
  --matrix contracts/relay_op_contract.json

cmake --build out/build/dev-mingw-cpu --target `
  registry_test infer_type_test operator_compilation_test `
  compiler_extension_contract_test check_relay_op_contract -j 4
```

后端变更还需运行对应的数值测试矩阵：

- `codegen_llvm_test`、`op_numeric_llvm_test`，以及 LLVM `onnx_importer_test`；
- `codegen_cuda_test` 和适用的 CUDA 硬件测试。

完整命令和工具链要求见 [构建说明](BUILDING.md)。

## 更新矩阵

算子变更必须先更新机器契约，重新生成并提交产物，再补充正向与负向测试；只有公共能力边界变化时才更新本文档。具体流程见 [编译器扩展契约](COMPILER_EXTENSION_CONTRACT.md)。
