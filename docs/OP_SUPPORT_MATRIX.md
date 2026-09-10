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

Tanh/Erf 的新增合同为 float32 静态 CPU/LLVM，真实 MiniMind-V 视觉链和拒绝边界见 [技术报告](implementation/M9_MINIMIND_V_VISION_REPORT.md)。尚无对应 CUDA 或 bounded 数值声明。

## 已声明算子

| 类别 | 规范算子 |
|---|---|
| 张量数学运算 | `add`, `subtract`, `mul`, `divide`, `where`, `sqrt`, `matmul`, `equal`, `neg`, `sigmoid`, `tanh`, `erf`, `pow` |
| 张量变换 | `nn_flatten`, `reshape`, `transpose`, `gather`, `cast`, `concatenate`, `split`, `slice`, `expand`, `shape_of`, `shape_expr`, `reshape_dynamic`, `expand_dynamic`, `constant_of_shape`, `squeeze`, `unsqueeze`, `trilu` |
| 张量归约 | `reduce_mean`（已知缺陷：keepdims=1 的 size-1 保留轴后跟非单例轴时 LLVM 数值错误，见 [G1 记录](implementation/G1_RECORD.md)；Transformer fixture 使用的尾轴归约不受影响） |
| 神经网络 | `nn_dense`, `nn_gemm`, `nn_layer_norm`, `nn_relu`, `nn_conv2d`, `nn_max_pool2d`, `nn_avg_pool2d`, `nn_global_avg_pool2d`, `softmax`, `masked_softmax` |

以上列表包含当前 41 个契约条目。不在机器契约中的算子，必须在对应的注册表、前端或编译器边界明确报错。形状值算子仅在 [M3 的受限路径](implementation/M3_SHAPE_VALUES.md) 中使用，不代表任意动态 shape 已开放。

M4/M5 补充边界：`neg`/`sigmoid`/`pow` 是 float32-only 子集（`pow` 为同 dtype 二元广播，经 `llvm.pow` 调用分派执行）；`expand` 的目标 shape 是导入期已解析进 `ExpandAttrs` 的常量控制输入（numpy `broadcast_to` 规则），动态 shape 输入未开放；ONNX `Unsqueeze` 在导入期规范化为既有 `reshape`，没有独立算子条目。

## 注册自动化状态

声明驱动的注册路径仍在渐进迁移。`add`、`equal`、`neg`、`sigmoid`、`pow`、`shape_of`、`masked_softmax` 当前使用生成式注册翻译单元与静态链接锚点（生成的 InferType/TE 符号按契约绑定，静态库内单一注册来源）。`masked_softmax` 复用 `SoftmaxAttrs`，由生成器声明 `TAttrs`。其他已声明算子（含 `expand` 的 `ExpandAttrs` 路径）在迁移前仍保留唯一的手工注册权威。同一个算子禁止同时使用生成式注册和手工注册。

迁移状态不改变运行时语义，也不改变目标后端的验证结论。

## 后端边界

### LLVM

LLVM 是主要的可执行 CPU 后端，要求 LLVM 20 或更新版本。算子要在 LLVM 上通过验证，必须在启用 LLVM 的构建中通过适用的逐单元编译、LLVM 代码生成、数值与端到端测试。禁用 LLVM 的 CPU 构建只能证明核心契约成立。

### CUDA

CUDA 支持范围刻意窄于 Relay 契约：

- `BindCudaThreads` v5 将静态行主序输出域展平到线程；多阶段计算以共有独立坐标划分各线程负责的归约行；
- 线程内串行归约须先无条件初始化。多阶段 producer 必须完整、有序，临时 Allocate 经逐维地址证明后压缩为私有存储，总量最多 64 KiB/线程；跨线程读取、写冲突、索引溢出、未证明的动态循环和临时存储均拒绝；
- 静态单阶段 Gather 可通过只读索引来源、分支区间、整数溢出及表地址证明；任意未证明的间接 `Load` 和多阶段间接访问仍拒绝；
- 找到 CUDA Toolkit 不代表本地设备能够启动内核；
- 不支持的 CUDA 图不会回退到 LLVM 或 CPU。

MatMul/Dense 的单阶段归约及 Softmax/MaskedSoftmax、LayerNorm、ReduceMean 的静态多阶段存储已分别取得设备数值证据。多阶段证明与非末轴、标量归约、广播 mask、注意力组合的验证见 [技术报告](implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md)。后续 [Gather/Pow 报告](implementation/GPU_GATHER_POW_REPORT.md) 和 [完整 prefill 报告](implementation/GPU_MINIMIND_PREFILL_REPORT.md) 补齐固定 B1/S16 八层模型的 CUDA 数值；静态容量 GPU 状态与四步 decode 已有后续 [报告](implementation/GPU_KV_STATE_REPORT.md)，bounded 单阶段输出、shape_of 与 rank-2 MatMul 已有 [基础报告](implementation/GPU_BOUNDED_CORE_REPORT.md)；有界 Softmax/MaskedSoftmax、固定轴 ReduceMean/RMS 和三头注意力已有 [多阶段报告](implementation/GPU_BOUNDED_REDUCTION_REPORT.md)；静态表有界索引与形状重排已通过完整八层模型的 CUDA 证明和 PTX 编译，见 [接入报告](implementation/GPU_BOUNDED_PREFILL_REPORT.md)；完整 bounded 模型设备数值及协作归约仍待验收。Relay Pow 仍只支持 float32，float64 Pow 仅为 TIR/backend 能力。

2026-09-09 的 Windows RTX 4070 Ti SUPER 基础专项已通过 5/5，含真实 add/constant/relu、非空一维 Where/Slice/Concatenate、异步复制与 CUPTI 活动，见 [实测报告](implementation/GPU_WINDOWS_VALIDATION_REPORT.md)。该局部证据不等于完整 Transformer 或各 NLP profile 已通过 CUDA 门禁。

后续专项 6/6 已增加 float32/64 MatMul/Dense、float32 批次广播 MatMul、三维 Where/Slice/Concatenate、标量与线程内 sum/max 的真实数值，见 [归约报告](implementation/GPU_OWNED_REDUCTION_REPORT.md)。该轮矩阵更新三格局部证据；最新 Gather 与固定模型验收仍保持 `implemented`，不以文本升为整体 `validated`。Slice 仍仅支持 `+1` 步长，尚无性能结论。

### C

C 后端输出可读源码，也可作为 AOT 构建组件。生成 C 源码可以用于诊断；只有当结果完成编译、加载、通过内核 ABI 启动并通过数值检查后，才能作为可执行证据。

## 明确的契约限制

- `nn_gemm` 在 `transA != 0` 时不能通过生产级 TE 路径执行。
- 当前值图 ABI 不支持函数元组参数。扁平多输出 Lowering 不等于支持元组参数。
- 默认 ONNX `Gather` 的常量索引逐值验证范围；运行时索引保留静态 shape/dtype/axis，越界按既有受保护 lowering 返回零，具体边界见 [导入器](ONNX_IMPORTER.md)。
- Relay `concatenate` 保持二元。静态 ONNX 的 1..N 输入通过同一 importer 归一化，真实调用见 [图文联合报告](implementation/M9_MINIMIND_V_JOINT_REPORT.md)；有界长度与定长片段的二元拼接见 [KV 追加报告](implementation/M3_KV_APPEND_REPORT.md)。
- Relay `split` 只开放静态 axis、恰好两个输出和常量分段长度；属性/initializer 形式及输出顺序、LLVM 数值见 [Split 报告](implementation/M4_SPLIT_REPORT.md)。动态分段长度与三路以上输出仍拒绝。
- `slice` 只支持 [ONNX 导入器](ONNX_IMPORTER.md) 中定义的静态精确、正步长子集。
- `nn_layer_norm` 只支持文档定义的静态精确仿射子集，不暴露 ONNX 的 Mean/InvStdDev 输出。
- `masked_softmax` 仅接入显式 Relay/FFI：float32/64、bool 单向广播 mask、固定 axis 与正归约长度。有限可见 logits 归一化，全 False 行和 False 元素为精确正零；静态及有界 CPU/LLVM 证据见 [全 mask 报告](implementation/M5_MASKED_SOFTMAX_REPORT.md)。普通 Softmax 不变；静态 CUDA 局部数值见 [多阶段报告](implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md)，有界 float32 广播 mask、变长与 masked NaN/Inf 的 CUDA 局部数值见 [后续报告](implementation/GPU_BOUNDED_REDUCTION_REPORT.md)；ONNX Attention 仍未开放。
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
