# TinyTVM 需求文档

状态：Draft  
日期：2026-06-04  
最近修订：2026-07-20
适用仓库：`hdkx-aicompiler`

## 1. 背景

本仓库已经具备一个 TVM 风格轻量 AI compiler/runtime 原型的主要分层：

```text
Relay IR -> Relay Pass -> TE/TOPI -> TIR -> TIR Pass -> C/LLVM Codegen -> Runtime
```

当前已有对象系统、Relay/TIR IR、部分算子注册、部分 TE/TOPI helper、Relay 到 TIR lowering、LLVM/CUDA 编译执行链路、CPU/CUDA DeviceAPI、静态 RuntimeSession、Disco 多 worker 执行计划原型和 profiling bundle 工具。

本需求文档的目标是定义“完整 TinyTVM”的可交付范围，使后续开发能围绕可运行闭环推进，而不是只停留在 IR dump、算子元数据注册或局部 smoke test。

## 2. 产品目标

TinyTVM 是一个教学和实验定位的轻量编译器/runtime，目标不是覆盖完整 TVM 生态，而是做到：

1. 能从手写 Relay 或小规模 ONNX 模型构建计算图。
2. 能完成静态 shape 模型的类型推导、优化、lowering、codegen 和运行。
3. 能用 CPU 后端稳定执行典型神经网络算子。
4. 能生成可读 C 源码，并支持 LLVM JIT 作为主执行后端。
5. 有清晰的 pass、算子、runtime 扩展方法。
6. 有系统化测试，确保新增算子和 pass 不破坏端到端链路。

## 3. 范围定义

### 3.1 MVP 范围

MVP 定义为“静态 shape、单机 CPU、单输出模型”的完整闭环：

```text
ONNX/手写 Relay
  -> Relay Function
  -> Type/Shape Infer
  -> Relay Optimize
  -> TE/TIR Lower
  -> TIR Optimize
  -> LLVM JIT or C Source
  -> Run with NDArray inputs
  -> Verify numeric output
```

MVP 必须支持以下模型：

- Elementwise add/mul/sub/div/sqrt。
- MLP：dense + bias + relu。
- CNN block：conv2d + bias + relu + maxpool/global_avg_pool。
- ResNet18 子图或完整 ResNet18 的静态 shape 编译运行，至少 CPU float32。

### 3.2 完整 TinyTVM 范围

完整版本在 MVP 上增加：

- ONNX importer 更完整，能处理常见视觉和 Transformer 静态图。
- Relay 支持 tuple、多输出、常量权重绑定、基础控制流。
- TE schedule primitives 真正影响 TIR。
- C backend 具备可编译 AOT 路径。
- Runtime 支持模块保存/加载、NDArray 参数绑定、profiling、cache。
- CUDA 作为实验后端，至少支持简单 elementwise 或 matmul kernel codegen。
- Disco 多 worker 从通信语义模拟推进到真实 kernel 调用。

### 3.3 非目标

以下内容不作为第一版完整 TinyTVM 的硬性范围：

- 完整 TVM Relay 语法和所有 pass。
- 自动张量化、AutoTVM、Ansor 或复杂 auto scheduler。
- 完整动态 shape 编译。
- 完整 ONNX opset 覆盖。
- 分布式训练。
- 生产级 CUDA/NCCL 性能。

## 4. 用户角色

1. 编译器开发者：新增算子、pass、lowering 或 codegen 能力。
2. 模型导入使用者：导入 ONNX 模型并运行 CPU 推理。
3. 教学使用者：阅读 IR、pass 前后变化和生成代码。
4. 性能调试者：查看 pass 耗时、kernel 耗时和 profiling bundle。

## 5. 功能需求

### R1. 构建与开发环境

R1.1 必须提供稳定的 CPU-only 构建路径。

- Windows 下至少支持 MSVC 或 MinGW 其中之一。
- Linux 下至少支持 GCC/Clang。
- CPU-only preset 不应依赖 CUDA、NCCL 或本机 GPU 工具链。

验收标准：

- `cmake --preset dev-ninja-cpu` 配置成功。
- `cmake --build --preset dev-ninja-cpu` 构建成功。
- 所有 CPU smoke test 可通过 custom target 运行。

R1.2 必须提供清晰的工具链说明。

- 说明 Windows 需要的 Visual Studio 组件，例如 C++ compiler、Windows SDK、`rc`、`mt`。
- 说明 MinGW 或 LLVM 配置方式。
- 说明 LLVM 可选依赖和关闭方式。

R1.3 构建开关必须能隔离功能。

- `KXC_ENABLE_LLVM=OFF` 时仍能构建基础 IR、pass 和 C source codegen 测试。
- `KXC_ENABLE_CUDA=OFF` 时不编译 CUDA-only smoke test。

### R2. Relay IR

R2.1 Relay 必须支持基础表达式节点。

- `Var`
- `Constant`
- `Call`
- `Function`
- `Tuple`
- `TupleGetItem`
- `Let`

MVP 中 `If` 可以保留 IR 表达能力，但不要求 lowering 到 executable kernel。

R2.2 Relay 节点必须有可调试打印。

- 打印 op name、attrs、输入输出类型。
- 打印结构应能稳定用于测试快照。

R2.3 Relay 类型系统必须支持 TensorType。

- 静态 shape。
- dtype 至少支持 `float32`、`int32`、`int64`、`bool`。
- shape 维度允许 `int64_t` 静态值。

### R3. Type/Shape Inference

R3.1 必须实现 Relay type/shape inference pass。

最低覆盖：

- elementwise broadcast。
- matmul/dense/gemm。
- conv2d NCHW/OIHW。
- maxpool2d/avgpool2d/global_avg_pool2d。
- reshape/flatten/transpose/squeeze/unsqueeze。
- concat/split/gather/slice。
- reduce_mean。
- softmax。

验收标准：

- 用户构图时无需手动给每个中间 `Call` 填 `TensorType`。
- `LowerToTIR` 不再依赖脆弱的手写中间类型。

R3.2 必须验证 attrs 和输入数量。

- op 输入数量不匹配应报明确错误。
- attrs 类型不匹配应报明确错误。
- shape 不合法应在 type inference 阶段失败，而不是 codegen 阶段崩溃。

### R4. Relay 算子系统

R4.1 算子注册名必须统一。

当前 repo 存在风险示例：

- `mul` 和 `multiply` 同时出现。
- `softmax` 和 `nn_softmax` 同时出现。
- `_make.sub` 使用 `sub`，但实际注册有 `subtract`。

需求：

- 建立统一 op name 表。
- ONNX importer、C++ helper、Relay op registration、pass、lowering 必须使用同一命名。
- 兼容别名可以保留，但必须映射到 canonical op。

R4.2 注册了可执行语义的 op 必须包含 `FRelayToTE`。

MVP 必须实现 `FRelayToTE`：

- `add`
- `subtract`
- `mul`
- `divide`
- `sqrt`
- `cast`
- `matmul`
- `nn_dense`
- `nn_gemm`
- `nn_relu`
- `nn_conv2d`
- `nn_max_pool2d`
- `nn_avg_pool2d`
- `nn_global_avg_pool2d`
- `nn_flatten`
- `reshape`
- `transpose`
- `reduce_mean`
- `softmax`

完整版本增加：

- `pow`
- `erf`
- `equal`
- `where`
- `gather`
- `slice`
- `concat`
- `split`
- `squeeze`
- `unsqueeze`
- `shape`
- `constant_of_shape`
- `expand`

R4.3 算子 attrs 必须集中定义。

- 不应在多个 `.cc` 中重复或隐式约定 attrs。
- attrs 应支持 `VisitAttrs()` 或等价打印/序列化能力。

### R5. ONNX 导入

R5.1 必须提供一个正式 ONNX importer。

当前 `python/onnx_to_cpp.py` 和 `python/gen_resnet18_ir_dump_cpp.py` 偏实验脚本。完整 TinyTVM 需要一个明确入口：

```text
onnx model -> Relay Function + params
```

R5.2 importer 必须保留 initializer 数据。

- 权重不能只创建空 NDArray。
- 支持从 ONNX TensorProto 读取 float32/int64 等数据。
- Relay `Constant` 应持有真实 `NDArray`。

R5.3 importer 必须处理常见静态 ONNX op。

MVP 覆盖 ResNet18 所需：

- `Conv`
- `Relu`
- `MaxPool`
- `Add`
- `GlobalAveragePool`
- `Flatten`
- `Gemm`

完整视觉模型覆盖：

- `BatchNormalization`
- `AveragePool`
- `Clip`
- `Sigmoid`
- `Resize`，可先限定 nearest/linear 静态参数。

Transformer 静态子集覆盖：

- `MatMul`
- `Add`
- `Mul`
- `Div`
- `Sub`
- `Pow`
- `Sqrt`
- `Erf`
- `Softmax`
- `LayerNorm` 模式，或由 primitive ops 组合。
- `Reshape`
- `Transpose`
- `Gather`
- `Concat`
- `Slice`
- `Unsqueeze`
- `Squeeze`
- `Cast`
- `Where`

R5.4 importer 输出必须可测试。

- 能导出 Relay 文本。
- 能导出 TIR 文本。
- 能运行 numeric compare，与 ONNX Runtime 或 numpy 参考结果比较。

### R6. TE/TOPI

R6.1 TE 表达能力必须覆盖 MVP 算子。

最低要求：

- placeholder。
- compute。
- producer load。
- reduce axis。
- sum/max/min reduction。
- call intrinsic。
- broadcast index 计算。

R6.2 TOPI helper 必须返回有效 Tensor。

- 不允许用空 Tensor 作为 placeholder 实现。
- `einsum` 如果未实现，应不进入完整范围或显式报错。
- `min`、`prod` 不得用 `sum` 代替并静默返回错误结果。

R6.3 TOPI 必须有单元测试。

- 每个 helper 至少有 shape test。
- 关键 helper 至少有 TIR lowering test。
- elementwise、reduction、conv、pool、dense 必须有 numeric test。

### R7. Relay 到 TIR Lowering

R7.1 MVP lowering 必须支持单输出 tensor。

当前已有基础能力，但需要稳定覆盖：

- elementwise。
- broadcast。
- reduction。
- conv2d。
- pool。
- dense/gemm。

R7.2 完整版本必须支持多输出。

- Relay `Tuple` 输出。
- `split` 等多输出 op。
- `TupleGetItem` 进入 lowering。

R7.3 常量处理必须完整。

- 常量权重应进入 PrimFunc 参数绑定或模块常量区。
- Runtime 运行时应知道用户输入和常量参数的顺序。
- 不能要求用户手动把所有常量作为 runtime args 传入，除非 API 明确暴露。

R7.4 lowering 错误必须可诊断。

- 无 `FRelayToTE` 的 op 应报告 op name。
- 不支持的 Relay node 应报告 node kind。
- 输出不是 compute tensor 应报告产生该输出的 op。

### R8. TIR IR 和 Pass

R8.1 TIR 必须能表达基础 CPU kernel。

最低节点：

- `PrimFunc`
- `Buffer`
- `Load`
- `Store`
- `For`
- `IfThenElse`
- `Allocate`
- `SeqStmt`
- arithmetic/logical expr。

R8.2 Pass pipeline 必须稳定可配置。

- 支持默认 pipeline。
- 支持按名称运行单个 pass。
- 未知 pass 报明确错误。

R8.3 MVP TIR pass 覆盖：

- constant folding。
- simplify expr。
- remove no-op。
- loop unroll，可限定小 extent。
- vectorize 标注或实际 vector lowering 二选一，MVP 可先标注。

R8.4 完整版本需要 pass manager。

- Pass metadata。
- opt level。
- required passes。
- disabled passes。
- pass instrumentation，用于 profiling。

### R9. Schedule

R9.1 TE schedule primitive 必须真实生效。

最低 primitive：

- split。
- fuse。
- reorder。
- vectorize。
- unroll。
- parallel，可先只在 IR 标注。

R9.2 默认 schedule 必须按 target 选择。

- CPU elementwise 使用简单 contiguous loop。
- reduction 使用合理 loop order。
- conv2d MVP 可先 direct convolution，后续再加 im2col 或 tiled schedule。

R9.3 schedule 需要可视化和测试。

- schedule 前后 TIR 可打印。
- 同一 compute 不同 schedule 生成不同 TIR。

### R10. Codegen

R10.1 LLVM JIT 是 MVP 主执行后端。

必须支持：

- 后端私有 packed ABI：`int32_t kernel(void** packed_args, uint64_t count)`。
- buffer 参数从 `packed_args` 解包。
- scalar int/float constants。
- load/store。
- nested for。
- if。
- allocate。
- math intrinsic：`exp`、`log`、`sqrt`、`tanh`，至少 float32。

R10.2 LLVM codegen 必须支持完整 MVP TIR 节点。

- unsupported expr/stmt 不得静默生成错误代码。
- numeric test 必须覆盖 codegen。

R10.3 C codegen 必须有两层能力。

MVP：

- 生成可读 C 源码。
- 作为调试输出。

完整版本：

- 编译 C 源码为动态库或对象文件。
- 加载 symbol。
- 与 `CompiledKernel` 共用 ABI。

R10.4 CUDA codegen 作为完整版本增强。

最低目标：

- elementwise add/mul。
- 1D/2D grid mapping。
- CUDA kernel launch wrapper。
- 与 DeviceAPI 的 GPU NDArray 对接。

### R11. Runtime 和 Module API

R11.1 Runtime 必须提供清晰的编译产物模型。

```cpp
CompiledModule module = Compiler::Compile(func, config);
RuntimeSession session(module);
Array<NDArray> outputs = session.Run({input});
```

公共接口必须使用 `NDArray`、`KernelSignature` 和 `DeviceStream`，不得保留 `Run(std::vector<void*>)` 或公开后端函数/模块指针。后端私有 launcher 只允许在强类型校验之后完成 ABI 打包。

R11.2 Runtime 必须区分输入、权重和输出。

- 用户输入由用户设置。
- 权重来自 Relay constants 或 params。
- 输出由 runtime 分配或由用户传入，两种模式需明确。

R11.3 NDArray 必须支持基础数据操作。

- shape。
- dtype。
- device。
- data pointer。
- copy from/to host。
- fill 或从 vector 创建。

R11.4 Module 必须支持保存调试产物。

- Relay after passes。
- TIR after passes。
- C source。
- LLVM IR。
- profiling bundle。

R11.5 动态 shape specialization 作为增强功能。

完整版本需要：

- shape signature 正确包含所有输入。
- cache key 区分 target、dtype、shape、op graph hash。
- 只允许 exact contract 命中；不得用模糊 shape 匹配复用不兼容 kernel。
- 后台编译、失败传播和 module 热替换必须建立独立的所有权与并发契约。

### R12. DeviceAPI

R12.1 CPU DeviceAPI 必须稳定。

- aligned allocation。
- CPU to CPU copy。
- workspace allocation。
- basic device attrs。

R12.2 CUDA DeviceAPI 必须可选。

- CUDA 关闭时构建和 CPU 测试不受影响。
- CUDA 开启时支持 device allocation、copy、stream。

R12.3 Device 和 Target 必须贯穿 compile。

- Target 决定 codegen backend 和 schedule。
- Device 决定 runtime allocation 和 copy。

### R13. Profiling

R13.1 MVP 必须记录编译阶段 profiling。

- Relay pass 耗时。
- LowerToTIR 耗时。
- TIR pass 耗时。
- Codegen 耗时。

R13.2 完整版本必须记录运行阶段 profiling。

- kernel 执行耗时。
- cache hit/miss。
- Device copy。
- memory allocation。

R13.3 Python agent 必须能分析 bundle。

- `analyze_bundle`
- `compare_bundles`
- `inspect_pass_trace`
- `explain_logs`

输出应能定位失败 pass、慢 pass、缺失 kernel、cache miss。

### R14. Disco 多 Worker

R14.1 MVP 中 Disco 可保持实验模块。

- 不阻塞单机 CPU TinyTVM 闭环。
- 执行计划 JSON 序列化可测试。
- CPU CCL 模拟可测试。

R14.2 完整版本需要真实 kernel invocation。

当前 executor 中 kernel node 主要做 copy/占位输出。完整版本要求：

- `KernelExecNode` 编译或引用真实 `CompiledKernel`。
- 每个 worker 能执行对应 kernel。
- 通信 op 前后数据一致。

R14.3 NCCL backend 必须明确状态。

- 如果未实现，应在文档和 build option 中标注 experimental。
- 不应默认进入普通测试路径。

### R15. 测试

R15.1 测试分层必须清晰。

最低测试层级：

- object/container unit tests。
- Relay op registration tests。
- Type/shape inference tests。
- TOPI compute shape tests。
- Relay to TIR lowering tests。
- TIR pass tests。
- Codegen numeric tests。
- Compiler API end-to-end tests。
- ONNX importer integration tests。

R15.2 每个可执行 op 必须有 numeric test。

示例：

- 输入小 shape。
- 运行 compiled module。
- 与 numpy 或手写 reference 比较。
- 误差阈值按 dtype 设定。

R15.3 必须有模型级测试。

MVP：

- MLP。
- Conv block。
- ResNet18 smoke，可先检查 top-1 shape 和若干数值。

完整版本：

- 至少一个视觉模型。
- 至少一个 Transformer 静态子图。

R15.4 CI 必须先跑 CPU-only。

- CPU-only 是主线。
- LLVM 可作为 required 或 optional，由项目决定。
- CUDA 测试单独 job。

### R16. 文档和开发者体验

R16.1 必须有 README 主入口。

内容：

- 项目定位。
- 快速构建。
- 运行第一个 Relay add。
- 导入 ONNX。
- 查看生成 C/LLVM/TIR。

R16.2 必须有扩展指南。

- 新增 op。
- 新增 attrs。
- 新增 type relation。
- 新增 TOPI compute。
- 新增 pass。
- 新增 codegen node。

R16.3 必须有当前能力矩阵。

建议维护 `docs/OP_SUPPORT_MATRIX.md`：

| Op | Register | Type Infer | FRelayToTE | Lower | LLVM Run | Test |
|---|---|---|---|---|---|---|

R16.4 错误排查文档必须覆盖常见失败。

- CMake toolchain。
- LLVM not found。
- missing FRelayToTE。
- unsupported TIR stmt。
- runtime args order mismatch。
- ONNX unsupported op。

## 6. 非功能需求

### N1. 正确性

- 所有 numeric tests 必须与 reference 对齐。
- 不允许用 placeholder 实现返回错误结果。
- 未实现功能必须显式失败。

### N2. 可维护性

- op 名称、attrs、type relation、FRelayToTE 应集中可查。
- pass pipeline 不应散落隐式顺序。
- docs 与测试应随功能更新。

### N3. 可观测性

- 编译失败应能定位到 stage、op、pass 或 TIR node。
- profiling bundle 应能复现关键路径。

### N4. 可移植性

- CPU-only 应跨 Windows/Linux。
- CUDA、LLVM、C backend 都应可通过 CMake option 隔离。

### N5. 性能

MVP 不追求极致性能，但必须避免明显不可用：

- ResNet18 CPU 推理可完成，不应因为指数级 IR 或极端低效 lowering 卡死。
- 编译时间应可通过 profiling 定位。

## 7. 当前主要缺口

本节源自 2026-06-04 repo 盘点，并在 2026-07-20 对已完成的构建、Compiler/Codegen 和 RuntimeSession 状态做了更新：

1. Windows MinGW CPU/LLVM 与 omen CUDA 矩阵已可运行；GitHub Workflow 当前手动关闭，仍缺少自动远端门禁。
2. 算子注册和 lowering 不一致：很多 op 只注册元数据，没有 `FRelayToTE`。
3. op name 不统一：`mul/multiply`、`softmax/nn_softmax`、`sub/subtract` 存在混用。
4. Type/shape inference 不完整，很多链路依赖手动 TensorType。
5. ONNX importer 仍是实验脚本，权重数据没有形成稳定 params/runtime 绑定。
6. TE/TOPI 中有 placeholder 或错误替代实现，例如 `einsum` 返回空 Tensor、`min/prod` 使用 sum 占位。
7. `LowerToTIR` 只支持单输出 compute tensor，不支持多输出和完整常量绑定。
8. C emitter 只生成诊断源码，尚未形成可加载 AOT artifact。
9. LLVM/CUDA 已接入 Compiler 主链路，但支持的 TIR 节点仍需按模型扩展。
10. RuntimeSession 已提供强类型 inputs、常量绑定和静态输出分配；动态输出和 module registry 尚未实现。
11. Disco executor 当前不执行真实 kernel，主要是 copy/占位行为。
12. 测试更偏 smoke，需要补齐 per-op numeric 和模型级端到端测试。

## 8. 里程碑

### M0. 构建可用

目标：任何开发者能在 CPU-only 环境构建和运行测试。

交付：

- 修复或补充 Windows CPU-only preset。
- 添加 MinGW 或 MSVC 工具链说明。
- 跑通 `pass_pipeline_test`、`profile_bundle_test`。

### M1. Relay Add 端到端稳定

目标：最小编译闭环稳定。

交付：

- 手写 Relay add。
- Type/shape infer。
- Relay pass。
- LowerToTIR。
- LLVM JIT。
- Runtime numeric test。

验收：

- `Compiler::Compile(add).Run()` 输出正确。
- 保存 Relay/TIR/C source/LLVM IR。

### M2. MVP 算子覆盖

目标：支持 MLP 和 CNN block。

交付：

- 完成 MVP op 的 `FRelayToTE`。
- 完成 type relation。
- 完成 per-op numeric tests。

验收：

- dense + relu 可运行。
- conv2d + relu + pool 可运行。
- global_avg_pool + flatten + gemm 可运行。

### M3. ONNX ResNet18 闭环

目标：ResNet18 从 ONNX 导入到 CPU 运行。

交付：

- 正式 ONNX importer。
- initializer 数据加载。
- params 绑定。
- ResNet18 integration test。

验收：

- 能从 `resnet18.onnx` 构建 Relay Function。
- 能编译到 LLVM JIT。
- 能用固定输入运行并与参考结果比较。

### M4. Runtime 易用 API

目标：从完整有序 CompiledModule 参数过渡到只传 inputs 的会话接口。

交付：

- `RuntimeSession::Run/RunAsync`。
- KernelSignature 参数顺序管理。
- 常量权重自动绑定。
- 静态输出自动分配和 NDArray host copy。

验收：

- 用户无需手写 `void*` args 顺序即可运行模型。

### M5. C AOT 和 Schedule

目标：让 TinyTVM 具备教学友好的 AOT 能力和真实 schedule 影响。

交付：

- C source 编译为动态库。
- schedule primitives 影响 TIR。
- TIR before/after schedule snapshot tests。

### M6. 增强后端和分布式实验

目标：保留研究扩展空间。

交付：

- CUDA elementwise/matmul prototype。
- Disco kernel node 真实调用。
- NCCL backend 状态明确。

## 9. 验收清单

完整 TinyTVM 第一版完成时，必须满足：

- CPU-only 构建无外部 GPU 依赖。
- 至少 20 个核心 Relay op 有 type infer、FRelayToTE、lowering、numeric test。
- 手写 Relay add、MLP、CNN block 端到端通过。
- ONNX ResNet18 可导入、编译、运行。
- C source 可导出，LLVM JIT 可执行。
- Runtime 提供非 packed args 的用户 API。
- profiling bundle 能记录编译和运行关键阶段。
- 文档包含 quickstart、op support matrix、new op guide、troubleshooting。
- 未实现功能显式报错，不返回错误占位结果。

## 10. 风险

1. LLVM/Windows 工具链配置可能拖慢主线，建议优先保证 MinGW 或 Linux CPU-only CI。
2. ResNet18 direct conv CPU 性能可能较慢，MVP 验收可先用小输入或子图，完整版本再优化。
3. 如果没有 type/shape inference，新增算子会持续依赖手工类型，端到端复杂度会快速失控。
4. 如果 op name 不统一，ONNX importer、helper 和 lowering 会反复出现“注册了但不能 lower”的问题。
5. 如果常量权重绑定不解决，ONNX 模型运行 API 会难以使用。

## 11. 建议下一步

优先顺序：

1. 修复 CPU-only 构建和测试入口。
2. 建立 op support matrix，统一 canonical op name。
3. 实现 type/shape inference pass 的 MVP 子集。
4. 补齐 MVP 算子的 `FRelayToTE` 和 numeric tests。
5. 正式化 ONNX importer，并处理 initializer 数据。
6. 在已完成的强类型 RuntimeSession 基础上实现动态输出 shape function 和 ExecutionPlan module registry。
