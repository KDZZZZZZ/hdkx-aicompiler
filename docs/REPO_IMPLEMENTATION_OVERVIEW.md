# hdkx-aicompiler 实现导览（历史快照）

> **状态：已归档。** 本文混有旧目录、whole-graph lowering 和单-entry
> RuntimeSession 描述，不再是当前实现入口。当前生产路径以
> `Compiler::Compile -> per-unit LowerGraph -> CompiledModule + ExecutablePlan -> RuntimeSession`
> 为准；当前边界见 [`ARCHITECTURE_STATUS.md`](ARCHITECTURE_STATUS.md)，历史基线审查见
> [`COMPILER_FOUNDATION_BASELINE_REVIEW.md`](COMPILER_FOUNDATION_BASELINE_REVIEW.md)。

本文档保留历史实现导览。

> 想沿着真实代码系统学习现代高级 C++，请打开 [`docs/cpp-tutorial/index.html`](cpp-tutorial/index.html)。教程以本仓库 C++17 实现为基线，并包含明确标注的 C++20/23 对照实验。

## 1. 项目定位

这个 repo 是一个轻量级 AI compiler/runtime 原型，整体设计接近 TVM 的分层方式：

```text
ONNX/手写模型
  -> Relay Function
  -> Relay pass pipeline
  -> Relay lowering to TE
  -> TE tensor DAG lowering to TIR PrimFunc
  -> TIR pass pipeline
  -> Codegen LLVM/CUDA
  -> CompiledKernel
  -> RuntimeSession / DeviceAPI / Disco execution
```

当前实现重点在：

- 自定义 Object/ObjectRef 对象系统。
- Relay 表达式 IR、算子注册和 Relay pass。
- TE/TOPI 风格的 tensor compute。
- TIR 表达式、语句和 TIR pass。
- Relay 到 TIR 的 lowering。
- LLVM ORC JIT 与 CUDA NVRTC/Driver codegen。
- 强类型 RuntimeSession，包括输入校验、常量绑定、静态输出分配和同步/异步执行。
- CPU/CUDA DeviceAPI。
- Disco 风格的多 worker 执行计划和 CPU CCL 模拟。
- profiling bundle 和 Python 离线分析工具。

## 2. 构建入口

主构建文件是 `CMakeLists.txt`，核心产物是静态库 `kxc_runtime`，以及若干测试/调试可执行文件。

关键开关：

| CMake 选项 | 默认值 | 作用 |
|---|---:|---|
| `KXC_ENABLE_CUDA` | `ON` | 检测 CUDA Toolkit，决定是否启用 CUDA DeviceAPI、NVRTC/Driver codegen 和 CUDA 测试 |
| `KXC_ENABLE_LLVM` | `ON` | 检测 LLVM，决定是否启用 LLVM codegen/JIT |
| `KXC_BUILD_RESNET18_IR_DUMP` | `OFF` | 显式构建 ResNet18 IR dump 诊断工具 |
| `KXC_BUILD_PASS_TESTS` | `ON` | 构建 pass 相关测试 |
| `KXC_BUILD_CODEGEN_TESTS` | `ON` | 在 LLVM 可用时构建 LLVM codegen 测试；CUDA codegen 测试随 pass 测试和 CUDA 特性构建 |

`KXC_RUNTIME_SOURCES` 汇总了 runtime 静态库使用的 C++ 源文件。CUDA 和 LLVM 通过编译定义 `KXC_USE_CUDA=<0|1>`、`KXC_USE_LLVM=<0|1>` 向源码暴露。

`CMakePresets.json` 提供了 Ninja debug 配置：

- `dev-ninja`：Debug，CUDA ON。
- `dev-ninja-cpu`：继承 `dev-ninja`，CUDA OFF。
- `dev-mingw-cpu`：继承 CPU 配置，使用 MinGW 且关闭 LLVM。

## 3. 顶层目录

| 路径 | 职责 |
|---|---|
| `include/` | 公共头文件，按 base/relay/te/tir/codegen/runtime/api 分层 |
| `src/` | C++ 实现，目录结构基本对应 `include/` |
| `test/` | C++ 测试、IR dump 示例、codegen smoke test |
| `python/` | ONNX 分析/代码生成脚本，以及 profiling agent 工具 |
| `docs/` | 设计文档、专题说明、模型报告、本文档 |
| `resnet18.onnx` | 本地用于导入/分析的 ResNet18 模型 |

## 4. 基础对象系统

核心文件：

- `include/base/object.h`
- `include/base/container.h`
- `include/base/expr.h`
- `include/base/packedfunc.h`
- `include/base/registry.h`
- `src/base/pass.cc`

### 4.1 Object 和 ObjectRef

所有 IR 节点、attrs、runtime 对象都继承 `Object`，再用轻量 handle 类继承 `ObjectRef` 对外持有。

`Object` 提供：

- 引用计数：`IncRef()` / `DecRef()`。
- 运行时类型 ID：`GetTypeId()`。
- 可选属性访问：`VisitAttrs()`。
- arena-aware `operator new`，使用 `thread_local Arena* current_arena`。

`ObjectRef` 提供：

- 引用计数式生命周期管理。
- `defined()` 判断。
- `As<T>()` 动态向下转换。
- 拷贝/移动语义。

类型注册通过两个宏完成：

```cpp
KXC_OBJECT_DECLARE
KXC_OBJECT_DEFINE(TypeName)
```

`TypeRegistry::Register(name)` 为每类对象分配 type index。当前实现偏简单，模板容器的 type id 共享同一个 `"Array"` 或 `"Map"` 注册名。

### 4.2 Array / Map / String

`include/base/container.h` 实现了 ObjectRef 包装的容器：

- `Array<T>`：内部是 `ArrayNode<T>`，持有 `std::vector<T>`。
- `Map<K,V>`：内部是 `MapNode<K,V>`，持有 `std::unordered_map<K,V>`。
- `String`：ObjectRef 包装的字符串。

这些容器用于 IR 节点字段，让 Relay/TIR/TE 节点都能被统一对象系统管理。

### 4.3 PackedFunc 和 Registry

`Registry` 是全局函数注册表，注册宏是：

```cpp
KXC_REGISTER_GLOBAL("name").set_body(...)
```

它用于把 C++ pass、runtime API、device API、disco API 暴露成字符串可查的函数。`ToPackedFunc` 把 C++ lambda 包装成统一调用接口。

典型注册点：

- `src/relay/transforms/pipeline.cc`
- `src/tir/transforms/pipeline.cc`
- `src/base/device_api.cc`
- `src/base/disco/*.cc`

## 5. Relay IR 和算子系统

核心文件：

- `include/relay/relay.h`
- `include/relay/op.h`
- `include/relay/op_attr_types.h`
- `include/relay/op_macros.h`
- `src/relay/relay.cc`
- `src/relay/op_attrs.cc`
- `src/relay/op_macros.cc`
- `src/relay/op/**`

### 5.1 Relay 节点

Relay 表达式节点继承自 `RelayNode`，而 `RelayNode` 又继承 `ExprNode`。当前主要节点：

| 节点 | 作用 |
|---|---|
| `TensorTypeNode` | Tensor shape/dtype 类型信息 |
| `VarNode` | 函数参数或局部变量 |
| `ConstantNode` | 持有 `runtime::NDArray` 的常量 |
| `CallNode` | 调用算子或函数 |
| `FunctionNode` | Relay 函数，包含 params/body |
| `TupleNode` | 元组 |
| `TupleGetItemNode` | 元组取项 |
| `IfNode` | 条件表达式 |
| `LetNode` | let binding |

`RelayNode` 上有 `virtual_device_`，用于后续设备规划、多设备 placement 和 memory scope 标注。

### 5.2 Op 和 Attrs

Relay 算子用 `relay::Op` 表示，内部 `OpNode` 包含：

- `name`
- `description`
- `arguments`
- `num_inputs`
- `attrs`

`attrs` 是 `std::unordered_map<std::string, std::any>`，当前重要 key 包括：

- `TAttrs`：attrs 类型名。
- `FRelayToTE`：Relay 算子 lowering 到 TE 的函数。

算子属性类集中在 `include/relay/op.h`，例如：

- `Conv2DAttrs`
- `DenseAttrs`
- `MaxPool2DAttrs`
- `SoftmaxAttrs`
- `ReshapeAttrs`
- `TransposeAttrs`
- `DeviceCopyAttrs`
- `CollectiveAttrs`

### 5.3 算子注册

算子通过 `KXC_REGISTER_OP(name)` 注册。注册实现由 `OpRegEntry` 修改 `OpNode` 字段和 attrs。

例如 `src/relay/op/tensor/math.cc` 中 `add` 注册了 `FRelayToTE`：

```cpp
KXC_REGISTER_OP(add)
    .describe(...)
    .set_num_inputs(2)
    .set_attr<FRelayToTE>("FRelayToTE", AddCompute);
```

`LowerToTIR` 遇到 `CallNode` 时会查 `Call.op.attrs["FRelayToTE"]`，执行它得到 TE tensor。

算子的 schema、type hook 和 lowering hook 统一放在 `src/relay/op/**` 的唯一注册块中；不再保留 metadata-only 的公共注册文件或历史别名。

## 6. TE 和 TOPI 层

核心文件：

- `include/te/te.h`
- `src/te/te.cc`
- `include/te/topi/*.h`

TE 是 Relay lowering 到 TIR 之间的 tensor expression 层。

主要对象：

| 类型 | 作用 |
|---|---|
| `te::Tensor` | 一个 tensor 值，包含 shape、dtype、producer op |
| `te::Operation` | producer 抽象基类 |
| `PlaceholderOp` | 输入/常量占位 |
| `ComputeOp` | 由 shape 和 compute lambda 定义的 tensor |
| `ProducerLoad` | 在 TE 表达式中读取其他 tensor |
| `Reduce` | reduction 表达式 |
| `Schedule/Stage` | 初步 schedule 数据结构，包含 split/fuse/vectorize/unroll 等接口 |

TOPI 层在 `include/te/topi/` 中，用 TE 表达常见算子：

- `broadcast.h`：broadcast/elemwise 相关。
- `elemwise.h`：基础元素级算子。
- `nn.h`：卷积、pooling、dense、softmax 等。
- `reduction.h`：reduce。
- `transform.h`：reshape/transpose/concat 等。

当前 lowering 对 schedule 的利用有限，主要是把 TE compute DAG 转成简单 loop nest。

## 7. TIR IR

核心文件：

- `include/tir/expr.h`
- `include/tir/stmt.h`
- `src/tir/expr.cc`
- `src/tir/stmt.cc`
- `include/tir/pass_utils.h`
- `src/tir/pass_utils.cc`
- `src/tir/pass/print_ir.cc`

### 7.1 PrimExpr

`tir::PrimExpr` 表示标量表达式。主要节点：

- `IntImm` / `FloatImm`
- `Var`
- 二元算子：`Add/Sub/Mul/Div/Mod/Min/Max`
- 逻辑算子：`EQ/LT/And/Or/Not`
- `Load`
- `Call`
- `Select`

`DataType` 用 `{code, bits, lanes}` 表示 dtype，类似 DLDataType。

### 7.2 Stmt 和 PrimFunc

`tir::Stmt` 表示语句。主要节点：

- `LetStmt`
- `Store`
- `For`
- `IfThenElse`
- `Allocate`
- `AttrStmt`
- `Block`
- `SeqStmt`
- `Evaluate`

`PrimFunc` 是 TIR 函数，字段包括：

- `params`：函数参数变量。
- `body`：函数体语句。
- `buffer_map`：参数变量到 buffer 的映射。
- `attrs`：如 `global_symbol`、`tir.noalias`、pass context 属性等。

### 7.3 TIR visitor/pass 基类

`include/base/pass.h` 提供：

- `TIRExprFunctor<R>`
- `TIRStmtFunctor<R>`
- `TIRPass`

各 TIR pass 继承 `TIRPass` 后覆写感兴趣的 visit 方法。

## 8. Pass 系统

核心文件：

- `include/base/pass.h`
- `src/base/pass.cc`
- `include/relay/transforms/*.h`
- `src/relay/transforms/*.cc`
- `include/tir/transforms/*.h`
- `src/tir/transforms/*.cc`

### 8.1 PassContext

`PassContext` 存储设备和 placement 元数据：

- 是否多设备。
- primary virtual device。
- virtual device 列表。
- default target。
- disco placement。

`PassContext::Scope` 通过线程本地状态设置当前 context。`LowerToTIR` 会从 Relay function 推断或读取当前 context，并把 context 属性附到 `PrimFunc.attrs`。

### 8.2 Relay pass pipeline

入口：

- `relay::RunRelayPassPipeline(func, pass_names)`
- 全局注册名：`kxc.relay.transform.run_pipeline`

默认 pass 顺序：

1. `fold_tuple_get_item`
2. `fold_constant`
3. `simplify_expr`
4. `canonicalize_cast`
5. `remove_standalone_reshapes`
6. `eliminate_dead_let`
7. `annotate_memory_scope`
8. `capture_post_dfs_index_in_spans`
9. `infer_type`

`optimize_default` 是特殊 pass name，会展开为上面这组顺序。`eliminate_common_subexpr`
仍可显式调用，但因结构键尚不包含 Constant 内容和完整 Call attrs，不进入默认链。

### 8.3 TIR pass pipeline

入口：

- `tir::RunTIRPassPipeline(func, pass_names)`
- 全局注册名：`kxc.tir.transform.run_pipeline`

默认 pass 顺序：

1. `fold_constant`
2. `simplify_expr`
3. `force_narrow_index_to_i32`
4. `convert_for_loops_serial`
5. `loop_partition`
6. `unroll_loop`
7. `vectorize_loop`
8. `remove_no_op`

当前 pass pipeline 是静态表驱动，不是通用 pass manager。加新 pass 需要：

1. 新增 `include/.../transforms/new_pass.h`。
2. 新增 `src/.../transforms/new_pass.cc`。
3. 在对应 `pipeline.cc` include 头文件。
4. 加入 pass table。
5. 如需默认启用，加入 default order。
6. 如需外部调用，注册 `KXC_REGISTER_GLOBAL`。

## 9. Relay 到 TIR 的 Lowering

核心文件：

- `include/relay/transforms/lower.h`
- `src/relay/backend/lower.cc`

入口：

```cpp
tir::PrimFunc LowerToTIR(Function func);
```

实现步骤：

1. 检查输入 `Function` 和 body。
2. 创建或读取当前 `PassContext`。
3. `RelayToTEConverter` 遍历 Relay 表达式：
   - 函数参数转 `te::placeholder`。
   - 常量转 `te::placeholder`，并记录到 `constant_tensors_`。
   - `CallNode` 查找 `FRelayToTE`，执行得到输出 `te::Tensor`。
   - `device.*` 通信 op 不进入普通 TIR lowering，要求先走 execution plan lowering。
4. 对输出 tensor 做 DFS，收集 TE operation 拓扑序。
5. 为输入、常量、输出和中间 tensor 分配 TIR buffer var。
6. `ExprLowerer` 把 TE 表达式转 TIR 表达式：
   - `ProducerLoad` 转 `tir::Load`。
   - TE 中的二元/逻辑表达式递归变换。
   - `Reduce` 留到 statement 层处理。
7. `LowerComputeStmt` 把每个 `ComputeOp` 转成：
   - 普通 compute：嵌套 data loops + `Store`。
   - reduce：先初始化输出，再加 reduction loops 做更新。
8. 中间 tensor 用 `tir::Allocate` 包住 body。
9. 构造 `tir::PrimFunc(params, body, buffer_map, attrs)`。

当前限制：

- 只支持单输出 function。
- 输出必须是 compute tensor。
- reduce 当前只支持单 source。
- 普通 lowering 不处理 `device.*` 通信 op。
- 常量 tensor 被建为参数/placeholder，数据内容不在 lowering 中内嵌。

## 10. Codegen

核心文件：

- `include/codegen/backend.h`
- `include/codegen/kernel_signature.h`
- `include/codegen/compiled_kernel.h`
- `include/codegen/codegen_c.h`
- `include/codegen/codegen_llvm.h`
- `include/codegen/llvm_jit.h`
- `include/codegen/codegen_cuda.h`
- `include/codegen/cuda_module.h`
- `src/codegen/kernel_signature.cc`
- `src/codegen/compiled_kernel.cc`
- `src/codegen/codegen_c.cc`
- `src/codegen/codegen_llvm.cc`
- `src/codegen/llvm_jit.cc`
- `src/codegen/codegen_cuda.cc`
- `src/codegen/cuda_module.cc`

### 10.1 后端类型

`CodeGenBackend` 当前定义：

- `kLLVM`：LLVM IR 经 ORC JIT 生成 CPU 机器码。
- `kCUDA`：CUDA C 经 NVRTC 生成 PTX，再由 Driver API 加载 GPU 内核。

公共调用契约不由函数指针或后端原生句柄定义，而由以下对象共同冻结：

- `KernelSignature`：symbol 及 input/constant/output 的有序 `KernelArgSpec`；
- `KernelLaunchMetadata`：执行设备、后端和 CUDA grid/block；
- `CompiledKernel`：强持有上述契约和后端私有 `KernelLauncher`。

`CSourceEmitter` 仍可把 PrimFunc 生成可读 C 源码，用于诊断和教学；它不是 `CodeGenBackend`，也没有接入可执行模块装载路径。

### 10.2 LLVM codegen 与 JIT

`CodeGenLLVM` 把优化后的 `tir::PrimFunc` 转为 LLVM Module。`LLVMJITEngine` 使用 ORC LLJIT 优化、验证并查找 `KernelSignature.symbol`，返回强类型 `CompiledKernel`。launcher 独占 module、context、函数地址和 LLJIT 生命周期；裸函数地址不会穿过 codegen 边界。

LLVM launcher 在调用线程同步完成内核执行，再返回已经完成的 `AsyncOperation`。

### 10.3 CUDA codegen 与模块加载

CUDA Target 先经 `BindCudaThreads` 建立启动维度，再由 `CodeGenCUDA` 生成 CUDA C。`CUDAModule` 使用 Target 的 compute capability 选择 NVRTC architecture，编译 PTX、加载 `CUmodule` 并解析入口 symbol。

CUDA launcher 仅在后端内部把已校验 NDArray 转成 Driver 参数数组。`AsyncOperation` 同时保活参数 Storage、stream 和 `CompiledKernel`，GPU 工作完成前不会卸载 `CUmodule`。

### 10.4 CompiledKernel 边界

`CompiledKernel::Launch(Array<NDArray>, DeviceStream)` 是唯一内部启动入口。公共层不暴露 `func_ptr`、`module_ptr` 或 `std::vector<void*>`；仅 LLVM/CUDA launcher 可以在完成强类型校验后进行后端 ABI 打包。

## 11. 编译 API 主链路

核心文件：

- `include/api/compile_config.h`
- `include/api/compiler.h`
- `src/api/compile_config.cc`
- `src/api/compiler.cc`

### 11.1 CompileConfig

`CompileConfigNode` 只包含：

- `target`：backend kind、物理设备和能力快照的唯一事实来源；
- `opt_level`：0 到 3 的确定性 Relay/TIR pass 策略；
- `profile_options`：编译阶段 profiling 与 artifact 捕获配置。

调用方统一使用 `CompileConfig::Create(target, opt_level)`。配置中不存在 AOT/JIT/Adaptive mode，也不携带 RuntimeSession、cache 或后台线程选项。

### 11.2 Compiler::Compile

Compiler 执行七个步骤；前六步由单向 `CompileResult` 状态机承载，最后的 assemble 在状态机完成后构造 `CompiledModule`：

```text
validate
  -> optimize_relay
  -> lower
  -> optimize_tir
  -> build_signature
  -> build_backend
  -> assemble
```

`opt_level` 只选择 pass 集合。`build_backend` 完全按 Target dispatch：`llvm` + CPU 进入 LLVM ORC JIT，`cuda` + CUDA Device 进入 CUDA emitter、NVRTC 和 Driver API。缺少对应构建特性时返回明确错误，Compiler 不创建 RuntimeSession。

### 11.3 CompiledModule

`CompiledModule` 一次性持有 Target、优化后 TIR、`KernelSignature`、`KernelLaunchMetadata`、常量表、`CompiledKernel` 和可选 profiling context。`Launch` 只接受按签名完整排序的 `Array<NDArray>` 与同 Device 的 `DeviceStream`，在进入后端前校验数量、dtype、shape、Device、连续布局、Storage range、alignment 和常量对象身份。

## 12. RuntimeSession

核心文件：

- `include/runtime/runtime_session.h`
- `src/runtime/runtime_session.cc`

### 12.1 RuntimeSession

`RuntimeSessionNode` 只强持有一个 ready `CompiledModule`，不持有 Compiler、cache、默认 stream、后台线程或可变调用状态。

- `Run(inputs)`：使用目标设备默认 stream，自动绑定模块常量、分配静态输出并等待完成；
- `RunAsync(inputs, stream)`：在显式同 Device stream 上提交，返回 `RunAsyncResult { outputs, completion }`；
- 输入在输出分配前完成校验，参数最终严格按 `KernelSignature` 顺序组装；
- completion 保活参数 Storage 和 stream；尚未完成的 CUDA launch 还保活后端 executable。

当前不实现动态输出 shape function、shape specialization、exact/fuzzy cache、后台重编译、热替换或 ExecutionPlan module registry。这些能力若后续引入，必须建立新契约，不能恢复旧 Adaptive Runtime 的裸参数接口。

## 13. Device、Target 和 NDArray

核心文件：

- `include/base/device.h`
- `include/base/device_api.h`
- `include/base/ndarray.h`
- `include/base/target.h`
- `include/base/virtual_device.h`
- `src/base/device.cc`
- `src/base/device_api.cc`
- `src/base/device/cpu_device_api.cc`
- `src/base/device/cuda_device_api.cc`
- `src/base/ndarray.cc`
- `src/base/target.cc`
- `src/base/virtual_device.cc`

### 13.1 Device

`Device` 是具有值语义的 ObjectRef，节点包含：

- `DeviceTypeCode`：CPU/CUDA/OpenCL/Metal/Unknown。
- `device_id`。

`DeviceManager` 按 `(type, id)` 规范驻留节点；`Device(type, id)`、`Device::CPU` 和 `Device::CUDA` 都返回该规范对象。

### 13.2 DeviceAPI

`DeviceAPI` 是设备 backend 抽象，提供：

- `SetDevice`
- `AllocDataSpace`
- `FreeDataSpace`
- `CopyDataSync` / `CopyDataAsync`
- `ZeroData`
- `GetDeviceAttributes`
- stream / event 相关接口

`DeviceAPIManager` 按 `DeviceTypeCode` 懒加载 backend：

- `kCPU` -> `GetCPUDeviceAPI()`
- `kCUDA` -> `GetCUDADeviceAPI()`

CPU backend：

- 使用 `_aligned_malloc` 或 `posix_memalign`。
- 只支持 CPU 到 CPU copy。
- target kind 返回 `"llvm"`。

CUDA backend：

- 编译期开关 `KXC_USE_CUDA`。
- 支持 `cudaMalloc/cudaFree`。
- 支持 Host/Device/Device copy。
- `DeviceStream` 显式持有 stream 所属 Device 和后端 handle。
- 提供默认/自有 stream、event 记录/查询/等待和显式 sync，不维护 thread-local current stream。
- target kind 返回 `"cuda"`。

### 13.3 NDArray

`runtime::NDArray` 是持有 `NDArrayNode` 的 ObjectRef：

- 节点持有 `DLTensor` 只读视图、shape/strides 元数据和 `Storage`；
- `NDArray::Empty/Zeros` 通过 Storage/DeviceAPI 按 alignment 分配；
- view 共享原 Storage，并校验连续布局、byte offset 和容量；
- 同步/异步 copy 统一经 Storage 边界，异步结果用 `AsyncOperation` 保活两端 Storage。

当前 dtype 支持集中在各使用点做字符串/DLDataType 转换，例如 `"float32"`、`"int64"`。

### 13.4 Target 和 VirtualDevice

`Target` 由 `BuildTarget(Device)` 创建：

- CPU target kind 通常是 `"llvm"`。
- CUDA target kind 是 `"cuda"`。
- 包含设备属性和 arch。

`VirtualDevice` 用于 Relay pass 和多设备规划，字段：

- `device`
- `target`
- `memory_scope`
- `virtual_device_id`

`AnnotateMemoryScopePass` 会补默认 memory scope，例如变量走 `global`，常量走 `const`。

## 14. Disco 多 worker 执行

核心文件：

- `include/base/disco_placement.h`
- `include/base/execution_plan.h`
- `include/base/disco/session.h`
- `include/base/disco/dref.h`
- `include/base/disco/executor.h`
- `include/base/disco/ccl_backend.h`
- `src/base/disco_placement.cc`
- `src/base/execution_plan.cc`
- `src/base/disco/*.cc`

### 14.1 Placement

`WorkerPlacement` 描述 worker 到 device/target/virtual device 的映射。

`DiscoPlacement` 维护：

- `workers`
- `vd_to_worker`
- `num_groups`

`BuildDiscoPlacement` 从 virtual devices 生成 worker placement。

### 14.2 ExecutionPlan

`ExecutionPlan` 是多设备执行计划：

- `nodes`：执行节点列表。
- `value_virtual_devices`：值到 virtual device。
- `input_value_ids`
- `constant_value_ids`
- `value_shapes`
- `value_dtypes`
- `num_values`
- `pass_ctx`
- `output_value`

执行节点分三类：

- `KernelExecNode`：计算节点，含 `op_name`、`kernel_symbol`、`primfunc`。
- `CommExecNode`：通信节点，含 `op_name` 和 attrs。
- `BarrierExecNode`：同步节点。

`ExecutionPlan` 支持 JSON 序列化/反序列化，便于落盘和调试。

### 14.3 DiscoSession 和 DRef

`DiscoSession` 是多 worker 抽象。当前实现是 `ThreadedSession`：

- 每个 worker 有 register file。
- `AllocateRegister()` 分配分布式寄存器编号。
- `DRef` 表示一个分布式值引用。
- `Get/Set(worker_id, DRef)` 读写某 worker 的数组值。

### 14.4 CCLBackend

`CCLBackend` 抽象通信原语：

- `Copy`
- `AllReduce`
- `BroadcastFromWorker0`
- `ScatterFromWorker0`
- `GatherToWorker0`
- `SendToWorker`
- `RecvFromWorker`
- `SyncWorker`

当前 `CpuCCLBackend` 是内存复制和 CPU 数组操作模拟：

- `Copy` 克隆 NDArray。
- `AllReduce` 目前只支持 sum。
- `Scatter/Gather` 按第 0 维切分/拼接。

`NcclCCLBackend` 目前是占位，所有方法抛出未实现。

### 14.5 ExecutionPlanExecutor

`ExecutionPlanExecutor::Execute(plan, initial_values)` 按 plan 顺序解释执行：

- Kernel node 当前主要做值复制或占位输出分配，并没有真正编译/调用 primfunc。
- Comm node 调 CCLBackend 的通信方法。
- Barrier node 调 `SyncWorker`。

所以 Disco 目前更像执行计划和通信语义验证层，不是完整分布式 kernel runtime。

## 15. Profiling 系统

核心文件：

- `include/base/profiling.h`
- `src/base/profiling.cc`
- `python/kxc_agent/**`
- `docs/PROFILING_AGENT_SYSTEM_V1.md`

### 15.1 C++ 侧

`ProfileContext` 管理一次 profiling bundle。它记录：

- span/event。
- log。
- artifacts，例如 pass 前后 IR。
- Perfetto/Chrome trace 事件。
- diagnostics 占位。

核心 RAII 类型：

- `ActivationScope`：在线程本地状态安装 `ProfileContext` 和 run id。
- `ScopedSpan`：析构时记录一次 duration event。

bundle 文件：

- `manifest.json`
- `events.jsonl`
- `trace.json`
- `summary.json`
- `diagnosis.json`
- `diagnosis.md`
- `artifacts/`

环境变量覆盖：

- `KXC_PROFILE_ENABLE`
- `KXC_PROFILE_BUNDLE_DIR`
- `KXC_PROFILE_LOG_LEVEL`
- `KXC_PROFILE_IR_MODE`
- `KXC_PROFILE_NVTX`
- `KXC_PROFILE_CUPTI`
- `KXC_PROFILE_RECORD_PASS_IR`
- `KXC_PROFILE_EXEC_PLAN_DETAILS`

采集点覆盖：

- `Compiler::Compile`
- Relay/TIR pass pipeline
- `LowerToTIR`
- `ExecutionPlanExecutor`
- CUDA CUPTI activity

### 15.2 Python agent

`python/kxc_agent/` 是离线分析控制面：

- `cli.py`：命令行入口。
- `services/bundle_loader.py`：加载 bundle。
- `services/schema.py`：校验 event schema。
- `services/diagnosis_engine.py`：规则分析。
- `tools/profile_run.py`：设置环境变量并运行目标命令。
- `tools/analyze_bundle.py`：分析单个 bundle。
- `tools/compare_bundles.py`：比较两个 bundle。
- `tools/inspect_pass_trace.py`：查看 Relay/TIR pass 耗时和 IR hash。
- `tools/explain_logs.py`：提取 log event。
- `memory/store.py`：本地记录 bundle 分析历史。

## 16. Python ONNX 工具

路径：

- `python/kxc_onnx/`
- `python/gen_resnet18_ir_dump_cpp.py`

`kxc_onnx` 是正式的 ONNX 解析、规格生成和参数导出入口。`gen_resnet18_ir_dump_cpp.py`
复用正式 importer spec 生成 `test/resnet18_ir_dump.cpp` 风格的诊断代码；它不是另一套 importer，
也不进入默认构建。旧的平行分析/报告/构图脚本和其版本控制生成物已经删除。

## 17. 测试和调试入口

主要测试：

| 文件 | 作用 |
|---|---|
| `test/pass_pipeline_test.cpp` | Relay/TIR pass 单元测试和 pipeline 顺序测试 |
| `test/codegen_llvm_test.cpp` | TIR/Relay 到 LLVM JIT 的 smoke test |
| `test/codegen_cuda_test.cpp` | CUDA source、NVRTC/Driver launch 和 RuntimeSession 数值测试 |
| `test/compiled_module_test.cpp` | CompiledModule 强类型参数、常量和启动契约测试 |
| `test/runtime_session_test.cpp` | RuntimeSession 输入校验、装配、输出和并发局部状态测试 |
| `test/object_test.cpp` | ObjectRef 类型、引用计数、Arena 和跨线程发布测试 |
| `test/resnet18_ir_dump.cpp` | 手写/生成 ResNet18 Relay graph，输出 Relay/TIR 文本 |
| `test/profile_bundle_test.cpp` | profiling bundle 结构和事件 smoke test |
| `test/cupti_smoke_test.cpp` | CUDA/CUPTI activity 采集 smoke test，CUDA 开启时构建 |

CMake custom targets：

- `run_pass_pipeline_test`
- `run_codegen_llvm_test`
- `run_resnet18_ir_dump`
- `run_profile_bundle_test`
- `run_cupti_smoke_test`

常见开发命令：

```powershell
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu
cmake --build out/build/dev-ninja-cpu --target run_pass_pipeline_test
```

具体 build 目录取决于 preset。

## 18. 典型扩展流程

### 18.1 新增 Relay 算子

1. 在 `include/relay/op.h` 中定义 attrs node/ref，如果需要 attrs。
2. 在 `src/relay/op/...` 中写 compute 函数，签名是 `FRelayToTE`：

   ```cpp
   te::Tensor MyCompute(const Attrs& attrs,
                        const Array<te::Tensor>& inputs,
                        const kxc::Type& out_type);
   ```

3. 用 `KXC_REGISTER_OP(my_op)` 注册：
   - `describe`
   - `set_num_inputs`
   - `add_argument`
   - `set_attr<TAttrs>`
   - `set_attr<FRelayToTE>`
4. 确保该 `.cc` 加入 `CMakeLists.txt` 的 `KXC_RUNTIME_SOURCES`。
5. 在 `LowerToTIR` 可覆盖的 TE/TIR 表达能力内实现 TOPI compute。
6. 添加 pass/lowering/codegen 测试。

### 18.2 新增 Relay pass

1. 继承或使用 `RelayPass`。
2. 覆写 `VisitCall`、`VisitLet` 等方法。
3. 返回新的 `Function`。
4. 接入 `src/relay/transforms/pipeline.cc` 的 pass table。
5. 加入 `test/pass_pipeline_test.cpp`。

### 18.3 新增 TIR pass

1. 继承 `TIRPass`。
2. 覆写 `VisitPrimFunc`、`VisitFor`、`VisitStore` 或表达式 visit。
3. 注意使用 `tir::pass_utils` 保持 no-op、SeqStmt、常量等处理一致。
4. 接入 `src/tir/transforms/pipeline.cc`。
5. 添加测试。

### 18.4 新增 codegen 能力

如果是 LLVM：

- 扩展 `CodeGenLLVM::GenExpr` 支持新的 TIR expr。
- 扩展 `CodeGenLLVM::GenStmt` 支持新的 TIR stmt。
- 必要时扩展 intrinsic 映射。
- 在 `test/codegen_llvm_test.cpp` 增加直接 TIR 和 Relay lowering 两类覆盖。

如果是 CUDA：

- 扩展 `CodeGenCUDA::GenExpr` / `GenStmt`。
- 同时补齐 CUDA source、NVRTC/Driver launch 和 RuntimeSession 数值测试。

## 19. 当前实现边界

阅读代码和改功能时需要注意这些边界：

- Relay type inference 不完整，很多地方依赖构图时手动填 `TensorType`。
- `LowerToTIR` 只支持单输出 compute tensor。
- TE schedule API 有雏形，但 lowering 当前主要生成朴素 loop nest。
- LLVM 和 CUDA 都已接入 Compiler 主链路，但各自支持的 TIR 节点集合仍有限。
- CUDA 已覆盖 NVRTC、Driver launch、RuntimeSession 数值和 CUPTI smoke；复杂 schedule 与算子覆盖仍需扩展。
- Disco execution plan executor 当前偏解释和数据移动模拟，不执行真实编译后 kernel。
- Python ONNX 转 C++ 脚本偏实验性，路径和覆盖算子需要按实际模型维护。

## 20. 建议的读代码顺序

如果是第一次读这个 repo，推荐顺序：

1. `include/base/object.h`、`include/base/container.h`
2. `include/relay/relay.h`、`include/relay/op.h`
3. `src/relay/op/tensor/math.cc` 和一个 NN op，例如 `src/relay/op/nn/convolution.cc`
4. `include/te/te.h`、`include/te/topi/nn.h`
5. `include/tir/expr.h`、`include/tir/stmt.h`
6. `src/relay/backend/lower.cc`
7. `src/relay/transforms/pipeline.cc`、`src/tir/transforms/pipeline.cc`
8. `src/api/compiler.cc`
9. `src/codegen/codegen_llvm.cc`、`src/codegen/llvm_jit.cc`
10. `src/runtime/runtime_session.cc`
11. `src/base/device_api.cc` 和 CPU/CUDA backend
12. `src/base/disco/executor.cc`
13. `include/base/profiling.h`、`src/base/profiling.cc`

这样能按“对象系统 -> IR -> 算子 -> lowering -> pass -> codegen -> runtime”的主线建立完整心智模型。
