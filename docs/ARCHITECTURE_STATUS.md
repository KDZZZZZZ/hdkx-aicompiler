# hdkx-aicompiler 架构与实现状态（历史快照）

> **状态：已归档。** 本文固定描述 `cdb4c6f`，不得作为当前 capability
> 声明。当前 per-unit 编译、CoreContract v1、capability/pipeline/identity/cache
> 事实见 [`COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md`](COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)
> 与 [`handoffs/compiler-foundation/core.md`](handoffs/compiler-foundation/core.md)，最终仍以当前源码/测试为准。
>
> 最后核对日期：2026-07-21
>
> 核对分支：**feature/compiler-codegen-runtime-contract**
>
> 核对提交：**cdb4c6f**

本文档描述仓库当前实际架构、各模块已经实现的能力、测试覆盖和已知缺口。
它是面向代码审查和后续实施的当前状态快照，不是目标设计文档。

阅读时应遵守以下优先级：

1. 当前源码和可重复测试结果是最终事实来源。
2. 本文档总结当前源码，但会随提交演进。
3. [DEVICE_MODEL.md](DEVICE_MODEL.md) 和 [2026-07-19-compiler-codegen-runtime-contract.md](plans/2026-07-19-compiler-codegen-runtime-contract.md) 主要描述目标与实施过程，不能单独证明能力已经落地。
4. [REPO_IMPLEMENTATION_OVERVIEW.md](REPO_IMPLEMENTATION_OVERVIEW.md)、docs/MODULE_GUIDE.md 和旧教程中仍有历史描述，相关偏差见本文第 12 节。

## 1. 状态定义与总体结论

本文使用以下状态：

| 状态 | 含义 |
|---|---|
| 已闭环 | 公共入口、核心实现和主要正反向测试都存在 |
| MVP 可用 | 主路径可执行并有数值测试，但 shape、dtype、属性或性能范围有限 |
| 部分实现 | 数据结构或部分路径存在，但仍有明确语义缺口 |
| 结构占位 | 类型和接口存在，不能用于真实计算 |
| 未验证 | 代码可能可达，但缺少当前环境下的端到端证据 |

当前总体判断：

| 子系统 | 当前状态 | 结论 |
|---|---|---|
| Object/ObjectRef/TypeInfo | 已闭环 | 引用计数、运行时类型、Arena 和对象容器已有独立测试 |
| Device/Storage/NDArray/Stream | 已闭环 | CPU 路径完整；CUDA 路径依赖 CUDA 构建和设备 |
| Relay IR 与算子注册 | MVP 可用 | 正式契约包含 19 个算子 |
| ONNX 前端 | 部分实现 | 支持 ResNet18 所需的 7 类 ONNX 节点，限定静态 shape 和单节点单输出 |
| Relay 到 TE/TIR lowering | MVP 可用 | 将整个 Relay Function 降成单个 PrimFunc，支持常量和多输出 |
| Relay Pass | 部分实现 | 大多是局部规则；O3 CSE 存在正确性风险 |
| TIR Pass | 部分实现 | 常量和清理规则可用；循环优化仍是启发式实现 |
| LLVM 后端 | MVP 可用 | 19 个正式算子均有 CPU LLVM 数值执行证据 |
| CUDA 后端 | 部分实现 | 仅一维逐元素 Compiler 主链有端到端证据 |
| KernelSignature/CompiledModule | 已闭环 | 参数 ABI、常量身份和启动前校验已有完整测试 |
| RuntimeSession | 已闭环 | 支持同步、异步、常量注入、静态输出和多输出 |
| Profiling | 已闭环 | Compiler、Pass 和 Runtime 可生成 profile bundle |
| ExecutionPlan/Disco | 结构占位 | 通信结构存在，但普通 KernelExec 不执行真实 kernel |
| OpenCL/Metal | 结构占位 | 只有设备类型身份，没有 DeviceAPI 或 codegen |

最重要的边界是：

- CPU LLVM 是当前唯一覆盖全部 19 个正式算子并能编译 ResNet18 的后端。
- CUDA 当前只能称为一维逐元素 MVP，不能称为通用算子后端。
- 当前 Compiler 将整个 Relay Function 降成一个 PrimFunc，不存在成熟的图级分区、多 kernel 调度或 module registry。
- 动态输入 shape 可作为签名哨兵参与校验，但当前 lowering 和 RuntimeSession 尚未形成完整动态 shape 执行链。
- ExecutionPlan/Disco 不在 Compiler 主路径中，当前不能用于真实计算。

## 2. 总体架构

### 2.1 分层视图

~~~text
ONNX / 手写 Relay
        |
        v
Frontend Importer
        |
        v
Relay Function
        |
        v
Relay InferType + Relay Pass
        |
        v
Relay -> TE Tensor DAG -> TIR PrimFunc
        |
        v
TIR Pass + 后端专用调度
        |
        +---------------------------+
        |                           |
        v                           v
LLVM IR + ORC JIT          CUDA C + NVRTC + PTX
        |                           |
        +-------------+-------------+
                      v
              KernelSignature
              LaunchMetadata
              CompiledKernel
                      |
                      v
               CompiledModule
                      |
                      v
               RuntimeSession
                      |
                      v
       NDArray / Storage / DeviceStream
                      |
                      v
             CPUDeviceAPI / CUDADeviceAPI
~~~

ExecutionPlan、Disco worker 和 CCL 是另一条实验性路径，目前没有接入上面的可执行模块主链。

### 2.2 模块边界

| 层 | 主要目录 | 核心职责 | 不负责 |
|---|---|---|---|
| 基础对象层 | [include/base](../include/base) | ObjectRef、TypeInfo、容器、PackedFunc、Registry、Arena | 张量计算语义 |
| 设备运行层 | [src/base](../src/base) | Device、Storage、NDArray、Stream、复制和设备属性 | 图优化和算子 lowering |
| Relay | [src/relay](../src/relay) | 高层表达式、算子注册、类型推导、Relay Pass | 设备内存分配 |
| TE/TOPI | [include/te](../include/te) | 描述 tensor compute 和归约 | 后端资源生命周期 |
| TIR | [src/tir](../src/tir) | 标量循环 IR、语句优化、CUDA thread binding | 模型导入 |
| Compiler API | [src/api](../src/api) | 固定阶段编译、ABI 冻结、模块组装 | 运行时输入自动装配 |
| Codegen | [src/codegen](../src/codegen) | LLVM/CUDA 发射、JIT/module 生命周期、launcher | Relay 图改写 |
| Runtime | [src/runtime](../src/runtime) | 输入校验、常量注入、输出分配、同步和异步启动 | 编译、缓存和 specialization |
| Frontend | [python/kxc_onnx](../python/kxc_onnx)、[src/frontend](../src/frontend) | ONNX 解析、参数序列化、Relay 构造 | 任意 ONNX 算子兼容 |
| Disco | [src/base/disco](../src/base/disco) | worker、DRef、通信和计划解释结构 | 当前不具备真实 kernel 执行 |
| Profiling | [src/base/profiling.cc](../src/base/profiling.cc) | span、事件、bundle 和分析数据 | 修改编译语义 |

### 2.3 必须保持的不变量

1. 物理 Device、VirtualDevice 放置和 Disco worker ID 是三种不同身份。
2. NDArray 是 Storage 的张量视图，不直接拥有裸分配策略。
3. Compiler 冻结的 KernelSignature 是后端和 Runtime 之间唯一参数 ABI。
4. CompiledModule 必须同时持有 Target、最终 PrimFunc、Signature、LaunchMetadata、常量和 executable。
5. RuntimeSession 只能消费 ready CompiledModule，不能隐式重新编译。
6. 异步操作必须保活参与执行的 Storage、Stream、event 和 executable。
7. CPU-only 构建不得要求 CUDA；CUDA 不可用时不得静默回退 CPU。
8. 公共对象边界使用 Array、Map、String 和 ObjectRef；算法内部临时图、memo 和主机缓冲可以继续使用 STL。

## 3. 基础对象与设备运行层

### 3.1 Object、ObjectRef、TypeInfo 与容器

核心入口：

- [object.h](../include/base/object.h)
- [container.h](../include/base/container.h)
- [arena.h](../include/base/arena.h)
- [packedfunc.h](../include/base/packedfunc.h)
- [registry.h](../include/base/registry.h)

当前职责：

- Object 提供侵入式引用计数和运行时 TypeInfo。
- ObjectRef 提供带动态类型检查的共享对象句柄。
- Array、Map、String 是可穿过对象系统、PackedFunc 和公共 IR API 的对象容器。
- Arena 只优化对象分配，不改变引用计数和逃逸对象的生命周期语义。
- PackedFunc/Registry 提供字符串名称到类型擦除调用入口的映射。

当前状态：

- 对象复制、移动、自赋值、成员逃逸、跨线程最后释放、超对齐分配和异常构造均有测试。
- Array/Map 使用家族 TypeInfo；Relay attrs 已迁移到对象容器。
- 公共 API 的容器迁移基本完成。
- Pass 内部的 std::vector、std::unordered_map 和 std::unordered_set 大多是局部算法状态，不应机械迁移。
- [op.h](../include/relay/op.h) 中 KXC_DECLARE_ATTRS_REF 会生成 self-friend 编译警告，应清理。

### 3.2 Device、Target 与设备发现

核心入口：

- [device.h](../include/base/device.h)
- [device_api.h](../include/base/device_api.h)
- [target.h](../include/base/target.h)

当前职责：

- Device 表示物理设备类型和编号。
- DeviceManager 枚举可用设备和设备诊断。
- DeviceAPI 路由分配、释放、复制、stream、event 和设备属性查询。
- Target 是编译期能力快照，保存后端 kind、设备身份和硬件属性。

当前状态：

- CPU 和 CUDA 有 DeviceAPI 实现。
- CPU 仅接受 cpu:0。
- CUDA Target 必须具有可用 compute capability 和 launch capability。
- OpenCL 和 Metal 只有 DeviceTypeCode，没有后端实现。
- Device PackedFunc 同时保留旧名称和显式 JSON 名称，属于待删除兼容表面。

### 3.3 Storage、NDArray、DeviceStream 与 AsyncOperation

核心入口：

- [storage.h](../include/base/storage.h)
- [ndarray.h](../include/base/ndarray.h)
- [device_stream.h](../include/base/device_stream.h)

当前职责：

- Storage 持有设备地址、容量、对齐、ownership 和释放策略。
- NDArray 持有 dtype、shape、strides、byte_offset 和 Storage 视图。
- DeviceStream 绑定一个物理设备和后端 stream handle。
- AsyncOperation 绑定完成 event，并保活 Storage 和 executable。

当前状态：

- 支持 owned、external 和 workspace Storage。
- NDArray 支持 Empty、Zeros、同步/异步复制、字节复制和 view。
- 启动前会校验连续性、范围、offset 和 alignment。
- CPU 默认 stream 和 CUDA stream/event 路径均有实现。
- 动态输出尚无 shape function，RuntimeSession 会明确拒绝自动分配。

## 4. 前端与 Relay

### 4.1 ONNX 前端

正式 Python importer 位于 [importer.py](../python/kxc_onnx/importer.py)，C++ 侧消费导入规格位于 [onnx_importer.cc](../src/frontend/onnx_importer.cc)。

当前正式映射：

| ONNX | Relay |
|---|---|
| Add | add |
| Conv | nn_conv2d |
| Relu | nn_relu |
| MaxPool | nn_max_pool2d |
| GlobalAveragePool | nn_global_avg_pool2d |
| Flatten | nn_flatten |
| Gemm | nn_gemm |

限制：

- 仅支持静态 shape MVP。
- 每个 ONNX 节点必须只有一个输出。
- 未支持算子立即报错，不生成 unknown_op。
- Conv 权重必须是 initializer。
- ResNet18 所需节点均可导入，并能进入 LLVM 编译。
- ResNet18 数值执行默认关闭，需要显式设置 KXC_RUN_RESNET18_EXEC。

### 4.2 Relay IR 与算子注册

核心入口：

- [relay.h](../include/relay/relay.h)
- [op.h](../include/relay/op.h)
- [op_macros.h](../include/relay/op_macros.h)
- [type_infer.cc](../src/relay/type_infer.cc)

正式算子必须同时具备：

1. 唯一规范名称。
2. 输入 schema。
3. attrs 类型。
4. FInferType。
5. FRelayToTE。
6. FFI 构造入口。
7. LowerToTIR 覆盖。
8. 后端数值测试。

当前契约脚本检查的 19 个算子全部满足上述结构要求，但结构契约通过不等于所有属性语义完整。

## 5. Compiler 与完整编译链

公共入口为 [compiler.h](../include/api/compiler.h) 中的 Compiler::Compile。
实现位于 [compiler.cc](../src/api/compiler.cc)。

### 5.1 七个固定阶段

| 阶段 | 输入 | 输出 | 当前行为 |
|---|---|---|---|
| validate | Relay Function、CompileConfig | Validated CompileResult | 校验 Function、Target、设备和 opt_level |
| optimize_relay | validated Relay | typed/optimized Relay | InferType、等级策略、再次 InferType |
| lower | optimized Relay | PrimFunc、常量表 | Relay 到 TE，再将整个 TE DAG 降为一个 PrimFunc |
| optimize_tir | lowered PrimFunc | optimized PrimFunc | TIR 等级策略；CUDA 再执行 thread binding |
| build_signature | 最终 PrimFunc、常量、Target | KernelSignature | 冻结参数顺序、角色、dtype、shape、device 和 alignment |
| build_backend | PrimFunc、Signature、Target | CompiledKernel、LaunchMetadata | LLVM ORC JIT 或 CUDA NVRTC/Driver |
| assemble | 全部阶段产物 | CompiledModule | 校验跨对象一致性并转移所有权 |

七阶段顺序是 Compiler 的公共可观测契约，不允许后端自行跳过 Signature 或直接暴露裸函数指针。

### 5.2 Relay 到 TE/TIR

核心实现位于 [lower.cc](../src/relay/backend/lower.cc)。

当前支持：

- Function 参数转成 TE placeholder。
- Relay Constant 转成稳定 key 的 ConstantBinding。
- Call 通过 FRelayToTE 构建 TE tensor DAG。
- Tuple 和 TupleGetItem。
- 多个输出 tensor。
- 中间 tensor 分配和依赖顺序。
- 参数顺序固定为 input、constant、output。

当前限制：

- 整个 Function 只生成一个 PrimFunc。
- 不支持 Relay If、Let 或函数值调用。
- 不支持嵌套 tuple 字段和重复输出 tensor。
- TE Reduce 只支持单 source 的 sum、max、min。
- 没有 graph partition、kernel fusion policy、module registry 或多 kernel 调度。
- 动态 shape 无法形成完整可执行路径。

### 5.3 Kernel ABI

KernelSignature 和 KernelLaunchMetadata 位于：

- [kernel_signature.h](../include/codegen/kernel_signature.h)
- [kernel_signature.cc](../src/codegen/kernel_signature.cc)

参数角色固定为：

~~~text
[输入参数...] [常量参数...] [输出参数...]
~~~

每个参数记录：

- name
- role
- dtype
- shape
- device
- alignment
- mutable_data
- constant_key

Signature 从最终优化后的 PrimFunc 构建，因此 CUDA thread binding、symbol 重命名和 Buffer 规范化必须在此之前完成。

## 6. 正式算子实现状态

以下 LLVM 状态表示当前数值样例通过，不代表所有 dtype、shape 和 attrs 组合都已覆盖。

| Relay 算子 | 类型推导/TE | LLVM | CUDA | ONNX | 已知限制 |
|---|---|---|---|---|---|
| add | 已实现 | 数值通过 | 一维 E2E 通过 | Add | 广播 MVP |
| subtract | 已实现 | 数值通过 | 未逐算子验证 | 无 | 广播 MVP |
| mul | 已实现 | 数值通过 | 未逐算子验证 | 无 | 广播 MVP |
| divide | 已实现 | 数值通过 | 未逐算子验证 | 无 | 广播 MVP |
| sqrt | 部分实现 | 数值通过 | 未逐算子验证 | 无 | 类型推导接受任意 tensor dtype，后端数学调用只接受浮点 |
| matmul | 部分实现 | 数值通过 | 当前调度不可达 | 无 | 仅 rank-2 |
| nn_dense | 部分实现 | 数值通过 | 当前调度不可达 | 无 | 类型推导使用 units/out_dtype，TE compute 未消费 |
| nn_gemm | 部分实现 | 数值通过 | 当前调度不可达 | Gemm | 支持 transB/alpha/beta，拒绝 transA=1 |
| nn_relu | 已实现 | 数值通过 | 一维 E2E 通过 | Relu | 当前 CUDA 仅验证一维 |
| nn_conv2d | 部分实现 | 数值通过 | 当前调度不可达 | Conv | 仅 NCHW/OIHW；compute 忽略 groups/channels/out_dtype，四向 padding 语义不一致 |
| nn_max_pool2d | 部分实现 | 数值通过 | 当前调度不可达 | MaxPool | 类型推导处理 dilation，compute 忽略 dilation |
| nn_avg_pool2d | 部分实现 | 数值通过 | 当前调度不可达 | 无 | 与 max pool 共用 attrs，compute 忽略 dilation |
| nn_global_avg_pool2d | MVP 可用 | 数值通过 | 当前调度不可达 | GlobalAveragePool | 静态 NCHW rank-4 |
| nn_flatten | MVP 可用 | 数值通过 | 未逐算子验证 | Flatten | 静态 shape 和 axis |
| reshape | 部分实现 | 数值通过 | 未逐算子验证 | 无 | schema 为单输入，type infer 仍接受 legacy shape input |
| transpose | MVP 可用 | 数值通过 | 未逐算子验证 | 无 | 静态 axes |
| cast | MVP 可用 | 数值通过 | 未逐算子验证 | 无 | 受当前标量 dtype 映射限制 |
| reduce_mean | MVP 可用 | 数值通过 | 当前调度不可达 | 无 | 静态 axes/keepdims |
| softmax | 部分实现 | 数值通过 | 当前调度不可达 | 无 | 直接 exp/sum/div，未减最大值，数值稳定性不足 |

算子侧最优先的语义修复：

1. 让 conv2d 的 groups、channels、padding、out_layout 和 out_dtype 在 type inference 与 TE compute 中完全一致。
2. 让 pool dilation 真正进入 compute，或在 schema 层拒绝非 1 dilation。
3. 让 dense 的 units/out_dtype 生效，或删除尚不支持的属性。
4. 统一 Gemm 对 transA 的类型推导和 lowering 能力。
5. 将 softmax 改为减最大值的稳定实现。
6. 让 sqrt 类型推导明确限制浮点 dtype。
7. 删除 reshape 的第二输入兼容推导路径。

## 7. Relay Pass

Pass 注册表位于 [relay/transforms/pipeline.cc](../src/relay/transforms/pipeline.cc)。

### 7.1 Compiler 等级策略

| opt_level | Relay 策略 |
|---:|---|
| 0 | 无显式优化；Compiler 仍在前后执行 InferType |
| 1 | fold_tuple_get_item、fold_constant、simplify_expr |
| 2 | O1 + canonicalize_cast、remove_standalone_reshapes、eliminate_dead_let |
| 3 | 与 O2 相同；不启用结构键尚不安全的 CSE 或调试/标注 Pass |

Compiler 在策略前后各执行一次强制 InferType；策略本身不重复插入 InferType。

### 7.2 逐 Pass 状态

| Pass | 实际能力 | 状态与限制 |
|---|---|---|
| infer_type | 为参数、Call、Tuple、If、Let 等写入 checked_type | 中等完整；仍限于当前 Relay 类型系统 |
| fold_tuple_get_item | 折叠 TupleGetItem(Tuple(...), i) | 窄但可靠 |
| fold_constant | 折叠两个标量 Constant 的部分算术和比较 | 使用 double 中间值，整数精度、符号和溢出语义不完整 |
| simplify_expr | 处理加零、减零、乘一、除一 | 局部规则；只接受 canonical op name |
| canonicalize_cast | 合并目标 dtype 相同的嵌套 cast | 窄但可靠 |
| remove_standalone_reshapes | 合并连续 reshape | 名称大于实际能力，不会删除一般独立 reshape |
| eliminate_common_subexpr | 在 Let 作用域合并结构键相同的值 | 仅允许显式调用；不在 Compiler O3 或 `optimize_default` 中，见 7.3 |
| eliminate_dead_let | 删除未使用且判定纯的 Let | 纯度模型把大多数非 device op 视为纯 |
| annotate_memory_scope | 为已有 VirtualDevice 节点补 const/global | 普通 Compiler 输入上通常无作用，并原地改共享 IR |
| capture_post_dfs_index_in_spans | 写入 DFS/dominator 调试编号 | 仓内无生产消费者，并原地改共享 IR |

### 7.3 CSE 正确性风险

[pass_utils.cc](../src/relay/pass_utils.cc) 使用 Relay debug printer 文本作为 ExprStructuralKey。
[print_ir.cc](../src/relay/pass/print_ir.cc) 对 Constant 只打印 shape/dtype，对 Call 不打印 attrs。

因此以下表达式可能得到相同结构键：

- 内容不同但 shape/dtype 相同的 Constant。
- attrs 不同的 reshape、cast、conv 或 pool Call。

显式调用 `eliminate_common_subexpr` 仍可能据此合并语义不同的 Let 值，形成静默误编译。
Compiler O3 和公共 `optimize_default` 均已移除该 Pass；修复结构键并增加 Constant/attrs
回归测试前，不应把它重新加入任何默认链。

## 8. TIR Pass 与 CUDA 调度

Pass 注册表位于 [tir/transforms/pipeline.cc](../src/tir/transforms/pipeline.cc)。

### 8.1 Compiler 等级策略

| opt_level | TIR 策略 |
|---:|---|
| 0 | Compiler 强制 fold_constant、simplify_expr |
| 1 | fold_constant、simplify_expr |
| 2 | O1 + force_narrow_index_to_i32、convert_for_loops_serial |
| 3 | 完整通用默认链 |

O3 通用默认链为：

~~~text
fold_constant
-> simplify_expr
-> force_narrow_index_to_i32
-> convert_for_loops_serial
-> loop_partition
-> unroll_loop
-> vectorize_loop
-> remove_no_op
~~~

CUDA 的 bind_cuda_threads 不在通用默认链中，由 Compiler 在上述优化完成后额外执行。

### 8.2 逐 Pass 状态

| Pass | 实际能力 | 状态与限制 |
|---|---|---|
| fold_constant | 标量整数/浮点算术、比较、逻辑和 select | 中等完整，仅立即数 |
| simplify_expr | 加零、减零、乘一、除一 | 局部规则 |
| force_narrow_index_to_i32 | 收窄可表示的 int64 IntImm | 不处理变量和一般表达式 |
| convert_for_loops_serial | 将 Parallel/Vectorized/Unrolled 改为 Serial | 保语义但丢失性能意图 |
| loop_partition | 对静态、无嵌套循环按固定因子 4 strip-mine | 启发式，不是一般 loop partition |
| unroll_loop | 展开 extent 0 到 8 的静态循环 | extent 0 边界缺少专门测试 |
| vectorize_loop | 将满足简单条件的循环标为 Vectorized | 仅标签；LLVM 当前仍生成标量循环 |
| remove_no_op | 删除 Evaluate(0)、空循环并扁平化 SeqStmt | 窄但可靠 |
| bind_cuda_threads | 将一个外层静态循环映射到 blockIdx.x/threadIdx.x | 只支持直线独立 Store |

### 8.3 CUDA O3 风险

BindCudaThreads 要求：

- 一个外层 Serial For。
- 静态正 extent。
- Store 下标正是循环变量。
- body 只能包含直线 Store、Let 和 SeqStmt。
- 不允许内层 For、Allocate、归约或读取正在写入的 Buffer。

但 O3 在 Bind 之前执行 partition、unroll 和 vectorize。
这些 Pass 可能删除外层循环、引入 Let 或改变 ForType，从而破坏 Bind 的前置条件。
当前 CUDA Compiler 测试使用 O2，没有覆盖 CUDA O3。

## 9. Codegen 与后端

### 9.1 LLVM

核心实现：

- [codegen_llvm.cc](../src/codegen/codegen_llvm.cc)
- [llvm_jit.cc](../src/codegen/llvm_jit.cc)

当前支持：

- 标量 Int、UInt、Float。
- 算术、比较、逻辑、select 和 cast。
- Load、Store、For、Allocate、IfThenElse、LetStmt、SeqStmt、Evaluate。
- exp、log、sqrt、floor、ceil、fabs。
- 私有 ABI：int32(void** data, uint64_t count)。
- 每个 CompiledKernel 独占 ORC LLJIT 生命周期。
- LLVM verifier 和 O0 到 O3 官方优化管线。
- 多输出、byte_offset、零长度张量和中间 Allocate。

当前限制：

- 不支持 vector-lane NDArray 参数。
- TIR Vectorized 标签不产生真正 SIMD lowering。
- 没有 AOT object/shared-library 导出。
- 没有 kernel cache、shape specialization 或后台重编译。

### 9.2 CUDA

核心实现：

- [codegen_cuda.cc](../src/codegen/codegen_cuda.cc)
- [cuda_module.cc](../src/codegen/cuda_module.cc)
- [bind_cuda_threads.cc](../src/tir/transforms/bind_cuda_threads.cc)

编译与执行链：

~~~text
PrimFunc
-> BindCudaThreads
-> CUDA C++ source
-> NVRTC
-> PTX
-> cuModuleLoadDataEx
-> cuModuleGetFunction
-> cuLaunchKernel
-> CUDA event
-> AsyncOperation
~~~

当前能力：

- CUDA C emitter 支持标量 dtype、基础表达式、数学调用、ThreadBinding、Serial/Unrolled For 和常量 extent Allocate。
- NVRTC 编译日志和 Driver 错误都有明确异常。
- CUmodule 与 CUfunction 由同一 launcher 保活。
- 异步操作保活参数 Storage 和 executable。
- 已验证一维 add、常量 add、relu 的 Compiler + RuntimeSession 路径。

当前限制：

- Compiler 可达调度仅支持一维逐元素直线 kernel。
- 归约、嵌套循环、共享内存调度、多维 grid/block 和复杂依赖均不可达。
- 其余 17 个算子没有 CUDA 逐算子端到端证据。
- 当前分支尚未在 Omen 上完成 fresh CUDA 全矩阵验证。

### 9.3 C source emitter

[codegen_c.cc](../src/codegen/codegen_c.cc) 仍被 LLVM 测试用于诊断源码输出。
它不是 Compiler 可执行后端，不应再被文档描述为正式 C backend。
后续应移入 debug/test target，或在删除诊断能力后整体移除。

## 10. CompiledModule、RuntimeSession 与 Profiling

### 10.1 CompiledKernel 和 CompiledModule

核心实现：

- [compiled_kernel.cc](../src/codegen/compiled_kernel.cc)
- [compiled_module.cc](../src/api/compiled_module.cc)

CompiledKernel 统一持有：

- KernelSignature
- KernelLaunchMetadata
- 后端 KernelLauncher

CompiledModule 在组装和 Launch 时校验：

- Target 与 launch device 一致。
- Signature 和 executable 共享同一契约对象。
- 常量表与 constant 参数严格一一对应。
- 参数数量、role、dtype、shape 和 device。
- NDArray 连续性、范围、alignment 和 byte_offset。
- 常量参数必须是模块持有的同一 NDArray 对象。
- stream 必须属于模块设备。

裸函数指针、CUmodule 和后端参数数组不会暴露到公共 API。

### 10.2 RuntimeSession

核心实现位于 [runtime_session.cc](../src/runtime/runtime_session.cc)。

RuntimeSession 的职责：

1. 持有一个 ready CompiledModule。
2. 根据 Signature 校验调用方 inputs。
3. 按 ABI 顺序插入模块 constants。
4. 为静态 output 参数分配 NDArray。
5. 调用唯一的 CompiledModule::Launch。
6. 同步 Run 等待完成后返回 outputs。
7. 异步 RunAsync 返回 outputs 和 completion。

RuntimeSession 不负责：

- 编译 Relay。
- 选择或缓存 kernel。
- 动态 shape specialization。
- 图分区和多 kernel 调度。
- 自动设备回退。

### 10.3 Profiling

Compiler 的七个阶段、Relay/TIR pipeline、Pass 和 Runtime 都可以写入 profile span。
ProfileContext 负责 run ID、事件、artifact 和 bundle 输出。

当前 profile_bundle_test 可以生成结构化 bundle。
profiling 不参与编译语义，不允许为了记录事件而改变 IR。

## 11. ExecutionPlan、Disco 与集合通信

核心实现：

- [execution_plan.h](../include/base/execution_plan.h)
- [multi_device.cc](../src/relay/transforms/multi_device.cc)
- [executor.cc](../src/base/disco/executor.cc)
- [ccl_cpu.cc](../src/base/disco/ccl_cpu.cc)

已经实现的结构：

- KernelExec、CommExec、BarrierExec 和 ExecutionPlan 对象。
- VirtualDevice 到 worker set 的放置解析。
- DRef 和 ThreadedSession。
- CPU CCL 模拟的 copy、broadcast、scatter、gather、allgather、allreduce、reduce_scatter。
- 通信 op 到 CommExec 的 lowering。

尚未实现的关键能力：

- 普通 Relay Call 会生成持有空 PrimFunc 的 KernelExec。
- ExecutionPlanExecutor 不查找或启动 CompiledModule。
- NCCL 集合通信后端尚未实现。

该路径目前只能验证放置、计划结构和 CPU 通信语义，不能执行 kernel 数值计算。
旧的 `kxc.disco.execute_plan*` 公共 FFI 已删除；直接执行 KernelExec 会明确抛出
`ExecutionPlan CompiledModule launch is not implemented`，不会再返回零张量或复制输入伪装执行成功。

## 12. 旧实现清理结果

### 12.1 已完成：正确性和伪执行

1. Relay O3 默认链已移除 `eliminate_common_subexpr`；该 Pass 仍保留给显式测试和后续修复。
2. CPU 和 CUDA 已使用独立 TIR O3 policy，CUDA 不再经过会破坏 thread binding 前置条件的通用循环 Pass。
3. ExecutionPlan KernelExec 已改为 fail closed，并删除旧公共执行 FFI 和伪造输出路径。

### 12.2 已完成：旧表面删除

1. 已删除空壳 `common_ops`、无效 `set_support_level`、Pass 旧算子别名和重复 lowering 包装层。
2. 已删除 `python/onnx_to_cpp.py` 及三个平行 ONNX 报告脚本；正式入口统一为 `kxc_onnx`。
3. 已删除临时 conv/maxpool 调试源码、手写 bat 和失效的 debug-examples CMake 开关/preset。
4. 已删除无 factory 的 NCCL 占位后端，以及只会抛异常的 `topi::einsum`、`topi::prod` 公共表面。
5. Device PackedFunc 只保留显式 `ListDevicesJSON`、`GetAllDeviceInfoJSON` 契约。
6. ResNet18 IR dump 保留为显式诊断工具，但 CMake 默认和所有通用 preset 均关闭。
7. 已删除版本控制中的模型报告、DOT 和 IR dump 生成物，并在 `.gitignore` 中阻止重新收录。

### 12.3 已完成：未注册算子残留

正式契约之外、没有注册/FFI/lowering 的以下 attrs 和 type rule 已删除：

- BatchNormAttrs
- ConcatAttrs
- ConstantAttrs
- ConstantOfShapeAttrs
- DivAttrs
- EqualAttrs
- ErfAttrs
- ExpandAttrs
- GatherAttrs
- SplitAttrs
- IdentityInferType
- PowInferType
- EqualInferType
- GreaterInferType
- ShapeInferType
- ConcatenateInferType
- SplitInferType
- WhereInferType
- GatherInferType

后续若实现这些算子，应从 contract、schema、type、lowering 和数值测试完整加入，不能只恢复单个声明。
`DeviceCopyAttrs` 和 `CollectiveAttrs` 仍被 multi-device IR 使用，因此保留。

### 12.4 保留的后续边界

1. Relay/TIR pipeline 仍有重复的名称表、instrumentation 和 PackedFunc 注册框架；这是独立重构，不在本次死代码删除中扩张。
2. `base::Tensor` 仍被 PackedFunc 特化和 Relay 公共头引用，必须先迁移 API，不能直接删除。
3. `AddAttrs`、`ReluAttrs` 等无字段 attrs 仍有 importer/诊断构图调用，待 schema 统一后再删除。
4. `CompileResult` 可以内部化，但其阶段状态机不应删除。
5. `CSourceEmitter` 仍由 LLVM 诊断测试使用，明确保持“只生成可读 C、不是 backend”的定位。

### 12.5 已知陈旧文档

- MODULE_GUIDE.md 仍描述 Adaptive Runtime、KernelCache、裸 func_ptr 和旧 CompiledKernel。
- cpp-tutorial/content/v2-runtime-system.js 仍描述旧裸 ABI 和生命周期。
- PASS_IMPLEMENTATION_SUMMARY.md 的默认 Pass 顺序与当前代码不完全一致。

后续文档更新应以本文和当前代码为基线，不应继续复制这些旧段落。

## 13. 当前测试证据

### 13.1 Windows LLVM 构建

本次核对配置：

~~~text
CMAKE_BUILD_TYPE=Release
KXC_ENABLE_LLVM=ON
KXC_ENABLE_CUDA=OFF
KXC_BUILD_PASS_TESTS=ON
KXC_BUILD_CODEGEN_TESTS=ON
KXC_BUILD_RESNET18_IR_DUMP=OFF
~~~

构建目录：

~~~text
out/build/windows-llvm-msys2
~~~

运行时需要将 C:\msys64\ucrt64\bin 放入 PATH。

### 13.2 Fresh 结果

| 验证项 | 结果 |
|---|---|
| Relay op contract | 19 checked，19 passed，0 failed |
| object_test | 通过 |
| device_info_test | 通过 |
| device_runtime_test | 通过 |
| pass_pipeline_test | 19 项通过 |
| infer_type_test | 9 项通过 |
| compiler_contract_test | 11 项通过 |
| kernel_signature_test | 9 项通过 |
| compiled_module_test | 8 项通过 |
| runtime_session_test | 9 项通过 |
| codegen_llvm_test | 12 项通过 |
| op_numeric_llvm_test | 19 个算子和 3 个组合模型通过 |
| onnx_importer_test | ResNet18 导入和 LLVM 编译通过 |
| profile_bundle_test | 通过并生成 bundle |

ResNet18 数值执行因未设置 KXC_RUN_RESNET18_EXEC 而跳过。
这意味着当前证据可以证明导入和编译，但不能证明完整 ResNet18 输出与参考框架一致。

### 13.3 CUDA 验证缺口

Omen 当前仍位于旧分支 device-info-query-contract@c3b007f，且工作区包含大量未提交修改。
本次没有覆盖、清理或同步远端工作区，因此不能声称当前提交已经通过 Omen CUDA 全矩阵。

当前可引用的 CUDA 证据仅包括既有测试中的：

- 一维 add。
- 一维 constant add。
- 一维 relu。
- CUDA module/event/异步保活测试。
- BindCudaThreads 的正负路径测试。

其余算子和 CUDA O3 均为未验证。

## 14. 建议实施顺序

后续工作应按以下顺序推进：

1. 先修复正确性风险：关闭不安全 CSE、隔离 ExecutionPlan 伪执行、拆分 CUDA O3。
2. 对齐算子 schema、type inference 和 TE compute，优先 conv、pool、dense、gemm、softmax、sqrt。
3. 扩展 CUDA 调度模型，使其支持嵌套循环、归约和必要的临时存储。
4. 为 19 个算子建立 CUDA 逐算子数值矩阵。
5. 在 Omen 当前分支执行 fresh build、测试、memcheck 和 sanitizer。
6. 开启 ResNet18 LLVM 数值执行并与参考输出比较。
7. 继续统一旧教程和 Pass 专题文档；不要恢复本节已删除的兼容入口或半实现 schema。

## 15. 审查入口

建议按以下顺序审查：

1. [src/api/compiler.cc](../src/api/compiler.cc)：七阶段管线和优化等级。
2. [src/relay/backend/lower.cc](../src/relay/backend/lower.cc)：Relay、TE、TIR 和参数 ABI。
3. [src/codegen/kernel_signature.cc](../src/codegen/kernel_signature.cc)：Signature 冻结规则。
4. [src/codegen/codegen_llvm.cc](../src/codegen/codegen_llvm.cc) 与 [src/codegen/llvm_jit.cc](../src/codegen/llvm_jit.cc)：CPU 后端。
5. [src/tir/transforms/bind_cuda_threads.cc](../src/tir/transforms/bind_cuda_threads.cc)、[src/codegen/codegen_cuda.cc](../src/codegen/codegen_cuda.cc) 与 [src/codegen/cuda_module.cc](../src/codegen/cuda_module.cc)：CUDA 后端。
6. [src/api/compiled_module.cc](../src/api/compiled_module.cc) 与 [src/runtime/runtime_session.cc](../src/runtime/runtime_session.cc)：运行边界。
7. [src/relay/type_infer.cc](../src/relay/type_infer.cc) 与 [src/relay/op](../src/relay/op)：19 个算子语义。
8. [src/relay/transforms](../src/relay/transforms) 与 [src/tir/transforms](../src/tir/transforms)：Pass。
9. [src/base/device_api.cc](../src/base/device_api.cc)、[src/base/ndarray.cc](../src/base/ndarray.cc) 和 [src/base/device_stream.cc](../src/base/device_stream.cc)：设备和异步生命周期。
10. [src/relay/transforms/multi_device.cc](../src/relay/transforms/multi_device.cc) 与 [src/base/disco/executor.cc](../src/base/disco/executor.cc)：尚未闭环的多设备路径。

## 16. 文档更新规则

发生以下变化时必须更新本文：

- 增减正式 Relay 算子。
- 修改 Compiler 阶段或 O0 到 O3 策略。
- 修改 KernelSignature 或内核参数顺序。
- 增加新的 DeviceAPI 或 codegen 后端。
- CUDA 支持范围发生变化。
- RuntimeSession 获得动态输出、缓存、specialization 或多 kernel 能力。
- ExecutionPlan 开始执行真实 CompiledModule。
- 清理第 12 节中的旧入口或文件。
- 测试矩阵或验证环境发生变化。
