# hdkx-aicompiler 架构

> **状态：** 当前架构权威文档
> **更新时间：** 2026-09-09
> **规则：** 若本文档与实现有冲突，以源码、机器可读契约和可复现测试为最终依据。

## 1. 系统边界

`hdkx-aicompiler` 是一个精简的 TVM 风格编译器与运行时，遵循四条核心约束：

1. Relay 算子与 Pass 都有机器可读契约；
2. 生产编译以 `PrimitiveUnit` 为单位，不把整张图视为单个内核；
3. 编译结果包含不可变模块，以及与运行时实现无关的执行计划；
4. 不支持的语义会在执行前明确报错，不做隐式回退。

默认主路径面向静态精确、单目标编译。LLVM 是主要 CPU 后端；CUDA 仍是实验性后端，只开放保守的调度子集。控制流、精确 Profile 路由、受限 bounded 动态图与自适应替换均为默认关闭的独立能力。开启完整门禁后，`Compiler::CompileBounded` 支持固定秩、有限 bounds 的 CPU/LLVM 逐元素运算、受限形状值链、MatMul→Softmax attention，以及静态权重的变长 RMSNorm/QKV 投影、实际 ONNX 形状控制链、拆头后的 Q/K 归一化、RoPE 组合链和 K/V GQA 重复；真实 MiniMind 完整八层的变长 prefill/decode 现已在 fresh-output plan 执行，见[完整 prefill 报告](implementation/M3_FULL_PREFILL_REPORT.md)和[完整 decode 报告](implementation/M3_FULL_DECODE_REPORT.md)；与持久 KV state 的联合见 [bounded 状态报告](implementation/M2_BOUNDED_STATE_REPORT.md)。

当前不作为生产承诺的能力包括：

- 通用动态秩、参差、数据相关或任意 shape 多态执行；
- CUDA 跨线程协作归约、未证明安全的间接访存或任意嵌套循环调度；
- 自动微分与训练能力；
- 分布式变长执行、CUDA worker、网络服务与持久 KV 一致性；
- 完整 ONNX opset 或全量算子覆盖。

## 2. 架构一览

主静态生产路径为：

```text
Relay Function + 不可变 CompileConfig
  -> PrepareRelayProgram
     -> 校验目标与配置
     -> 执行规范化 Relay 流水线
     -> 生成类型完备的 ANF 与残留控制画像
  -> BuildValueGraph
  -> PartitionValueGraph
     -> 有序 PrimitiveUnit[]
  -> CompilePrimitiveUnits
     -> Relay 编译单元 -> TE 计算
     -> 感知目标的 TE 调度 -> 不可变静态 te::Program -> TIR
     -> 规范化 TIR 流水线
     -> KernelSignature + 编译产物身份
     -> 获取原语缓存
     -> LLVM 或 CUDA 后端编译
  -> AssembleCompiledGraph
     -> CompiledModule + ExecutablePlan + ArtifactPin[]
  -> RuntimeSession
       -> 校验输入、分配并绑定值、按计划顺序启动内核
```

静态单 Call 与首个融合 region 均经 `te::Program` 冻结完整 DAG、调度、边界、Target 和规范 TIR pipeline。优化等级 3 在 CPU:0 对相邻、纯、同形 float32/float64 的 `add → sqrt` 选择一个 PrimitiveUnit；内部 add 结果必须没有外部消费者。默认等级 2 保留独立单元。Program v1 字节替代静态 schedule-only 候选身份并进入 artifact key v4，compiler execution contract 为 v5；TE→TIR、cache、后端与 RuntimeSession owner 保持原路径。metadata-only 形状控制输入仍保留 ABI；exact-profile 与普通编译共用同一可见计划边界。实际 kernel 数从 2 到 1、数值、缓存和寿命证据见 [M8 报告](implementation/M8_TE_PROGRAM_REPORT.md)。现有内部 Allocate 仍可能存在，尚未证明模型级加速。

可选的结构化控制流路径复用准备、原语编译、签名、编译产物与所有权逻辑：

```text
PrepareRelayProgram
  -> LowerPreparedRelayToControlPlanWithSidecar
  -> CompilePrimitiveUnits
  -> BindControlPlanForRuntime
  -> CompiledControlFlowGraph
  -> ControlRuntimeSession
```

`Compiler::Compile` 会拒绝残留控制拓扑。独立的
`Compiler::CompileControlFlowExact` 入口仅在 `KXC_ENABLE_CONTROL_RUNTIME=ON` 时可用，
当前要求使用受支持的精确 CPU 控制子集和真实 LLVM 编译产物。

受限 bounded 生产路径为：

```text
RestrictedSymbolicShapeAdapter::Prepare
  -> adapter-minted BoundedCompileRequest
  -> Compiler::CompileBounded
     -> BoundedCompilePreparation
        -> DynamicUnitShapeContract[] + graph-input guards
     -> CompilePrimitiveUnits(bounded overload)
        -> dynamic TE -> serial TIR -> KernelSignature
        -> ModuleInvocationContract（与 extent ABI 共用同一 ordered vector）
        -> 既有 primitive cache 与 LLVM backend
     -> CompiledModule + kDynamicFreshOutputV1 ExecutablePlan + ArtifactPin[]
  -> RuntimeSession（同一 CompiledGraph 可运行多个合法 shape）
```

该路径支持固定秩 direct `InputAxis`/`Const`、有限范围与整除约束下的 `relu`、`sqrt`、`sigmoid`、`cast`、可证明广播的 `add`/`mul`/`divide`/`pow` 与 rank≥2 MatMul、Transpose/Softmax/MaskedSoftmax、静态归约轴 ReduceMean 和受限形状值链，目标为 CPU:0/LLVM。它不做隐式路由、运行时编译或 CUDA 回退。

受限 adapter 的 applicability v14 支持共享 Call DAG、重复逻辑实参、明确的非空 tuple 结果和动态 slice 的双 shape anchor。物化只重注类型到既有 Relay 私有快照，保留结果字段顺序及仍被消费的输出；值编号由 ValueGraph 独占。运行结果按已有 tensor-leaf 顺序展开，重复输出和 tuple 型算子实参仍拒绝。方法与 LLVM 证据见 [M3 图结构报告](implementation/M3_GRAPH_STRUCTURE_REPORT.md)。静态常量通过既有模块常量池绑定；bounded preparation 使用相同的生产 Relay pipeline 规范化代表图和逻辑图，再比较 ValueGraph，避免 ANF 提升调用造成值编号错位。当前 preparation v3、unit shape contract v9，见 [M3 多 token 报告](implementation/M3_MULTITOKEN_REPORT.md)。

ONNX 前端另有显式 `kxc.onnx_shape_source.v1`：Python 保留 Shape/Gather/Unsqueeze/Concat/Reshape 控制链，C++ `LoadONNXShapeSource` 返回尚未类型校验的 source；调用方须把 function 与 declared_output_types 一并传给 adapter，在编译/cache 前完成来源证明和代表输出校验。默认静态 loader 拒绝该格式。真实四组 B/S 的拆头与 Q/K RMSNorm 结果见 [M3/M4 拆头报告](implementation/M3_ONNX_HEADS_REPORT.md)。在可证明的整数控制链内，ConstantOfShape/Equal/Where 和形状向量恒等 reshape 会在准备阶段消解；数据 Squeeze/Unsqueeze 保留 TE 动态 extent，实际 K/V GQA 四组 B/S 的复制结果精确一致，见 [GQA 报告](implementation/M3_GQA_REPORT.md)。固定 head_dim 的 Slice 控制现可证明并折叠，Slice/Concat 保留非操作轴的 TE 动态 extent，真实 RMSNorm/QKV/heads/RoPE 组合链四组 B/S、204 次 LLVM 调用通过，见 [RoPE 报告](implementation/M3_ROPE_REPORT.md)。参数和常量按既有 ValueGraph 名称区分，不依赖 ShapeProgram 排序后的位置。实际第一层 attention 已联合 RoPE/GQA、动态 `[S,S]` 的 float32 `-inf` 填充、Trilu、Softmax、固定商 Reshape -1 和输出投影，见 [因果 attention 报告](implementation/M3_CAUSAL_ATTENTION_REPORT.md)。ModuleInvocationContract ABI v4 支持同一输入内严格引用更早轴的相等 guard；shape-only lowering 也显式记录仅用于输入 buffer 形状的 extent，并拒绝输入 payload 读取。完整八层 bounded prefill 现已通过，含静态表 Gather、图内 `0:S` 位置前缀和 FFN/残差，见 [完整 prefill 报告](implementation/M3_FULL_PREFILL_REPORT.md)。Slice 的 shape anchor 由生产 lowering 显式标记为只读形状元数据；TE→TIR 检查其 ABI 归属并拒绝 payload 读取。GraphSemanticKey/ShapeProfileKey 共享不可变规范字节，避免逐 unit 请求复制模型权重；完整字节身份语义保持不变。Concatenate v3 可将一个有界动态长度与定长片段拼接，内部形状支持 Symbol+非负常量，循环及输出分配共同消费 offset；bounded schedule 为 v3，见 [KV 追加报告](implementation/M3_KV_APPEND_REPORT.md)。图输入仍只绑定直接 Symbol/Const。完整八层变长 decode 已接通 shape-source、P:P+1 位置窗口、受限 Add/Sub 和 P+1 正数证明，见 [完整 decode 报告](implementation/M3_FULL_DECODE_REPORT.md)。Slice v4 的 window_size 和共用 shape-expression kind=2 经已有 consumer 执行并进入身份。图输入仍只绑定直接 Symbol/Const；任意运行时 shape 条件仍待完成。请求级 CPU/LLVM 等长 batching 已通过独立合同接入 RuntimeSession，见 [批处理报告](implementation/M2_REQUEST_BATCHING_REPORT.md)。`BindBoundedStateOutputs` 通过独立 `kBoundedStatefulExternalV1` 将原输入绑定为会话状态，并为 kernel 提供复用的连续有效前缀。模块输出表达式须证明 P+append；CPU 执行成功后复制新片段并提交长度。物理 batch/容量及 prefix 配对进入计划 ABI v11，kernel ABI 保持不变；真实 LLVM prefill 交接、四组 B/P 和连续 greedy 证据见 [状态报告](implementation/M2_BOUNDED_STATE_REPORT.md)。

请求批处理在上述 bounded state 计划上显式声明 `RequestBatchingContract`，计划 ABI 为 v12；物理 batch 轴是有限请求槽位。`AdmitRequest/EnqueueRequest/RunNextBatch/ReleaseRequest` 在同一个 RuntimeSession owner 中接通初值、快照、等长同形 FIFO 分组、有效前缀收集、追加散写和槽位复用。每请求最多排队一步，CPU/LLVM 与 CUDA 返回前完成提交；不同 P 分批执行，生产者明确声明行独立性。请求状态和 queue 不参与 cache 身份。全部 logits/16 份 KV 对照及 5418→3870 次 kernel 提交证据见 [请求批处理报告](implementation/M2_REQUEST_BATCHING_REPORT.md)。

显式 `masked_softmax` 接入同一 shape resolver 与生产 lowering：bool mask 仅能广播到 logits，正归约长度必须可证明；全 False 行输出精确正零。静态 TE Program 区分 axis 身份，常量 mask 仍由各图绑定。普通 Softmax 不变，CPU/LLVM 静态和变长注意力证据见 [全 mask 报告](implementation/M5_MASKED_SOFTMAX_REPORT.md)。

复制观测沿用 AsyncOperation 完成回调：submit/complete 共用 copy_id 和提交时的区间关联，CPU 异步接口保留实测执行耗时；图外复制也记录完成并保活上下文。诊断器只将明确的 host_execute 事件用于主机耗时判断，host_observed_complete 不代表设备计时。方法和验证见 [复制完成观测报告](implementation/M1_COPY_EVENT_REPORT.md)。CUDA Storage 复制另经 OnCopyScopeEnter 建立 copy_launch，span_id 即 copy_id；设备 DMA 指向该 launch，host submit/complete 保留进入作用域前的调用者，避免改变原父节点。同步完成独立分配 span_id，异步回调只使用提交时捕获的关联。真实受控 pending 与跨线程完成见 [CUDA 复制报告](implementation/M1_CUDA_COPY_REPORT.md)。

MiniMind-V 的固定视觉阶段沿用同一 ONNX/Relay/TE/LLVM/RuntimeSession 路径：新增 float32 Tanh/Erf 接通两种 GELU；固定输入的 Shape 证明仅作为显式导出适配，不进入默认或变长运行时。完整 12 层编码器和投影层已验证，模型边界、参考卷积修正和数值证据见 [视觉链报告](implementation/M9_MINIMIND_V_VISION_REPORT.md)。

固定单图的完整图文 prefill 与会话 decode 也已通过，见 [联合推理报告](implementation/M9_MINIMIND_V_JOINT_REPORT.md)。静态 ONNX Concat 的 1..N 输入归一到已有二元算子；导出时 Python marker 扫描的固定布局由模型调用者显式校验。真实 prefill 的 16 份 KV 经 `InitializeState` 进入既有 RuntimeSession，四步复用容量 decode 产物，不新增状态 owner 或 ABI。此项尚不覆盖多图、图文变长或 GPU。后续 [有界图文报告](implementation/M9_MINIMIND_V_BOUNDED_REPORT.md) 把 marker 扫描移到显式 host 步骤：视觉 token 写入定容 `visual_slots`，与嵌入表拼成静态 `[6592,768]` 的 Gather 表，语言侧以受限形状 S≤224 编译一次，覆盖 0～3 张图与任意位置；视觉编码器仍是固定单图合同，GPU 未验证。

## 3. 模块与依赖方向

公开头文件位于 `include/kxc/<module>/`，实现位于 `src/<module>/`。
`tools/architecture/check_include_layers.py` 会校验允许的头文件包含方向。

| 模块 | 职责 | 不应承担 |
|---|---|---|
| `support`, `ffi`, `ir` | 对象与容器基础设施、注册表、通用 IR 工具 | 编译器或运行时策略 |
| `runtime` | 设备、流、NDArray 与存储、内核 ABI、已编译模块、执行计划、会话 | Relay、TE、Pass 选择、缓存变更或编译 |
| `profiling` | Span、Bundle 序列化及可选 CUPTI 采集 | 编译决策 |
| `target` | 不可变目标与设备能力快照 | 运行时分配或后端编译 |
| `pass` | 与 IR 无关的 Pass 元数据与校验 | 变换实现 |
| `tir` | 低层 IR、变换、打印、CUDA 线程绑定契约 | 图级路由 |
| `te` | 计算 DAG、TOPI 辅助与保留的调度原语 | 目标发现或 CUDA 启动权威 |
| `relay` | 高层 IR、算子注册、属性、类型推导、Relay 变换与 Relay-to-TE 绑定 | 运行时分配 |
| `frontend` | ONNX 导入规范与 Relay 重建 | 编译器执行策略 |
| `codegen` | TIR-to-C/LLVM/CUDA 后端发射与可执行内核创建 | 图拓扑或运行时值路由 |
| `compiler` | 准备、拓扑、分区、Lowering 编排、身份、缓存与发布 | 运行时执行 |
| `distributed` | 显式计划、Worker、CPU 集合通信及已绑定模块的静态进程内执行 | 编译、缓存查询、网络服务或 KV 状态 owner |
| `shape` | 共享的符号形状与精确形状契约数据 | 编译器缓存或运行时执行 |

核心单向边界如下：

```text
前端 -> Relay/Pass -> TE/TIR -> 代码生成
                    \       |
                     编译器
                        |
                        v
                  运行时数据契约
                        |
                        v
                  RuntimeSession
```

编译器与代码生成层可以引用中立的运行时契约，例如 `NDArray`、`Device`、`KernelSignature`
与 `ExecutablePlan`。运行时不得包含或调用编译器、Relay、TE、TIR 或注册表的实现细节。

`src/runtime/internal/` 下保留一条窄接口：已编译模块会记录 `Target`，将编译目标绑定到对象。
头文件层级检查器将其建模为 `runtime_executable`，不会允许会话数据面反向依赖编译器。

## 4. 配置与目标权威

`CompileConfig::Create` 会快照三个编译输入：

- 已校验 `Target`，包含设备标识和相关能力；
- 优化级别 `0..3`；
- 性能分析选项。

`Target` 是后端选择的唯一权威。后端字符串、运行时会话或进程全局状态不能独立决定代码生成。
一次编译调用会把同一份不可变目标快照用于 Pass 解析、调度、编译产物身份、代码生成、模块元数据与运行时计划校验。

Relay 放置策略与配置目标必须一致；冲突会报错，编译器不会悄悄把任务迁移到其他设备。

## 5. Relay 算子契约

`contracts/relay_op_contract.json` 是机器可读的算子契约，记录规范名称与关键 Schema 元数据。
生成器会产出：

- `src/relay/generated/relay_op_contract.inc`（用于 `OperatorSpec`）；
- `src/relay/generated/relay_op_registration.cc`（用于声明了生成式注册的条目）。

生成的注册翻译单元会编译并锚定内置注册表，避免静态链接时被链接裁剪静默移除。
手写的类型推导与 TE 回调仍是普通 C++ 函数；生成绑定只能引用满足签名与可见性规则的符号。

当前处于渐进迁移阶段。没有生成式 `registration` 对象的契约项仍可通过手写注册接入；
已有生成式 `registration` 的条目不能再保留手写注册。

`python/tools/check_relay_op_contract.py` 会校验契约是否最新、注册是否唯一、回调绑定、FFI 覆盖、测试以及支持矩阵。
声明算子不代表它能在每个目标上执行；目标专属支持还取决于 Lowering、调度、代码生成与数值测试。

见 [编译器扩展契约](COMPILER_EXTENSION_CONTRACT.md) 与
[Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md)。

## 6. Pass 与流水线契约

`contracts/pass_contract.json` 是唯一的 Pass 元数据来源。生成的 `PassSpec` 描述身份、
IR 方言、作用域、阶段、优化级别、实现绑定、不变量、分析状态转换、目标谓词、确定性、幂等性与线程安全声明。

`PipelineResolver` 将编译请求转换为规范化的 `NormalizedPipeline`，该规范值记录：

- 精确的 Pass 顺序与出现次数；
- 目标能力快照与必需谓词；
- 不变量与分析状态转换；
- 编译产物身份使用的规范字节与指纹。

`PipelineExecutor` 会在执行前后读取同一份契约，并拒绝绑定漂移、缺失前置条件、目标不匹配、过期分析转换与未证明的不变量。当前可执行不变量包括
Relay `checked_type`/`anf` 与 TIR `prim_func_defined`。

`PassContext` 是单次调用的上下文。显式流水线重载会直接接收它；线程局部兼容 API 仍在使用，并会恢复嵌套状态。

见 [Pass 契约](PASS_CONTRACT.md)。

## 7. 静态图与 `PrimitiveUnit`

Relay 准备完成后，`BuildValueGraph` 会校验静态拓扑并创建逻辑值契约。它只接收受支持的一阶静态 Relay 子集，并拒绝原生 `If`/`While`；
这类场景应走可选控制流路径。

`PartitionValueGraph` 会创建紧凑、有序的 `PrimitiveUnit`，每个普通计算调用对应一个编译单元边界。
每个编译单元的输入输出、常量绑定、副作用、别名、已检查类型与设备都会在进入后端编译前冻结。

`CompilePrimitiveUnits` 是唯一的生产级原语后端路径。对每个编译单元：

1. 校验已冻结的编译单元与逻辑值；
2. 仅对该编译单元执行 TE/TIR Lowering；
3. 执行规范化 TIR 流水线；
4. 构造 `KernelSignature`；
5. 构造完整的编译产物键；
6. 获取或发布原语缓存；
7. 返回就绪的 `ArtifactPin`。

bounded overload 复用同一实现，只把 `DynamicUnitShapeContract` 注入 per-unit Lowering，并在缓存获取和后端编译前校验动态 `KernelSignature` 与 `ModuleInvocationContract`。`CompiledPrimitive` 持有本次图的 invocation applicability；可复用的 `CachedPrimitive` 只持有代码与物理 ABI。

全图 `relay::LowerToTIR` 仍有用于兼容性和 IR 测试的用途，但不构成生产能力证据。

## 8. TE 调度与 TIR

TE 计算与调度解耦：

- `ComputeOp` 描述张量计算；
- `Schedule`/`Stage` 记录循环变换；
- TE 到 TIR 的 Lowering 会校验并落实调度；
- TIR 流水线执行与目标相关的低层变换。

当前保留的调度原语为：

- `split`
- `reorder`
- `vectorize`
- `unroll`
- `parallel`

这些原语会影响最终循环结构，并进入附着于 `PrimFunc` 与编译产物键的规范调度契约。非法或不安全用法会直接报错。
`fuse`、`tile`、`bind`、`thread_axis`、`compute_at`、tensorization、autotuning 均未实现。

CPU 默认调度保守处理单射与归约循环；CUDA TE 循环保持串行。bounded 动态路径使用独立的 serial policy：只有 `InputAxis` 表达式生成有序 `uint64[1]` extent buffer，`Const` 轴保持静态 TIR extent；同一顺序同时生成模块 invocation scalar，并进入调度与编译产物身份。

bounded serial policy v2 支持由已登记 extent 驱动的归约循环；内部 TE 临时张量必须提供来自同一 invocation 合同的上界，并通过既有静态大小检查。实际循环和 LLVM scoped 分配使用本次长度。受限 adapter 的 MatMul batch 前缀、逐元素广播只接受相同维度表达式或常量 1 的可证明关系；Softmax 的归约长度下界必须大于零，ReduceMean 的归约轴须静态且为正。无 runtime extent 的静态权重变换复用默认 TE 调度，可与动态调用共存。
`tir::BindCudaThreads` v8 是唯一的 CUDA 输出独立性证明和启动元数据入口。它将静态行主序的多维输出坐标展平，每个线程拥有一个输出元素；先初始化、再读取同一元素的串行归约留在线程内，标量和空输出也有显式绑定与 guard。
多阶段张量先检查完整初始化和生产者顺序，消除只读输入复制/转换，再以共有独立坐标划分线程并压缩私有 Allocate（每线程总量最多 64 KiB）。每维地址都需证明不会进位到其他线程坐标；不支持的存储、跨线程/未初始化读取和未证明的动态阶段继续拒绝。静态单阶段只读 Gather 以分支区间证明索引表地址、负索引归一化、整数宽度及表地址。有界路径现复用这份保护证明，同时按实际紧凑 shape 校验索引张量地址；动态表、写入表或未证明的间接读取继续拒绝。形状重排还可证明正除数的商/余数、单例广播、静态表容量和分支内坐标范围；符号前缀加常量尾段的拼接按 j<P 分区证明，分支事实不会用于其他路径。TE 保持 serial，Pass schema v8 与 `cuda-nvrtc-driver-v8` 进入原有 pipeline/artifact identity。

MatMul 基础见 [输出归约报告](implementation/GPU_OWNED_REDUCTION_REPORT.md)，归一化见 [多阶段报告](implementation/GPU_MULTISTAGE_REDUCTION_REPORT.md)，静态 Gather/Pow 见 [报告](implementation/GPU_GATHER_POW_REPORT.md)。完整八层 B1/S16 静态模型已获得 [CUDA 数值证据](implementation/GPU_MINIMIND_PREFILL_REPORT.md)，静态容量 GPU state 与四步 decode 已有 [联合证据](implementation/GPU_KV_STATE_REPORT.md)。bounded 单阶段输出、形状值及 rank-2 MatMul 已有 [基础证据](implementation/GPU_BOUNDED_CORE_REPORT.md)；共有前缀行所有权与私有上界 scratch 支撑的 Softmax/MaskedSoftmax、固定轴 ReduceMean/RMS 与三头注意力已有 [数值证据](implementation/GPU_BOUNDED_REDUCTION_REPORT.md)。完整 bounded MiniMind 的 742 个原语已通过 CUDA 证明、源码生成和 PTX 编译，真实 GPU 数值及设备活动验收仍待执行，见 [接入报告](implementation/GPU_BOUNDED_PREFILL_REPORT.md)。同步内存生产者的默认流完成保证见 [修复报告](implementation/GPU_SYNC_MEMORY_REPORT.md)。

## 9. 身份、缓存与编译产物所有权

编译器使用三类身份：

| 身份 | 包含 | 排除 |
|---|---|---|
| `GraphSemanticKey` | 整图 Relay 语义 | 目标与编译器策略 |
| `UnitSemanticKey` | 规范化编译单元计算、属性、边界契约、副作用与别名 | 图编号、存储 ID、对象地址和链接符号 |
| `PrimitiveArtifactKey` | 编译单元语义、目标能力指纹、规范流水线、ABI 版本、精确调度契约与后端版本 | 请求热度、图内 ID、可变缓存状态 |

摘要只用于索引；完整的规范内容决定相等性。

原语缓存是编译器私有的权威组件，负责请求合并、等待者、失败传播、背压、淘汰与 Pin 生命周期。
`ArtifactPin` 是不可变的对外持有视图，不具备查找、发布或变更能力。

`AssembleCompiledGraph` 要求每个有序编译单元恰好对应一个就绪原语。它校验签名与目标兼容性后发布新的不可变图，不会再次运行 Relay、Lower TIR 或访问缓存。

## 10. 运行时执行模型

`CompiledGraph` 包含：

- 就绪的 `CompiledModule`；
- 不可变 `ExecutablePlan`；
- 保留的 `ArtifactPin`；
- 图语义键。

`ExecutablePlan` 与具体运行时实现无关。其 `ValueSpec` 条目描述逻辑值 ID、物理存储 ID、形状、数据类型、设备、输入/常量/输出角色、别名、状态、写模式、活跃区间和 `valid_bytes`；
`KernelCall` 条目描述符号及有序的逻辑输入输出。

`ExecutablePlan` 还可选携带 `structured_schedule`：一组结构化 region，其 task 只能是 kernel（按 `call_index` 引用本计划的调用点）、branch（CPU 标量 bool 谓词 + 独立 then/else region + Phi 绑定）或 loop（condition-before-body、carried 绑定、`max_trip_count`）。它是**正交方面**而非新的 `ExecutablePlanMode`：`mode` 仍决定分配与状态合同，`calls()` 仍是每个调用点恰好一个模块入口和一个 artifact pin，拓扑只决定“执行哪些调用点、以什么顺序”。没有 schedule 时按线性路径执行；有 schedule 时由同一 `RuntimeSession` 遍历 region，复用同一参数准备、模块调用和完成管理，不新增 allocator、launch 通道或 state owner。结构化拓扑进入 plan ABI identity；region/task id 是本地定位符，实际谓词值与迭代次数是运行时数据，不进身份。

结构化计划当前仍限定静态模式：与 bounded/state 的组合（region 边界的状态更新、region-aware shape 证明）尚未实现，见 [M10 C3 计划](implementation/M10_C3_UNIFIED_CONTROL_PLAN.md) §7。

`RuntimeSession` 在执行前校验模块与计划的边界：

- 模块入口与计划调用一一匹配；
- 符号与内核签名完全一致；
- 当前会话单一执行设备；
- 数据类型、形状、角色、常量键、对齐和元数均匹配；
- 存储复用、别名链、状态值与写入顺序有效；
- 静态模式不接受运行时范围绑定。

`Run` 使用模块的默认设备流；`RunAsync` 返回输出与 `AsyncOperation`，并在任务完成前保留存储与模块所有权。状态缓冲区由会话持有，有状态运行会被串行化。`RuntimeSession::Validate` 只验证模块与计划，不分配状态；`RunAsyncWithModule` 在一次调用中消费合同相同的显式模块，保留原默认模块、状态和计划，不负责判断数学等价或路由选择。会话不会编译缺失变体，不检查 Relay，也不改写原语缓存。

`kDynamicFreshOutputV1` 模式的生产 producer 是 `CompileBounded`；`RuntimeSession` 只接收通过相同契约校验的 CPU/LLVM 或获准 CUDA module/plan。它在首个 launch 前按 wildcard input-axis 顺序统一检查 graph bounds、整除和 shared-symbol 等式；每个 call 再由模块契约解析输出与 extent scalar。输出满足 `logical == physical == valid`，每次运行和每个中间值均使用独立 fresh storage。静态常量仍由模块注入，可与静态调用共存；受限 producer 拒绝直接返回常量。该模式拒绝 state、alias、donation、storage reuse、常量动态轴和控制计划；CUDA 还必须通过后端地址与线程所有权证明。完成对象会保活模块、计划、中间存储和先前 operation。

持久有效长度有两个独立版本：`kDynamicStatefulV1` 的 production KV 编译器生成 runtime extent ABI 与原地 alias 追加；`kStaticStatefulExternalV1` 由 `ExecutablePlan::BindStateOutputs` 将静态 past/present 输入输出挂到会话状态。后者保留固定 kernel 形状，将私有 present 尾段复制到已提交 cursor，在所有 CPU/LLVM 或 CUDA kernel 与同步状态复制完成后提交长度；`RunAsync` 返回前已完成。它仅允许同一 CPU:0 或 CUDA 设备的 float32 state/source，拒绝可写 state 输入、kernel alias/donation 和容量越界。`InitializeState` 通过显式前缀复制接收已完成的 prefill 输出。真实 MiniMind 的实现和证据见 [M2 报告](implementation/M2_MINIMIND_STATE_REPORT.md)与 [GPU 状态报告](implementation/GPU_KV_STATE_REPORT.md)。静态状态返回的是已完成句柄，跨 stream 使用仍逐步提交；没有增加异步状态队列。

CUDA bounded fresh-output 的 launch 按有限上界生成、guard/地址按实际形状计算，上界进入原有 TE schedule identity；模块通过现有 `uint64[1]` 设备参数传入实际维度并保活。bounded state 的 CUDA 准入已复用同一 prefix/append owner，完整模型 consumer 已接入但设备数值/复制顺序尚未验收，见 [状态接入报告](implementation/GPU_BOUNDED_STATE_REPORT.md)；请求批处理现已接入同设备 CUDA 与显式 stream，但设备验收待执行，见 [批处理接入报告](implementation/GPU_REQUEST_BATCHING_REPORT.md)。证明、82 条设备 kernel 与 182 次 scalar DMA 的基础记录见 [报告](implementation/GPU_BOUNDED_CORE_REPORT.md)。有界多阶段归约和注意力新增 204 条设备 kernel 与 456 次 scalar DMA，逐 call 核验 DMA 完成先于 kernel 开始，见 [多阶段报告](implementation/GPU_BOUNDED_REDUCTION_REPORT.md)。后续完整 bounded decode 的 774 个原语已通过源码/PTX 编译，见 [编译接入报告](implementation/GPU_BOUNDED_DECODE_REPORT.md)；本版本 GPU 执行与状态验收尚未完成。

## 11. 形状与自适应控制面

静态编译仍要求精确形状。可选形状层不会把运行时变成通用动态编译器。

- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT` 启用生产级精确形状适配器，复用准备、原语编译与图组装链路；
- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE` 启用源码树内的实验性决策层，把受限符号模板解析为精确值，并校验生成的精确编译产物；它也能铸造不可由裸 `Function` 伪造的 `BoundedCompileRequest`，但 adapter 本身不编译、缓存、分配或执行；
- `KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH` 启用显式 `Compiler::CompileBounded`。它要求 dynamic module ABI、restricted/exact shape gates 和可用 LLVM >= 20 或 CUDA 后端；编译结果仍是普通不可变 `CompiledGraph`，其 plan ABI 包含 bounded gate/version、graph guards 与每个 module invocation contract；
- `ExactProfileRouteTable` 是调用方持有的有限路由表，用于查询已发布的精确图；未命中会报错，不允许产生隐式编译副作用；
- `KXC_ENABLE_ADAPTIVE_HOT_SWAP` 启用边界受限的全图替换控制面。替换过程复用 `CompilePrimitiveUnits` 与 `AssembleCompiledGraph`，旧代际通过租约保活。

adaptive 合同 v7 的 `RunAsync/RunNextBatch` 从实际 lease 填写 generation、route、ABI、variant 与 validation receipt；调用方不能覆盖这些保留字段。preparation v5 在静态无状态与静态外部状态计划之外，新增由 adapter 请求授权的有界入口，核对输入范围与逐个调用连接；组合开关下的完整八层 CPU 模型已执行 742 次 prefill 与 3,096 次 decode kernel，代际 1→2→1→1、16 份 KV 地址不变、extent 4→8，见 [有界热替换报告](implementation/M6_BOUNDED_REPORT.md)。有状态 candidate 不共享 RuntimeSession，由 `CreateStatefulSession` 为单个请求或一组有界请求创建状态；每步固定当前 lease，并经 `RunAsyncWithModule` 使用原会话计划、KV 和长度表。完整八层静态容量 CPU MiniMind 已执行代际 1→2→1→1、健康隔离与回滚，16 份 KV 地址不变、extent 16→20，见 [状态热替换报告](implementation/M6_STATEFUL_REPORT.md)。CPU 请求批处理热替换已通过完整八层模型，队列和槽位沿同一 RuntimeSession 保留，代际 1→2→2→1→1；四配置全量回归 55/55、56/56、70/70、75/75 及集成审计通过，见 [请求热替换报告](implementation/M6_REQUEST_BATCHING_REPORT.md)。CompiledGraph 通过公开的 BindStateOutputs、BindBoundedStateOutputs、BindRequestBatching 派生受原合同验证的状态产物。CUDA 热替换尚未验收；不同状态容量/布局/Plan ABI 的迁移仍拒绝。有界路由按完整显式身份查询，不做重叠范围搜索。原静态无状态证据见 [M6 报告](implementation/M6_RUNTIME_REPORT.md)。

`DispatchKey`、`PlanAbiFingerprint` 和 `PlanVariantKey` 共享不可变 canonical 字节；route/隔离容器直接持有这些已有身份，flight/负缓存共享一次构建的完整字节。哈希仅选择 bucket，相等性仍比较完整 canonical 内容，逻辑 metadata 预算不变。首次身份与 flight 字节构建仍有大模型成本，实测内存见状态热替换报告。

任何路径都不会对“近似兼容形状”执行模糊缓存匹配。

## 12. 前端与 ONNX

`python/kxc_onnx/` 下的 Python 代码解析 ONNX Proto，生成静态导入规格与参数字节流。
C++ 前端重建器（`src/frontend/onnx_importer.cc`）构造真实的 Relay 表达式与常量，执行类型推导并校验输出契约。

未知秩与不支持的符号维度会直接拒绝，但保留 ONNX 文档中的限定轴 `default_batch` 绑定例外。
不支持的算子或语义子集会给出上下文化诊断，不会静默丢弃节点或在主机侧执行。

## 13. 性能分析与工作台

当前性能分析 Span 覆盖编译准备、原语编译、汇总组装、Pass 执行、
形状与自适应编译，以及已经安装 Span 的分布式计划和通信执行路径。可选 CUPTI 可关联受支持的 CUDA 活动。
`RuntimeSession` 的 run、kernel submit/exec、分配与复制已接入 runtime observer。调用方的 `ExecutionMetadata` 随运行事件关联，真实 MiniMind bundle 已携带 export receipt、stage、extent、generation 和既有 builder 计算出的 plan ABI digest；这些观测字段不改变 kernel ABI。
Profile Bundle 可包含事件 JSONL、Chrome/Perfetto Trace、IR 编译产物与 Manifest。

runtime 的同步 OnScopeEnter/退出钩子让 profiling 适配器在真实提交线程安装 CUPTI run/span。唯一 kernel_launch span 保留 call_index 与模型字段，设备 activity 的 parent 指向该 launch，再连接 run；既有 kernel_submit/exec 的 parent 仍为 run。全局 CUPTI collector 按 profile session_id 路由并分别对齐时钟；context 注销与回调写入共用锁。device_execute 与主机提交/观测完成时间严格分开，完整模型和并发证据见 [CUDA 关联报告](implementation/M1_CUDA_CORRELATION_REPORT.md)。

性能分析只负责观测，不会改变 Pass 顺序、选择目标、在缓存未命中时触发编译，或放宽运行时契约。

`tools/workbench/` 下的工作台读取有界 Bundle，用于检查与对比。Fixture 是合成回归数据，不能证明后端或分布式内核已经具备生产支持。

## 14. 分布式子系统

`include/kxc/distributed/` 与 `src/distributed/` 提供与具体运行时实现无关的执行计划、Worker、会话、
JSON 序列化、放置策略与 CPU 集合通信后端。该子系统与单目标编译器发布链路彼此独立。

通信节点通过 `CCLBackend` 执行；kernel 节点消费外部绑定的 `CompiledModule` 与完整 `KernelSignature` 字节。执行前校验整份计划的 use/def、worker/Target、张量与通信合同；随后各 worker 线程调用真实 CPU/LLVM kernel，等待 completion 后再进入下一节点。每次执行独立持有 value Map，DRef 最后引用释放时清理寄存器张量。

distributed JSON 现在写出 v3 的有序 `output_value_ids`，读取 v2 时把旧单输出转为一个元素的列表，仍拒绝 v1 与未知字段。旧单输出构造和 `output_value` 是受校验的首输出视图，`ExecuteForOutputs` 返回全部输出，预检核对每个输出的 worker 可用性。Target 规范字节继续由 `Target::CanonicalBytes()` 提供，单目标 compiler helper、Kernel ABI 与 artifact 身份格式不变。

`ExecutionPlan::FromExecutablePlan` 通过纯 `RuntimeSession::Validate` 验证已编译的静态 fresh-output 图，再按显式输入/调用 worker 和输出 worker 生成现有 kernel/copy 节点。常量由模块注入，共享数据的跨 worker 副本按来源和目的地复用。完整八层 B1/S16 MiniMind prefill 已用两套放置执行 1,300 次 CPU/LLVM kernel，全部 17 个输出与单机逐位相等，并对齐 401,408 个独立 ONNX 值，见 [整图报告](implementation/M7_MODEL_REPORT.md)。旧 2-worker 小图与 4-worker/2-group CCL 证据见 [M7 首版报告](implementation/M7_DISTRIBUTED_REPORT.md)。当前仍不支持动态 extent、state/alias、CUDA worker、跨机器服务或自动分区/重叠调度。

## 15. 特性开关与后端

| 选项 | 默认值 | 边界 |
|---|---:|---|
| `KXC_ENABLE_LLVM` | ON | 仅当发现 LLVM >= 20 时可用；否则 LLVM 测试与编译不可用 |
| `KXC_ENABLE_CUDA` | ON | 仅当发现 CUDA Toolkit 时启用；显式 CPU 预设会关闭 |
| `KXC_ENABLE_CONTROL_RUNTIME` | OFF | CPU/LLVM 上的精确控制拓扑 |
| `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI` | OFF | 非 `const` 模块调用契约；只有受支持 plan mode 才能执行 |
| `KXC_ENABLE_SHAPE_PRODUCTION_EXACT` | OFF | 精确形状适配器 |
| `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE` | OFF | 受限决策型符号形状层，启用后进入精确形状适配器 |
| `KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH` | OFF | 显式 bounded LLVM/CUDA `CompileBounded`；要求前述 dynamic/restricted/exact gates 与至少一个可用后端 |
| `KXC_ENABLE_ADAPTIVE_HOT_SWAP` | OFF | 有界全图自适应替换 |

C 后端会输出可读源码，主要用于诊断和 AOT 构建。LLVM 产出可执行的 CPU 编译产物；CUDA 在 Toolkit 与设备可用时生成、加载并启动受支持的内核。
后端可用性不会自动扩大算子或调度契约。

## 16. 验证

最小架构校验链：

```powershell
python tools/architecture/check_docs.py --root .
python tools/architecture/check_include_layers.py --root .
python tools/architecture/check_public_headers.py --root . --compile
python python/tools/check_relay_op_contract.py --root . `
  --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . `
  --matrix contracts/pass_contract.json
git diff --check
```

对于已配置的构建目录：

```powershell
cmake --build out/build/dev-mingw-cpu --parallel
ctest --test-dir out/build/dev-mingw-cpu `
  --output-on-failure --no-tests=error
```

LLVM 与 CUDA 数值测试要求对应工具链可用。合成 CUDA 目标测试只验证契约行为，不能代表完整的硬件执行结果。若某后端被跳过或不可用，应记录为验证缺口，不能记为后端验证成功。

## 17. 架构变更规则

架构相关 PR 必须一次性更新所有受影响的权威项：

1. 公共头文件与实现；
2. 适用时更新机器可读的算子与 Pass 契约；
3. 生成产物与契约检查器；
4. 正向测试与失败即拒绝测试；
5. 本文档；
6. 公开工作流变化时，更新对应的算子、Pass、ONNX 或构建文档。

不要在 `docs/` 中新增按日期划分的实施计划、迁移交接或会议纪要；历史性说明请放在 issue 与 PR 中，
本文件只保留当前系统状态。
