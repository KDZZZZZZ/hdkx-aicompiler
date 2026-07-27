# hdkx-aicompiler 架构

> **状态：** 当前架构权威文档
> **更新时间：** 2026-07-27
> **规则：** 若本文档与实现有冲突，以源码、机器可读契约和可复现测试为最终依据。

## 1. 系统边界

`hdkx-aicompiler` 是一个精简的 TVM 风格编译器与运行时，遵循四条核心约束：

1. Relay 算子与 Pass 都有机器可读契约；
2. 生产编译以 `PrimitiveUnit` 为单位，不把整张图视为单个内核；
3. 编译结果包含不可变模块，以及与运行时实现无关的执行计划；
4. 不支持的语义会在执行前明确报错，不做隐式回退。

当前主路径面向静态精确、单目标编译。LLVM 是主要 CPU 后端；CUDA 仍是实验性后端，只开放保守的调度子集。控制流、符号形状决策、精确 Profile 路由、动态模块调用与自适应替换均为默认关闭的独立能力。

当前不作为生产承诺的能力包括：

- 通用动态、参差、数据相关或多态执行；
- 任意 CUDA 归约、间接访存或嵌套循环调度；
- 自动微分与训练能力；
- 分布式已编译内核执行；
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
     -> 感知目标的 TE 调度 -> TIR
     -> 规范化 TIR 流水线
     -> KernelSignature + 编译产物身份
     -> 获取原语缓存
     -> LLVM 或 CUDA 后端编译
  -> AssembleCompiledGraph
     -> CompiledModule + ExecutablePlan + ArtifactPin[]
  -> RuntimeSession
       -> 校验输入、分配并绑定值、按计划顺序启动内核
```

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
| `distributed` | 与运行时实现无关的分布式计划、Worker 与 CPU 集合通信模拟 | 当前单目标编译器发布 |
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

CPU 默认调度保守处理单射与归约循环；CUDA TE 循环保持串行。
`tir::BindCudaThreads` 是当前唯一能证明受支持的一维独占写映射并创建启动元数据的权威入口。
它会在代码生成前拒绝嵌套归约、写冲突和间接加载索引。

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

`RuntimeSession` 在执行前校验模块与计划的边界：

- 模块入口与计划调用一一匹配；
- 符号与内核签名完全一致；
- 当前会话单一执行设备；
- 数据类型、形状、角色、常量键、对齐和元数均匹配；
- 存储复用、别名链、状态值与写入顺序有效；
- 不接受未支持的运行时范围绑定。

`Run` 使用模块的默认设备流；`RunAsync` 返回输出与 `AsyncOperation`，并在任务完成前保留存储与模块所有权。状态缓冲区由会话持有，有状态运行会被串行化。会话不会编译缺失变体，不检查 Relay，也不改写原语缓存。

非 `const` 模块调用契约可在 `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI` 下构建，但在动态图内存规划完成前，`RuntimeSession` 仍会拒绝执行。

## 11. 形状与自适应控制面

静态编译仍要求精确形状。可选形状层不会把运行时变成通用动态编译器。

- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT` 启用生产级精确形状适配器，复用准备、原语编译与图组装链路；
- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE` 启用源码树内的实验性决策层，把受限符号模板解析为精确值，并校验生成的精确编译产物；它不独立完成编译、缓存、分配或执行；
- `ExactProfileRouteTable` 是调用方持有的有限路由表，用于查询已发布的精确图；未命中会报错，不允许产生隐式编译副作用；
- `KXC_ENABLE_ADAPTIVE_HOT_SWAP` 启用边界受限的全图替换控制面。替换过程复用 `CompilePrimitiveUnits` 与 `AssembleCompiledGraph`，旧代际通过租约保活。

任何路径都不会对“近似兼容形状”执行模糊缓存匹配。

## 12. 前端与 ONNX

`python/kxc_onnx/` 下的 Python 代码解析 ONNX Proto，生成静态导入规格与参数字节流。
C++ 前端重建器（`src/frontend/onnx_importer.cc`）构造真实的 Relay 表达式与常量，执行类型推导并校验输出契约。

未知秩与不支持的符号维度会直接拒绝，但保留 ONNX 文档中的限定轴 `default_batch` 绑定例外。
不支持的算子或语义子集会给出上下文化诊断，不会静默丢弃节点或在主机侧执行。

## 13. 性能分析与工作台

当前性能分析 Span 覆盖编译准备、原语编译、汇总组装、Pass 执行、
形状与自适应编译，以及已经安装 Span 的分布式计划和通信执行路径。可选 CUPTI 可关联受支持的 CUDA 活动。
通用的 `RuntimeSession`、分配和拷贝路径尚未完整覆盖，不能根据合成工作台场景推断这些能力。
Profile Bundle 可包含事件 JSONL、Chrome/Perfetto Trace、IR 编译产物与 Manifest。

性能分析只负责观测，不会改变 Pass 顺序、选择目标、在缓存未命中时触发编译，或放宽运行时契约。

`tools/workbench/` 下的工作台读取有界 Bundle，用于检查与对比。Fixture 是合成回归数据，不能证明后端或分布式内核已经具备生产支持。

## 14. 分布式子系统

`include/kxc/distributed/` 与 `src/distributed/` 提供与具体运行时实现无关的执行计划、Worker、会话、
JSON 序列化、放置策略与 CPU 集合通信后端。该子系统与单目标编译器发布链路彼此独立。

通信节点通过 `CCLBackend` 执行；分布式内核节点尚不能解析并启动 `CompiledModule`，
执行器只会记录 Span，并抛出 `ExecutionPlan CompiledModule launch is not implemented`。
因此，分布式计划、控制与通信测试可以覆盖对应逻辑，但分布式数值内核执行仍未形成受支持能力。

## 15. 特性开关与后端

| 选项 | 默认值 | 边界 |
|---|---:|---|
| `KXC_ENABLE_LLVM` | ON | 仅当发现 LLVM >= 20 时可用；否则 LLVM 测试与编译不可用 |
| `KXC_ENABLE_CUDA` | ON | 仅当发现 CUDA Toolkit 时启用；显式 CPU 预设会关闭 |
| `KXC_ENABLE_CONTROL_RUNTIME` | OFF | CPU/LLVM 上的精确控制拓扑 |
| `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI` | OFF | 非 `const` 模块调用契约，不等同于通用动态会话内存规划 |
| `KXC_ENABLE_SHAPE_PRODUCTION_EXACT` | OFF | 精确形状适配器 |
| `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE` | OFF | 受限决策型符号形状层，启用后进入精确形状适配器 |
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
