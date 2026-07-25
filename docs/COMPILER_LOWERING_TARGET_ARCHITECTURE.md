# Compiler Relay、ControlPlan 与 Primitive Lowering 目标架构

> **状态：** Proposed
> **文档类型：** 目标架构与迁移合同，不是当前能力声明
> **编写日期：** 2026-07-25
> **取证基线：** `compiler-foundation-acceptance-cleanup` / `affea3ecf8a7`
> **适用范围：** Relay 程序准备、静态数据流规划、控制流规划、primitive lowering、artifact 编译及 runtime plan 绑定

## 1. 执行摘要

当前代码中以下主链路是合理的，必须保留：

```text
Relay 控制结构
  -> ControlPlan lowering
       -> Region / Branch / Loop / Phi / value routing
       -> 叶子 primitive
            -> TE
            -> TIR
            -> Kernel ABI
            -> backend artifact
```

需要收口的不是“控制流规划”和“primitive 物理 lowering”这两个阶段，而是它们之间及其前后的平行合同：

1. 静态编译与控制流编译分别准备和解释 Relay。
2. Operator Call 的 arity、attrs、type relation 和 lowering binding 被多处重复验证。
3. 静态 `ValueGraph` 与 `ControlPlanBuilder` 分别展开 tensor leaf、分配 value id 并重建 value contract。
4. 静态路径使用正式 `CompilationUnit`，控制流路径使用 `task_id -> Function` sidecar。
5. 控制流路径把每个叶子 Function 重新送入完整 `Compiler::Compile`，重复 graph pipeline、ValueGraph 和 partition。
6. 公共 `relay::LowerToTIR` 仍保留 whole-graph lowering，与正式 per-unit lowering 并行。
7. `ValueSpec`、`ControlValueSpec` 与 `ControlExecutionValueSpec` 重复表达 dtype、shape 和 device。

目标架构建立以下唯一权威：

```text
PreparedRelayProgram
  -> ResolvedRelayCall
  -> LogicalValueContract
  -> PrimitiveUnit
  -> CompilePrimitiveUnits
  -> PrimitiveArtifact
```

静态数据流 planner 与控制流 planner 可以保留不同拓扑，但必须消费和产出上述共同合同。

## 2. 目标、非目标与约束

### 2.1 功能目标

- 支持静态数据流与 static-exact 控制流使用同一套 Relay Call 语义。
- 支持 `If`、bounded `While`、Phi 和 loop-carried value 的显式控制拓扑。
- 每个 ordinary compute Call 形成一个正式 `PrimitiveUnit`。
- 静态与控制流路径使用同一 primitive lowering、TIR pipeline、Kernel ABI、artifact identity、cache 和 backend 编译实现。
- Runtime 只接收已绑定、不可变、无需 Relay/TE/TIR/Compiler 回调的执行计划。
- 保留当前 fail-closed 行为；目标架构不扩大已批准的 operator、shape、device 或 backend 能力。

### 2.2 非功能目标

| 属性 | 目标 |
|---|---|
| 确定性 | 相同 Relay 语义、target 和 pipeline 产生相同 `UnitSemanticKey` 与 `PrimitiveArtifactKey` |
| 可维护性 | Operator Call 合同解析、tensor leaf 展开、primitive 编译各只有一个实现 |
| 可测试性 | analysis、planner、primitive compiler、plan binder 可以分别做纯单元测试 |
| 性能 | 每个 primitive 只准备和 lowering 一次；backend 支持批量编译，不按叶子重复运行完整 graph compiler |
| 资源安全 | artifact pin、module owner 和 control-plan retention 只有一个明确所有者链 |
| 可观测性 | graph、region、task、unit、artifact 的诊断身份分层且可关联 |
| 兼容性 | 删除 legacy public API 前，仓库内调用方全部迁移并提供明确的编译期失败信息 |

### 2.3 非目标

- 不把 `ControlPlan` 和 `ExecutablePlan` 强行合成一个拓扑类型。
- 不把 `If`、`While`、Phi 或 loop-carried value 塞进普通静态 `ValueGraph`。
- 不让 Runtime 读取 Relay、Operator registry、TE、TIR 或 Compiler cache。
- 不在本次收口中引入 dynamic shape、shape specialization、adaptive publication 或 hot swap。
- 不改变当前“一 ordinary Call 一 primitive unit”的 partition 策略；未来融合必须通过新的显式 partition policy。

### 2.4 约束

- C++17。
- 现有 `Compiler::Compile` 继续是静态数据流生产入口。
- `Compiler::CompileControlFlowExact` 在 capability 完全对齐前继续是显式、独立、默认关闭的控制流入口。
- `CompiledModule`、`ExecutablePlan`、`ControlExecutionPlan` 与 artifact cache 的不可变性不能倒退。
- 诊断字符串不能成为 ABI、artifact identity 或 binding authority。

### 2.5 安全与资源边界

- Relay 输入、attrs、shape、device 和 target snapshot 在进入 lowering 前必须视为未验证数据。
- analysis 和 planner 不得执行用户 kernel、backend callback 或 Runtime launch。
- shape element count、tensor byte count、参数数量和 plan id 必须在转换边界做溢出与范围检查。
- compiler function object、registry entry、裸指针和诊断文本不得穿过 Compiler/Runtime 边界。
- artifact 只有在 TIR、signature、launch metadata 和 backend executable 全部验证后才能发布。
- module、constant payload、artifact pin 和异步执行对象必须通过不可变 owner 链保活。
- 未识别的 operator、pass、control node、alias/effect 或 placement 一律 fail closed。

## 3. 当前架构与双轨位置

### 3.1 当前三条路径

```text
正式静态路径
Compiler::Compile
  -> ValidateInput
  -> Relay pipeline
  -> CapabilityVerifier
  -> ValueGraph
  -> PartitionedGraph / CompilationUnit
  -> LowerCompilationUnit
  -> TIR / ABI / cache / backend
  -> CompiledModule + ExecutablePlan

控制流路径
Compiler::CompileControlFlowExact
  -> InferTypePass
  -> NormalizeToANF
  -> internal executable capability
  -> ControlPlanBuilder
  -> task_id -> frozen Function sidecar
  -> 每个 Function 再调用 Compiler::Compile
  -> BindControlPlanForRuntime
  -> ControlExecutionPlan

兼容路径
relay::LowerToTIR
  -> InferTypePass
  -> RelayToTEConverter
  -> whole-graph PrimFunc
```

### 3.2 双轨清单

| 责任 | 当前实现 A | 当前实现 B/C | 漂移风险 |
|---|---|---|---|
| Relay 准备 | [`src/compiler/compiler.cc`](../src/compiler/compiler.cc) | [`src/compiler/control_flow/relay_control_plan.cc`](../src/compiler/control_flow/relay_control_plan.cc) | pass、ANF、capability 顺序和模式不一致 |
| Operator Call 合同 | [`src/compiler/control_flow/executable_capability.cc`](../src/compiler/control_flow/executable_capability.cc) | `relay_control_plan.cc`、[`src/compiler/lowering/lowered_graph.cc`](../src/compiler/lowering/lowered_graph.cc)、[`src/compiler/lowering/relay_to_tir.cc`](../src/compiler/lowering/relay_to_tir.cc) | arity、attrs、type relation、lowering binding 规则分叉 |
| Tensor leaf 展开 | [`src/compiler/graph/value_graph.cc`](../src/compiler/graph/value_graph.cc) | `relay_control_plan.cc`、`executable_capability.cc` | nested tuple、output arity、静态形状规则漂移 |
| Logical value | `ValueInfo` | `ControlValueSpec` | dtype、shape、device、origin 和 locator 重复建模 |
| Primitive 边界 | `CompilationUnit` | `task_id -> Function` sidecar | semantic key、参数顺序、常量顺序和 device 边界可能不一致 |
| Relay 到 TE | `LowerCompilationUnit` | `RelayToTEConverter` | whole-graph 与 per-unit cardinality 和常量语义不同 |
| Runtime value | `runtime::ValueSpec` | `ControlExecutionValueSpec` | runtime tensor contract 存在第二套权威 |

## 4. 目标高层架构

```text
                           +-------------------------+
Relay Function ---------->| PrepareRelayProgram     |
CompileConfig / Target --->| - mode-safe pipeline    |
                           | - typed ANF             |
                           | - one capability engine |
                           +------------+------------+
                                        |
                              PreparedRelayProgram
                                        |
                    +-------------------+-------------------+
                    |                                       |
         +----------v-----------+               +-----------v----------+
         | StaticDataflowPlanner|               | ControlFlowPlanner   |
         | ordered dataflow     |               | region/branch/loop   |
         +----------+-----------+               +-----------+----------+
                    |                                       |
          PreparedStaticPlan                     PreparedControlPlan
                    |                                       |
                    +-------------------+-------------------+
                                        |
                              vector<PrimitiveUnit>
                                        |
                           +------------v-------------+
                           | CompilePrimitiveUnits    |
                           | - unit -> TE             |
                           | - TE -> TIR              |
                           | - TIR pipeline           |
                           | - Kernel ABI             |
                           | - identity/cache/backend |
                           +------+-------------+-----+
                                  |             |
                    +-------------v--+       +--v----------------+
                    | Static binder  |       | Control binder    |
                    | module + plan  |       | bound regions     |
                    +--------+-------+       +---------+---------+
                             |                         |
                   CompiledGraph            CompiledControlFlowGraph
                             |                         |
                   RuntimeSession            ControlRuntimeSession
```

## 5. 分层责任

| 层 | 唯一责任 | 可以依赖 | 禁止依赖 |
|---|---|---|---|
| Relay | IR、OperatorSpec、attrs、type relation、lowering binding 注册 | support、target type vocabulary | Compiler cache、Runtime session |
| Compiler analysis | mode-safe pipeline、typed ANF、Call 合同解析、logical tensor contract | Relay、Pass、Target | backend handle、Runtime executor |
| Planner | value routing、partition、region、branch、loop、Phi、依赖 | analysis contract、identity | TE/TIR 实现细节、cache mutation |
| Primitive lowering | 一个 `PrimitiveUnit` 到 TE/TIR | resolved call、logical values、TE/TIR | graph/control topology |
| Primitive compiler | TIR pass、Kernel ABI、artifact key、cache、backend | lowering、identity、codegen | Relay traversal、control execution |
| Plan binder | 将已编译 primitive 绑定到静态或控制执行拓扑 | artifacts、runtime plan contracts | Operator registry、TE/TIR |
| Runtime | plan validation、storage、launch、completion | immutable module/plan/value contract | Relay、Compiler、cache、registry |

依赖方向必须保持：

```text
Relay/Pass
   -> Compiler Analysis
       -> Planner
           -> Primitive Lowering
               -> Primitive Compiler
                   -> Plan Binder
                       -> Runtime Contracts

Runtime Executor 不得反向依赖以上 Compiler 层。
```

## 6. 核心合同

以下接口为目标形态示意，名称可以在实现评审中微调，但责任边界不可改变。

### 6.1 `PreparedRelayProgram`

```cpp
enum class RelayProgramMode {
    kStaticDataflow,
    kControlFlow,
};

struct PreparedRelayProgram {
    Function typed_anf;
    RelayProgramMode mode;
    Target target;
    CompilerExecutionContract execution_contract;
};

PreparedRelayProgram PrepareRelayProgram(
    Function function,
    const CompileConfig& config,
    RelayProgramMode mode);
```

职责：

- 解析并冻结 target 与 pipeline contract。
- 执行该 mode 明确允许的 Relay passes。
- 建立 typed ANF。
- 调用唯一 capability engine。
- 不分配 runtime storage，不创建 TIR，不访问 primitive cache。

控制流 mode 只运行明确声明为 control-safe 的 graph passes。未知 pass 必须拒绝，不得默认假定可以跨 `If`/`While` 改写。

### 6.2 `ResolvedRelayCall`

```cpp
struct ResolvedRelayCall {
    Call call;
    relay::OperatorSpec spec;
    relay::Attrs attrs;
    relay::FInferType infer_type;
    std::variant<relay::FRelayToTE, relay::FRelayToTEMulti> lowering;
    std::vector<Type> input_types;
    std::vector<Type> output_leaf_types;
};

ResolvedRelayCall ResolveRelayCall(
    const Expr& call_expr,
    const OperatorCapabilityPolicy& policy);
```

这是以下规则的唯一执行点：

- Call target 必须是 registry 中的正式 `Op` 实例。
- `OperatorSpec` schema 必须有效。
- input arity 与 attrs type 必须匹配。
- type relation binding 与 lowering binding 必须存在且类型正确。
- checked type 必须与 type relation 重新计算结果一致。
- output leaf count 必须与 `OperatorSpec` 一致。
- pure/deterministic/alias/device 等能力由显式 policy 判断。

除该模块外，Compiler 不得直接 `any_cast<FInferType>`、`any_cast<FRelayToTE>` 或按字符串查找 lowering key。

### 6.3 `LogicalValueContract`

```cpp
using ValueId = std::int64_t;

enum class LogicalValueOrigin {
    kParameter,
    kConstant,
    kPrimitiveOutput,
    kPhi,
    kLoopCarried,
};

struct LogicalValueContract {
    ValueId id;
    Type checked_type;
    Device device;
    LogicalValueOrigin origin;
    Expr source;
    std::string source_locator;
};
```

约束：

- `checked_type` 是 compiler 侧唯一 dtype/shape 权威。
- value id 只标识计划内连线，不进入 primitive semantic key。
- storage id 不属于 logical value。
- control planner 可以增加 Phi/loop-carried origin，但不得重建 string dtype/shape ABI。
- compiler/runtime 边界只进行一次 `Type -> runtime::ValueSpec` 转换。

### 6.4 `PrimitiveUnit`

```cpp
using PrimitiveUnitId = std::int64_t;

struct PrimitiveUnit {
    PrimitiveUnitId id;
    String symbol;
    ResolvedRelayCall call;
    std::vector<ValueId> argument_value_ids;
    std::vector<ValueId> boundary_input_value_ids;
    std::vector<ValueId> output_value_ids;
    Device device;
    UnitSemanticKey semantic_key;
};
```

字段语义：

- `argument_value_ids` 保留 Call 原始参数顺序与重复值。
- `boundary_input_value_ids` 是物理 ABI 边界输入，按正式 constant ordering 规则排列。
- `output_value_ids` 与 OperatorSpec output leaf 一一对应。
- `semantic_key` 只组合 operator、attrs、logical argument mapping、tensor contract 与必要 lowering policy。
- graph-local unit id、value id、symbol 和 source locator 不得进入数学语义。

静态 planner 与控制 planner 都必须产出这个类型，不允许再建立 `task_id -> Function` 平行 sidecar。

### 6.5 `CompilePrimitiveUnits`

```cpp
struct CompiledPrimitive {
    PrimitiveUnitId unit_id;
    PrimitiveArtifactKey artifact_key;
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel kernel;
    ArtifactPin pin;
};

std::vector<CompiledPrimitive> CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract);
```

唯一流水线：

```text
PrimitiveUnit
  -> boundary TE tensors
  -> registered operator lowering
  -> LowerTensorGraphToTIR
  -> TIR pipeline
  -> KernelSignature
  -> PrimitiveArtifactKey
  -> primitive cache transaction
  -> backend batch compile
  -> immutable artifact pin
```

该接口接受一批 units，以保留 LLVM/CUDA backend batch build 和 cache owner/waiter 顺序。

## 7. Planner 设计

### 7.1 `StaticDataflowPlanner`

保留当前 `ValueGraph -> PartitionedGraph` 的基本职责：

- 接收不含 `If`/`While` 的 `PreparedRelayProgram`。
- 建立确定性的 ordinary value routing。
- 为每个 ordinary Call 生成一个 `PrimitiveUnit`。
- 生成有序 `KernelCall` 拓扑。
- 输出静态 `runtime::ExecutablePlan` 草案。

它不再：

- 重新解析 OperatorSpec binding。
- 自己实现 tensor leaf 展开规则。
- 直接调用 backend。

### 7.2 `ControlFlowPlanner`

保留当前 ControlPlan 中真正独有的逻辑：

- Region。
- Branch 与 Phi。
- condition-before-body bounded While。
- loop-carried value。
- live-in/live-out。
- task dependency。
- control-specific placement 约束。

它不再：

- 手工重新执行 InferType、ANF 或另一套 capability engine。
- 重复验证 arity、attrs、type relation 和 lowering binding。
- 保存 `kernel_ref` 作为 binding authority。
- 构造叶子 Function sidecar。
- 表达独立的 dtype/shape/device value ABI。

目标内部类型应命名为 `PreparedControlPlan` 或 `UnresolvedControlPlan`，并位于 Compiler internal namespace。它不是 Runtime 可直接执行的对象。

### 7.3 为什么两个 planner 不合并

静态数据流与控制流具有不同的拓扑不变量：

| 静态数据流 | 控制流 |
|---|---|
| 全序或 DAG kernel calls | Region 内任务与跨 Region 转移 |
| value producer 唯一 | Phi 选择、loop backedge |
| 无条件执行 | predicate 与 bounded iteration |
| 普通 storage lifetime | 分支选择与 loop-carried retention |

强行使用一个 planner 会把大量 mode 条件散落到同一个 visitor 中。目标是共享语义合同，不共享拓扑算法。

## 8. Runtime 边界

### 8.1 静态路径

```text
CompiledPrimitive[]
  + static topology
  + runtime::ValueSpec[]
  -> CompiledModule
  -> ExecutablePlan
  -> CompiledGraph
```

### 8.2 控制流路径

```text
CompiledPrimitive[]
  + PreparedControlPlan
  + runtime::ValueSpec[]
  -> BoundControlKernel[]
  -> ControlExecutionPlan
  -> CompiledControlFlowGraph
```

### 8.3 必须满足的不变量

- Runtime plan 不含 Relay `Expr`、`Call`、OperatorSpec 或 compiler callback。
- 每个 kernel task 绑定一个 ready artifact、正式 symbol、KernelSignature 和 retention owner。
- `runtime::ValueSpec` 是静态与控制流执行计划共享的 tensor/value contract。
- source locator 可以作为独立诊断元数据存在，但不参与 ABI 或 artifact identity。
- 未绑定 plan 不能由 public Runtime API 构造或执行。
- Runtime 不得根据 `kernel_ref`、诊断文本或 source locator 查找 artifact。

## 9. Public API 目标

保留：

```cpp
class Compiler {
public:
    static CompiledGraph Compile(Function, CompileConfig);
    static CompiledControlFlowGraph CompileControlFlowExact(
        Function, CompileConfig);
};
```

删除正式 public API：

```cpp
relay::LowerToTIR(Function);
relay::LoweredFunction;
relay::ConstantBinding;
```

原因：

- 它们暴露的是 compiler internal primitive lowering 中间态。
- whole-graph 单 PrimFunc 语义与正式 per-unit module/plan 语义不一致。
- focused 测试需求不能成为安装 public API 的理由。

迁移后：

- 生产调用方使用 `Compiler`。
- compiler 单元测试链接 internal test target。
- TE/TIR focused 测试使用 `test/support/primitive_lowering.h` 或直接测试 `LowerTensorGraphToTIR` 的 internal seam。

## 10. 目标文件布局

```text
include/kxc/compiler/
  compiler.h
  capability.h
  compile_config.h
  identity.h

include/kxc/runtime/
  executable_plan.h
  control_execution_plan.h
  compiled_module.h
  session.h

src/compiler/
  analysis/
    relay_program.cc
    resolved_relay_call.cc
    logical_value.cc
  planning/
    static_dataflow_plan.cc
    control_flow_plan.cc
  lowering/
    primitive_to_te.cc
    te_to_tir.cc
  primitive_compiler.cc
  control_flow/
    control_plan_adapter.cc
    production_control_flow.cc
  internal/
    analyzed_relay_program.h
    primitive_unit.h
    primitive_compiler.h
    prepared_static_plan.h
    prepared_control_plan.h
```

以下 public 文件退出安装集：

```text
include/kxc/compiler/lowering/relay_to_tir.h
```

以下当前文件的职责迁移：

| 当前文件 | 目标处置 |
|---|---|
| `src/compiler/lowering/relay_to_tir.cc` | 拆出 `te_to_tir.cc`，删除 public whole-graph converter |
| `src/compiler/lowering/lowered_function.cc` | 移入 internal primitive lowering result |
| `src/compiler/lowering/lowered_graph.cc` | 拆为 static planner consumer 与 primitive lowering |
| `src/compiler/graph/value_graph.cc` | 收敛为 static planner，复用 common analysis |
| `src/compiler/graph/partition.cc` | 产出正式 `PrimitiveUnit` |
| `src/compiler/control_flow/relay_control_plan.cc` | 只保留 control topology，产出同一 `PrimitiveUnit` |
| `src/compiler/control_flow/internal_lowering.h` | 删除 `kernel_functions`，改为 `primitive_units` |
| `src/compiler/control_flow/production_control_flow.cc` | 调用 `CompilePrimitiveUnits`，不再递归调用完整 `Compiler::Compile` |
| `src/compiler/control_flow/control_plan.h` | 改为 Compiler internal unresolved plan，移出 `kxc::runtime` namespace |
| `src/compiler/control_flow/control_plan_adapter.cc` | 只做 unresolved-to-bound typestate conversion |
| `include/kxc/runtime/control_execution_plan.h` | value contract 复用 `runtime::ValueSpec` |

## 11. 迁移方案

迁移必须保持每一步可构建、可回退并有回归证据。

### 阶段 0：锁定行为

- 保留并扩展 `compiler_contract_test`、`operator_compilation_test`。
- 保留 `relay_control_plan_test`、`control_runtime_integration_test`。
- 增加静态与控制路径对同一 leaf Call 生成相同 semantic key 的测试。
- 增加 argument duplicate、constant ordering、多输出和 tuple leaf 的交叉路径测试。

### 阶段 1：统一 Operator Call 合同

- 新增 `ResolvedRelayCall`。
- 将 capability、static lowering 和 control planner 切换到同一 resolver。
- 删除本地 `ValidateInputArity`、`ValidateAttrs`、`ValidateOperatorBindings`。
- 验收：Compiler 内只有 resolver 可以读取 OperatorSpec implementation binding。

### 阶段 2：统一 logical tensor/value contract

- 抽取 tensor leaf 展开与 placement 解析。
- `ValueGraphBuilder` 和 `ControlFlowPlanner` 使用同一 `LogicalValueContract`。
- 保留各自确定性的 value-id 分配算法，但禁止复制 dtype/shape 权威。
- 验收：control preparation 不再创建 string dtype/shape contract。

### 阶段 3：引入正式 `PrimitiveUnit`

- 扩展当前 `CompilationUnit` 为共同 `PrimitiveUnit`。
- 静态 partition 与 control planner 均产出该类型。
- Control task 只引用 `PrimitiveUnitId`。
- 删除 `task_id -> Function` sidecar 和 `kernel_ref` binding 语义。

### 阶段 4：抽取 primitive compiler

- 从 `compiler.cc` 抽取 `CompilePrimitiveUnits`。
- 静态和控制路径共享 TIR、signature、identity、cache 与 backend batch build。
- Control path 不再为每个 leaf 递归执行完整 `Compiler::Compile`。

### 阶段 5：收口 Runtime value plan

- `ControlExecutionPlan` 复用 `runtime::ValueSpec`。
- unresolved `ControlPlan` 移入 Compiler internal namespace。
- adapter 只负责 plan typestate 与 artifact binding。

### 阶段 6：删除 legacy whole-graph lowering

- 迁移仓库内全部 `LowerToTIR` 测试调用。
- 从 `KXC_PUBLIC_HEADERS` 删除 `relay_to_tir.h`。
- 删除 `RelayToTEConverter` 与公开 `LowerToTIR`。
- 将共享 TE-to-TIR 代码迁到 `te_to_tir.cc`。
- public-header compile check 必须证明旧 include 不再可用。

## 12. 失败模式与处理

| 失败模式 | 检测位置 | 处理 |
|---|---|---|
| stale checked type | `ResolveRelayCall` | 在 planner 前拒绝，报告 Relay locator |
| attrs 或 arity 不匹配 | `ResolveRelayCall` | fail closed，不进入 unit 生成 |
| mode 不支持 If/While | `PrepareRelayProgram` capability policy | 报告所需 capability |
| control-unsafe graph pass | pipeline resolver | 配置阶段拒绝 |
| value placement 冲突 | planner | 要求显式 copy task 或拒绝 |
| primitive ABI 与 task value 顺序不一致 | plan binder | 绑定前拒绝 |
| artifact cache 编译失败 | `CompilePrimitiveUnits` | 保留分类失败状态，不发布半成品 |
| backend batch 部分返回 | primitive compiler | 整批失败或只发布已完整验证的独立 lease |
| missing artifact retention | immutable compiled bundle | 构造失败，Runtime 不接收 plan |
| diagnostic text 漂移 | identity tests | identity 只比较 canonical bytes |

## 13. 可观测性

统一后保留以下关联字段：

```text
graph_semantic_key
relay_program_mode
region_id
task_id
primitive_unit_id
unit_semantic_key
primitive_artifact_key
entry_symbol
pipeline_fingerprint
cache_hit
```

约束：

- `region_id`、`task_id`、`unit_id` 是一次 plan 内的 locator，不是跨图 semantic identity。
- `entry_symbol` 是 module lookup key，不是 mathematical identity。
- profiling 可以消费 canonical identity digest，但不能拥有 identity hash 算法。

## 14. 验证矩阵

| 层 | 必须验证 |
|---|---|
| Relay Call resolver | fixed/variadic arity、attrs type、type relation、single/multi lowering、effect/alias policy |
| Logical values | parameter、constant、tuple、nested tuple、duplicate argument、placement |
| Static planner | stable ids、ordered units、multi-output、constant ordering、plan determinism |
| Control planner | If/Phi、bounded While、loop-carried values、captures、live-in/out、dependency determinism |
| Cross-path | 相同 leaf Call 的 `UnitSemanticKey`、KernelSignature 和 artifact cache 命中一致 |
| Primitive lowering | input/constant/output ABI、shape overflow、dtype、multi-output、symbol identity |
| Primitive compiler | cache owner/waiter、failure publication、LLVM/CUDA batch build、pin retention |
| Plan binder | missing/extra binding、signature mismatch、value order mismatch、immutable snapshot |
| Runtime | 不依赖 Relay/TE/TIR/Compiler；并发 Run 不共享可变 value table |
| Public API | `LowerToTIR` header 不安装；正式样例只使用 `Compiler` |

最低回归集合：

```text
compiler_contract_test
compiler_identity_test
operator_compilation_test
graph_partition_test
relay_control_plan_test
control_plan_test
control_runtime_integration_test
runtime_session_test
kernel_signature_test
infer_type_test
pass_pipeline_test
check_relay_op_contract
```

## 15. ADR

### ADR-001：保留 ControlPlan 与 primitive lowering 两阶段

**状态：** Proposed

**背景：** 控制拓扑与 kernel 物理实现具有不同不变量。前者包含 Region、Phi 和 loop backedge，后者包含 TE/TIR、ABI 和 backend artifact。

**决策：** 保留两个阶段，通过正式 `PrimitiveUnit` 单向连接。

**正面影响：**

- 控制 planner 不理解 TE/TIR。
- primitive compiler 不理解 Branch/Loop。
- 两层可以独立测试和扩展。

**负面影响：**

- 需要显式 plan binder。
- 需要维护 `PrimitiveUnitId` 到 artifact 的绑定表。

**拒绝的替代方案：**

- 一个 visitor 同时生成 control topology 和 TIR：职责混合，难以验证。
- Runtime 遇到 control task 时动态回调 Compiler：破坏不可变执行边界。

### ADR-002：静态与控制流共享 Relay Call 和 value 合同，不共享 topology builder

**状态：** Proposed

**决策：** 建立 `PreparedRelayProgram`、`ResolvedRelayCall` 和 `LogicalValueContract`；保留独立的 StaticDataflowPlanner 与 ControlFlowPlanner。

**正面影响：**

- 消除 OperatorSpec 和 tensor contract 多重权威。
- 不牺牲两类拓扑各自清晰的不变量。

**负面影响：**

- analysis contract 必须支持两种 planner 所需的最小公共信息。
- mode policy 需要严格测试。

**拒绝的替代方案：**

- 完全独立的两个 compiler front-end：继续产生规则漂移。
- 一个巨型通用 planner：大量 mode 分支，控制与静态不变量混杂。

### ADR-003：控制流叶子使用正式 `PrimitiveUnit`

**状态：** Proposed

**决策：** 删除 `task_id -> Function` sidecar；control task 引用与静态 partition 相同的 `PrimitiveUnit`。

**正面影响：**

- semantic identity、常量顺序和 ABI 边界一致。
- 控制流不再重复运行完整 graph compiler。
- backend 可以批量编译控制流全部 leaf units。

**负面影响：**

- 需要从 `compiler.cc` 抽取 primitive batch compiler。
- 当前依赖 `CompiledGraph` 作为临时 artifact bundle 的代码需要迁移。

### ADR-004：删除 public whole-graph `LowerToTIR`

**状态：** Proposed

**决策：** 正式 public API 只保留返回 immutable compiled bundle 的 `Compiler` 入口；TE/TIR focused seam 变成 test/internal API。

**正面影响：**

- production capability 只有一个权威入口。
- 不再把 whole-graph 单 PrimFunc 误认为正式多 entry 编译模型。
- compiler intermediate 不再污染 public headers。

**负面影响：**

- 仓库内大量 focused tests 需要迁移。
- 未知外部调用方可能需要一个弃用周期。

**替代方案：**

- 永久保留 deprecated API：拒绝，因为它仍要求维护第二套 Relay-to-TE 前端。
- 将多个 PrimFunc 伪装成一个 `LoweredFunction`：拒绝，因为返回类型无法正确表达正式结果。

### ADR-005：只有 bound immutable plan 可以进入 Runtime

**状态：** Proposed

**决策：** unresolved control plan 保持 Compiler internal；Runtime 只接收使用正式 artifact 和 `runtime::ValueSpec` 完成绑定的 `ControlExecutionPlan`。

**正面影响：**

- Runtime 无需理解 compiler provenance。
- plan/module/signature/value contract 在执行前一次性验证。
- retention owner 明确。

**负面影响：**

- compiler/runtime adapter 是必须维护的显式边界。

## 16. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 一次性重写过大 | 回归难定位 | 按第 11 节分阶段，每阶段保持 green |
| common analysis 变成巨型万能层 | 新的耦合中心 | 只提供 Call、value、policy 数据，不生成任何 topology |
| control-safe pass 定义不足 | 控制语义被错误改写 | 默认拒绝，逐个 pass 声明并测试 |
| identity 在迁移中变化 | cache miss 或错误复用 | 增加跨路径 canonical bytes golden tests |
| `LowerToTIR` 外部用户未知 | API break | release note、deprecated 周期或明确 major-version 变更 |
| 复用 `ValueSpec` 时混入 storage 语义 | logical/physical 再次混淆 | logical analysis 不持有 storage id；只在 runtime plan 构造时赋值 |
| primitive compiler 抽取破坏 backend batching | 编译性能回退 | 接口以 batch 为单位，不设计成单 unit callback |

## 17. 完成定义

满足以下全部条件后，才可以认为本目标架构收口完成：

- [ ] `Compiler::Compile` 与 `CompileControlFlowExact` 共享 `PrepareRelayProgram`。
- [ ] Compiler 内只有一个 `ResolvedRelayCall` 合同解析实现。
- [ ] tensor leaf 和 logical tensor contract 只有一个权威实现。
- [ ] 静态与控制 planner 均产出同一 `PrimitiveUnit`。
- [ ] ControlPlan 不再包含或旁挂 frozen Function。
- [ ] 控制流叶子不再递归调用完整 `Compiler::Compile`。
- [ ] TIR、Kernel ABI、artifact identity、cache、backend 使用唯一 primitive compiler。
- [ ] unresolved ControlPlan 位于 Compiler internal namespace。
- [ ] Runtime control plan 复用正式 `runtime::ValueSpec`。
- [ ] public 安装集不再包含 `relay_to_tir.h`。
- [ ] 仓库内不存在 `relay::LowerToTIR` 调用。
- [ ] Runtime public headers 和实现不依赖 Relay、TE、TIR 或 Compiler internal。
- [ ] 第 14 节测试矩阵全部通过。
- [ ] 文档、样例、public-header compile check 与真实代码一致。
