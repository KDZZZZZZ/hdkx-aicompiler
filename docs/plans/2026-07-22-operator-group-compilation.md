# 按算子独立编译与算子/Pass 规范实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 将当前“整张 Relay 图编译为唯一 PrimFunc/Kernel”的流程改造成“每个可计算算子独立形成编译单元、独立生成 PrimFunc/Kernel、Runtime 按图计划调度”，同时建立统一的算子规范和 Pass 规范；不新增任何具体算子或具体 Pass 算法。

**Architecture:** Relay 仍负责整图语义和图级变换；Compiler 为图值建立稳定身份，并把每个普通计算 `Call` 固定划分为一个 `CompilationUnit`，随后逐单元执行 lowering、TIR pipeline、Kernel ABI 构建和后端编译。`OperatorSpec` 决定一个算子是否具备独立编译所需的完整语义，`PassSpec` 决定 Pass 可作用的 IR 层级、作用域和前后置条件；Runtime 只消费多入口 `CompiledModule` 与不含 Relay/TE/TIR 的 `ExecutablePlan`。

**Tech Stack:** C++17、Relay/TE/TIR、ObjectRef、PackedFunc Registry、LLVM ORC JIT、CUDA/NVRTC、NDArray、DeviceStream、JSON contract、CMake/Ninja。

---

## 1. 核心结论

本计划的主目标保持不变：

```text
Relay Function
  -> 图级规范化与类型检查
  -> 稳定 value id
  -> 每个普通计算 Call 一个 CompilationUnit
  -> 每个 CompilationUnit 一个 PrimFunc
  -> 每个 PrimFunc 一个可寻址 kernel entry
  -> ExecutablePlan 按数据依赖顺序调用多个 kernel
```

这里的“按算子独立编译”是强制迁移门禁，不是可选示例：

- 一个普通计算 `Call` 必须且只能属于一个编译单元。
- 一个编译单元在本计划内必须且只能包含一个普通计算 `Call`。
- 一个合法编译单元必须且只能 lowering 为一个 `PrimFunc`。
- 一个 `PrimFunc` 必须对应一个稳定 symbol、一个 `KernelSignature` 和一个 module entry。
- 图输入、常量、中间值和图输出通过稳定 value id 连接，不通过对象地址或局部序号连接。

本计划不会引入任何具体算子实现，也不会引入任何具体 Pass 算法。未来的跨算子优化属于独立计划，不得弱化本计划的一算子一编译单元验收门禁。

## 2. 当前模型与目标模型

### 2.1 当前模型

```text
Relay Function（整图）
  -> RelayToTEConverter 递归构造整张 TE DAG
  -> 收集所有 TE operation
  -> 拼接为一个 SeqStmt
  -> 唯一 PrimFunc(global_symbol="main")
  -> 唯一 KernelSignature / CompiledKernel / CompiledModule entry
  -> RuntimeSession 单次 Launch
```

当前限制集中在：

- `src/compiler/lowering/relay_to_tir.cc` 以整张 Function 为 lowering 单元。
- `src/compiler/internal/compile_state.h` 只保存一份 PrimFunc、Signature 和 kernel 状态。
- `src/runtime/internal/compiled_module_node.h` 只表达单入口 executable。
- `src/runtime/session.cc` 只组装并启动一次 kernel。

### 2.2 目标模型

```text
Relay Function（图级语义）
  -> OperatorSpec/PassSpec 契约校验
  -> value graph
  -> CompilationUnit[0..N]（每单元一个算子）
  -> PrimFunc[0..N]
  -> KernelEntry[0..N]
  -> ExecutablePlan.calls[0..N]
  -> RuntimeSession 多次 Launch
```

### 2.3 性能边界

按算子独立编译的主要价值是：独立 schedule、独立缓存、独立后端选择、独立 profiling 和可控的图执行边界。代价是 kernel launch 和中间张量读写增加。因此本计划先完成正确、稳定、可测的多 kernel 架构；批量后端编译、缓存和内存复用在主链闭环后处理，但不通过新增具体优化 Pass 来掩盖架构问题。

## 3. 范围与非目标

### 3.1 本计划交付

- 每个普通计算算子独立成为 `CompilationUnit`。
- 每个单元独立 lowering、执行 TIR pipeline、构建 ABI 并生成 kernel entry。
- `CompiledModule` 支持按 symbol 查询和启动多个 entry。
- `RuntimeSession` 根据 `ExecutablePlan` 管理中间值并顺序调度 kernel。
- 算子注册使用统一 `OperatorSpec`，明确独立编译所需能力。
- Pass 注册使用统一 `PassSpec`，明确图级/单元级作用域和不变量。
- contract checker、架构测试、CPU/LLVM 数值回归与 Omen CUDA 验证。

### 3.2 本计划不做

- 不新增、补写或修改任何具体算子的计算、类型推导或 lowering 实现。
- 不新增、补写或修改任何具体 Pass 的改写算法。
- 不改变现有默认 Pass 的语义和顺序，除非为保持原顺序做等价注册迁移。
- 不实现跨算子合并、目标相关图优化或算子组合策略。
- 不引入新的第三方依赖。
- 不让 Runtime、Codegen 公共头依赖 Relay、TE、TIR 或具体后端类型。

## 4. 算子规范

`OperatorSpec` 不是算子算法实现，而是判断算子能否进入独立编译主链的完整契约。

### 4.1 必备字段

| 字段组 | 内容 | 独立编译要求 |
|---|---|---|
| 身份 | canonical name、schema version、category | 名称全局唯一且稳定 |
| 输入 | 固定/可变 arity、参数名、可选性 | 边界参数顺序可确定 |
| 属性 | attrs type key、字段 schema、默认值、合法域 | 属性可稳定序列化和 hash |
| 输出 | 输出数规则、输出 type relation | 编译前可确定 ABI 输出槽数量 |
| 类型 | type relation key、shape/dtype 约束 | lowering 前必须完成类型检查 |
| 语义 | purity/effect、determinism、alias/in-place | 默认采用最保守值 |
| Lowering | lowering kind、implementation key、target capability | 不允许空 hook 或占位结果 |
| 生命周期 | constant/input/output 访问语义 | 可计算单元边界和常量归属 |

建议结构只保存元数据：

```cpp
struct OperatorSpec {
    String name;
    int schema_version;
    String category;
    InputArity input_arity;
    Array<ArgumentSpec> arguments;
    AttrSchema attrs;
    OutputArity output_arity;
    String type_relation_key;
    EffectKind effect;
    bool deterministic;
    AliasContract alias_contract;
    LoweringContract lowering;
};
```

禁止把 `std::function`、TE Tensor、TIR 节点或 backend handle 存入 `OperatorSpec`。实现通过稳定 registry key 绑定。

### 4.2 注册与校验规则

- `OpRegEntry` 必须绑定完整 spec，再绑定 type/lowering implementation key。
- `Op::Get` 只能查找已注册算子，不得自动创建空算子。
- 缺失 schema、arity 冲突、attrs type 冲突、输出数不明确、lowering 能力缺失和重复名称必须在 registry freeze/check 阶段失败。
- 普通计算算子只有在 spec 完整且 lowering 能力有效时，才可成为 `CompilationUnit`。
- 非计算节点或具有特殊执行语义的节点必须由 spec 显式标记，不能伪装成普通 compute unit。
- `test/relay_op_contract.json` 是机器可读事实源，C++ registry、FFI、前端映射与测试由 checker 反查。

### 4.3 文件边界

**Files:**

- Modify: `include/kxc/relay/op.h`
- Modify: `include/kxc/relay/op_attr_types.h`
- Modify: `include/kxc/relay/op_macros.h`
- Modify: `src/relay/op_registry.cc`
- Modify: `test/relay_op_contract.json`
- Modify: `python/tools/check_relay_op_contract.py`
- Modify: `docs/relay-op-integration/00-contract.md`
- Modify: `docs/OPERATOR_REGISTRATION_GUIDE.md`

不修改 `src/relay/op/**` 下任何具体算子实现文件。

## 5. Pass 规范

`PassSpec` 只规定 Pass 如何被安全编排，不定义任何具体改写规则。

### 5.1 必备字段

| 字段组 | 内容 | 约束 |
|---|---|---|
| 身份 | canonical name、schema version | 对应 IR 域内唯一 |
| IR | dialect、input kind、output kind | 禁止跨 IR 隐式转换 |
| 作用域 | graph、compilation-unit、PrimFunc、module | 必须显式声明 |
| 阶段 | phase、required opt level | 阶段逆序必须失败 |
| 依赖 | required passes/analyses/invariants | 执行前完整校验 |
| 效果 | preserved/invalidated analyses、may_change_ir | 未声明即保守失效 |
| 行为 | deterministic、idempotent、thread-safe、target-dependent | 作为可测试承诺 |
| 实现 | implementation registry key | spec 不保存函数对象 |

```cpp
struct PassSpec {
    String name;
    int schema_version;
    IRDialect dialect;
    PassScope scope;
    String phase;
    int opt_level;
    Array<String> required_invariants;
    Array<String> produced_invariants;
    Array<String> preserved_analyses;
    Array<String> invalidated_analyses;
    bool deterministic;
    bool idempotent;
    bool thread_safe;
    bool target_dependent;
    String implementation_key;
};
```

### 5.2 独立编译下的 Pass 边界

- 图级 Pass 只在划分 `CompilationUnit` 之前作用于完整 Relay Function。
- 单元级 Relay Pass 只能读取本单元及其显式边界输入，禁止递归进入 producer 单元。
- TIR Pass 逐 PrimFunc 独立执行；失败信息必须带 unit id 和 symbol。
- Module Pass 可以观察多个 entry，但不得改变已冻结的 Kernel ABI，除非 spec 显式声明 ABI 变更阶段。
- 同一 per-unit Pass pipeline 对每个单元使用相同规范解析逻辑；target-dependent Pass 从 `PassContext` 和 unit target 读取能力。
- 本计划不新增任何具体 Pass，不决定新的默认优化组合。

### 5.3 文件边界

**Files:**

- Create: `include/kxc/pass/pass.h`
- Create: `src/pass/pass.cc`
- Modify: `include/kxc/pass/context.h`
- Modify: `include/kxc/relay/transforms/pipeline.h`
- Modify: `src/relay/transforms/pipeline.cc`
- Modify: `include/kxc/tir/transforms/pipeline.h`
- Modify: `src/tir/transforms/pipeline.cc`
- Create: `test/pass_contract.json`
- Create: `python/tools/check_pass_contract.py`
- Modify: `test/pass_pipeline_test.cpp`
- Create: `docs/PASS_CONTRACT.md`
- Modify: `docs/PASS_IMPLEMENTATION_SUMMARY.md`
- Modify: `CMakeLists.txt`

不修改 `src/relay/transforms/**`、`src/tir/transforms/**` 下任何具体 Pass 算法文件。

## 6. 新算子/Pass 实现准入与跨层交互规范

这一模块规定后续扩展如何接入本架构。它不在本计划中新增任何具体算子或具体 Pass，但所有后续实现都必须通过这些准入门禁。

### 6.1 新算子实现要求

新增算子必须按以下顺序落地，禁止先写 lowering、再补 schema：

1. **Contract**：先在 `test/relay_op_contract.json` 声明 canonical name、schema version、arity、attrs、输出数规则、effect、alias、lowering kind、target capability、FFI/前端映射和测试要求。
2. **Schema**：定义或复用 attrs Object，明确每个字段的类型、默认值、合法域和序列化顺序；不得通过未记录的约定解释 attrs。
3. **Registration**：通过 `OperatorSpec` 注册完整元数据、type relation key 和 lowering implementation key；注册完成后必须通过 freeze/check。
4. **Type relation**：验证输入数量、rank、shape、dtype、属性和输出数量；错误必须包含 operator name、参数位置和违反的约束。
5. **Lowering**：只消费当前 `Call` 的显式输入、attrs、checked type 和 target capability；不得递归 lowering producer `Call`，不得读取整图隐式状态。
6. **Compilation unit**：普通 compute op 必须独立形成一个 unit；特殊执行语义必须由 spec 显式声明，不能依赖 operator name 白名单。
7. **Backend capability**：未支持的 target 必须在编译期明确失败；禁止空 tensor、空 PrimFunc、无操作 kernel 或静默 CPU fallback。
8. **Tests**：至少覆盖 schema 正反例、type relation 正反例、单算子 lowering、PrimFunc ABI、多输出/常量/alias（如适用）、支持后端数值结果和不支持后端诊断。
9. **Documentation**：更新 operator contract 文档和支持矩阵；文档内容必须能由 checker 与注册信息交叉验证。

新算子实现还必须满足以下不变量：

- 同一输入、attrs、checked type 和 target 产生稳定的结构身份。
- 每个逻辑输出对应独立、明确的 value id；不得复制同一个 tensor 引用冒充多输出。
- purity、effect 和 alias 未声明时按最保守语义处理，Pass 不得自行推断。
- 常量必须通过 unit 边界和稳定 key 传递，不得从全局 map 按局部序号猜测。
- 新算子不能要求 Runtime 理解 operator name；Runtime 只消费 plan、signature 和 module entry。

### 6.2 新 Pass 实现要求

新增 Pass 必须按以下顺序落地：

1. **Contract**：先在 `test/pass_contract.json` 声明 canonical name、schema version、dialect、scope、phase、opt level、依赖、不变量、analysis 保留/失效关系和 target dependence。
2. **Registration**：注册 `PassSpec` 与唯一 implementation key；禁止在多个 pipeline 表中重复维护 Pass 元数据。
3. **Input checks**：执行前验证 IR dialect、scope、phase、required invariants、required analyses 和 target capability。
4. **Transformation**：实现只处理声明范围内的 IR，不访问更高层或更低层私有状态。
5. **Metadata**：明确保持或重建 checked type、virtual device、span、unit id、symbol、buffer/ABI metadata；未声明保持的 analysis 一律失效。
6. **Output checks**：执行后验证 produced invariants；如果变更会使类型或 shape 信息失效，必须通过 spec 触发相应分析重建。
7. **Diagnostics**：失败信息至少包含 pass name、dialect、scope、phase，以及适用时的 unit id/symbol。
8. **Tests**：至少覆盖命中、不命中、非法输入、元数据保持、确定性；声明幂等时必须覆盖二次执行；声明 thread-safe 时必须有并发验证或明确的可证明无共享状态设计。
9. **Pipeline**：新增 Pass 默认不自动进入任何 pipeline；是否加入默认 pipeline 必须单独评审并更新 contract 顺序。

所有 Pass 必须遵守：

- 需要识别算子能力时查询 `OperatorSpec` 的通用字段，禁止在框架 Pass 中硬编码具体 operator name。
- Relay 图级 Pass 只能在 unit 划分前改变拓扑。
- unit 冻结后，单元级 Relay/TIR Pass 不得跨 unit 读取、合并或重排计算。
- TIR Pass 每次只接收一个 PrimFunc；Module Pass 不得静默改变已冻结 ABI。
- Pass 不得修改 operator/pass registry，也不得依赖注册顺序产生结果。

### 6.3 各层交互关系

```text
Frontend / FFI
  -> 根据 OperatorSpec 构造合法 Call + Attrs
Relay Registry
  -> 提供 schema、type relation、effect、lowering capability
Relay Type/Graph Pass Pipeline
  -> 查询 OperatorSpec；产生类型完整、语义合法的 Relay Function
Value Graph / Unit Partition
  -> 每个普通 compute Call 冻结为一个 CompilationUnit
Unit Lowering
  -> 仅对当前 Call lowering，产生一个 PrimFunc
Per-PrimFunc TIR Pipeline
  -> 根据 PassSpec 在单 unit 范围变换并保持 ABI 前置不变量
Kernel ABI / Codegen
  -> 产生稳定 Signature、symbol 和 executable entry
CompiledModule + ExecutablePlan
  -> 用 symbol 和 value id 连接多个独立 kernel
RuntimeSession
  -> 只校验 ABI、分配 value、调度 kernel；不识别算子和 Pass
```

| 层 | 消费 | 产出 | 与算子的关系 | 与 Pass 的关系 | 禁止 |
|---|---|---|---|---|---|
| Frontend/FFI | 外部模型、attrs | Relay `Call` | 按 schema 构造并规范化属性 | 不执行编译 Pass | 绕过 registry 构造未知 op |
| Relay Registry | operator contract | `OperatorSpec` 查询结果 | 提供身份、类型、effect、lowering 能力 | 为 Pass 提供通用能力查询 | 自动创建空 op、名称白名单分支 |
| Type/Graph 层 | Relay Function、spec | checked Relay Function | 验证每个 Call 的 schema/type | 只运行 graph scope Pass | 未重建类型就输出失效 checked type |
| Unit Partition | checked Relay、effect | value graph、units、plan draft | 一个普通 Call 固定为一个 unit | 不是可插入的具体优化 Pass | 多 Call unit、遗漏或重复归属 |
| Unit Lowering | 单 unit、checked type、target | 单 PrimFunc | 仅调用当前 op 的 lowering binding | 可运行 unit scope 的规范化入口 | 递归进入 producer unit |
| TIR Pipeline | 单 PrimFunc、PassContext | 优化后单 PrimFunc | 不再依赖 operator 实现细节 | 仅运行 PrimFunc scope Pass | 跨 unit 合并、改变 symbol 身份 |
| ABI/Codegen | PrimFunc、target | signature、kernel entry | operator identity 仅作诊断元数据 | Module Pass 受 ABI phase 限制 | 将 backend handle 暴露到公共头 |
| Runtime | module、plan、NDArray | 图输出、completion | 不读取 operator registry | 不加载或执行编译 Pass | 按 operator name 分配或调度 |

### 6.4 算子与 Pass 的交互规则

- Pass 读取算子语义只能通过 `OperatorSpec`；如果现有字段不足，应先扩展通用 contract schema，而不是添加具体名称判断。
- Relay Pass 新建或替换 `Call` 时，目标 op 必须已注册，attrs 必须通过 schema 校验，arity 必须匹配。
- Pass 改变 Call、attrs、shape 或 dtype 后，必须声明类型/shape analysis 失效，并在后续阶段重建；禁止保留陈旧 checked type。
- Pass 删除、复制或重排 Call 时，必须遵守 effect、determinism 和 alias contract；不纯或未知语义默认不可删除、复制或交换顺序。
- 图级 Pass 完成后才分配最终 value id 和 unit id，避免拓扑改写造成身份漂移。
- unit id 冻结后，per-unit Pass 只能更新 unit 内 IR，不能改变 unit 输入输出集合；需要改变边界时必须回到图级阶段重新划分。
- PrimFunc symbol、KernelSignature 和 `KernelCall` symbol 必须一一对应；Pass 不得单独改写其中一方。
- profiling 关联使用 operator identity、pass canonical name、unit id 和 symbol，不使用对象地址或源文件路径作为身份。

### 6.5 变更影响矩阵

| 变更 | 必须更新 | 必须验证 | 不应修改 |
|---|---|---|---|
| 新增算子 | operator contract、schema/registration、type/lowering binding、测试、文档 | checker、type、独立 lowering、ABI、数值或明确 unsupported | Pass framework、Runtime 调度逻辑 |
| 修改算子 attrs/输出 | schema version、兼容性说明、type/lowering 测试 | 旧模型兼容性、hash、cache、ABI | 无关算子实现 |
| 新增 Pass | pass contract、implementation binding、测试、文档 | scope、依赖、metadata、determinism、pipeline 显式入口 | Operator registry、Runtime |
| 修改 Pass 保证 | contract version、invariant/analysis 声明 | 下游 pipeline、缓存失效、重复执行 | 无关 Pass 算法 |
| 新增通用算子能力 | `OperatorSpec` schema、checker、查询 API | 所有已注册算子的默认/显式值 | 具体名称白名单 |
| 改变 unit 边界 | 独立架构计划、partition contract、plan/ABI/runtime 测试 | N Call = N unit 门禁是否仍成立 | 以普通 Pass 偷渡边界变化 |

### 6.6 规范落地文件

**Files:**

- Create: `docs/COMPILER_EXTENSION_CONTRACT.md`
- Modify: `docs/OPERATOR_REGISTRATION_GUIDE.md`
- Modify: `docs/relay-op-integration/00-contract.md`
- Modify: `docs/PASS_CONTRACT.md`
- Modify: `docs/PASS_IMPLEMENTATION_SUMMARY.md`
- Create: `test/compiler_extension_contract_test.cpp`
- Modify: `python/tools/check_relay_op_contract.py`
- Modify: `python/tools/check_pass_contract.py`
- Modify: `CMakeLists.txt`

`compiler_extension_contract_test` 使用测试内匿名 operator/pass fixture 验证交互规则，不向生产 registry 引入具体算子或具体 Pass。

## 7. 核心数据结构

### 7.1 Compiler 私有结构

```cpp
struct CompilationUnit {
    int64_t unit_id;
    String symbol;
    Expr call;
    Array<int64_t> input_value_ids;
    Array<int64_t> output_value_ids;
    String structural_hash;
};

struct LoweredPrimitive {
    int64_t unit_id;
    String symbol;
    tir::PrimFunc prim_func;
    Map<String, runtime::NDArray> constants;
};

struct LoweredGraph {
    Array<LoweredPrimitive> primitives;
    runtime::ExecutablePlan plan;
    Map<String, runtime::NDArray> constants;
};
```

`CompilationUnit.call` 必须是单个普通计算 `Call`。任何包含多个普通计算 `Call` 的 unit 都违反本计划门禁。

### 7.2 Runtime 公共契约

```cpp
class ValueSpecNode final : public Object {
public:
    int64_t value_id{-1};
    int64_t storage_id{-1};
    Array<int64_t> shape;
    DLDataType dtype{};
    Device device;
    bool is_input{false};
    bool is_constant{false};
    bool is_output{false};
};

class KernelCallNode final : public Object {
public:
    String symbol;
    Array<int64_t> input_values;
    Array<int64_t> output_values;
};

class ExecutablePlanNode final : public Object {
public:
    Array<ValueSpec> values;
    Array<KernelCall> calls;
    Array<int64_t> input_values;
    Array<int64_t> output_values;
};
```

`ExecutablePlan` 不保存 Relay、TE、TIR、Pass 中间状态或 backend executable。

## 8. 实施任务

### Task 1：锁定整图单 kernel 基线和多 kernel 目标契约

**Files:**

- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/runtime_session_test.cpp`
- Create: `test/operator_compilation_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 增加 characterization test，证明当前多算子图只生成一个 PrimFunc、一个 signature 和一次 launch。
2. 增加目标契约测试：包含 N 个普通计算 `Call` 的图必须生成 N 个 compilation units、N 个 PrimFunc 和 N 个 `KernelCall`。
3. 增加 chain、branch、diamond、tuple 和多输出图 fixture，锁定 value id、调用顺序和 live-out。
4. fixture 复用现有已支持算子或测试内抽象节点，不新增生产算子。
5. 先运行测试，确认目标契约因缺少多 kernel 架构而失败。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target operator_compilation_test compiler_contract_test runtime_session_test -j 4
```

### Task 2：建立 OperatorSpec 与 PassSpec 基础设施

**Files:** 使用第 4.3 节和第 5.3 节列出的文件。

**Steps:**

1. 为缺字段、重复身份、非法作用域、缺失实现 key 和非确定序列化写失败测试。
2. 实现只含元数据的 `OperatorSpec`、`PassSpec` 和 registry freeze/check。
3. 将现有 operator/pass 注册迁移为等价 spec 绑定，保持现有语义与默认 pipeline 顺序。
4. 增加 `check_pass_contract`，升级 `check_relay_op_contract`。
5. 验证没有修改任何具体算子或具体 Pass 算法文件。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  registry_test pass_pipeline_test check_relay_op_contract check_pass_contract `
  check_include_layers check_public_headers -j 4
```

### Task 3：固化扩展准入与跨层交互门禁

**Files:** 使用第 6.6 节列出的文件。

**Steps:**

1. 用匿名 fixture 为不完整算子规范、越界 lowering、错误 Pass scope、陈旧 checked type 和 ABI symbol 不一致写失败测试。
2. 扩展 operator/pass checker，使 contract、registry、implementation key 和测试覆盖可交叉验证。
3. 在 unit partition 入口拒绝 spec 不完整或不允许普通 compute lowering 的 `Call`。
4. 在 pipeline 入口拒绝 dialect/scope/phase 不匹配的 Pass。
5. 在 lowering 和 ABI 边界加入 unit id、operator identity、symbol 一致性校验。
6. 编写 `docs/COMPILER_EXTENSION_CONTRACT.md`，以第 6 节作为规范事实源，不加入具体实现示例。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  compiler_extension_contract_test registry_test pass_pipeline_test `
  check_relay_op_contract check_pass_contract check_include_layers -j 4
```

Expected: 所有跨层违规在进入 backend/runtime 前失败，并包含 operator/pass/unit/symbol 中适用的稳定身份。

### Task 4：建立 runtime-neutral ExecutablePlan

**Files:**

- Create: `include/kxc/runtime/executable_plan.h`
- Create: `src/runtime/executable_plan.cc`
- Create: `src/runtime/internal/executable_plan_validation.h`
- Create: `test/executable_plan_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 为非法 value id、重复 producer、输入未定义、调用逆序、输出缺失和 shape/dtype/device 缺失写失败测试。
2. 实现 `ValueSpec`、`KernelCall`、`ExecutablePlan` ObjectRef 和完整 validation。
3. 约束每个非 input/constant value 恰好有一个 producer，call 输入在调用前可用。
4. 第一版使用 `storage_id == value_id`，暂不复用中间存储。
5. 将新公共头加入 runtime FILE_SET 并运行头文件检查。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target executable_plan_test check_include_layers check_public_headers -j 4
```

### Task 5：构建稳定 value graph 和一算子一单元划分

**Files:**

- Create: `src/compiler/internal/value_graph.h`
- Create: `src/compiler/internal/compilation_unit.h`
- Create: `src/compiler/graph/value_graph.cc`
- Create: `src/compiler/graph/partition.cc`
- Create: `test/graph_partition_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 从已完成类型检查的 Relay Function 构建 DAG，为 param、constant、call output 和 tuple field 分配稳定 value id。
2. 身份只依赖拓扑位置和结构内容，不依赖 `Object*` 地址或 unordered 容器顺序。
3. 根据 `OperatorSpec` 区分普通 compute call 与非计算/特殊节点。
4. 为每个普通 compute call 创建且只创建一个 `CompilationUnit`。
5. 计算每个单元的外部输入和所有 live-out，生成 plan 草稿。
6. 对含多个普通 compute call 的 unit、遗漏 call 或重复归属明确失败。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target graph_partition_test -j 4
```

### Task 6：按单算子 CompilationUnit 独立 lowering

**Files:**

- Modify: `include/kxc/compiler/lowering/relay_to_tir.h`
- Modify: `src/compiler/lowering/relay_to_tir.cc`
- Modify: `src/compiler/lowering/lowered_function.cc`
- Create: `src/compiler/internal/lowered_graph.h`
- Create: `src/compiler/lowering/lowered_graph.cc`
- Modify: `test/operator_compilation_test.cpp`
- Modify: `test/infer_type_test.cpp`

**Steps:**

1. 将现有整图 lowering 拆成边界 placeholder、单 call TE 转换、TE DAG 到 PrimFunc 三个私有阶段。
2. 单元外部值只能从边界映射读取，禁止递归 lowering producer call。
3. 根据 `OperatorSpec` 校验输入数、输出数、attrs、type relation 和 lowering kind。
4. 支持单算子多输出，保持现有 output metadata 契约。
5. 常量 key 使用图级稳定 value id；每个 PrimFunc 只携带本算子实际使用的常量。
6. 每个 PrimFunc 写入唯一 symbol、unit id、operator identity 和 structural hash。
7. 对整图含 N 个普通 call 的 fixture，断言恰好生成 N 个 PrimFunc。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target operator_compilation_test infer_type_test kernel_signature_test -j 4
```

### Task 7：把 Compiler 状态机改为多单元状态

**Files:**

- Modify: `src/compiler/internal/compile_state.h`
- Modify: `src/compiler/compile_state.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/internal/kernel_abi_builder.h`
- Modify: `src/compiler/kernel_abi_builder.cc`
- Modify: `test/compiler_contract_test.cpp`

**Steps:**

1. 将单个 `tir_`、`signature_`、`kernel_` 字段替换为按 unit 顺序保存的 primitive 集合。
2. 每次阶段迁移校验 unit id、symbol、PrimFunc、signature、metadata 和 kernel 一一对应。
3. 根据 `PassSpec.scope` 对每个 PrimFunc 独立执行现有 TIR pipeline。
4. 对每个 primitive 独立构建 Kernel ABI 和 backend kernel。
5. profiling artifact 使用稳定 unit/symbol 路径；错误包含 stage、unit id、symbol 和 operator identity。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target compiler_contract_test kernel_signature_test pass_pipeline_test -j 4
```

### Task 8：把 CompiledModule 升级为多入口模块

**Files:**

- Modify: `include/kxc/runtime/compiled_module.h`
- Modify: `src/runtime/internal/compiled_module_node.h`
- Modify: `src/runtime/compiled_module.cc`
- Modify: `src/codegen/internal/compiled_kernel.h`
- Modify: `src/codegen/common/compiled_kernel.cc`
- Modify: `test/compiled_module_test.cpp`

**Steps:**

1. 定义私有 `KernelEntry{signature, launch_metadata, executable}`，按 symbol 存入 module。
2. 公共 API 提供 `HasFunction(symbol)`、按 symbol 查询 metadata 和 `Launch(symbol, args, stream)`。
3. module 常量池按稳定 key 去重；entry 只能访问自身 signature 声明的常量。
4. 删除无 symbol 的单入口假设，同时保持单 entry 是多入口模块的合法特例。
5. 保留现有 NDArray、device、alignment、stream 和异步保活校验。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target compiled_module_test -j 4
```

LLVM 构建：

```powershell
cmake --build out/build/windows-llvm-msys2 --target compiled_module_test codegen_llvm_test -j 4
```

### Task 9：让 RuntimeSession 执行多 kernel 图计划

**Files:**

- Modify: `include/kxc/runtime/session.h`
- Modify: `src/runtime/internal/session_node.h`
- Modify: `src/runtime/session.cc`
- Create: `src/runtime/internal/value_table.h`
- Modify: `test/runtime_session_test.cpp`
- Modify: `test/operator_compilation_test.cpp`

**Steps:**

1. Session 同时持有 `CompiledModule` 和 `ExecutablePlan`。
2. 构造时验证每个 call symbol 存在，且 value specs 与对应 signature 一致。
3. `Run` 绑定图输入和常量，分配中间值/输出，并按 calls 顺序启动 kernel。
4. 第一版在同一 stream 串行执行，不做 storage reuse。
5. `RunAsync` 聚合 completion，并保活 plan、value table、module 和所有中间 storage。
6. 对 chain、branch、diamond、tuple、多输出和共享常量做端到端数值回归。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target runtime_session_test operator_compilation_test op_numeric_llvm_test -j 4
```

### Task 10：切换 Compiler::Compile 和现有调用方

**Files:**

- Modify: `include/kxc/compiler/compiler.h`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/frontend/onnx_importer.cc`
- Modify: `test/codegen_llvm_test.cpp`
- Modify: `test/codegen_cuda_test.cpp`
- Modify: `test/onnx_importer_test.cpp`
- Modify: `test/resnet18_ir_dump.cpp`

**Steps:**

1. `Compiler::Compile` 返回同时包含多入口 module 和 plan 的可执行产物。
2. 删除调用方对唯一 `global_symbol=main`、唯一 signature 和唯一 launch metadata 的假设。
3. 保持 `CompilePrimitive` 为 Compiler 私有 helper，不增加第二套公共编译入口。
4. 复用现有算子覆盖与模型 fixture 验证 N 个普通 call 对应 N 个 kernel entry。
5. 比较迁移前后图输出，禁止以结构断言代替数值验证。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target `
  codegen_llvm_test op_numeric_llvm_test onnx_importer_test `
  compiler_contract_test runtime_session_test operator_compilation_test -j 4
```

### Task 11：批量后端编译、缓存与中间存储复用

**Files:**

- Modify: `src/codegen/llvm/internal/codegen_llvm.h`
- Modify: `src/codegen/llvm/codegen_llvm.cc`
- Modify: `src/codegen/llvm/internal/llvm_jit.h`
- Modify: `src/codegen/llvm/llvm_jit.cc`
- Modify: `src/codegen/cuda/internal/codegen_cuda.h`
- Modify: `src/codegen/cuda/codegen_cuda.cc`
- Modify: `src/codegen/cuda/internal/cuda_module.h`
- Modify: `src/codegen/cuda/cuda_module.cc`
- Create: `src/compiler/cache/primitive_cache.cc`
- Create: `src/runtime/memory_plan.cc`
- Modify: `test/codegen_llvm_test.cpp`
- Modify: `test/codegen_cuda_test.cpp`
- Modify: `test/runtime_session_test.cpp`

**Steps:**

1. 同一 target 的多个 PrimFunc 共享一个 backend module/JIT 资源，同时保持每个 symbol 独立寻址。
2. cache key 包含 PrimFunc structural hash、target、opt level、ABI version 和 backend version。
3. 禁止仅按 operator name、shape 或注册顺序命中缓存。
4. 根据最后使用位置为不重叠的中间值分配可复用 storage id。
5. input、constant、output、alias value 和异步存活值不得错误复用。
6. 记录编译时间、cache 命中率、kernel 数、launch 时间、峰值存储和端到端耗时。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target codegen_llvm_test op_numeric_llvm_test runtime_session_test -j 4
```

Omen CUDA：

```bash
cmake --build out/build/omen-cuda --target codegen_cuda_test runtime_session_test cupti_smoke_test -j 4
```

### Task 12：接入 distributed ExecutionPlan

**Files:**

- Modify: `include/kxc/distributed/execution_plan.h`
- Modify: `src/distributed/execution_plan.cc`
- Modify: `src/distributed/executor.cc`
- Modify: `src/compiler/distributed/multi_device.cc`
- Modify: `src/relay/distributed/plan_adapter.cc`
- Modify: `test/compiler_contract_test.cpp`

**Steps:**

1. distributed kernel node 只保存 symbol、value ids 和 worker placement，不保存 `tir::PrimFunc`。
2. 用真实 compiled symbol 替换空 PrimFunc 占位路径。
3. executor 从 worker module registry 查找 symbol，并按 value ids 启动对应 entry。
4. 通信、copy、barrier 和 compute 的依赖必须显式验证。
5. 保持 distributed 依赖 runtime plan，禁止 runtime 反向依赖 distributed/compiler。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target compiler_contract_test pass_pipeline_test -j 4
```

## 9. 验收门禁

### 9.1 算子独立编译

- 含 N 个普通计算 `Call` 的 Relay 图恰好生成 N 个 `CompilationUnit`。
- 每个 unit 恰好包含一个普通计算 `Call`，恰好生成一个 PrimFunc 和一个 kernel entry。
- chain、branch、diamond、tuple、多输出和共享常量的 value id、依赖和数值结果正确。
- 禁止退化为把多个 unit 的 TIR 重新拼回一个 PrimFunc。
- symbol 和 cache identity 在重复编译中稳定。

### 9.2 算子规范

- 每个公开算子有且只有一个完整 `OperatorSpec`。
- contract、C++ registry、FFI、前端映射和测试不存在身份漂移。
- 未注册名称不能被静默创建。
- 本计划未新增或修改任何具体算子算法。

### 9.3 Pass 规范

- 每个可调用 Pass 有且只有一个 `PassSpec` 和实现绑定。
- graph/unit/PrimFunc/module 作用域明确，pipeline 在执行前拒绝非法组合。
- 现有默认 pipeline 顺序和行为保持不变。
- 本计划未新增或修改任何具体 Pass 算法。

### 9.4 扩展准入与跨层交互

- `compiler_extension_contract_test` 覆盖新增算子/Pass 的最小准入条件和主要越界失败路径。
- Pass 只能通过 `OperatorSpec` 查询算子能力，不存在具体 operator name 白名单。
- graph、unit、PrimFunc、module 四种 Pass scope 在执行前可验证，unit 冻结后不能跨边界改写。
- Call、unit、PrimFunc、KernelSignature、module entry 和 `KernelCall` 的稳定身份可以逐层追踪并校验。
- Backend 和 Runtime 不依赖 operator/pass registry；Runtime 不包含算子或 Pass 分支。

### 9.5 Runtime 与架构

- Runtime 公共头零 Relay/TE/TIR include。
- Codegen 后端类型保持私有。
- 单入口 API 假设全部删除，多入口 module 可独立启动每个 symbol。
- 公共头独立编译和 include layer check 通过。
- RuntimeSession 同步/异步资源生命周期测试通过。

### 9.6 性能

- 记录迁移前整图基线与按算子编译后的编译时间、kernel 数、launch 时间、峰值内存和端到端耗时。
- 性能下降必须可归因到 launch、中间存储或后端编译，而不是语义错误。
- 批量 module 与缓存不能改变独立 kernel entry 和数值结果。
- 不以减少 kernel 数作为本计划验收条件。

## 10. 全量验证矩阵

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  object_test packed_func_test registry_test type_registration_test `
  pass_pipeline_test compiler_extension_contract_test infer_type_test `
  executable_plan_test graph_partition_test `
  operator_compilation_test compiler_contract_test kernel_signature_test `
  compiled_module_test runtime_session_test check_relay_op_contract `
  check_pass_contract check_include_layers check_public_headers -j 4
```

```powershell
cmake --build out/build/windows-llvm-msys2 --target `
  codegen_llvm_test op_numeric_llvm_test onnx_importer_test `
  operator_compilation_test runtime_session_test -j 4
```

Omen CUDA 继续使用仓库 `AGENTS.md` 规定的 CUDA 12.9/CUPTI flags：

```bash
cmake --build out/build/omen-cuda --target \
  codegen_cuda_test runtime_session_test cupti_smoke_test \
  operator_compilation_test check_relay_op_contract check_pass_contract -j 4
```

涉及 value table、module 共享资源、异步 completion 或 storage reuse 后，追加 ASan+UBSan；CUDA 路径追加 Compute Sanitizer。生成的 profiling/CUPTI 输出不得提交。

## 11. 建议提交序列

1. `test: characterize whole-graph single-kernel lowering`
2. `feat: define operator and pass specifications`
3. `test: enforce compiler extension contracts`
4. `feat: define runtime executable plans`
5. `feat: partition relay graphs into per-operator units`
6. `feat: lower operators into independent tir functions`
7. `refactor: track multiple primitives through compiler stages`
8. `feat: support multi-entry compiled modules`
9. `feat: execute kernel graphs in runtime sessions`
10. `refactor: switch compiler clients to graph executables`
11. `perf: batch backend compilation and reuse storage`
12. `feat: execute compiled kernels in distributed plans`

每个提交必须独立构建并通过当时启用的测试。不得把具体算子实现、具体 Pass 算法或与 0721 文件架构无关的修改混入这些提交。
