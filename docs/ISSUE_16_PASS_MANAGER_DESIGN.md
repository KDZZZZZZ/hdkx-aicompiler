# Issue #16: 结构化 Pass Manager 设计文档

关联 issue: [#16](https://github.com/KDZZZZZZ/hdkx-aicompiler/issues/16)

## 1. 背景

当前 Relay/TIR pass pipeline 由两个静态表驱动：

- Relay: [`src/relay/transforms/pipeline.cc`](../src/relay/transforms/pipeline.cc)
- TIR: [`src/tir/transforms/pipeline.cc`](../src/tir/transforms/pipeline.cc)

现有实现已经具备三个基础能力：

1. `RunRelayPassPipeline` 和 `RunTIRPassPipeline` 可以按字符串列表运行 pass。
2. 未知 pass 名称会立即报错。
3. pipeline 内部已经记录 per-pass profiling span、IR hash 和可选 IR artifact。

但它仍然不是结构化 pass manager。主要问题是：

1. pass 没有统一 metadata，无法表达 `opt_level`、依赖、target 适用性和是否可配置。
2. `optimize_default` 是硬编码别名，不是根据 `opt_level` 生成默认 pipeline。
3. `CompileConfig.opt_level` 目前只影响 LLVM JIT 优化，不影响 Relay/TIR pass pipeline。
4. 用户无法从 `CompileConfig` 选择或禁用 Relay/TIR pass。
5. instrumentation 逻辑重复写在 Relay/TIR pipeline 中，不是统一 hook。
6. 测试只覆盖静态顺序、未知 pass 和 `optimize_default` 幂等，没有覆盖 disabled pass、required pass 展开、opt-level 默认 pipeline 和 instrumentation hook。

Issue #16 的目标是把这些散落逻辑收敛成可配置、可测试、可扩展的 pass manager，同时保留现有简单 API。

## 2. 参考实现

### 2.1 TVM

TVM 的 pass infrastructure 使用 `PassInfo` 承载 pass 元信息，包括 `name`、`opt_level` 和 `required`。`PassContext` 承载编译期配置，包括 `opt_level`、`required_pass`、`disabled_pass`、`config` 和 instruments。`Sequential` 负责执行 pass 列表，并在后端解析依赖与禁用规则。

对 TinyTVM 的启发：

- 必须把 pass 的“名字到函数”提升为“名字到元信息加函数”。
- `opt_level` 应该由 pass metadata 和 pipeline config 共同决定，而不是由 caller 手写默认列表。
- required pass 应该由 pass manager 展开，不能要求每个 caller 手工维护依赖。
- disabled pass 和 required pass 冲突时必须有明确错误。

参考：

- <https://tvm.apache.org/docs/arch/pass_infra.html>
- <https://tvm.apache.org/docs/reference/api/python/transform.html>

### 2.2 MLIR

MLIR 将 pass 管理拆成 `PassManager` 和嵌套的 `OpPassManager`，并提供 `PassInstrumentation` hook，用于 pipeline/pass/analysis 的 before/after/failure 事件。MLIR 还强调 pass 应避免全局可变状态，方便未来并行和复用。

对 TinyTVM 的启发：

- 当前 TinyTVM 不需要 MLIR 的嵌套 pass manager，但需要统一 instrumentation hook。
- pass 失败应该中断后续 pipeline，并由 hook 记录失败信息。
- pass metadata 应包含稳定名称和描述，方便自动生成文档、错误信息和测试断言。
- 不引入全局可变 pass 实例状态，pass function 保持无状态或局部状态。

参考：

- <https://mlir.llvm.org/docs/PassManagement/>

### 2.3 LLVM New Pass Manager

LLVM New PM 使用 `PassBuilder` 根据优化等级构造默认 pipeline，并允许前端、后端或插件通过 callback 插入 pass。LLVM 的 instrumentation callback 由 pass manager 在 pass 执行前后调用，callback 可以观察 pass 名称和 IR 单元，也可以控制是否执行。

对 TinyTVM 的启发：

- 默认 pipeline 应该由 builder 从 `opt_level` 和 target 生成。
- 后端相关 pass 不应该硬塞进所有 target 的默认 pipeline。
- hook 不应该直接修改 IR，避免 instrumentation 影响优化行为。
- 兼容手写 pass 列表，同时为默认 pipeline 提供结构化构建入口。

参考：

- <https://llvm.org/docs/NewPassManager.html>
- <https://www.llvm.org/docs/doxygen/PassInstrumentation_8h.html>

## 3. 设计目标

### 3.1 功能目标

1. 为 Relay/TIR pass 增加统一 metadata：
   - `name`
   - `ir_kind`
   - `opt_level`
   - `required`
   - `default_enabled`
   - `configurable`
   - `target_kinds`
   - `traceable`
   - `description`
2. 支持按 `opt_level` 展开默认 Relay/TIR pipeline。
3. 支持从 `CompileConfig` 选择和禁用 Relay/TIR pass。
4. 支持 required pass 自动展开。
5. 支持统一 instrumentation hook，复用现有 profiling 和 IR snapshot 行为。
6. 保留现有兼容入口：
   - `kxc::relay::RunRelayPassPipeline(const Function&, const Array<String>&)`
   - `kxc::tir::RunTIRPassPipeline(const tir::PrimFunc&, const Array<String>&)`
   - `kxc.relay.transform.run_pipeline`
   - `kxc.tir.transform.run_pipeline`

### 3.2 非目标

本 issue 不实现以下能力：

1. 不引入 MLIR 级别的嵌套 pass manager。
2. 不引入 LLVM 风格 analysis manager 和 analysis cache。
3. 不支持 pass plugin 动态加载。
4. 不改变 Relay/TIR pass 的具体优化语义。
5. 不把算子 lowering、TOPI、LLVM codegen 行为混入本 PR。

## 4. 核心模型

### 4.1 IR kind

```cpp
enum class PassIRKind {
    kRelay,
    kTIR,
};
```

`PassIRKind` 用于隔离 Relay 和 TIR registry。Relay pipeline 不能选择 TIR pass，TIR pipeline 不能选择 Relay pass。

### 4.2 PassInfo

```cpp
struct PassInfo {
    std::string name;
    PassIRKind ir_kind;
    int opt_level{0};
    std::vector<std::string> required;
    bool default_enabled{true};
    bool configurable{true};
    std::vector<std::string> target_kinds;
    bool traceable{true};
    std::string description;
};
```

字段语义：

| 字段 | 语义 |
| --- | --- |
| `name` | 稳定 pass 名，必须唯一 |
| `ir_kind` | Relay 或 TIR |
| `opt_level` | 默认 pipeline 中启用该 pass 所需的最低优化等级 |
| `required` | 运行该 pass 前必须先运行的 pass |
| `default_enabled` | 是否参与默认 pipeline |
| `configurable` | 是否允许用户在 `CompileConfig` 中禁用或显式选择 |
| `target_kinds` | 为空表示所有 target 可用，否则只允许指定 target |
| `traceable` | 是否允许 instrumentation 捕获 IR snapshot |
| `description` | 错误信息和文档使用的中文说明 |

约束：

1. `name` 必须用 canonical snake_case。
2. 同一个 `ir_kind` 下不允许重复 `name`。
3. `required` 只能引用同一 `ir_kind` 下的 pass。
4. `opt_level` 范围为 `0..3`。
5. `configurable=false` 的 pass 表示基础不变量 pass，例如未来可能的 verifier。用户不能禁用它。

### 4.3 PassEntry

Relay 和 TIR 的 pass 函数类型不同，因此 manager 只共享元信息、展开规则和 instrumentation 执行框架，真正调用处保留 typed wrapper。

```cpp
using RelayPassFunc = std::function<Function(const Function&)>;
using TIRPassFunc = std::function<tir::PrimFunc(const tir::PrimFunc&)>;

struct RelayPassEntry {
    PassInfo info;
    RelayPassFunc run;
};

struct TIRPassEntry {
    PassInfo info;
    TIRPassFunc run;
};
```

### 4.4 PassPipelineConfig

```cpp
struct PassPipelineConfig {
    int opt_level{2};
    std::vector<std::string> selected_passes;
    std::vector<std::string> required_passes;
    std::vector<std::string> disabled_passes;
    std::string target_kind;
    bool capture_ir{true};
};
```

字段语义：

| 字段 | 语义 |
| --- | --- |
| `opt_level` | 默认 pipeline 展开等级 |
| `selected_passes` | 非空时表示用户显式指定基础 pass 顺序 |
| `required_passes` | 无论 opt level 如何都必须运行的 pass |
| `disabled_passes` | 禁用 pass 列表 |
| `target_kind` | 来自 `CompileConfig.target->kind` |
| `capture_ir` | 是否允许 pass manager 调用 IR snapshot hook |

规则：

1. `selected_passes` 为空时，manager 根据 `opt_level` 展开默认 pipeline。
2. `selected_passes` 非空时，manager 以用户给定顺序为基础，不再自动加入同等级默认 pass。
3. `required_passes` 会被合并到基础 pipeline，并在依赖展开时保持先于依赖者。
4. `disabled_passes` 是硬禁用。如果默认 pipeline 命中 disabled pass，则跳过；如果显式选择或依赖需要 disabled pass，则立即报错。
5. `capture_ir=false` 时 instrumentation 仍记录 timing，但不写 IR artifact。

## 5. Pipeline 展开规则

### 5.1 默认 pipeline

默认 pipeline 由 registry 顺序和 `PassInfo` 决定：

```text
default pass = default_enabled
             && info.opt_level <= config.opt_level
             && target applicable
             && not disabled
```

默认顺序必须显式保存在 registry 构造函数中，不能依赖 `unordered_map` 遍历顺序。

### 5.2 显式 pipeline

显式 pipeline 来自：

1. `RunRelayPassPipeline(func, pass_names)`。
2. `RunTIRPassPipeline(func, pass_names)`。
3. `CompileConfig.relay_passes` 或 `CompileConfig.tir_passes`。

显式 pipeline 的规则：

1. 未知 pass 名立即报错，错误信息包含 IR kind 和可用 pass 列表。
2. 显式选择 target 不适用的 pass，立即报错。
3. 显式选择 `configurable=false` 的 pass 可以执行，但不能被禁用。
4. 显式重复 pass 允许执行多次。

### 5.3 required pass 展开

required pass 使用 DFS 展开：

```text
Expand(pass):
  validate pass exists
  validate pass is not disabled
  validate pass target applicable
  for req in pass.required:
      Expand(req)
  append pass
```

错误规则：

1. required pass 不存在，报错。
2. required pass 被 disabled，报错。
3. required pass 和 target 不匹配，报错。
4. required pass 出现环，报错并输出环路径。

去重规则：

1. 自动插入的 required pass 在同一个 pipeline 中只插入一次。
2. 用户显式写两次同一个 pass 时，保留两次执行。

### 5.4 `optimize_default` 兼容

为了保留旧 API，`Run*PassPipeline(..., {"optimize_default"})` 继续可用。

兼容规则：

1. `optimize_default` 是 legacy alias，不作为普通 pass 注册。
2. legacy alias 展开为当前 AOT 默认行为等价的 pipeline。
3. 新代码应使用 `PassPipelineConfig{.opt_level = ...}` 或 `CompileConfig.opt_level`。
4. 文档和测试要覆盖 legacy alias 与新默认 pipeline 的对应关系。

## 6. CompileConfig 接入

`CompileConfigNode` 增加 pass 配置字段：

```cpp
Array<String> relay_passes;
Array<String> tir_passes;
Array<String> required_relay_passes;
Array<String> required_tir_passes;
Array<String> disabled_relay_passes;
Array<String> disabled_tir_passes;
```

选择这种直接字段，而不是新增复杂 object，原因是：

1. 当前对象系统已有 `Array<String>`，实现和 Python/registry 绑定成本低。
2. Relay/TIR pass 配置先满足 MVP，不提前引入通用 config map。
3. 后续如果需要 per-pass options，再扩展为 `Map<String, ObjectRef>` 或专门 `PassOptions`。

`Compiler::Compile` 使用规则：

1. 从 `CompileConfig` 构造 Relay `PassPipelineConfig`。
2. Relay pipeline 使用 `config.opt_level`，不再根据 `CompileMode` 手写 `{ "optimize_default" }`。
3. 从 `CompileConfig` 构造 TIR `PassPipelineConfig`。
4. TIR pipeline 同样使用 `config.opt_level`。
5. `CompileMode::kJIT` 可以通过工厂函数设置较低 `opt_level`，而不是绕过 pass manager。

## 7. Instrumentation 设计

现有 profiling 行为保留，但从 Relay/TIR pipeline 中抽出为统一 hook。

### 7.1 Hook 接口

```cpp
class PassInstrumentation {
public:
    virtual ~PassInstrumentation() = default;

    virtual void BeforePipeline(const PassPipelineConfig& config, PassIRKind kind) {}
    virtual void AfterPipeline(const PassPipelineConfig& config, PassIRKind kind) {}
    virtual void BeforePass(const PassInfo& info, const std::string& before_text) {}
    virtual void AfterPass(const PassInfo& info,
                           const std::string& before_text,
                           const std::string& after_text) {}
    virtual void AfterPassFailed(const PassInfo& info,
                                 const std::string& before_text,
                                 const std::exception& error) {}
};
```

MVP 只需要一个内置 instrumentation：

- `ProfilingPassInstrumentation`

它负责：

1. 创建 `relay_pipeline` / `tir_pipeline` span。
2. 创建 `relay_pass` / `tir_pass` span。
3. 记录 `pass_name`。
4. 记录 `ir_before_hash`、`ir_after_hash`、`ir_changed`。
5. 按 `ProfileOptions.ir_capture_mode` 写 before/after/failed IR artifact。
6. pass 失败时记录 error log。

### 7.2 IR 打印函数

由于 Relay 和 TIR 的打印函数不同，typed pipeline 在调用 manager 时提供 printer：

```cpp
using IRPrinter<IR> = std::function<std::string(const IR&)>;
```

Relay 使用 `relay::pass::ToText`，TIR 使用 `tir::pass::DumpPrimFunc`。

### 7.3 Hook 不修改 IR

Instrumentation 只能观察，不允许修改 IR。这样可以保证打开 profiling 不改变编译结果。

## 8. 文件改动计划

### 8.1 新增基础设施

新增：

- `include/base/pass_manager.h`
- `src/base/pass_manager.cc`

职责：

1. 定义 `PassIRKind`、`PassInfo`、`PassPipelineConfig`。
2. 实现 pipeline 展开、依赖解析、错误格式化。
3. 实现通用 instrumentation hook 调用框架。

### 8.2 Relay 改造

修改：

- `include/relay/transforms/pipeline.h`
- `src/relay/transforms/pipeline.cc`

职责：

1. 用 `RelayPassEntry` 替代静态 `unordered_map<string, RelayPassFunc>`。
2. 为每个 Relay pass 填写 `PassInfo`。
3. 增加新入口：

```cpp
Function RunRelayPassPipeline(const Function& func, const PassPipelineConfig& config);
```

4. 保留旧入口，并转成 `PassPipelineConfig`。

### 8.3 TIR 改造

修改：

- `include/tir/transforms/pipeline.h`
- `src/tir/transforms/pipeline.cc`

职责同 Relay。

### 8.4 CompileConfig 改造

修改：

- `include/api/compile_config.h`
- `src/api/compile_config.cc`
- `src/api/compiler.cc`

职责：

1. 增加 Relay/TIR pass 配置字段。
2. 工厂函数设置默认 `opt_level`。
3. `Compiler::Compile` 改为从 `CompileConfig` 构造 pass pipeline config。

### 8.5 测试改造

修改：

- `test/pass_pipeline_test.cpp`
- `test/profile_bundle_test.cpp`
- `CMakeLists.txt`

新增测试点：

1. Relay/TIR 未知 pass 报错。
2. opt-level 默认 pipeline 展开。
3. disabled pass 从默认 pipeline 中移除。
4. disabled pass 被显式选择时报错。
5. required pass 自动插入到依赖者前面。
6. required pass 与 disabled pass 冲突时报错。
7. required pass 环检测。
8. target 不适用 pass 的默认跳过和显式报错。
9. legacy `optimize_default` 兼容。
10. profiling bundle 中包含 per-pass event、hash、changed 字段和可选 artifact。

## 9. MVP PassInfo 建议

Relay:

| pass | opt_level | required | default | 说明 |
| --- | --- | --- | --- | --- |
| `infer_type` | 0 | 无 | 是 | 保持类型不变量 |
| `fold_tuple_get_item` | 1 | `infer_type` | 是 | 折叠 tuple get item |
| `fold_constant` | 1 | `infer_type` | 是 | 常量折叠 |
| `simplify_expr` | 1 | `infer_type` | 是 | 基础代数化简 |
| `canonicalize_cast` | 2 | `infer_type` | 是 | cast 规范化 |
| `remove_standalone_reshapes` | 2 | `infer_type` | 是 | reshape 清理 |
| `eliminate_common_subexpr` | 2 | `infer_type` | 是 | 公共子表达式消除 |
| `eliminate_dead_let` | 2 | `infer_type` | 是 | 删除 dead let |
| `annotate_memory_scope` | 3 | `infer_type` | 是 | memory scope 标注 |
| `capture_post_dfs_index_in_spans` | 3 | `infer_type` | 是 | span index 捕获 |

TIR:

| pass | opt_level | required | default | 说明 |
| --- | --- | --- | --- | --- |
| `fold_constant` | 1 | 无 | 是 | 常量折叠 |
| `simplify_expr` | 1 | 无 | 是 | 基础表达式化简 |
| `force_narrow_index_to_i32` | 2 | 无 | 是 | 索引收窄 |
| `convert_for_loops_serial` | 2 | 无 | 是 | codegen 前 loop type 规范化 |
| `loop_partition` | 3 | 无 | 是 | loop partition |
| `unroll_loop` | 3 | 无 | 是 | loop unroll |
| `vectorize_loop` | 3 | 无 | 是 | loop vectorize |
| `remove_no_op` | 1 | 无 | 是 | 清理 no-op |

说明：

1. Relay `infer_type` 放在 opt level 0，是为了让 pipeline 默认保持类型不变量。
2. `Compiler::Compile` 中当前手工调用 `InferTypePass` 的位置需要在实现 PR 中重新评估，避免重复运行。
3. TIR `convert_for_loops_serial` 仍放在默认 pipeline 中，保证当前 LLVM codegen 不直接接收 unsupported loop type。

## 10. 错误信息格式

错误信息必须包含：

1. IR kind。
2. pass name。
3. 失败原因。
4. 可用 pass 或冲突链路。

示例：

```text
Unknown Relay pass 'foo'. Available passes: infer_type, fold_constant, simplify_expr, ...
Relay pass 'eliminate_common_subexpr' requires disabled pass 'infer_type'.
TIR pass dependency cycle detected: a -> b -> a.
TIR pass 'vectorize_loop' is not applicable to target 'c'.
```

## 11. 落地顺序

建议拆成三个 PR：

1. PR A: 本设计文档。
2. PR B: 新增 `base/pass_manager` 和纯展开规则测试，不改变现有 Relay/TIR 行为。
3. PR C: Relay/TIR pipeline 接入 pass manager，`CompileConfig` 接入 opt-level/selected/disabled 配置，补 profiling 和兼容测试。

如果 PR C 过大，可以继续拆成：

1. Relay 接入。
2. TIR 接入。
3. CompileConfig 和 compiler 接入。

## 12. 验收标准

Issue #16 完成时，必须满足：

1. `RunRelayPassPipeline` 和 `RunTIRPassPipeline` 旧入口仍可用。
2. 未知 pass 名清晰失败。
3. 默认 pipeline 可由 `opt_level` 展开。
4. `CompileConfig` 可以选择和禁用 Relay/TIR pass。
5. required pass 可自动展开。
6. required/disabled 冲突、依赖环、target 不匹配都清晰失败。
7. profiling bundle 记录 per-pass timing、IR hash 和可选 IR snapshot。
8. 测试覆盖 pass ordering、disabled pass、required pass expansion 和 instrumentation output。

