# Compiler Foundation：静态 Runtime plan 交接

> **状态：** Region task-DAG 实验轨已删除。唯一的静态数据面是
> `CompiledModule + ExecutablePlan + RuntimeSession`。

## 当前契约

`Compiler::Compile` 返回 `CompiledGraph`，其中保留不可变的 `CompiledModule`、
`ExecutablePlan`、graph semantic key 和按 plan 顺序持有的 `ArtifactPin`。pin 是编译
结果的生命周期属性；它不再经由 Runtime manifest 或 session API 传播。

`RuntimeSession(module, plan)` 在构造期验证：

- module 已就绪，module entry 与 ordered `KernelCall` 一一匹配；
- signature、静态 shape、dtype、device、constant key 和 constant value contract；
- 单 device、单 stream 的静态 ordered plan 边界。

`Run` 保持同步接口；`RunAsync` 依次提交 ordered calls，并让最终 completion 保活
module、plan、每次运行的 value table、输入/中间/输出 storage 和先前操作。它不查询
Compiler、cache 或 artifact generation，也不选择或替换 executable。动态
`CompiledModule` invocation ABI 仍是 module 层的独立能力；静态 `RuntimeSession`
在具备动态 graph memory plan 前继续明确拒绝非静态 invocation contract。

## 已删除的重复轨

已删除 default-OFF Region task-DAG gate、task-plan/task-executor sources 与 tests、
Runtime execution-mode selection、fallback/observer/manifest DTO，以及
`CompiledGraph::plan_variant()`。没有兼容别名或静默降级入口。

控制流继续由独立的 `ControlExecutionPlan` / `ControlRuntimeSession` 实验路径处理；它
不复用或依赖已删除的 Region task-DAG schema。

## 验证

CPU build 应以 LLVM/CUDA 关闭、其余现有 feature gates 按需开启，执行完整 CTest、
include-layer 与 public-header checks。`runtime_session_test` 仍覆盖 ordered plan 的
输入校验、常量绑定、同步/异步执行、storage 生命周期和 memory-plan reuse。
