# KXC Runtime 抽象层架构指南（历史/专题材料）

> **状态：已归档。** 对象系统等背景仍可参考，但路径、FFI 与
> ExecutionPlan 示例不代表当前公共 API。当前 `RuntimeSession` 只消费 ready
> `CompiledModule + ExecutablePlan`，不负责编译、缓存、shape prediction 或
> fallback；以 `include/kxc/runtime/session.h`、`src/runtime/session.cc` 和
> `test/runtime_session_test.cpp` 为事实源。

本文档保留历史运行时抽象背景。

## 1. 核心对象系统 (Object System)

整个运行时系统的基石是侵入式引用计数对象系统，它允许 C++ 对象在 Python 中被安全地管理和访问，实现了零拷贝的对象传递。

### 1.1 C++ 层：`Object` 与 `ObjectRef`

- **`kxc::Object`**: 所有运行时对象的基类。
  - **功能**: 维护一个原子引用计数器 (`_refCount`)。
  - **内存管理**: 通过 `IncRef()` 和 `DecRef()` 手动管理生命周期。当引用计数降为 0 时，自动调用 `delete this`。
  - **RTTI**: 通过 `GetTypeId()` 提供运行时类型识别。

- **`kxc::ObjectRef`**: `Object` 的智能指针封装。
  - **功能**: 类似于 `std::shared_ptr`，但在对象内部存储引用计数（侵入式）。
  - **设计目的**: 作为函数参数和返回值，确保跨语言边界时的内存安全。
  - **关键特性**: 是所有句柄类（Handle Class）的基类。

**代码示例 (C++):**
```cpp
// 定义一个继承自 Object 的类
class MyObj : public kxc::Object {
public:
    // 唯一的类型 ID
    const kxc::TypeIndex GetTypeId() const override { return 7; }
    std::string name = "MyObject";
};

// 使用 ObjectRef 进行管理
ObjectRef CreateMyObj() {
    // 此时引用计数为 0
    MyObj* obj = new MyObj(); 
    // 包装进 ObjectRef，引用计数 +1
    return ObjectRef(obj); 
}
```

### 1.2 Python 层：映射与绑定

通过 `pybind11` 的自定义 `type_caster`，C++ 的 `ObjectRef` 会自动转换为对应的 Python 类。

- **自动向下转型**: 如果 C++ 返回 `ObjectRef`，但底层是 `MyObj`，Python 端会自动获得 `MyObj` 类型的实例。
- **引用保持**: Python 对象持有 C++ 对象的引用，防止其在 Python 使用期间被释放。

---

## 2. 类型擦除与函数调用 (PackedFunc)

为了在 C++ 和 Python 之间动态注册和调用函数，我们使用了 `PackedFunc` 机制。这是一种类型擦除的函数包装器。

### 2.1 C++ 层：`PackedFunc` 与 `Registry`

- **`PackedFunc`**: 通用函数签名 `void(Args, RetValue*)`。
  - **Args**: 包含参数值 (`Value` union) 和类型码 (`TypeCode`) 的数组。
  - **RetValue**: 用于存储返回值的容器。
  - **自动转换**: `ArgConverter` 模板自动将 `Args` 中的原始数据转换为 C++ 类型（如 `int`, `std::string`, `ObjectRef`）。

- **`Registry`**: 全局函数注册表。
  - 允许在 C++ 中注册函数，并在 Python 中按名称查找。

**注册示例 (C++):**
```cpp
// 定义普通 C++ 函数
int Add(int a, int b) { return a + b; }

// 注册到全局表
KXC_REGISTER_GLOBAL("my_module.add")
.set_body(ToPackedFunc(Add));
```

### 2.2 Python 层：调用注册函数

Python 端可以通过 `get_global_func` 获取并调用这些函数。

```python
import kxc_runtime

# 获取函数
add_func = kxc_runtime.get_global_func("my_module.add")

# 调用 (自动处理类型转换)
result = add_func(10, 20)
print(result) # 输出 30
```

---

## 3. 设备抽象 (Device Abstraction)

运行时需要统一管理异构硬件（CPU, GPU, etc.）。

### 3.1 `Device` 类

表示一个逻辑计算设备。

- **属性**:
  - `device_type` (`DeviceTypeCode`): 设备类型枚举 (CPU=0, GPU=1, etc.)。
  - `device_id` (int): 设备索引。
- **管理**: 通过 `DeviceManager` 单例进行缓存管理，确保相同的 `(type, id)` 对应同一个 `Device` 对象实例。

### 3.2 `DeviceAPI` 接口

定义了底层硬件操作的统一接口（纯虚基类）。

- **核心方法**:
  - `AllocDataSpace`: 分配显存/内存。
  - `FreeDataSpace`: 释放内存。
  - `CopyDataSync` / `CopyDataAsync`: 设备间同步或流绑定异步拷贝。
  - `CreateStream` / event API: 管理后端 stream 与完成事件。

### 3.3 使用方法

**Python 端创建设备:**
```python
cpu_dev = kxc_runtime.Device.cpu()
cuda_dev = kxc_runtime.Device.cuda(0)
```

**C++ 端获取 API:**
```cpp
Device cpu = Device::CPU();
Device cuda = Device::CUDA(0);
runtime::NDArray host = runtime::NDArray::Zeros(
    {256}, runtime::DataTypeFromString("float32"), cpu);
runtime::NDArray device_array = host.CopyTo(cuda);
```

---

## 4. 内存管理与数据传输

公共 API 只暴露 `Storage`、`NDArray`、`DeviceStream` 和 `AsyncOperation`。
裸指针仅存在于 `DeviceAPI` 后端、DLPack 边界和内核 ABI，不编码成 Python 整数。

### 4.1 核心 API

1. `Storage::Alloc` / `Storage::FromExternal` 管理物理内存及其唯一释放责任。
2. `NDArray::Empty` / `NDArray::Zeros` 创建带 dtype、shape 和 Device 的张量视图。
3. `NDArray::CopyTo` / `CopyFrom` 执行所有权安全的同步复制。
4. `NDArray::CopyFromAsync` 返回持有 stream、event 和参与 Storage 的 `AsyncOperation`。

### 4.2 完整示例 (Python)

```python
import kxc_runtime
cpu = kxc_runtime.Device.cpu()
array = kxc_runtime.NDArray.zeros([256], "float32", cpu)
view = memoryview(array)  # 仅连续 cpu:0 NDArray 可导出 buffer
```

## 5. 扩展指南

### 如何添加新的 C++ 类并暴露给 Python?

1.  **定义类**: 继承自 `kxc::Object`，实现 `GetTypeId`。
2.  **编写绑定**: 在 `py_bindings.cpp` 中使用 `py::class_` 绑定。
3.  **注册工厂函数**: 编写一个返回 `ObjectRef` 的 C++ 函数，并使用 `KXC_REGISTER_GLOBAL` 注册。
4.  **Python 使用**: 通过 `get_global_func` 获取工厂函数创建实例。

### 如何添加新的设备后端?

1.  **继承 `DeviceAPI`**: 实现分配、释放、清零、同步/异步复制、stream、event 和属性查询契约。
2.  **注册 API**: 在 `DeviceAPIManager::GetAPI` 中添加新的 `DeviceTypeCode` 分支，返回新的 API 单例。
3.  **编译**: 将新的 `.cc` 文件加入构建系统（如 `build_pybind.bat`）。

---

## 6. 编译优化：设备通用 vs 设备特定（三个位置）

在 KXC 里，你可以把“优化/调度”放在三层来做：

1. **Relay（图级/算子级）Pass**：改写计算图结构，适合做算子融合、代数化简、布局变换等。
2. **TE（算子级）Schedule**：固定算子语义不变，只改变循环与并行策略，适合做 `tile/vectorize/unroll/bind`。
3. **TIR（低层语句）Pass**：直接改写 `Stmt`/`PrimExpr`，适合做 loop 标注、内存/线程域注入、低层 canonicalization。

下面分别说明“怎么注册/怎么实现”以及如何区分 **设备通用** 和 **设备特定**。

### 6.1 Relay Pass：注册与实现

**入口/基类**

- Relay 的 IR 节点定义见 `include/base/relay.h:1`。
- Pass 基础设施见 `include/base/pass.h:1`：`RelayPassFunctor` 负责按节点类型分发，`RelayPass` 默认做 copy-on-write 的 Mutator。

**实现方式（无需全局注册，直接在 pipeline 里调用）**

你可以像 `test/test_pass.cpp:15` 那样继承 `kxc::RelayPass`，覆盖你关心的 `Visit*`：

```cpp
class MyPass : public kxc::RelayPass {
protected:
  kxc::Expr VisitCall(const kxc::CallNode* op, const kxc::Expr& ref) override {
    // 1) 先递归改写子节点
    // 2) 判断是否需要改写
    // 3) changed==false 时返回 ref，实现 copy-on-write
  }
};
```

**设备通用优化（示例：代数化简 / 常量折叠）**

- 这类 Pass 不需要 target 信息；例如“`x + 0 -> x`”、“`mul(const, const)` 预计算”等。
- 参考已有的 Pass 文档：`docs/FoldConstant.md`。

**设备特定优化（示例：标注/分派）**

当你需要做“针对某设备的改写”，你需要把 target/device 信息带到 Relay 上。

当前代码库里 `CallNode` 有一个 `ObjectRef attrs` 字段（见 `include/base/relay.h:90`），这使得你可以把“目标信息”作为 `ObjectRef` 附着在 `Call` 上。

- 示例 Pass：`AnnotateDevice` 在 `test/test_pass.cpp:74`。
- 它的行为是：如果 `Call.attrs` 为空，就写入一个目标对象；如果已存在，则保持不变。

这种做法的作用类似 TVM 的“pass context/target”，但目前是用 `attrs` 临时承载。后续如果你引入显式的 `Target`/`PassContext`，可以把这条链路替换掉。

### 6.2 TE Schedule：注册与实现

**入口/对象模型**

- TE 算子与 Tensor 表达：`include/te/te.h`。
- Schedule 关键类型：`te::Schedule` / `te::Stage` / `te::IterVar`（见 `include/te/te.h:20`、`include/te/te.h:98`）。

**“注册”是什么意思**

在 TE 层，通常不做“全局注册某个 schedule”。惯用方式是：

1. 在 build/lowering pipeline 里，根据 `target` 选择一个 schedule 函数。
2. schedule 函数对 `Schedule s` 做原语调用（split/tile/bind/...）。

当前仓库已经提供了若干 schedule 原语：

- `Stage::split/fuse/reorder/tile`：`include/te/te.h:394` 起。
- `Stage::vectorize/unroll/parallel/bind`：`include/te/te.h:503` 起。

**设备通用 schedule（示例：CPU 友好的 tile + vectorize）**

- 典型策略：外层 `tile` 改善 cache locality，内层 `vectorize` 触发 SIMD。
- 示例见 `test/test_schedule_api.cpp:43`。

**设备特定 schedule（示例：GPU thread 绑定）**

- GPU 常见策略：把 innermost 轴 `bind(threadIdx.x)`，并配合 `blockIdx.x`/`vthread` 等。
- 当前最小示例：`test/test_schedule_api.cpp:105`，其中：
  - `thread_axis(IntImm(64), "threadIdx.x")` 创建 thread 轴。
  - `stage.bind(inner, tx)` 将 `inner` 设为 `IterVarType::kThreadIndex` 并写入 `thread_tag`。

注意：这一步只是“标注”与“意图表达”。要真正生成 CUDA kernel，还需要后续 lowering/codegen 支持把 `IterVarType::kThreadIndex` 翻译为 `tir::AttrStmt(thread_extent=...)` 或等价结构。

### 6.3 TIR Pass：注册与实现

**入口/IR 结构**

- TIR `Stmt` 节点：`include/tir/stmt.h:1`。
- 典型 loop 结构：`ForNode` 带 `ForType`（见 `include/tir/stmt.h:71`）。

**实现方式（当前是“函数式 pass”）**

仓库目前没有完整的 `StmtMutator`/`PassManager` 基础设施，所以建议按“纯函数改写”的方式写 pass：

```cpp
kxc::tir::Stmt MyTIRPass(const kxc::tir::Stmt& s);
```

示例 `VectorizeSerialLoops`：`test/test_tir_structure.cpp:68`。

**设备通用 TIR 优化**

- 例如：对 `ForType` 做规范化、对表达式做常量传播、把 `SeqStmt` 扁平化等。

**设备特定 TIR 优化**

- CPU：把热点 loop 标注为 `ForType::Parallel`（对应 OpenMP/pthreads 的 lowering）。
- GPU：把 loop/iter 绑定到 thread/block，并用 `AttrStmt(thread_extent, ...)` 注入线程域信息。
  - `AttrStmtNode` 定义见 `include/tir/stmt.h:154`。

在 TVM 的典型链路里，TE 的 `bind(threadIdx.x)` 会在 lowering 时生成 TIR 的 `AttrStmt(thread_extent=...)`；在本仓库里你可以先用 TIR Pass 直接生成/校正这些标注，作为后续 codegen 的输入。

## 7. ExecutionPlan JSON Workflow (Phase-1)

Phase-1 supports exporting ExecutionPlan to JSON at compile time, then loading JSON at runtime.

### 7.1 Compile-side APIs

- `kxc.relay.transform.lower_to_exec_plan_json(func) -> string`
- `kxc.relay.transform.lower_to_exec_plan_json_file(func, path) -> string`

### 7.2 Runtime-side APIs

- `kxc.disco.execute_plan_json(session, json_text) -> Map<int, DRef>`
- `kxc.disco.execute_plan_json_output(session, json_text) -> DRef`
- `kxc.disco.execute_plan_json_file(session, path) -> Map<int, DRef>`
- `kxc.disco.execute_plan_json_file_output(session, path) -> DRef`

### 7.3 C++ Serialization APIs

- `SerializeExecutionPlanToJson(const ExecutionPlan&)`
- `DeserializeExecutionPlanFromJson(const std::string&)`
- `LoadExecutionPlanFromJsonFile(const std::string&)`
- `SaveExecutionPlanToJsonFile(const ExecutionPlan&, const std::string&)`

### 7.4 Notes

- `schema_version` is fixed to `1` in Phase-1.
- `kernel_symbol` is reserved for Phase-2 kernel dispatch integration.
- Phase-1 keeps `KernelExec` runtime behavior as placeholder.

## Pass Pipeline APIs (2026-02 update)

The repository provides lightweight pass pipelines for Relay and TIR.

### Relay Pipeline Entry

- `kxc.relay.transform.run_pipeline(func, pass_names)`

### Relay Single Pass Entry

- `kxc.relay.transform.fold_tuple_get_item(func)`
- `kxc.relay.transform.fold_constant(func)`
- `kxc.relay.transform.simplify_expr(func)`
- `kxc.relay.transform.canonicalize_cast(func)`
- `kxc.relay.transform.remove_standalone_reshapes(func)`
- `kxc.relay.transform.eliminate_common_subexpr(func)`
- `kxc.relay.transform.eliminate_dead_let(func)`
- `kxc.relay.transform.annotate_memory_scope(func)`
- `kxc.relay.transform.capture_post_dfs_index_in_spans(func)`

Relay default alias:

- `optimize_default` ->
  `fold_tuple_get_item` ->
  `fold_constant` ->
  `simplify_expr` ->
  `canonicalize_cast` ->
  `remove_standalone_reshapes` ->
  `eliminate_dead_let` ->
  `annotate_memory_scope` ->
  `capture_post_dfs_index_in_spans` ->
  `infer_type`

`eliminate_common_subexpr` 仍是显式单 Pass 入口，但当前不属于默认别名。

### TIR Pipeline Entry

- `kxc.tir.transform.run_pipeline(func, pass_names)`

### TIR Single Pass Entry

- `kxc.tir.transform.fold_constant(func)`
- `kxc.tir.transform.simplify_expr(func)`
- `kxc.tir.transform.force_narrow_index_to_i32(func)`
- `kxc.tir.transform.convert_for_loops_serial(func)`
- `kxc.tir.transform.loop_partition(func)`
- `kxc.tir.transform.unroll_loop(func)`
- `kxc.tir.transform.vectorize_loop(func)`
- `kxc.tir.transform.remove_no_op(func)`

TIR default alias:

- `optimize_default` ->
  `fold_constant` ->
  `simplify_expr` ->
  `force_narrow_index_to_i32` ->
  `convert_for_loops_serial` ->
  `loop_partition` ->
  `unroll_loop` ->
  `vectorize_loop` ->
  `remove_no_op`

### C++ APIs

- `kxc::relay::RunRelayPassPipeline(const Function&, const Array<String>&)`
- `kxc::tir::RunTIRPassPipeline(const tir::PrimFunc&, const Array<String>&)`

The pipeline throws a runtime error when pass names are unknown.
