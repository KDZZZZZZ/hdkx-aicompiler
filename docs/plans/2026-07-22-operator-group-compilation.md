# 按算子/融合组编译实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** 将当前“整张 Relay 图降低并编译为唯一 PrimFunc/Kernel”的流程，改造成“图级编排 + 每个算子或融合组独立 PrimFunc/Kernel”的 TVM 风格编译与执行流程。

**Architecture:** Relay 仍表示整图；Compiler 先建立稳定的数据流值编号并划分 primitive group，再对每个 group 独立执行 Relay/TE → TIR、TIR pass、Kernel ABI 构建和后端编译。Runtime 只消费不含 Relay/TE/TIR 的可执行计划，以 value table 驱动多个 kernel，管理常量、中间张量、输出与异步生命周期。第一阶段关闭融合，做到一算子一 PrimFunc；第二阶段增加模式分类与融合，最终语义是一融合组一 PrimFunc，而不是机械地永久保持一算子一 kernel。

**Tech Stack:** C++17、Relay/TE/TIR、LLVM ORC JIT、CUDA/NVRTC、NDArray、DeviceStream、CMake/Ninja。

---

## 1. 结论与术语

### 1.1 当前模型

当前链路是：

```text
Relay Function（整图）
  -> RelayToTEConverter 递归构造整张 TE DAG
  -> CollectOpsDFS 收集所有 TE operation
  -> 所有计算拼为一个 SeqStmt
  -> 唯一 PrimFunc(global_symbol="main")
  -> 唯一 KernelSignature / CompiledKernel / CompiledModule
  -> RuntimeSession 单次 Launch
```

直接证据：

- `src/compiler/lowering/relay_to_tir.cc:509-541` 把整个函数体转换为一个 TE DAG，并从所有输出递归收集所有 operation。
- `src/compiler/lowering/relay_to_tir.cc:602-654` 把所有 operation 拼成一个 `SeqStmt`，最后构造唯一 `PrimFunc`。
- `src/compiler/lowering/relay_to_tir.cc:640-642` 固定写入 `global_symbol=main`。
- `src/compiler/internal/compile_state.h` 和 `src/compiler/compile_state.cc` 的状态机每个阶段只保存一个 `PrimFunc`、一个 `KernelSignature` 和一个 `CompiledKernel`。
- `src/runtime/internal/compiled_module_node.h` 只保存一个可执行入口。
- `src/runtime/session.cc` 每次运行只组装一次参数并调用一次 `CompiledModule::Launch`。

### 1.2 目标模型

“像 TVM 一样按算子编译”应解释为：

```text
Relay Function（图级语义）
  -> 稳定值编号 + primitive group 划分
  -> group 0 -> PrimFunc 0 -> kernel 0
  -> group 1 -> PrimFunc 1 -> kernel 1
  -> group N -> PrimFunc N -> kernel N
  -> runtime ExecutablePlan 按依赖顺序调度 kernel
```

TVM 的最终边界通常是“一个 primitive/fused function 对应一个 PrimFunc”，并非永久的一原始算子一 kernel。严格按算子拆分是本项目的安全过渡态；模式融合完成后，`conv2d + bias/add + relu` 等合法组合可以重新成为一个 PrimFunc。

## 2. 二者区别

| 维度 | 当前整图单 PrimFunc | 按算子/融合组多个 PrimFunc |
|---|---|---|
| 编译单位 | 整个 Relay Function | 单算子或 primitive group |
| 运行单位 | 一次 kernel launch | 图执行器按计划多次 launch |
| 中间值 | PrimFunc 内部 `Allocate` | runtime value table 中的 NDArray/存储槽 |
| 调度 | 全图共享一次 TIR/CUDA 调度决策 | 每个 group 独立 schedule，可按算子特征选择策略 |
| 编译缓存 | 图任一处改变通常重编整图 | 可按 group hash 缓存和复用 |
| 后端选择 | 整图只能走一个统一后端入口 | 可逐 group 选择 LLVM、CUDA 或外部库 |
| 编译时间 | 大图容易形成大 TIR/LLVM/CUDA TU | 可增量、并行、按 group 编译，但入口数增加 |
| 执行开销 | launch 少，中间值可局部化 | launch 多且有中间内存流量，需要融合和内存规划补偿 |
| 图优化 | 跨算子优化天然可见，但边界不可控 | 图优化与 kernel 优化分层，融合策略显式可测 |
| 控制流/动态性 | 单 kernel 很难表达一般图执行 | graph/VM 层可以调度、分支、shape function 和设备通信 |
| 可观测性 | profiling 只能看到一个大 kernel | 可看到每个 op/group 的耗时、shape、后端和失败点 |
| 分布式 | 计算和通信不易穿插 | plan 可显式交织 kernel、copy、collective、barrier |

整图单 PrimFunc 并非没有价值：对短小静态图，它可以减少 launch，并让中间结果留在局部存储。但是它把图级编排、算子调度和后端 kernel 生成耦合在一起；模型增大后，会限制算子级 schedule、外部库 dispatch、编译缓存、独立调优和分布式编排。

## 3. 关键架构决定

### ADR-1：采用“两阶段拆分”，不直接一步实现复杂融合

**决定：** 第一阶段每个普通 Relay `Call` 独立成组；第二阶段引入 `OpPatternKind` 和融合分组。

**原因：** 多 kernel ABI、常量归属、中间张量生命周期和运行时调度必须先有独立的正确性闭环。若一开始同时实现后支配树融合，失败时难以判断是分组错误、lowering 错误还是 runtime 错误。

**代价：** 第一阶段性能可能显著低于当前整图 kernel；该阶段是迁移门禁，不是最终性能目标。

### ADR-2：保留 `CompiledModule` 作为 opaque runtime 对象，但把它升级为多入口模块

**决定：** `CompiledModule` 内部从单个 kernel 改成 `symbol -> KernelEntry`，公共 API 改为按 symbol 查询签名、元数据和启动函数。删除无 symbol 的单入口假设。

**原因：** 这最接近 TVM `runtime::Module` 的职责，也允许 LLVM/CUDA 的多个入口共享一个 JIT/module 资源。0721 迁移已经把 Node 和后端资源隐藏在 `src/runtime/internal`，因此可以在不向公共头暴露 LLVM/CUDA/TIR 的前提下修改内部布局。

### ADR-3：新增 runtime-neutral `ExecutablePlan`，不直接复用 distributed `ExecutionPlan`

**决定：** 新计划只含 value id、shape、dtype、device、storage id、kernel symbol 和输入输出 id，不包含 Relay、TE、TIR 或 `CompiledKernel`。

**原因：** 当前 distributed `ExecutionPlan` 同时包含放置、通信、barrier 和 `tir::PrimFunc`，且 `ExecuteKernel` 尚未接入真实 module。单设备核心执行计划应位于 runtime；distributed 层后续组合或适配它，不能让 runtime 反向依赖 distributed/compiler。

### ADR-4：常量 key 和 kernel symbol 必须基于图级稳定身份

**决定：** 常量 key 使用图 value id 或结构哈希，例如 `constant_17`；kernel symbol 使用稳定 group id 与结构哈希，例如 `fused_nn_conv2d_add_relu_3a7c9e2b`。禁止使用对象地址或每组局部 ordinal 作为全模块身份。

**原因：** 每个 group 都从 0 开始编号会造成常量 key 冲突；非确定符号会破坏 cache、profiling、序列化和测试稳定性。

## 4. 目标数据结构

Compiler 私有结构：

```cpp
struct PrimitiveGroup {
    int group_id;
    String symbol;
    Array<Expr> nodes;
    Array<Expr> inputs;
    Array<Expr> outputs;
};

struct LoweredPrimitive {
    int group_id;
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

Runtime 公共契约：

```cpp
class ValueSpecNode final : public Object {
public:
    int value_id{-1};
    int storage_id{-1};
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
    Array<int> input_values;
    Array<int> output_values;
};

class ExecutablePlanNode final : public Object {
public:
    Array<ValueSpec> values;
    Array<KernelCall> calls;
    Array<int> input_values;
    Array<int> output_values;
};
```

`ExecutablePlan` 不保存 TIR。TIR、group 内容、pass 中间态和结构哈希只留在 Compiler 私有实现与 profiling artifact 中。

## 5. 实施任务

### Task 1：锁定当前单 kernel 行为并建立多 kernel 失败测试

**Files:**

- Modify: `test/compiler_contract_test.cpp`
- Modify: `test/runtime_session_test.cpp`
- Create: `test/operator_group_compilation_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 增加 characterization test，证明 `add -> relu` 当前只生成 `global_symbol=main` 的一个 PrimFunc。
2. 增加目标契约测试：同一图应得到两个稳定 symbol、两个 KernelSignature 和两条 KernelCall；测试先失败。
3. 增加 diamond 图和多输出图 fixture，锁定 value id、调用顺序和 live-out 数量。
4. 增加重复常量值、同一常量被多个 group 消费的 fixture，要求 module 级常量只保存一份稳定绑定。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target operator_group_compilation_test compiler_contract_test runtime_session_test -j 4
```

Expected: characterization tests pass；新的多 kernel 契约测试因缺少 `ExecutablePlan`/多入口 module 而失败。

### Task 2：建立 runtime-neutral ExecutablePlan

**Files:**

- Create: `include/kxc/runtime/executable_plan.h`
- Create: `src/runtime/executable_plan.cc`
- Create: `src/runtime/internal/executable_plan_validation.h`
- Create: `test/executable_plan_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 先写非法 value id、重复 producer、缺失输出、拓扑逆序、shape/dtype/device 缺失的失败测试。
2. 实现 `ValueSpec`、`KernelCall`、`ExecutablePlan` ObjectRef 与完整 validation。
3. 约束每个非 input/constant value 恰好有一个 producer；每条 call 的输入必须在此前可用；module 输出必须已产生。
4. 首版令 `storage_id == value_id`，暂不做复用，确保内存生命周期语义简单。
5. 将新公共头加入 `kxc_runtime` FILE_SET，并运行公共头独立编译检查。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target executable_plan_test check_include_layers check_public_headers -j 4
```

Expected: plan validation、include layer 和 public header checks 全部通过。

### Task 3：建立图值编号和“一算子一组”分区器

**Files:**

- Create: `src/compiler/internal/graph_partition.h`
- Create: `src/compiler/graph/value_graph.cc`
- Create: `src/compiler/graph/partition.cc`
- Create: `test/graph_partition_test.cpp`
- Modify: `CMakeLists.txt`

**Steps:**

1. 从已完成 `InferType` 的 Relay Function 构建 DAG；为 param、constant、call output 和 tuple field 分配确定 value id。
2. 使用表达式拓扑顺序和结构内容生成稳定 id，不依赖 `Object*` 地址。
3. 首版每个非通信 `Call` 形成一个 `PrimitiveGroup`；通信 op、tuple 元操作和控制流不得伪装成 compute group。
4. 计算每个 group 的外部输入和 live-out；同一值被多个 consumer 使用时只产生一次。
5. 输出 runtime-neutral plan 草稿和 compiler-private group 列表。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target graph_partition_test -j 4
```

Expected: chain、branch、diamond、tuple、多输出、共享常量和拓扑稳定性测试通过。

### Task 4：将 Relay/TE lowering 改为按 group 生成 PrimFunc

**Files:**

- Modify: `include/kxc/compiler/lowering/relay_to_tir.h`
- Modify: `src/compiler/lowering/relay_to_tir.cc`
- Modify: `src/compiler/lowering/lowered_function.cc`
- Create: `src/compiler/internal/lowered_graph.h`
- Create: `src/compiler/lowering/lowered_graph.cc`
- Modify: `test/infer_type_test.cpp`
- Modify: `test/operator_group_compilation_test.cpp`

**Steps:**

1. 把现有整图 lowering 中可复用部分拆成“边界 placeholder 建立”“组内 TE 转换”“TE DAG 到 PrimFunc”三个私有阶段。
2. 对 group 外部输入创建 placeholder；converter 遇到组外 Expr 时只能从边界映射读取，禁止递归跨越 group。
3. 支持一个 group 多个 live-out，复用现有 multi-output 参数与 `kxc.output_count` 契约。
4. 常量 binding 使用图级 key；每个 PrimFunc 的 `kxc.constant_keys` 只包含本 group 使用的常量子集。
5. 每个 PrimFunc 写入唯一 `global_symbol`、`kxc.group_id` 和可诊断的 op 列表/结构 hash。
6. 返回 `LoweredGraph`，不再返回唯一 `LoweredFunction` 作为整图事实。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target operator_group_compilation_test infer_type_test kernel_signature_test -j 4
```

Expected: `add -> relu` 产生两个 PrimFunc；每个 PrimFunc 只含自身 group 的计算和 ABI 参数。

### Task 5：把 Compiler 状态机改成多 primitive 状态

**Files:**

- Modify: `src/compiler/internal/compile_state.h`
- Modify: `src/compiler/compile_state.cc`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/compiler/internal/kernel_abi_builder.h`
- Modify: `src/compiler/kernel_abi_builder.cc`
- Modify: `test/compiler_contract_test.cpp`

**Steps:**

1. 将状态字段从单个 `tir_`/`signature_`/`kernel_` 改为同序、同 symbol 的 primitive 集合。
2. 每次状态迁移验证 group id、symbol、PrimFunc、signature、metadata 和 backend kernel 一一对应。
3. `OptimizeTIR`、CUDA thread binding 和 `BuildKernelSignature` 对每个 primitive 独立执行。
4. profiling stage 同时记录 group count、逐组 hash/size/symbol，并把 artifact 路径改为 `.../groups/<symbol>/...`。
5. 任一 group 失败时错误包含 stage、group id、symbol 和 op 列表。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target compiler_contract_test kernel_signature_test -j 4
```

Expected: 状态机拒绝数组长度不一致、重复 symbol、signature 错配和半编译 group。

### Task 6：把 CompiledModule 升级为多入口模块

**Files:**

- Modify: `include/kxc/runtime/compiled_module.h`
- Modify: `src/runtime/internal/compiled_module_node.h`
- Modify: `src/runtime/compiled_module.cc`
- Modify: `src/codegen/internal/compiled_kernel.h`
- Modify: `src/codegen/common/compiled_kernel.cc`
- Modify: `test/compiled_module_test.cpp`

**Steps:**

1. 定义私有 `KernelEntry{signature, launch_metadata, executable}`，按 symbol 存入 module。
2. 公共 API 改为 `HasFunction(symbol)`、`signature(symbol)`、`launch_metadata(symbol)` 和 `Launch(symbol, args, stream)`。
3. module 级常量池去重；每个 entry 只能通过自身 signature 中的 constant key 访问常量。
4. 删除无 symbol 的唯一入口假设；单 primitive module 仍只是 entries 数量为 1 的正常模块。
5. `CompiledModule::Launch` 保留现有 NDArray、stream、device、alignment 和异步资源保活校验，并在错误中加入 symbol。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target compiled_module_test codegen_llvm_test -j 4
```

Expected: 一个 module 可独立启动两个 symbol；未知/重复 symbol 明确失败；原 ABI 校验不退化。

### Task 7：让 RuntimeSession 执行图计划

**Files:**

- Modify: `include/kxc/runtime/session.h`
- Modify: `src/runtime/internal/session_node.h`
- Modify: `src/runtime/session.cc`
- Create: `src/runtime/internal/value_table.h`
- Modify: `test/runtime_session_test.cpp`
- Modify: `test/operator_group_compilation_test.cpp`

**Steps:**

1. Session 同时持有 `CompiledModule` 和 `ExecutablePlan`；构造时验证所有 KernelCall symbol 都存在且 signature 与 value specs 一致。
2. `Run` 建立 value table：绑定输入和常量，按 ValueSpec 分配中间值/输出，按 calls 顺序 launch。
3. 首版在同一 stream 串行执行；每次 launch 的 completion 在后继消费前必须满足同 stream 顺序与生命周期要求。
4. `RunAsync` 返回所有 graph output 和一个聚合 completion，保活整个 value table、module 和中间 storage。
5. 最后一个 consumer 完成前不得提前回收中间值；首版不复用 storage。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target runtime_session_test operator_group_compilation_test op_numeric_llvm_test -j 4
```

Expected: add→relu、MLP、CNN chain 和多输出图通过；逐 kernel 数值与整图结果一致。

### Task 8：切换 Compiler::Compile 并完成旧单入口迁移

**Files:**

- Modify: `include/kxc/compiler/compiler.h`
- Modify: `src/compiler/compiler.cc`
- Modify: `src/frontend/onnx_importer.cc`
- Modify: `test/codegen_llvm_test.cpp`
- Modify: `test/codegen_cuda_test.cpp`
- Modify: `test/onnx_importer_test.cpp`
- Modify: `test/resnet18_ir_dump.cpp`

**Steps:**

1. `Compiler::Compile` 返回包含多入口 module 和 plan 的可执行产物；RuntimeSession 不再从唯一 signature 推断整图输入输出。
2. 删除测试和调用点对 `global_symbol=main`、唯一 signature、唯一 launch metadata 的假设。
3. 增加 `CompilePrimitive` 私有 helper，禁止创建第二套公共编译入口。
4. 对 ResNet18 记录 group/kernel 数量、编译时间、峰值中间内存和端到端数值。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target codegen_llvm_test op_numeric_llvm_test onnx_importer_test compiler_contract_test runtime_session_test -j 4
```

Expected: LLVM 全量数值测试和 ResNet18 执行通过；无单入口 API 残留。

### Task 9：引入 TVM 风格算子模式分类与融合

**Files:**

- Modify: `include/kxc/relay/op_attr_types.h`
- Modify: `src/relay/op/nn/activation.cc`
- Modify: `src/relay/op/nn/convolution.cc`
- Modify: `src/relay/op/nn/dense.cc`
- Modify: `src/relay/op/nn/pooling.cc`
- Modify: `src/relay/op/nn/softmax.cc`
- Modify: `src/relay/op/tensor/math.cc`
- Modify: `src/relay/op/tensor/reduce.cc`
- Modify: `src/relay/op/tensor/transform.cc`
- Modify: `src/compiler/graph/partition.cc`
- Modify: `test/graph_partition_test.cpp`

**Steps:**

1. 新增 `OpPatternKind`：elementwise、broadcast、injective、commutative-reduce、out-elementwise-fusable、opaque。
2. 为全部已注册 Relay op 显式注册 pattern；没有 pattern 的 op 默认 opaque，不能跨边界融合。
3. 第一版融合规则只支持线性链：elementwise/injective 生产者与单 consumer 融合；reduce/conv/dense 只允许吸收后继 elementwise。
4. 第二版加入 post-dominator/路径检查，正确处理 diamond；同时设置最大 group depth、最大参数数和 target-specific 禁止规则。
5. 保留 `fusion_level=0` 的一算子一组模式作为调试和差分基线。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target graph_partition_test operator_group_compilation_test op_numeric_llvm_test -j 4
```

Expected: `conv2d + add + relu` 合为一个 group；不合法 diamond、多个重型 op、通信 op 和 opaque op 不被错误融合。

### Task 10：批量后端编译、缓存与内存规划

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

1. LLVM 将同一 target 的多个 PrimFunc 放入一个 LLVM module/JIT resource；每个 KernelEntry 共享资源所有权。
2. CUDA 将多个生成函数合入一个 NVRTC translation unit/CUmodule，并逐 symbol 获取 `CUfunction`。
3. cache key 至少包含 PrimFunc 结构 hash、Target、opt level、ABI version 和后端版本；禁止仅按 op name 或 shape 命中。
4. 基于最后使用位置分配 storage id，使不重叠的中间值复用同一 storage；input、constant、output 和异步存活值不得错误复用。
5. 对 view/reshape 等 alias 语义另设明确规则，未实现前继续生成真实 kernel，不做隐式 alias。

**Verify:**

```powershell
cmake --build out/build/windows-llvm-msys2 --target codegen_llvm_test op_numeric_llvm_test runtime_session_test -j 4
```

Omen CUDA:

```bash
cmake --build out/build/omen-cuda --target codegen_cuda_test runtime_session_test cupti_smoke_test -j 4
```

Expected: 多入口共享 backend module；cache 命中不改变数值；memory planner 降低峰值存储且 ASan/UBSan/Compute Sanitizer 无错误。

### Task 11：接入 distributed ExecutionPlan 并清理重复计划模型

**Files:**

- Modify: `include/kxc/distributed/execution_plan.h`
- Modify: `src/distributed/execution_plan.cc`
- Modify: `src/distributed/executor.cc`
- Modify: `src/compiler/distributed/multi_device.cc`
- Modify: `src/relay/distributed/plan_adapter.cc`
- Modify: `test/compiler_contract_test.cpp`

**Steps:**

1. 删除 `KernelExecNode` 内的 `tir::PrimFunc`；保留 symbol、value ids 和 worker set，或直接组合 runtime `KernelCall`。
2. 将当前创建空 `tir::PrimFunc()` 的路径替换为真实 compiled symbol 绑定。
3. `ExecutionPlanExecutor::ExecuteKernel` 从 worker module registry 查找 symbol，并按 value ids 调用真实 CompiledModule。
4. 通信节点作为不可融合边界；kernel、copy、collective、barrier 的拓扑和 stream 依赖必须显式验证。
5. 保持 distributed 依赖 runtime plan，runtime 不依赖 distributed。

**Verify:**

```powershell
cmake --build out/build/dev-mingw-cpu --target compiler_contract_test pass_pipeline_test -j 4
```

Expected: 原先的 “ExecutionPlan CompiledModule launch is not implemented” 失败测试改为真实 kernel 执行闭环。

## 6. 验收门禁

### 正确性

- chain、branch、diamond、tuple、多输出、共享常量均生成确定的 plan 和 symbol。
- `fusion_level=0` 下每个普通 Call 恰好一个 PrimFunc。
- 默认融合下每个 primitive group 恰好一个 PrimFunc。
- LLVM/CUDA 与原整图结果在现有数值容差内一致。
- ResNet18 实际执行通过，不只检查 import 或 IR dump。

### 架构

- runtime 公共头 0 个 Relay/TE/TIR include。
- codegen 头仍全部私有。
- distributed 可以依赖 runtime executable plan，反向依赖为 0。
- `include/kxc/**` 公共头独立编译和 include layer check 通过。

### 性能

- 记录 `fusion_level=0`、默认融合和旧整图基线的编译时间、kernel 数、launch 时间、峰值中间内存和端到端耗时。
- 默认融合不得比严格按算子模式拥有更多 kernel。
- 后端批量编译后，一个 target/module 只创建一个 LLVM JIT resource 或 CUDA CUmodule。
- 不以“kernel 数减少”代替数值正确性和 sanitizer 门禁。

### 验证矩阵

```powershell
cmake --build out/build/dev-mingw-cpu --target `
  object_test packed_func_test registry_test type_registration_test `
  pass_pipeline_test infer_type_test executable_plan_test graph_partition_test `
  operator_group_compilation_test compiler_contract_test kernel_signature_test `
  compiled_module_test runtime_session_test check_include_layers check_public_headers -j 4
```

```powershell
cmake --build out/build/windows-llvm-msys2 --target `
  codegen_llvm_test op_numeric_llvm_test onnx_importer_test `
  operator_group_compilation_test runtime_session_test -j 4
```

Omen CUDA 继续使用仓库 `AGENTS.md` 规定的 CUDA 12.9/CUPTI flags，并运行：

```bash
cmake --build out/build/omen-cuda --target \
  codegen_cuda_test runtime_session_test cupti_smoke_test \
  operator_group_compilation_test check_relay_op_contract -j 4
```

涉及 value table、module 共享资源、异步 completion 和 storage reuse 后，必须追加 ASan+UBSan；CUDA 运行 Compute Sanitizer。生成的 profiling/CUPTI 输出不得提交。

## 7. 建议提交序列

1. `test: characterize whole-graph single-kernel lowering`
2. `feat: define runtime executable plan`
3. `feat: partition relay graph into primitive groups`
4. `feat: lower primitive groups to independent tir functions`
5. `refactor: track multiple primitives through compiler stages`
6. `feat: support multi-entry compiled modules`
7. `feat: execute kernel graphs in runtime sessions`
8. `refactor: switch compiler clients to graph executables`
9. `feat: fuse primitive groups by operator pattern`
10. `perf: batch backend compilation and reuse intermediate storage`
11. `feat: execute compiled kernels in distributed plans`

每个提交都应能单独构建并通过当时已启用的测试；不要把 0721 文件架构迁移的未提交改动和本功能实现混为一个不可审阅的大提交。
