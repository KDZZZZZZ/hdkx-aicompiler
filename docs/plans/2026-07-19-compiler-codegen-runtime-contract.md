# Compiler 与 Codegen 运行契约实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 建立由 `Target` 驱动、产物自描述、CPU/CUDA ABI 隔离的 Compiler 与 Codegen 链路，使 `Compiler` 输出携带 `KernelSignature` 和启动元数据的 `CompiledModule`，并只通过 `NDArray + DeviceStream` 启动内核。

**架构：** Relay lowering 同时产出 `PrimFunc` 和稳定常量绑定，随后由签名构建器将参数角色、dtype、shape、device、alignment 固化为 `KernelSignature`。Compiler 根据 `Target` 选择 LLVM 或 CUDA 后端；`CompiledModule` 负责统一校验和生命周期，后端 launcher 只处理各自 ABI。删除现有裸 `void*` 公共调用、伪 Adaptive 编译和未实现的可执行 C backend，不保留兼容重载。

**技术栈：** C++17、项目 Object/ObjectRef 与 `Array`/`Map` 容器、Relay/TE/TIR、LLVM ORC JIT、CUDA NVRTC + Driver API、CMake/Ninja、ASan/UBSan/TSan、Compute Sanitizer。

---

## 1. 范围与完成标准

本计划实现 `docs/DEVICE_MODEL.md` 第 8、9 节，并为第 10 节提供稳定消费接口。完成时必须满足：

1. `Compiler::Compile` 的结果包含目标设备、唯一入口符号、`KernelSignature`、常量绑定、启动元数据和可执行后端资源。
2. 公共 API 中不存在 `std::vector<void*>`、公开 `func_ptr`、`module_ptr` 或字符串 metadata map。
3. CPU 和 CUDA 只共享 `CompiledModule::Launch(Array<NDArray>, DeviceStream)`；参数打包完全位于各自 launcher 内部。
4. `Launch` 在提取任何数据指针前完成参数数量、角色、dtype、shape、device、连续性、alignment 和 stream 校验。
5. CPU LLVM 路径使用 `NDArray` 完成现有数值测试；CUDA 路径至少完成 Relay elementwise add/relu 的真实编译、异步启动和回拷数值测试。
6. CUDA 编译必须经过显式 GPU thread-binding pass；没有合法绑定时明确拒绝，不能生成“单线程串行 GPU kernel”作为静默回退。
7. `Target` 是后端选择的唯一事实来源：`cpu` 选择 LLVM，`cuda` 选择 CUDA；构建未启用对应后端时明确失败。
8. 旧 `kAdaptive`、`CompiledModule::Run(...)`、`KernelRunner` 裸参数路径和伪 shape fuzzy cache 不再参与 Compiler/Codegen。RuntimeSession 的完整强类型重写另立第 10 节计划。
9. LLVM 是长期保留的一等 CPU 后端，不是 CUDA 落地前的临时过渡方案；后续工作不得以 C emitter、解释器或 CUDA backend 替换、弱化 LLVM 验证矩阵。
10. 所有本计划新增或修改的公共结构体、Object/ObjectRef、枚举、函数和关键字段都必须有中文职责注释；资源所有权、ABI 打包、校验顺序、异步保活和异常回收等非直观关键语句必须就地添加中文解释。

## 2. 目标接口

### 2.1 编译配置

删除把执行策略、后端和目标混在一起的旧配置。目标接口为：

```cpp
class CompileConfigNode final : public Object {
public:
    Target target;
    int opt_level{2};
    profiling::ProfileOptions profile_options;
    KXC_OBJECT_DECLARE
};

class CompileConfig : public ObjectRef {
public:
    static CompileConfig Create(Target target, int opt_level = 2);
    void Validate() const;
};
```

约束：

- 删除 `CompileMode`、`backend`、`suggested_input_shapes`、`background_threads`、`enable_hot_swap` 和 `cache_dir`。
- 当前所谓 AOT 实际仍是进程内 ORC JIT，因此不继续保留错误命名；真正可序列化 AOT artifact 后续单独设计。
- `opt_level` 只允许 0 到 3。
- `Target` 必须 defined，`kind/device_type/device_id` 必须一致，并且能力快照必须来自 `BuildTarget(Device)`。

### 2.2 内核签名与启动元数据

新增对象系统类型，公共集合全部使用自定义容器：

```cpp
enum class KernelArgRole {
    kInput,
    kConstant,
    kOutput,
};

class KernelArgSpecNode final : public Object {
public:
    String name;
    KernelArgRole role{KernelArgRole::kInput};
    DLDataType dtype{};
    Array<int64_t> shape;
    Device device;
    uint64_t alignment{1};
    bool mutable_data{false};
    String constant_key;
    KXC_OBJECT_DECLARE
};

class KernelSignatureNode final : public Object {
public:
    String symbol;
    Array<KernelArgSpec> arguments;
    KXC_OBJECT_DECLARE
};

class KernelLaunchMetadataNode final : public Object {
public:
    Device device;
    CodeGenBackend backend{CodeGenBackend::kLLVM};
    Dim3 grid{1, 1, 1};
    Dim3 block{1, 1, 1};
    uint64_t dynamic_shared_memory_bytes{0};
    KXC_OBJECT_DECLARE
};
```

`rank` 只由 `shape.size()` 计算，不在节点中保存重复状态。`KernelSignature::Validate()` 必须验证角色分段、名称唯一、常量 key 唯一、dtype 合法、alignment 为 2 的幂和输出存在；输入允许唯一动态维哨兵 `-1`，常量和输出在尚无 shape function 时必须全部静态。

### 2.3 Lowering 产物

`LowerToTIR` 不再只返回 `PrimFunc`：

```cpp
class ConstantBindingNode final : public Object {
public:
    String key;
    runtime::NDArray value;
    int64_t param_index{-1};
    KXC_OBJECT_DECLARE
};

class LoweredFunctionNode final : public Object {
public:
    tir::PrimFunc prim_func;
    KXC_OBJECT_DECLARE

private:
    Array<ConstantBinding> constants_;
};

LoweredFunction LowerToTIR(Function function);
```

常量 key 使用确定性 Relay 遍历序号 `relay.constant.<N>`。同一个 Relay Function 重复编译必须得到相同 key；不得在 Compiler 中重新扫描 Relay 猜测常量顺序。

由于项目 `Array<T>` 当前复制后共享可变节点，所有已经 Validate 的契约数组都必须私有保存，并通过深拷贝访问器返回。TIR attrs 中的常量 key 使用专用 `ConstantKeyList` 对象，不直接从任意 `ObjectRef` 静态恢复 `Array<String>`，避免模板元素类型擦除造成未定义行为。

### 2.4 编译模块

```cpp
class CompiledModule : public ObjectRef {
public:
    AsyncOperation Launch(const Array<runtime::NDArray>& ordered_arguments,
                          const DeviceStream& stream) const;

    KernelSignature signature() const;
    KernelLaunchMetadata launch_metadata() const;
    Map<String, runtime::NDArray> constants() const;
    Target target() const;
    bool IsReady() const;
    String GetStatus() const;
};
```

`CompiledKernel` 降为 codegen 内部对象。其节点保存强类型 launcher、签名和后端资源，不再公开 `func_ptr`、`jit_resource`、`CUmodule` 或 `CUfunction`。

## 3. 实施任务

### Task 1: 修复测试门禁并锁定现有 lowering 行为

**文件：**
- 修改：`test/codegen_llvm_test.cpp`
- 新建：`test/compiler_contract_test.cpp`
- 修改：`CMakeLists.txt`
- 修改：`.github/workflows/ci.yml`

**步骤：**

1. 将 `codegen_llvm_test` 的打印式检查改为统一 `TEST_CHECK`/异常汇总，任一失败时 `main` 返回非零。
2. 删除该测试中的 Adaptive 用例；它验证的是固定 Relay 被错误缓存到任意 shape key 的伪 specialization。
3. 在 `compiler_contract_test` 写 characterization test，锁定 lowering 当前的 `input -> constant -> output` 顺序、多输出计数和 `buffer_map` dtype/shape。
4. 在尚未引入新接口前，只锁定当前可观察契约：lowering 计数 attrs、buffer_map、CompileConfig 工厂产生的 target/opt level，以及未实现 backend 的明确拒绝；undefined 和新 Validate API 的负例随 Task 5 一起加入。
5. 为所有新增 executable 添加 `run_*` target，并在 LLVM CI job 中运行 `run_codegen_llvm_test` 和 `run_compiler_contract_test`。
6. 运行并确认本任务提交始终可构建：characterization tests 全部通过；不得提交依赖后续 `KernelSignature` API 的预期编译失败测试。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target compiler_contract_test
```

预期：在新增强类型契约前编译失败，失败点必须是缺少 `KernelSignature`/目标 API，而不是测试基础设施错误。

**提交：**

```bash
git add test/codegen_llvm_test.cpp test/compiler_contract_test.cpp CMakeLists.txt .github/workflows/ci.yml
git commit -m "test: lock compiler and codegen contracts"
```

### Task 2: 建立 KernelSignature 对象模型

**文件：**
- 新建：`include/codegen/kernel_signature.h`
- 新建：`include/codegen/backend.h`
- 新建：`src/codegen/kernel_signature.cc`
- 新建：`test/kernel_signature_test.cpp`
- 修改：`include/codegen/codegen.h`
- 修改：`CMakeLists.txt`

**步骤：**

1. 先写失败测试，覆盖三个角色、参数顺序、dtype code/bits/lanes、静态/动态 shape、计算 rank、device、alignment、mutable_data 和 constant_key。
2. 写非法签名测试：重复名称、重复 constant key、常量无 key、非定值参数带 key、角色顺序回退、小于 -1 的 shape、非 2 的幂 alignment、无输出和动态输出无 shape function。
3. 实现 `KernelArgSpecNode/ObjectRef`、`KernelSignatureNode/ObjectRef`、`Dim3` 和 `KernelLaunchMetadataNode/ObjectRef`，每个结构体和函数添加中文职责注释，关键校验分支解释失败原因。
4. 实现确定性的 `ToString()`，供日志、测试和将来 artifact manifest 使用；此阶段不引入 JSON schema。
5. 将测试接入 CPU-only 构建，确保它不依赖 LLVM/CUDA。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target run_kernel_signature_test
```

预期：全部正例、负例通过。

**提交：**

```bash
git add include/codegen/kernel_signature.h src/codegen/kernel_signature.cc test/kernel_signature_test.cpp CMakeLists.txt
git commit -m "feat: define kernel signature and launch metadata"
```

### Task 3: 让 Relay lowering 保留常量和参数语义

**文件：**
- 修改：`include/relay/transforms/lower.h`
- 修改：`include/relay/transforms/multi_device.h`
- 修改：`src/relay/backend/lower.cc`
- 修改：`src/relay/transforms/multi_device.cc`
- 修改：`test/infer_type_test.cpp`
- 修改：`test/compiler_contract_test.cpp`
- 修改：`test/onnx_importer_test.cpp`
- 修改：`test/profile_bundle_test.cpp`
- 修改：`test/resnet18_ir_dump.cpp`
- 修改：`test/tmp_conv_lower.cpp`

**步骤：**

1. 将所有 `LowerToTIR` 调用点迁移为读取 `LoweredFunction::prim_func`，不保留旧返回类型重载。
2. 在 `RelayToTEConverter::VisitConstant` 同时记录原始 `NDArray` 和确定性 key，不再只生成无法回溯 payload 的 `const_N` tensor。
3. lowering 结束时返回 `Array<ConstantBinding>`；保证 binding 顺序与常量参数顺序一一对应。
4. 将角色计数、`output_param_start` 和每个 constant key 写入可结构化读取的 TIR attrs；不得依赖变量名解析角色。
5. 添加多常量、重复常量值、多输出测试，证明 key 唯一且重复编译结果稳定。
6. 添加计数/attrs/buffer_map 不一致的构造用例，为下一任务的 signature builder 提供负例。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target run_infer_type_test run_compiler_contract_test
```

预期：lowering 和契约测试全部通过。

**提交：**

```bash
git add include/relay/transforms/lower.h src/relay/backend/lower.cc src/relay/transforms/multi_device.cc test
git commit -m "refactor: preserve constants in relay lowering"
```

### Task 4: 从 PrimFunc 构建并验证 KernelSignature

**文件：**
- 修改：`include/codegen/kernel_signature.h`
- 修改：`src/codegen/kernel_signature.cc`
- 修改：`test/kernel_signature_test.cpp`
- 修改：`test/compiler_contract_test.cpp`

**步骤：**

1. 新增 `BuildKernelSignature(PrimFunc, Target, symbol)` 的失败测试。
2. 按 `PrimFunc::params` 原顺序读取 `buffer_map`，按 lowering attrs 分配 input/constant/output 角色。
3. 将 TIR `DataType` 严格映射为 `DLDataType`，保留 lanes；静态 shape 只接受可求值非负整数。
4. alignment 优先取 `Buffer::data_alignment`；为 0 时使用后端最低保证值，CPU 为元素对齐、CUDA 为至少 256 字节分配对齐但签名只声明实际需要值。
5. constant 参数从 lowering metadata 读取 key；缺失、数量不一致或 key 重复直接失败。
6. 检测动态 shape；输入动态 shape 可以在后续 specialization 中扩展，本阶段 Compiler 统一拒绝动态输出。
7. 测试多输入、多常量、多输出、bool/int/float、零尺寸和非法 attrs。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target run_kernel_signature_test run_compiler_contract_test
```

预期：Signature 与 PrimFunc 参数逐项一致，所有畸形 metadata 在 codegen 前失败。

**提交：**

```bash
git add include/codegen/kernel_signature.h src/codegen/kernel_signature.cc test/kernel_signature_test.cpp test/compiler_contract_test.cpp
git commit -m "feat: derive kernel signatures from lowered tir"
```

### Task 5: 收敛 Codegen 抽象并删除虚假的可执行 C backend

**文件：**
- 修改：`include/codegen/codegen.h`
- 修改：`include/codegen/codegen_c.h`
- 修改：`src/codegen/codegen_c.cc`
- 修改：`include/api/compile_config.h`
- 修改：`src/api/compile_config.cc`
- 修改：`test/codegen_llvm_test.cpp`
- 修改：`test/compiler_contract_test.cpp`

**步骤：**

1. 删除未被实际管线使用的 `CodeGenBase`、含 `void* module_ptr` 的 `CodeGenResult` 和字符串 metadata map。
2. `CodeGenBackend` 只保留真正可执行的 `kLLVM`、`kCUDA`；C emitter 改名为 `CSourceEmitter`，明确仅用于诊断源码，不是 Compiler backend。
3. 将 `CompileConfig` 改为 `Create(Target, opt_level)`，删除 `CompileMode` 和独立 backend 字段，不提供 AOT/JIT/Adaptive 兼容工厂。
4. 实现 `CompileConfig::Validate()`，覆盖 undefined、opt level、target kind/device type/device id/capability 一致性。
5. Compiler backend dispatch 以后只读取 `Target`：CPU 要求 LLVM build，CUDA 要求 CUDA build，其他 target 明确 unsupported。
6. 修改 C emitter 测试，只断言 source emission 和编译诊断，不再宣称 C backend 可执行。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target run_compiler_contract_test
```

预期：LLVM/CUDA 未启用的 target 产生精确错误；不存在 C backend 选择路径。

**提交：**

```bash
git add include/codegen include/api src/codegen src/api test/codegen_llvm_test.cpp test/compiler_contract_test.cpp
git commit -m "refactor: make target the codegen backend authority"
```

### Task 6: 实现强类型 CompiledModule 与统一启动校验

**文件：**
- 修改：`include/api/compiler.h`
- 修改：`src/api/compiler.cc`
- 修改：`include/codegen/compiled_kernel.h`
- 修改：`src/codegen/compiled_kernel.cc`
- 新建：`test/compiled_module_test.cpp`
- 修改：`CMakeLists.txt`

**步骤：**

1. 先写 `compiled_module_test`，使用假的记录型 launcher，证明合法参数按签名顺序到达 backend。
2. 将 `CompiledModule` 转为 `ObjectRef`，使缓存、异步 completion 和热替换都持有稳定对象生命周期。
3. 删除两个 `Run(std::vector<void*>)`、公开构造函数和 `CompiledKernel::operator()`；不保留 deprecated wrapper。
4. 在 `CompiledModule::Launch` 中按顺序校验：module ready、stream defined、stream/device 一致、参数数量、NDArray defined、角色约束、dtype、rank、shape、contiguous、byte_offset、alignment。
5. 只有全部校验通过后，才构造内部 `Array<Storage>` 并调用 launcher；错误信息包含 symbol、参数序号、参数名、expected 和 actual。
6. CPU default stream 可以显式传 `DeviceStream::Default(cpu)`；禁止 undefined stream 隐式选择设备。
7. `constants()` 返回 lowering 保存的稳定 key 到 NDArray 映射；Launch 本身不自动插入常量或分配输出，这些是 RuntimeSession 第 10 节职责。
8. 为零尺寸、view byte_offset、错误 dtype lanes、错误 device、错误 stream、非连续 view 和不足 alignment 写负例。

**验证：**

```powershell
cmake --build out/build/dev-ninja-cpu --target run_compiled_module_test
```

预期：合法调用一次到达假 launcher；所有负例在 launcher 调用计数仍为 0 时失败。

**提交：**

```bash
git add include/api/compiler.h src/api/compiler.cc include/codegen/compiled_kernel.h src/codegen/compiled_kernel.cc test/compiled_module_test.cpp CMakeLists.txt
git commit -m "feat: launch compiled modules with typed ndarrays"
```

### Task 7: 迁移 CPU LLVM ABI 和 JIT 生命周期

**文件：**
- 修改：`include/codegen/codegen_llvm.h`
- 修改：`src/codegen/codegen_llvm.cc`
- 修改：`include/codegen/llvm_jit.h`
- 修改：`src/codegen/llvm_jit.cc`
- 修改：`include/codegen/compiled_kernel.h`
- 修改：`src/codegen/compiled_kernel.cc`
- 修改：`test/codegen_llvm_test.cpp`
- 修改：`test/op_numeric_llvm_test.cpp`
- 修改：`test/onnx_importer_test.cpp`

**步骤：**

1. 保留 CPU 内部专用 call-frame ABI，例如 `int32_t(void** data, size_t count)`；它只能存在于 LLVM launcher 内，不进入 api/runtime/Python 头文件。
2. LLVM wrapper 在入口先比较 count 与 Signature 参数数，再把 slot 映射为 TIR buffer；内部 ABI 不与 CUDA `kernelParams` 共用。
3. `LLVMJITEngine::Compile` 接收 `KernelSignature` 和 `Target`，返回持有 ORC JIT 资源的内部 launcher；删除 `void* jit_resource` 手工类型擦除。
4. launcher 同步调用 CPU 函数，检查返回码，并返回 `AsyncOperation::Completed(stream, retained_storage)`。
5. 将 LLVM、数值和 ONNX 测试全部从主机数组迁为 CPU `NDArray`；输出由测试显式分配并按 Signature 顺序组合。
6. 增加常量、多输出、中间 Allocate、bool/int/float、零元素和带 byte_offset 的连续 view 测试。
7. 增加错误 symbol、LLVM verifier 失败、opt level 边界和 module/JIT 析构测试。

**验证：**

```bash
cmake -S . -B out/build/compiler-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=ON \
  -DKXC_BUILD_PASS_TESTS=ON \
  -DKXC_BUILD_CODEGEN_TESTS=ON
cmake --build out/build/compiler-llvm --target \
  run_compiled_module_test \
  run_codegen_llvm_test \
  run_op_numeric_llvm_test \
  run_onnx_importer_test -j 4
```

预期：四个 target 全部以 0 退出，数值结果与迁移前一致。

**提交：**

```bash
git add include/codegen src/codegen test
git commit -m "feat: execute llvm kernels through ndarray launch"
```

### Task 8: 重构 Compiler 为显式阶段管线

**文件：**
- 修改：`include/api/compiler.h`
- 修改：`src/api/compiler.cc`
- 新建：`include/api/compile_result.h`
- 新建：`src/api/compile_result.cc`
- 修改：`test/compiler_contract_test.cpp`
- 修改：`test/profile_bundle_test.cpp`

**步骤：**

1. 将 Compiler 内部拆为 `Validate -> OptimizeRelay -> Lower -> OptimizeTIR -> BuildSignature -> BuildBackend -> AssembleModule`，每阶段返回强类型结果。
2. pass 集合由 `opt_level` 决定，不再由已删除的 AOT/JIT mode 选择硬编码字符串；测试 0/1/2/3 的确定性顺序。
3. 用 `CompileConfig::target` 建立或合并 `PassContext`；IR placement 与 config target 冲突时失败，不得静默覆盖。
4. 入口符号由 lowered `global_symbol` 规范化生成；同进程多次编译使用 module-local symbol/resource，不再依赖一个全局固定 `kxc_kernel_main`。
5. `CompileResult` 保存优化后 Relay、PrimFunc、Signature、常量绑定、诊断 artifact 和 backend executable，最后组装为 `CompiledModule`。
6. profiling 为每阶段记录 target、device、symbol、IR hash、opt level 和 backend；失败 span 必须携带具体阶段。
7. 删除 Compiler 对 `RuntimeSession` 的构造和 include；本阶段 Compiler 永远返回真实编译产物。

**验证：**

```bash
cmake --build out/build/compiler-llvm --target run_compiler_contract_test run_profile_bundle_test -j 4
```

预期：阶段顺序、target 传播、symbol 和 profiling 字段均由测试证明。

**提交：**

```bash
git add include/api src/api test/compiler_contract_test.cpp test/profile_bundle_test.cpp
git commit -m "refactor: split compiler into typed stages"
```

### Task 9: 增加 TIR CUDA thread-binding 语义和调度 pass

**文件：**
- 修改：`include/tir/stmt.h`
- 修改：`src/tir/stmt.cc`
- 新建：`include/tir/transforms/bind_cuda_threads.h`
- 新建：`src/tir/transforms/bind_cuda_threads.cc`
- 修改：`include/tir/transforms/pipeline.h`
- 修改：`src/tir/transforms/pipeline.cc`
- 修改：`src/tir/ir_printer.cc`
- 新建：`test/cuda_schedule_test.cpp`
- 修改：`CMakeLists.txt`

**步骤：**

1. 先定义可打印、可访问、可重写的 `ThreadBinding` TIR 节点，显式表示 `blockIdx.{x,y,z}` 和 `threadIdx.{x,y,z}`；不要用裸字符串 AttrStmt 让后端猜语义。
2. `BindCudaThreads` 第一阶段只接受可证明无跨迭代写冲突的外层数据并行 loop；将其映射为一维 grid/block，block size 依据 Target capability 选择并上限 256。
3. 对 reduction、共享内存、动态 extent 或无法证明独立的 loop 明确标记 unsupported；不得错误并行化。
4. 计算并写入 `KernelLaunchMetadata` 所需 work size、grid、block 和 shared memory。
5. 用 Target 的 max threads/block、grid limits 和 shared memory 限制做编译期校验。
6. 添加 elementwise add/relu 正例，以及 reduction、超限 block、缺 target capability、错误 thread tag 负例。
7. 更新所有 TIR visitor/printer/pass，确保新节点不会被旧 pass 丢失。

**验证：**

```bash
cmake --build out/build/omen-cuda --target run_cuda_schedule_test run_pass_pipeline_test -j 4
```

预期：合法 elementwise TIR 含确定 thread binding；不支持模式在 codegen 前给出明确错误。

**提交：**

```bash
git add include/tir src/tir test/cuda_schedule_test.cpp CMakeLists.txt
git commit -m "feat: bind tir loops to cuda threads"
```

### Task 10: 实现 CUDA C/NVRTC codegen 与 module 生命周期

**文件：**
- 新建：`include/codegen/codegen_cuda.h`
- 新建：`src/codegen/codegen_cuda.cc`
- 新建：`include/codegen/cuda_module.h`
- 新建：`src/codegen/cuda_module.cc`
- 修改：`include/codegen/compiled_kernel.h`
- 修改：`src/codegen/compiled_kernel.cc`
- 新建：`test/codegen_cuda_test.cpp`
- 修改：`CMakeLists.txt`

**步骤：**

1. 从已绑定 TIR 生成 `extern \"C\" __global__` CUDA C；buffer 参数使用正确 dtype 指针，thread binding 映射为 CUDA builtin。
2. 仅实现 Task 9 已声明支持的表达式/语句集合；遇到未知节点直接报错，不输出注释占位代码。
3. 使用 NVRTC 编译 PTX，架构从 Target compute capability 生成，例如 GTX 1650 为 `compute_75`；完整编译日志进入异常和 profiling artifact。
4. 使用 CUDA Driver API `cuModuleLoadDataEx` 和 `cuModuleGetFunction` 创建强类型 RAII `CudaModuleLauncher`。
5. 析构时在正确 device context 下卸载 module；module 必须比所有 pending `AsyncOperation` 活得更久。
6. CUDA-off 构建不编译 CUDA 源文件；请求 cuda target 时返回明确的 build-feature 错误。
7. CMake 在 CUDA backend target 上链接 `CUDA::nvrtc` 和 `CUDA::cuda_driver`，不把依赖泄漏到 CPU-only runtime。
8. 测试 CUDA source、NVRTC 错误日志、符号 lookup、module unload 和不支持 TIR 拒绝。

**验证：**

```bash
cmake --build out/build/omen-cuda --target codegen_cuda_test -j 4
cd /home/oops/repo/hdkx-aicompiler/test
../out/build/omen-cuda/codegen_cuda_test
```

预期：GTX 1650 上生成 PTX、加载 module 并解析入口符号，所有负例返回精确错误。

**提交：**

```bash
git add include/codegen src/codegen test/codegen_cuda_test.cpp CMakeLists.txt
git commit -m "feat: compile bound tir with nvrtc"
```

### Task 11: 实现 CUDA 异步 Launch 和 ABI 保活

**文件：**
- 修改：`src/codegen/cuda_module.cc`
- 修改：`src/codegen/compiled_kernel.cc`
- 修改：`include/base/device_stream.h`
- 修改：`src/base/device_stream.cc`
- 修改：`test/codegen_cuda_test.cpp`
- 修改：`test/device_runtime_test.cpp`

**步骤：**

1. 抽取 `StorageCopyAsync` 已有的 hardened submit 模板：先建立 event/保活对象，再 enqueue；enqueue 失败时同步回收或永久保活，不能留下 use-after-free。
2. launcher 为每个 NDArray 建主机 `void* pointer_values[i]`，再令 `kernel_params[i] = &pointer_values[i]`；禁止直接把 device pointer 数组传给 `cuLaunchKernel`。
3. 从 `DeviceStream::backend_handle` 获取 CUDA stream，设置目标 device，使用 metadata 的 grid/block/shared memory 启动。
4. launch 成功后记录 CUDA event，返回 `AsyncOperation::Pending(stream, event, retained_storage, compiled_kernel)`。
5. event 完成前必须同时保活输入、输出、常量 Storage，CompiledKernel、CUmodule 和参数宿主存储；参数宿主存储若只在 `cuLaunchKernel` 调用期间需要，代码注释必须说明 Driver 已同步复制参数值的依据。
6. 增加 stream mismatch、错误设备指针、launch limit、丢弃 NDArray/module 外部引用后仍正确完成的测试。
7. 增加 Relay add/relu -> Compiler -> CUDA module -> Launch -> async D2H -> 数值校验的端到端用例。

**验证：**

```bash
cd /home/oops/repo/hdkx-aicompiler
/usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  out/build/omen-cuda/codegen_cuda_test
```

预期：数值测试通过，Compute Sanitizer 报告 0 error，异步析构测试无非法访问。

**提交：**

```bash
git add include/base/device_stream.h src/base/device_stream.cc src/codegen test/codegen_cuda_test.cpp test/device_runtime_test.cpp
git commit -m "feat: launch cuda kernels asynchronously"
```

### Task 12: 删除旧 RuntimeSession 编译耦合并完成全仓迁移

**文件：**
- 修改：`include/runtime/runtime_session.h`
- 修改：`src/runtime/runtime_session.cc`
- 删除：`include/runtime/kernel_runner.h`
- 删除：`src/runtime/kernel_runner.cc`
- 修改或删除：`include/runtime/kernel_cache.h`
- 修改或删除：`src/runtime/kernel_cache.cc`
- 修改或删除：`include/runtime/background_compiler.h`
- 修改或删除：`src/runtime/background_compiler.cc`
- 修改或删除：`include/runtime/shape_predictor.h`
- 修改或删除：`src/runtime/shape_predictor.cc`
- 修改：所有调用 `CompiledModule::Run` 或旧 `CompileConfig` 工厂的 `test/`、`examples/` 和 Python binding 文件
- 修改：`CMakeLists.txt`

**步骤：**

1. 全仓搜索旧接口，迁移到 `CompileConfig::Create` 和 `CompiledModule::Launch`。
2. 删除 `Compiler` 内 Adaptive 分支和旧裸参数 `RuntimeSession::Run`；不添加临时 adapter。
3. 如果 RuntimeSession 尚无第 10 节强类型实现，则将其从公共 Compiler 产物中移除并停止构建旧 cache/background 模块，避免保留不可用半迁移路径。
4. 删除 `KernelRunner`；其唯一职责是转发裸参数并热换 module，新的 ObjectRef module 和未来 session cache 可直接持有稳定句柄。
5. 删除 fuzzy shape 命中。现有编译根本未消费 shape，较大 shape kernel 运行较小输入没有正确性依据。
6. 若仍需为下一阶段保留 cache 类型，只允许 `KernelCache` 存取 `CompiledModule` 值句柄或 `shared_ptr<const CompiledModule>` 快照，不能返回 `unordered_map` 元素裸指针。
7. 更新 `docs/DEVICE_MODEL.md` 第 8、9 节接口为最终代码形式，并明确第 10 节尚待实现的边界。
8. 运行 `rg` 证明公共层无旧 ABI。

**验证：**

```bash
rg -n "vector<void\*>|func_ptr|module_ptr|CompiledModule::Run|CompileConfig::(AOT|JIT|Adaptive)" include src test
```

预期：公共 Compiler/Codegen/RuntimeSession 路径 0 命中；后端内部 CPU call frame 若仍使用 `void**`，只能命中私有 `.cc` 实现和对应 ABI 注释。

**提交：**

```bash
git add -u include/runtime src/runtime include/api src/api test examples CMakeLists.txt docs/DEVICE_MODEL.md
git add include/runtime src/runtime include/api src/api test examples CMakeLists.txt docs/DEVICE_MODEL.md
git commit -m "refactor: remove legacy compiler runtime paths"
```

### Task 13: 完整验证、sanitizer 和 CI 收口

**文件：**
- 修改：`.github/workflows/ci.yml`
- 修改：`CMakeLists.txt`
- 仅在测试发现问题时修改对应实现文件

**步骤：**

1. CPU-only 配置验证 object、container、device、NDArray、signature、Compiler 失败契约和 Relay pass。
2. LLVM 配置验证 Compiler、LLVM codegen、完整算子数值和 ONNX importer。
3. omen CUDA 配置验证真实设备编译、stream、异步 launch、CUPTI 和 Compute Sanitizer。
4. 对 ObjectRef、launcher、JIT/module 和 AsyncOperation 生命周期变更运行 ASan/UBSan。
5. 对保留的 module cache 或异步热替换运行 TSan；若本计划已删除旧 cache，则记录为第 10 节 RuntimeSession 的强制门禁。
6. 执行 Relay op contract，预期 19 checked、19 passed、0 failed。
7. 执行 `git diff --check`、本地与 omen `git status --short`，只删除本任务生成的 profile/CUPTI 输出，保留所有既有脏文件。
8. 将全部新 `run_*` target 加入 CI，避免只构建不运行。

**Windows/CPU 验证：**

```powershell
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu
cmake --build out/build/dev-ninja-cpu --target `
  run_object_test `
  run_device_runtime_test `
  run_kernel_signature_test `
  run_compiled_module_test `
  run_compiler_contract_test `
  run_infer_type_test `
  run_pass_pipeline_test `
  check_relay_op_contract
```

**LLVM 验证：**

```bash
cmake --build out/build/compiler-llvm --target \
  run_kernel_signature_test \
  run_compiled_module_test \
  run_compiler_contract_test \
  run_codegen_llvm_test \
  run_op_numeric_llvm_test \
  run_onnx_importer_test \
  check_relay_op_contract -j 4
```

**omen CUDA 验证：**

```bash
cmake --build out/build/omen-cuda --target \
  object_test \
  device_runtime_test \
  kernel_signature_test \
  compiled_module_test \
  compiler_contract_test \
  cuda_schedule_test \
  codegen_cuda_test \
  cupti_smoke_test \
  check_relay_op_contract -j 4

cd /home/oops/repo/hdkx-aicompiler/test
../out/build/omen-cuda/object_test
../out/build/omen-cuda/device_runtime_test
../out/build/omen-cuda/kernel_signature_test
../out/build/omen-cuda/compiled_module_test
../out/build/omen-cuda/compiler_contract_test
../out/build/omen-cuda/cuda_schedule_test
../out/build/omen-cuda/codegen_cuda_test
../out/build/omen-cuda/cupti_smoke_test
```

**生命周期 sanitizer：**

```bash
cmake -S . -B out/build/omen-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_CODEGEN_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build out/build/omen-asan --target compiled_module_test -j 4

ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  out/build/omen-asan/compiled_module_test
```

预期：所有测试 0 退出；ASan/UBSan/Compute Sanitizer 无错误；无新增 warning；Relay contract 19/19。

**提交：**

```bash
git add CMakeLists.txt .github/workflows/ci.yml
git commit -m "ci: enforce compiler and codegen verification"
```

## 4. 分阶段交付顺序

为降低一次性迁移风险，PR 内按以下顺序保持每个提交可审查：

1. 测试严格化与 characterization。
2. KernelSignature 和 lowering 常量契约。
3. CompileConfig/Codegen 抽象收敛。
4. CompiledModule 强类型 Launch。
5. CPU LLVM 全量迁移。
6. Compiler 阶段化和 Target 驱动。
7. TIR CUDA thread binding。
8. CUDA NVRTC/Driver module。
9. CUDA async Launch。
10. 删除旧 RuntimeSession/裸 ABI。
11. 全矩阵验证与 CI 门禁。

CPU LLVM 完成点和 CUDA 完成点应分别形成可运行提交；不得在中间提交保留“cuda target 静默走 LLVM”或“新旧 Run 同时存在”的状态。

## 5. 明确留到后续的内容

以下属于 `DEVICE_MODEL.md` 第 10 节及之后，不在本计划中实现：

- `RuntimeSession::Run/RunAsync(Array<NDArray>)` 的输出自动分配。
- session 常量表与 `constant_key` 自动绑定。
- 动态输出 shape function 和动态 shape specialization。
- 合法的 exact cache key、后台编译、失败传播和热替换。
- ExecutionPlan 的 module registry 与真实 `ExecuteKernel`。
- Disco 多 worker 的 kernel 分发和通信重叠。
- 可序列化 AOT artifact、磁盘 cache 和跨进程加载。

这些模块只能消费本计划产生的 `KernelSignature`、`CompiledModule` 和 `AsyncOperation`，不得重新暴露裸指针 ABI。
