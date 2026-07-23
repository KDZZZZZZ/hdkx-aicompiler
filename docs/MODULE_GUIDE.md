# hdkx-aicompiler 模块介绍（历史文档）

> **状态：已归档。** 本文包含已删除的 Adaptive Runtime、fuzzy
> `KernelCache`、旧目录与单 PrimFunc 主链，只用于历史取证，不能作为当前
> API/capability 指南。当前事实入口为
> [`COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md`](COMPILER_FOUNDATION_ARCHITECTURE_REVIEW.md)、
> [`COMPILER_EXTENSION_CONTRACT.md`](COMPILER_EXTENSION_CONTRACT.md) 和当前源码/测试。

本文档从宏观视角介绍历史仓库模块。

## 总体架构

```
ONNX 模型 / 手写构图
       │
       ▼
┌─────────────────┐     Frontend (onnx_importer)
│   ONNX 导入器    │
│  frontend/       │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     Relay IR
│  Relay 表达式     │     relay::Function / Call / Var / Constant
│  + Pass 流水线    │     → 常规优化 + 内存标注 + layout 调整
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     LowerToTIR
│  TE Tensor 计算   │     te::ComputeOp / PlaceholderOp / Reduce
│  te/ + topi/     │     → 将 Relay 算子展开为 Tensor 计算 DAG
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     TIR
│  TIR PrimFunc    │     tir::For / Store / Load / Allocate
│  + Pass 流水线    │     → 循环变换 + 向量化 + 常量折叠
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     Codegen
│  Codegen         │     CodeGenLLVM → LLVM IR → JIT
│  (LLVM / C)      │     CodeGenC → C 源码（调试用）
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     Runtime
│  RuntimeSession  │     自适应 shape 缓存 + 后台编译 + kernel 执行
│  KernelCache     │
└─────────────────┘
```

底层支撑：

- **base/**：Object 对象系统、NDArray、DeviceAPI、Registry、PackedFunc、Profiling
- **api/**：Compiler::Compile 主入口、CompileConfig
- **disco/**：多 worker 执行计划 + CCL 通信抽象

---

## 1. base — 基础对象系统

**职责**：提供所有 IR 节点和运行时对象的基石。包括 Object/ObjectRef 引用计数体系、全局类型注册表、PackedFunc 调用桥、NDArray 张量数据结构、Device/DeviceAPI 抽象、Profiling 埋点框架和 Pass 基类。

**头文件入口**：`include/base/`，共 16 个头文件。

| 子模块 | 核心文件 | 职责 |
|--------|---------|------|
| 对象系统 | `object.h`, `container.h`, `expr.h` | Object（引用计数 + 运行时类型 ID）+ ObjectRef 智能 handle；Array/Map/String 容器；ExprNode 基类 |
| 类型系统 | `typemanager.h`, `tensor.h` | 类型元数据管理；TensorType 描述（shape/dtype） |
| PackedFunc | `packedfunc.h`, `registry.h` | `KXC_REGISTER_GLOBAL("name")` 注册全局函数，跨模块调用桥 |
| 设备抽象 | `device.h`, `device_api.h`, `target.h`, `virtual_device.h` | Device/ObjectRef；DeviceAPI（分配、清零、复制、stream/event）；Target 描述编译目标；VirtualDevice（设备 + target + memory scope） |
| NDArray | `ndarray.h`, `storage.h`, `device_stream.h` | Storage 管理物理内存；NDArray 表示 DLTensor 视图；AsyncOperation 保活异步资源 |
| Pass 框架 | `pass.h` | TIRExprFunctor / TIRStmtFunctor / TIRPass / RelayPass 基类 |
| Profiling | `profiling.h` | ProfileContext / ScopedSpan / ActivationScope；生成 manifest/events/trace 文件 |
| Arena | `arena.h` | thread_local arena 分配器，Object 可选使用 |
| Disco 基础 | `disco_placement.h`, `execution_plan.h`；`disco/session.h`, `disco/worker.h`, `disco/executor.h`, `disco/dref.h`, `disco/ccl_backend.h` | WorkerPlacement / DiscoPlacement；ExecutionPlan（Kernel/Comm/Barrier 节点）；DiscoSession / Worker / Executor / DRef / CCLBackend 抽象 |
| Python 桥 | `py_runtime.h` | Python ↔ C++ 运行时互操作接口 |

**实现路径**：`src/base/`，以及 `src/base/device/`（CPU/CUDA DeviceAPI）、`src/base/disco/`（Disco 实现）。

**依赖关系**：base 模块**不依赖** relay/te/tir/codegen/runtime，是编译链中的最底层。

---

## 2. relay — Relay IR 与算子系统

**职责**：定义 Relay 中间表示的 IR 节点、算子注册框架、算子属性类型、pass 流水线和类型推断。Relay 是编译器前端优化层，对标 TVM Relay。

**头文件入口**：`include/relay/`

| 子模块 | 核心文件 | 职责 |
|--------|---------|------|
| IR 定义 | `relay.h` | RelayNode / VarNode / ConstantNode / CallNode / FunctionNode / TupleNode / TupleGetItemNode / IfNode / LetNode / TensorTypeNode |
| 算子注册 | `op.h`, `op_attr_types.h`, `op_macros.h` | OpNode（name/description/arguments/num_inputs/attrs）；`KXC_REGISTER_OP(name)` 宏；属性类型：Conv2DAttrs / DenseAttrs / Pool2DAttrs / SoftmaxAttrs / ReshapeAttrs / TransposeAttrs / DeviceCopyAttrs / CollectiveAttrs |

**实现路径**：`src/relay/`

- `src/relay/relay.cc` — IR 节点实现
- `src/relay/op_attrs.cc`, `src/relay/op_macros.cc` — 属性工厂和注册框架
- `src/relay/op/**` — 各算子具体实现（math.cc / nn/convolution.cc / nn/pooling.cc 等）
- `src/relay/transforms/` — pass 实现（fold_constant、simplify_expr、annotate_memory_scope 等）
- `src/relay/pass/` — pass 注册与流水线编排
- `src/relay/backend/` — LowerToTIR 实现
- `src/relay/type_infer.cc` — 类型推断

**典型扩展**：新增算子 → `src/relay/op/` 下新建文件 → `KXC_REGISTER_OP` 注册 → 设置 `FRelayToTE` 属性 → 加入 CMakeLists.txt。

---

## 3. te / topi — Tensor Expression 计算层

**职责**：提供 Tensor 计算原语和常见算子的 TE 表达。位于 Relay 和 TIR 之间，作为 lowering 的桥梁层。

**头文件入口**：`include/te/`

| 子模块 | 核心文件 | 职责 |
|--------|---------|------|
| TE 核心 | `te.h` | te::Tensor / Operation / PlaceholderOp / ComputeOp / ProducerLoad / Reduce / Schedule / Stage |
| TOPI | `topi.h`, `topi/broadcast.h`, `topi/elemwise.h`, `topi/nn.h`, `topi/reduction.h`, `topi/transform.h` | 常见算子的 TE compute 实现：broadcast、elemwise、conv/pool/dense/softmax、reduce、reshape/transpose/concat |

**实现路径**：
- `src/te/te.cc` — TE 核心对象实现
- `include/te/topi/*.h` — TOPI 实现（header-only）

**依赖关系**：依赖 base（Object/ObjectRef）、tir（DataType）。不依赖 relay。

**当前状态**：Schedule API 有雏形但 lowering 主要生成朴素 loop nest；Schedule（split/fuse/vectorize/unroll）可供扩展。

---

## 4. tir — TIR 中间表示

**职责**：底层标量 IR，包含 PrimExpr（标量表达式）和 Stmt（语句），以及 TIR 级别的 pass 流水线。对标 TVM TIR。

**头文件入口**：`include/tir/`

| 子模块 | 核心文件 | 职责 |
|--------|---------|------|
| 表达式 | `expr.h` | PrimExpr：IntImm / FloatImm / Var / Add / Sub / Mul / Div / Mod / Min / Max / EQ / LT / And / Or / Not / Load / Call / Select / DataType |
| 语句 | `stmt.h` | Stmt：LetStmt / Store / For / IfThenElse / Allocate / AttrStmt / Block / SeqStmt / Evaluate；PrimFunc（params/body/buffer_map/attrs） |
| 工具 | `pass_utils.h` | no-op 检测、SeqStmt 合并、常量提取等公用逻辑 |
| Pass | `pass/`, `transforms/` | TIRPass 基类；具体 pass：fold_constant / simplify_expr / unroll_loop / vectorize_loop / loop_partition / convert_for_loops_serial / force_narrow_index_to_i32 / remove_no_op |

**实现路径**：`src/tir/`

- `src/tir/expr.cc`, `src/tir/stmt.cc` — IR 节点实现
- `src/tir/pass_utils.cc` — 共用工具
- `src/tir/pass/` — 各 pass 实现 + TIR 打印
- `src/tir/transforms/pipeline.cc` — pass 流水线注册（`kxc.tir.transform.run_pipeline`）

**依赖关系**：依赖 base（Object、PackedFunc、Registry），不依赖 relay/te。

---

## 5. pass — Pass 流水线系统

**职责**：组织 Relay 和 TIR 两个层级的变换流水线，提供 PassContext 传递设备和 placement 上下文。

**核心文件**：

| 层级 | 流水线注册 | 默认 pass 顺序 |
|------|-----------|---------------|
| Relay | `src/relay/transforms/pipeline.cc`（`kxc.relay.transform.run_pipeline`） | fold_tuple_get_item → fold_constant → simplify_expr → canonicalize_cast → remove_standalone_reshapes → eliminate_dead_let → annotate_memory_scope → capture_post_dfs_index_in_spans → infer_type |
| TIR | `src/tir/transforms/pipeline.cc`（`kxc.tir.transform.run_pipeline`） | fold_constant → simplify_expr → force_narrow_index_to_i32 → convert_for_loops_serial → loop_partition → unroll_loop → vectorize_loop → remove_no_op |

**PassContext**（`include/base/pass.h`）：存储多设备标志、primary virtual device、virtual device 列表、default target、disco placement。通过 `PassContext::Scope` 线程本地作用域设置。

**扩展 pass 流程**：
1. 新增 `include/relay/transforms/`（或 `include/tir/transforms/`）头文件
2. 新增 `src/relay/transforms/`（或 `src/tir/transforms/`）实现
3. 在对应 `pipeline.cc` 加入 include 和 pass table
4. 如需外部调用，注册 `KXC_REGISTER_GLOBAL`

---

## 6. codegen — 代码生成

**职责**：将 TIR PrimFunc 编译为可执行代码。当前支持两个后端：LLVM JIT（主路径）和 C 源码生成（调试/备用）。

**头文件入口**：`include/codegen/`

| 组件 | 核心文件 | 职责 |
|------|---------|------|
| 抽象 | `codegen.h` | CodeGenBackend 枚举（kLLVM / kC / kCUDA） |
| C 后端 | `codegen_c.h`, `src/codegen/codegen_c.cc` | CodeGenC::Generate(PrimFunc, name) → C 源码字符串；CompiledModule::SaveCSource(path) 保存 |
| LLVM 后端 | `codegen_llvm.h`, `src/codegen/codegen_llvm.cc` | CodeGenLLVM：PrimFunc → LLVM Module（GenExpr / GenStmt / var_map_） |
| JIT 引擎 | `llvm_jit.h`, `src/codegen/llvm_jit.cc` | LLVMJITEngine：利用 LLVM ORC LLJIT，将 module 编译为可调用函数指针 |
| 编译产物 | `compiled_kernel.h`, `src/codegen/compiled_kernel.cc` | CompiledKernelNode（backend / func_ptr / kernel_name / jit_resource）；`operator()(args)` 统一调用 |

**编译开关**：
- `KXC_ENABLE_LLVM=ON`（默认）→ CodeGenLLVM + LLVMJITEngine 启用
- 若 LLVM 未找到 → `KXC_USE_LLVM=0`，此时 `Compiler::Compile` 要求 LLVM 后端

**当前局限**：
- LLVM codegen 是唯一完整可执行后端，C codegen 能生成源码但未接入完整编译执行路径
- CUDA codegen 为枚举预留，尚无实现

---

## 7. api — 编译 API 主入口

**职责**：对外暴露 `Compiler::Compile` 接口和 `CompileConfig` 配置。是用户（C++ 或 Python）调用编译链的唯一入口。

**核心文件**：

| 文件 | 职责 |
|------|------|
| `include/api/compile_config.h` / `src/api/compile_config.cc` | CompileConfigNode：mode（AOT / JIT / Adaptive）、opt_level、target、backend、自适应参数、profiling 参数；工厂方法 `AOT()` / `JIT()` / `Adaptive()` |
| `include/api/compiler.h` / `src/api/compiler.cc` | `Compiler::Compile(Function func, CompileConfig cfg)` → `CompiledModule`；主流程：RelayPass → LowerToTIR → TIRPass → CodeGen → JIT |

**编译模式**：

| 模式 | pass 选择 | 产物形态 |
|------|----------|---------|
| AOT | Relay + TIR 均走 `optimize_default`（完整流水线） | `prim_func_` + `CompiledKernel` |
| JIT | Relay + TIR 走轻量 `fold_constant` + `simplify_expr` | `prim_func_` + `CompiledKernel` |
| Adaptive | 不立即同步编译，创建 RuntimeSession + warmup | `relay_func_` + `RuntimeSession` |

---

## 8. runtime — 运行时与自适应编译

**职责**：管理自适应运行时的 kernel 缓存、shape 统计、后台编译和 kernel 执行。用于 Adaptive 编译模式。

**头文件入口**：`include/runtime/`

| 组件 | 核心文件 | 职责 |
|------|---------|------|
| RuntimeSession | `runtime_session.h`, `src/runtime/runtime_session.cc` | 管理自适应运行：保存原始 Relay function；创建后台编译器线程池；`Run(args, input_shapes)` 流程：shape 记录 → cache 查询 → 同步/后台编译 → 执行 |
| ShapePredictor | `shape_predictor.h`, `src/runtime/shape_predictor.cc` | shape 频率统计 + recently 队列（最多 200）；`ShouldCompile(sig, count_threshold=2, ratio=0.05)` 决策是否启动编译 |
| KernelCache | `kernel_cache.h`, `src/runtime/kernel_cache.cc` | exact match（shape 完全一致）+ fuzzy match（维度一致且 cached ≥ query，选最近） |
| BackgroundCompiler | `background_compiler.h`, `src/runtime/background_compiler.cc` | 优先队列 + 线程池；CompileTask（priority + function + config + on_complete）；后台调用 Compiler::Compile |
| KernelRunner | `kernel_runner.h`, `src/runtime/kernel_runner.cc` | kernel 执行封装 |

**自适应运行流程**：

```
RuntimeSession::Run(args, input_shapes)
  1. input_shapes → ShapeSignature
  2. ShapePredictor::Record(sig)        ← 记录频率
  3. KernelCache::GetExact(sig)         ← 精确命中？
     ├─ 命中 → 直接运行
     └─ 未命中 → GetFuzzy(sig)
          ├─ fuzzy 命中 → 运行 fallback，可选后台优化
          └─ 全部未命中 → 同步编译一次 → 插入 cache → 运行
  4. MaybeScheduleOptimization(sig)     ← 高频 shape 调度后台编译
```

---

## 9. device — 设备与张量管理层

**职责**：设备抽象（CPU/CUDA）、NDArray 张量数据结构和 Target 描述。是 codegen 和 runtime 的硬件接口层。

**核心文件**：

| 组件 | 关键文件 | 职责 |
|------|---------|------|
| Device | `include/base/device.h`, `src/base/device.cc` | DeviceNode + Device handle；DeviceManager 线程安全驻留 |
| DeviceAPI | `include/base/device_api.h`, `src/base/device_api.cc` | DeviceAPI 抽象：分配、释放、清零、同步/异步复制、stream/event、属性查询 |
| CPU DeviceAPI | `src/base/device/cpu_device_api.cc` | aligned_malloc 分配；CPU→CPU copy；target kind = "llvm" |
| CUDA DeviceAPI | `src/base/device/cuda_device_api.cc` | cudaMalloc/Free/Memset；Host/Device/Device copy；显式 stream/event；target kind = "cuda"；编译开关 KXC_USE_CUDA |
| NDArray | `include/base/ndarray.h`, `src/base/ndarray.cc` | 基于 Storage 的 dtype/shape/stride/offset 张量视图 |
| Target | `include/base/target.h`, `src/base/target.cc` | BuildTarget(Device) → Target（kind + arch + 属性） |
| VirtualDevice | `include/base/virtual_device.h`, `src/base/virtual_device.cc` | virtual_device 用于 Relay pass 的设备规划和 memory scope 标注 |

---

## 10. disco — 多 Worker 执行系统

**职责**：多设备执行计划 + 通信抽象。目前实现偏执行计划和通信语义验证层，不是完整分布式 kernel runtime。

**核心文件**：

| 组件 | 关键文件 | 职责 |
|------|---------|------|
| Placement | `include/base/disco_placement.h`, `src/base/disco_placement.cc` | WorkerPlacement / DiscoPlacement（worker → device mapping）；BuildDiscoPlacement |
| ExecutionPlan | `include/base/execution_plan.h`, `src/base/execution_plan.cc` | 执行计划：KernelExecNode / CommExecNode / BarrierExecNode；JSON 序列化 |
| Worker | `include/base/disco/worker.h`, `src/base/disco/worker.cc` | Worker 抽象基类，定义 worker 生命周期 |
| DiscoSession | `include/base/disco/session.h`, `src/base/disco/threaded_session.cc` | ThreadedSession（register file + AllocateRegister + Get/Set） |
| DRef | `include/base/disco/dref.h`, `src/base/disco/dref.cc` | 分布式值引用 |
| Executor | `include/base/disco/executor.h`, `src/base/disco/executor.cc` | ExecutionPlanExecutor::Execute(plan, values) → 按 plan 顺序解释执行 |
| CCL | `include/base/disco/ccl_backend.h`, `src/base/disco/ccl_cpu.cc` | CCLBackend 抽象：Copy / AllReduce / Broadcast / Scatter / Gather / Send / Recv / SyncWorker；当前只有用于语义验证的 CpuCCLBackend |

**当前执行计划节点类型**：

| 节点 | 作用 | 执行语义 |
|------|------|---------|
| KernelExecNode | 计算节点 | 尚未绑定 CompiledModule；执行时明确报错，不返回伪造输出 |
| CommExecNode | 通信节点 | 调用 CCLBackend 对应方法（AllReduce/Scatter/Gather 等） |
| BarrierExecNode | 同步节点 | 调用 SyncWorker |

---

## 11. profiling — 性能分析系统

**职责**：在编译和运行的关键路径上埋点，收集 span/event/artifact/trace，生成 profiling bundle 供离线分析。

**C++ 侧**：`include/base/profiling.h` / `src/base/profiling.cc`

| 类型 | 作用 |
|------|------|
| ProfileContext | 管理一次 profiling bundle（span/event/log/artifact/trace/diagnostics） |
| ActivationScope | RAII 安装 ProfileContext + run id 到线程本地 |
| ScopedSpan | RAII 记录 duration event |

**Bundle 文件结构**：

```
{bundle_dir}/
├── manifest.json         # 元数据
├── events.jsonl          # 事件流
├── trace.json            # Chrome trace / Perfetto 兼容
├── summary.json          # 汇总
├── diagnosis.json        # 诊断结果
├── diagnosis.md          # 诊断报告
└── artifacts/            # pass 前后 IR 等
```

**采集点**：
- `Compiler::Compile`
- Relay/TIR pass pipeline
- `LowerToTIR`
- `RuntimeSession`
- `BackgroundCompiler`
- `ExecutionPlanExecutor`
- CPU/CUDA DeviceAPI

**环境变量覆盖**：`KXC_PROFILE_ENABLE` / `KXC_PROFILE_BUNDLE_DIR` / `KXC_PROFILE_LOG_LEVEL` / `KXC_PROFILE_IR_MODE` / `KXC_PROFILE_NVTX` / `KXC_PROFILE_CUPTI` / `KXC_PROFILE_RECORD_PASS_IR` / `KXC_PROFILE_EXEC_PLAN_DETAILS`

**Python agent**（`python/kxc_agent/`）：

| 入口 | 职责 |
|------|------|
| `cli.py` | 命令行入口 |
| `tools/profile_run.py` | 设置环境变量并运行目标命令 |
| `tools/analyze_bundle.py` | 分析单个 bundle |
| `tools/compare_bundles.py` | 对比两个 bundle |
| `tools/inspect_pass_trace.py` | 查看 pass 耗时和 IR hash |
| `tools/explain_logs.py` | 提取 log event |
| `services/bundle_loader.py` | 加载 bundle |
| `services/diagnosis_engine.py` | 规则分析诊断 |
| `memory/store.py` | bundle 分析历史记录 |

---

## 12. frontend — ONNX 导入器

**职责**：将 ONNX 模型解析并导入为 Relay Function。连接外部模型格式和内部编译链。

**核心文件**：
- `include/frontend/onnx_importer.h`
- `src/frontend/onnx_importer.cc`

**已支持算子**（见 `docs/OP_SUPPORT_MATRIX.md`）：Conv、Relu、BatchNormalization、MaxPool、AveragePool、GlobalAveragePool、Add、Mul、Gemm、Reshape、Transpose、Softmax、Concat、Flatten 等。

**Python 辅助脚本**（`python/kxc_onnx/`）：

| 文件 | 职责 |
|------|------|
| `importer.py` | Python 侧 ONNX 导入逻辑 |
| `spec.py` | 算子规格描述 |

---

## 13. Python 工具集

**路径**：`python/`

| 文件 | 职责 |
|------|------|
| `python/kxc_agent/` | Profiling agent（分析工具，见第 11 节） |
| `python/kxc_onnx/` | ONNX 导入 Python 侧（见第 12 节） |
| `python/gen_resnet18_ir_dump_cpp.py` | 生成 `test/resnet18_ir_dump.cpp` 风格模型构造代码 |

**注意**：正式 ONNX 解析和校验统一走 `kxc_onnx`；IR dump 生成器仅是显式诊断工具，不是 CMake runtime 构建的一部分。

---

## 14. 测试

**测试目录**：`test/`

| 测试文件 | 覆盖内容 |
|---------|---------|
| `pass_pipeline_test.cpp` | Relay/TIR pass 单元测试 + pipeline 顺序 |
| `codegen_llvm_test.cpp` | TIR/Relay → LLVM JIT smoke test |
| `resnet18_ir_dump.cpp` | ResNet18 Relay graph 构造 + dump |
| `profile_bundle_test.cpp` | profiling bundle 结构 + 事件 |
| `cupti_smoke_test.cpp` | CUDA/CUPTI activity 采集（CUDA ON 时构建） |
| `infer_type_test.cpp` | Relay 类型推断 |
| `op_numeric_llvm_test.cpp` | 算子数值正确性（LLVM JIT 执行） |
| `onnx_importer_test.cpp` / `onnx_importer_py_test.py` | ONNX 导入器 C++ / Python 测试 |
| `device_info_test.cpp` | 设备信息查询 |

**CMake 自定义 target**（`cmake --build ... --target <target>`）：

```
run_pass_pipeline_test
run_codegen_llvm_test
run_resnet18_ir_dump
run_profile_bundle_test
run_cupti_smoke_test
```

**常用构建命令**：

```powershell
# 配置（CPU-only，Ninja）
cmake --preset dev-ninja-cpu

# 构建全部
cmake --build --preset dev-ninja-cpu

# 运行单测
cmake --build out/build/dev-ninja-cpu --target run_pass_pipeline_test
```

---

## 15. 构建系统

**构建系统**：CMake >= 3.20，预设配置在 `CMakePresets.json`。

**核心产物**：静态库 `kxc_runtime`（由 `KXC_RUNTIME_SOURCES` 汇总）+ 若干测试/调试可执行文件。

**关键 CMake 选项**：

| 选项 | 默认 | 作用 |
|------|------|------|
| `KXC_ENABLE_CUDA` | ON | 检测 CUDA Toolkit，控制 CUDA DeviceAPI 编译 |
| `KXC_ENABLE_LLVM` | ON | 检测 LLVM，控制 LLVM codegen/JIT 编译 |
| `KXC_BUILD_RESNET18_IR_DUMP` | OFF | 显式构建 ResNet18 IR dump 诊断工具 |
| `KXC_BUILD_PASS_TESTS` | ON | 构建 pass 测试 |
| `KXC_BUILD_CODEGEN_TESTS` | ON | 构建 codegen 测试 |

**Preset**：`dev-ninja`（Debug + CUDA ON） / `dev-ninja-cpu`（Debug + CUDA OFF） / `dev-mingw-cpu`（Debug + MinGW CPU-only）

---

## 16. 模块依赖关系总结

```
          ┌─────────────────────────────┐
          │  frontend (onnx_importer)   │
          │  import ONNX → Relay Func   │
          └──────────┬──────────────────┘
                     │ depends on relay + base
                     ▼
          ┌─────────────────────────────┐
          │  relay (IR + op + pass)     │
          │  + relay::backend/lower     │
          └──────────┬──────────────────┘
                     │ depends on te + tir + base
                     ▼
          ┌─────────────────────────────┐
          │  te / topi (Tensor Expr)    │
          └──────────┬──────────────────┘
                     │ depends on tir + base
                     ▼
          ┌─────────────────────────────┐
          │  tir (TIR IR + pass)        │
          └──────────┬──────────────────┘
                     │ depends on base
                     ▼
          ┌─────────────────────────────┐
          │  codegen (LLVM / C)         │
          │  + api::Compiler::Compile   │
          └──────────┬──────────────────┘
                     │ depends on tir + base
                     ▼
          ┌─────────────────────────────┐
          │  runtime                    │
          │  (Session / Cache / Predict)│
          └─────────────────────────────┘

base ───── 被所有模块依赖 ───── 无上层依赖
disco ──── 被 runtime + relay 引用 ──── 依赖 base
profiling ─ 埋点在以上所有模块 ──── 依赖 base
```

---

## 17. 推荐阅读顺序

第一次读代码时，按以下顺序建立完整心智模型：

1. `include/base/object.h`、`include/base/container.h` — 对象系统基石
2. `include/base/packedfunc.h`、`include/base/registry.h` — 函数注册机制
3. `include/relay/relay.h`、`include/relay/op.h` — Relay IR 与算子
4. `src/relay/op/tensor/math.cc`、`src/relay/op/nn/convolution.cc` — 算子注册实例
5. `include/te/te.h`、`include/te/topi/nn.h` — TE 计算表达
6. `include/tir/expr.h`、`include/tir/stmt.h` — TIR 底层 IR
7. `src/relay/backend/lower.cc` — Relay → TIR lowering 核心
8. `src/relay/transforms/pipeline.cc`、`src/tir/transforms/pipeline.cc` — pass 流水线
9. `src/api/compiler.cc` — 编译主链路
10. `src/codegen/codegen_llvm.cc`、`src/codegen/llvm_jit.cc` — 代码生成
11. `src/runtime/runtime_session.cc` — 自适应运行时
12. `src/base/device_api.cc`、`src/base/device/cpu_device_api.cc` — 设备抽象
13. `src/base/disco/executor.cc` — 多 worker 执行
14. `include/base/profiling.h`、`src/base/profiling.cc` — 性能分析
