# hdkx-aicompiler 编译说明

本项目使用 `CMake`、`Ninja` 和 `C++17`。要求 `CMake 3.23` 或更高版本。
`CUDA` 与 `LLVM` 为可选项；显式的 CPU-only 配置不需要这两套工具链。

## 预设

| 预设 | 编译器 | CUDA | LLVM |
|---|---|---:|---:|
| `dev-ninja` | CMake 默认 | 请求启用 | 请求启用 |
| `dev-ninja-cpu` | CMake 默认 | OFF | 请求启用 |
| `dev-mingw-cpu` | MinGW `g++` | OFF | OFF |

“请求启用”表示 CMake 会查找该依赖。`LLVM` 需版本 20 或以上。
如果可选依赖缺失，CMake 会报告对应后端未构建；该构建下该后端测试不作为有效证据。

## Windows 可移植 CPU 构建

先决条件：

- `CMake 3.23` 或更新版本；
- `Ninja`；
- 路径中可见 `MinGW g++`；
- 安装 `Python 3`，用于契约与架构检查。

```powershell
where.exe cmake
where.exe ninja
where.exe g++
python --version

cmake --preset dev-mingw-cpu
cmake --build --preset dev-mingw-cpu
ctest --test-dir out/build/dev-mingw-cpu `
  --output-on-failure --no-tests=error
```

## Windows MSVC CPU 构建

使用已就绪 `cl.exe`、`link.exe`、`rc.exe`、`mt.exe` 的
Visual Studio x64 Native Tools Shell：

```powershell
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu
ctest --test-dir out/build/dev-ninja-cpu `
  --output-on-failure --no-tests=error
```

若未安装 LLVM，可显式关闭以获取确定性的核心构建：

```powershell
cmake -S . -B out/build/dev-msvc-core -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug `
  -DKXC_ENABLE_CUDA=OFF `
  -DKXC_ENABLE_LLVM=OFF
cmake --build out/build/dev-msvc-core --parallel
ctest --test-dir out/build/dev-msvc-core `
  --output-on-failure --no-tests=error
```

## Linux LLVM 构建

安装 LLVM 20 开发文件；若不在默认搜索路径，请将 `LLVM_DIR` 指向对应目录：

```bash
cmake -S . -B out/build/dev-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=ON \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build out/build/dev-llvm --parallel
ctest --test-dir out/build/dev-llvm \
  --output-on-failure --no-tests=error
```

启用 LLVM 的验证必须包含 `codegen_llvm_test`、`op_numeric_llvm_test`，
以及 `onnx_importer_test` 的 LLVM 子集。

## Bounded 动态图 all-gates 构建

`KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH` 默认关闭，且配置时强制要求：

- `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON`
- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON`
- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON`
- 实际可用的 LLVM >= 20 或 CUDA 后端（至少一个）

独立 CPU/LLVM all-gates 配置示例：

```bash
cmake -S . -B out/build/dynamic-production-e2e-all-gates -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=ON \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DKXC_ENABLE_CONTROL_RUNTIME=ON \
  -DKXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON \
  -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON
cmake --build out/build/dynamic-production-e2e-all-gates --parallel
ctest --test-dir out/build/dynamic-production-e2e-all-gates \
  --output-on-failure --no-tests=error
```

`bounded_dynamic_graph_llvm_test` 证明一次 `CompileBounded` 后，同一
`CompiledGraph` 在 `N=2` 与 `N=6` 上执行 `add -> relu -> sqrt`，运行期间不访问原语缓存。该 gate 不启用 route、adaptive 对齐、state/alias/reuse、控制流合并或 CUDA fallback。

## CUDA 构建

CUDA 支持需要兼容的 NVIDIA 驱动与工具包。工具包发现与设备可用性是独立条件：
编译通过不代表本地 GPU 一定能成功启动内核。

```bash
cmake -S . -B out/build/dev-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build out/build/dev-cuda --parallel
ctest --test-dir out/build/dev-cuda \
  --output-on-failure --no-tests=error
```

仅在有可用 GPU 的机器上运行 CUDA/CUPTI 硬件测试：

```bash
ctest --test-dir out/build/dev-cuda \
  --output-on-failure --no-tests=error \
  --label-regex '^hardware$'
```

合成的 `cuda_schedule_test` 用例属于 CPU/core 测试矩阵的一部分，
仅验证调度契约，不触发真实硬件发射。

有界 CUDA 可在上述 CUDA 配置中同时开启 dynamic module ABI、restricted/exact shape 和 bounded 四个门禁；可设 `KXC_ENABLE_LLVM=OFF`。硬件消费测试为 `bounded_cuda_test`，要求实际数值标记及 CUPTI bundle 通过。单阶段有界输出、形状值和 rank-2 MatMul 见 [基础报告](implementation/GPU_BOUNDED_CORE_REPORT.md)；Softmax/MaskedSoftmax、固定轴 ReduceMean/RMS 和三头注意力见 [多阶段报告](implementation/GPU_BOUNDED_REDUCTION_REPORT.md)。完整 bounded 模型和 GPU 请求批处理仍需独立验收。

完整变长 MiniMind 的两个独立入口均使用 `KXC_MINIMIND_BOUNDED_PREFILL_DIR` 指定实际导出 fixture。`minimind_bounded_cuda_codegen_test` 使用显式 synthetic `sm_89` target，将全部 742 个原语生成 CUDA 源码并用 NVCC 编译为 PTX，不需要 GPU；`minimind_bounded_prefill_cuda_test` 执行四组 B/S、双流与未来 token 扰动，必须同时通过全部 logits/KV 数值和 CUPTI 关联检查。缺 fixture 或设备时返回 77，不能算作硬件验收。两者的证据边界与复现命令见 [接入报告](implementation/GPU_BOUNDED_PREFILL_REPORT.md)。

完整变长 decode 的无设备编译入口为 `minimind_bounded_decode_cuda_codegen_test`，fixture 由 `KXC_MINIMIND_BOUNDED_DECODE_DIR` 指定；全部 774 个原语通过源码/PTX 编译。此入口不执行 GPU，不替代 decode 数值或状态验收，见 [技术报告](implementation/GPU_BOUNDED_DECODE_REPORT.md)。

`minimind_bounded_decode_cuda_test` 是后续真实设备入口，同时需要上述 decode fixture 和 `KXC_MINIMIND_BOUNDED_PREFILL_DIR`。它验证普通/持久状态 decode、实际 prefill 交接与两条 stream 上的连续追加，且须通过 CUPTI kernel/extent/状态复制的关联与顺序检查。缺设备返回 77；本地 CPU 验证不能替代该门禁，见 [状态报告](implementation/GPU_BOUNDED_STATE_REPORT.md)。

## 特性开关

默认关闭的架构开关有：

- `KXC_ENABLE_CONTROL_RUNTIME`
- `KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI`
- `KXC_ENABLE_SHAPE_PRODUCTION_EXACT`
- `KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE`
- `KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH`
- `KXC_ENABLE_ADAPTIVE_HOT_SWAP`

仅在验证矩阵包含其测试时再开启该开关。
开关边界定义见 [架构总览](ARCHITECTURE.md)。

## 文档与契约检查

```powershell
python tools/architecture/check_docs.py --root .
python tools/architecture/check_include_layers.py --root .
python python/tools/check_relay_op_contract.py --root . `
  --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . `
  --matrix contracts/pass_contract.json
```

`profile_bundle_test` 与 `cupti_smoke_test` 会在 `test/` 下生成输出目录，
属于临时产物，不得提交到仓库。

## 完整模型的 Windows CUDA 测试

`minimind_prefill_cuda_test` 使用现有 decode-loop fixture 的静态 prefill 文件。设置 `KXC_MINIMIND_CUDA_PREFILL_DIR` 后，由 `ctest -R minimind_prefill_cuda_test -V` 验证全部 logits 与 16 个 K/V；未设置时以返回码 77 明确跳过。MSVC 目标通过 `/STACK:8388608` 为现有递归图遍历预留 8 MiB 栈。实际环境、源码和 fixture 校验及数值结果见 [报告](implementation/GPU_MINIMIND_PREFILL_REPORT.md)。

请求批处理的 CUDA 入口为 `request_batching_cuda_test` 和 `minimind_request_batching_cuda_test`。前者覆盖队列、状态、stream、生命周期与失败；后者要求 `KXC_MINIMIND_BOUNDED_DECODE_DIR`，通过同一完整模型的 `--cuda-batching` 入口核验数值及 CUPTI。没有设备返回 77。`request_batching_axis1_cuda_codegen_test` / `request_batching_axis2_cuda_codegen_test` 无需设备，编译共享小图的全部 CUDA 原语为 PTX；这些编译结果不能替代设备门禁，见 [技术报告](implementation/GPU_REQUEST_BATCHING_REPORT.md)。

## 带 KV 的静态 CPU 热替换

在 `KXC_ENABLE_ADAPTIVE_HOT_SWAP=ON` 的 LLVM 构建中，`adaptive_state_model_test` 复用现有完整 state fixture 的全部 ONNX 参考，执行四步代际 1/2/1/1。设置 `KXC_MINIMIND_ADAPTIVE_STATE_DIR` 指向 `out/fx_minimind_cuda_state`；该目录名称不改变测试的 CPU/LLVM 执行设备。缺少 fixture 时返回 77。该完整模型测试使用 `RUN_SERIAL`，避免大模型身份准备与其他模型/JIT 测试同时占用内存。

```bash
KXC_MINIMIND_ADAPTIVE_STATE_DIR="$PWD/out/fx_minimind_cuda_state" \
OPENBLAS_NUM_THREADS=1 \
ctest --test-dir out/build/adaptive-llvm --output-on-failure --no-tests=error \
  -R 'adaptive_state_model_test|adaptive_runtime_test'
```

方法、数值、内存与剩余边界见 [技术报告](implementation/M6_STATEFUL_REPORT.md)。

## 带 KV 的有界 CPU 热替换

在上述动态生产组合配置中同时启用 LLVM 与 `KXC_ENABLE_ADAPTIVE_HOT_SWAP`，`adaptive_runtime_test` 会执行有界小图、范围及连接反例；`adaptive_bounded_state_model_test` 复用完整 bounded prefill/decode fixture，执行真实 prefill 交接及四步代际 1/2/1/1。后者使用 `RUN_SERIAL`、300 秒超时；缺少 fixture 返回 77。独立静态 prefill 子用例还需 `KXC_MINIMIND_ADAPTIVE_PREFILL_DIR`，否则该子用例明确跳过。

```bash
KXC_MINIMIND_ADAPTIVE_PREFILL_DIR="$PWD/out/fx_decode_stateful" \
KXC_MINIMIND_BOUNDED_PREFILL_DIR="$PWD/out/fx_minimind_bounded_prefill" \
KXC_MINIMIND_BOUNDED_DECODE_DIR="$PWD/out/fx_minimind_bounded_decode" \
OPENBLAS_NUM_THREADS=1 \
ctest --test-dir out/build/adaptive-bounded-llvm --output-on-failure --no-tests=error \
  -R '^(adaptive_runtime_test|adaptive_bounded_state_model_test)$'
```

本次完整组合构建另启用 `KXC_ENABLE_CONTROL_RUNTIME` 复验既有控制流入口；该开关不是热替换的依赖。模型 fixture 的来源、数值、事件顺序及限制见 [技术报告](implementation/M6_BOUNDED_REPORT.md)。CPU 编译与大模型测试按阶段串行执行，避免同时占用内存。

## 请求批处理热替换

同一 adaptive/bounded/LLVM 组合配置提供 `adaptive_request_batch_model_test`，只需已有完整 bounded decode fixture。它执行四个请求的五个批次、指定 kernel 换代、队列保留、退出和回滚；每个请求的 logits 与全部 16 份 KV 均有独立执行对照。缺 fixture 返回 77，完整测试使用 `RUN_SERIAL` 和 300 秒超时。

```bash
KXC_MINIMIND_BOUNDED_DECODE_DIR="$PWD/out/fx_minimind_bounded_decode" \
OPENBLAS_NUM_THREADS=1 \
ctest --test-dir out/build/adaptive-bounded-llvm --output-on-failure --no-tests=error \
  -R '^adaptive_request_batch_model_test$'
```

调用方可直接从编译产物派生状态与请求计划；以下片段沿用 `kxc::api::adaptive::hot_swap::preparation` 的请求类型，`bounded_authority` 来自已有有界 adapter：

```cpp
auto batch_graph = compiled.BindBoundedStateOutputs(bindings, physical_shapes)
                           .BindRequestBatching(2);
auto lease = controller.CompileAndPublish({ProductionCompileRequest(
    bounded_authority, batch_graph, {selected_unit})});
ProductionExecutionRequest route(lease->dispatch_key(), lease->plan_abi());
auto session = controller.CreateStatefulSession(route);
auto id = session.AdmitRequest(seed_states, seed_extent);
session.EnqueueRequest(id, {token});
auto batch = controller.RunNextBatch(session, stream);
```

`StatefulSession` 在这里拥有一组请求的队列和槽位；每个批次固定自己的 lease。静态容量状态的公开入口为 `CompiledGraph::BindStateOutputs`。CPU 数值、队列和事件证据及 CUDA 边界见 [技术报告](implementation/M6_REQUEST_BATCHING_REPORT.md)。

## 静态整图分发与 MiniMind prefill

`distributed_runtime_test` 验证公开整图绑定、有序多输出、共享依赖、JSON v2 迁移与 v3 往返，以及真实 worker/CCL 的成功和失败路径。完整模型入口复用既有 `fx_decode_stateful` 的静态 B1/S16 prefill fixture，缺少环境变量返回 77；在 LLVM 配置中注册为 `distributed_minimind_model_test`，顺序运行、180 秒超时。

```bash
cmake --build out/build/dev-ninja-cpu --parallel 2 --target distributed_runtime_test
KXC_MINIMIND_DISTRIBUTED_DIR="$PWD/out/fx_decode_stateful" \
OPENBLAS_NUM_THREADS=1 \
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R '^(distributed_runtime_test|distributed_minimind_model_test)$'
```

已编译的静态图通过 `ExecutionPlan::FromExecutablePlan(compiled.module(), compiled.plan(), placement, input_workers, call_workers, output_worker)` 绑定。调用方按返回计划的 `input_value_ids` 顺序，将输入 DRef 放到声明的 worker，再调用 `ExecutionPlanExecutor::ExecuteForOutputs`，按原图输出顺序取得结果。常量由模块持有并注入；JSON 保存计划，模块由调用方另行绑定。

该路径按显式放置生成已有 kernel/copy 节点；完整模型的结果、事件和限制见 [整图报告](implementation/M7_MODEL_REPORT.md)。
