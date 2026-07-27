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
- `KXC_ENABLE_LLVM=ON`，并实际发现 LLVM >= 20

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
