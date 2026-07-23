# hdkx-aicompiler 模块指南

> **状态：Track01 仓内 closure 已实现并通过 CPU 验证，待 LLVM 绿色记录与终审。**
> 本页只说明当前模块边界和入口；单个 Relay 节点/目标的生产批准仍以
> executable capability 与对应 backend CI 为准。

## 生产编译入口

唯一的生产能力入口是：

```text
kxc::api::Compiler::Compile
  -> typed Relay graph
  -> value graph / per-unit partition
  -> per-unit lowering and code generation
  -> CompiledModule + ExecutablePlan
  -> RuntimeSession
```

相关路径：

- 公共 API：`include/kxc/compiler/compiler.h`
- 编译实现：`src/compiler/compiler.cc`
- value graph / partition：`src/compiler/graph/`
- per-unit lowering：`src/compiler/lowering/lowered_graph.cc`
- runtime-neutral plan：`include/kxc/runtime/executable_plan.h`
- 执行器：`src/runtime/session.cc`

`RuntimeSession` 是已冻结 `CompiledModule + ExecutablePlan` 的数据面执行器；它
不是编译、缓存、shape 选择或后端回退控制面。

## whole-graph lowering 的边界

`relay::LowerToTIR`（声明在
`include/kxc/compiler/lowering/relay_to_tir.h`，实现在
`src/compiler/lowering/relay_to_tir.cc`）保留给兼容性和 focused
TE/TIR 测试。它把一个 Relay Function 降为一个 `PrimFunc`，不能作为当前生产
capability、缓存 identity 或多 entry runtime 的证据。

新功能必须通过 `Compiler::Compile` 的 per-unit 路径接入并验证。测试可使用
`LowerToTIR` 描述历史/局部行为，但不得据此宣称 production support。

## 模块边界

| 模块 | 主要路径 | 职责 | 非职责 |
|---|---|---|---|
| 公共契约 | `include/kxc/compiler/`、`include/kxc/runtime/` | compiler config、capability、identity、pipeline、module/plan API | 直接执行未验证 Relay 节点 |
| Relay / Pass | `include/kxc/relay/`、`src/relay/`、`src/pass/` | IR、operator/pass contract、类型与图级变换 | runtime allocation 或 cache policy |
| Compiler | `src/compiler/` | capability 检查、partition、per-unit lowering、backend 组装 | 在 session 内作动态编译选择 |
| TIR / codegen | `include/kxc/tir/`、`src/tir/`、`src/codegen/` | unit-local TIR 与后端发射 | 图级 value routing |
| Runtime | `include/kxc/runtime/`、`src/runtime/` | plan 验证、storage、launch、completion 保活 | Relay registry、Compiler 内部状态 |
| Frontend | `include/kxc/frontend/`、`src/frontend/`、`python/kxc_onnx/` | 受限 ONNX 导入与规格转换 | 将 unknown shape 静默视为 production dynamic shape |

依赖必须保持单向：Compiler 可依赖 Relay/TIR/Codegen 和 runtime plan contract；
Runtime 不反向依赖 Compiler、Relay、TE、TIR 或 registry。

## 验证入口

CMake 将可执行 C++ 测试和结构检查注册到 CTest：

```bash
cmake -S . -B out/build/cpu -G Ninja \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=OFF \
  -DKXC_BUILD_PASS_TESTS=ON -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/build/cpu --parallel
ctest --test-dir out/build/cpu --output-on-failure --label-regex '(^|;)cpu(;|$)'
```

`cpu` 标签包括所有已构建的 CPU/core C++ executables、Relay/pass contract、
include-layer 和 public-header compile 检查。`cuda_schedule_test` 在该集合中仅
验证合成 CUDA Target 的结构契约，不查询或启动 CUDA 硬件。LLVM 数值/codegen 与
CUDA/CUPTI 硬件测试分别带有 `llvm` 与 `cuda;hardware` 标签，CPU-only job 不运行它们。

## 当前 closure 状态

**Ready for supervisor re-review；不自称 Core complete。**

- [x] executable capability 使用真实 normalized per-unit compile proof
- [x] public `ProductionArtifactCacheAdapter` transaction/singleflight 接入真实 primitive cache，并保活 production pin
- [x] normalized pipeline 在每步后执行已证明 invariant 的 validator
- [x] 本地 CPU CTest、contract、include/public-header：27/27
- [x] LLVM workflow 包含 relocation/cache、codegen、numeric、ONNX compile
- [ ] LLVM-enabled builder 实际绿色记录与目标后端逐项批准（本机/当前会话无 LLVM 记录）

历史 Adaptive Runtime、fuzzy cache、旧目录名和“whole graph 是主链”的说明均不再是
当前指南。详见 [`COMPILER_EXTENSION_CONTRACT.md`](COMPILER_EXTENSION_CONTRACT.md)、
[`OP_SUPPORT_MATRIX.md`](OP_SUPPORT_MATRIX.md) 和
[`handoffs/compiler-foundation/core.md`](handoffs/compiler-foundation/core.md)。
