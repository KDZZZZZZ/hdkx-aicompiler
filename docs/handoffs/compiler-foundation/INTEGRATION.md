# Compiler Foundation 集成交接

> **集成分支：** `integration/compiler-foundation`
>
> **上游源码基线：** `dev@3b95aca188122ff52ebdb2f43d390d21273ff3e2`
>
> **W1 集成提交：** `e05d9d7`
>
> **状态：** W1 isolated/default-off baseline 已集成；W2 production adapters 和跨轨闭环尚未完成。本页不是 dynamic Shape、hot swap、dynamic control flow、Transformer 或 GPU attention 的生产能力声明。

## 1. 已合并轨道

按 GitHub Flow 保留了六个 feature branch 的 merge 边界：

1. `feature/compiler-foundation-core`
2. `feature/compiler-foundation-shape`
3. `feature/compiler-foundation-adaptive`
4. `feature/compiler-foundation-control-flow`
5. `feature/compiler-foundation-runtime-plan`
6. `feature/compiler-foundation-nlp-gpu`

对应集成结果：

- Core capability / normalized pipeline / semantic identity / full-key cache / artifact pin / production singleflight transaction。
- `kxc::shape::experimental::v1` exact 与 guarded contract/fake；未接生产 Relay/Compiler/Runtime。
- `kxc::api::experimental::adaptive::v1` static-exact coordinator/slot/lease；未接真实 Compiler/RuntimeSession。
- Relay ANF、lexical Let、static-exact `ControlPlan` preparation/reference；生产 `Compiler::Compile` 继续拒绝 `If`。
- default-OFF Region/Task DAG contract、validator、memory planner 和 RuntimeSession task mode；不支持 dynamic Shape、ControlFlow、Library ABI、multi-stream/device。
- finite-logit exact-static NLP reference、stable softmax、batched matmul、fail-closed ONNX contract；不支持真实 KV cache、完整 Transformer 或 CUDA reduction/attention。

## 2. 主集成修正

跨分支语义合并额外关闭了以下问题：

- 将 `NormalizeToANF` 注册为 Pass contract，并纳入唯一 `NormalizedPipeline`、canonical identity 和 executable `anf` invariant；没有在 Compiler 外偷偷追加 pass。
- Core capability verifier 接受经过验证的 lexical `Let`，但继续拒绝 `If`；这与 ANF pipeline 和 ValueGraph 的 Let lowering 一致。
- 统一 production TE output 的数量、definedness、dtype、rank 与 static shape 验证。
- 修复 batched matmul 中引用类型 `Array` 的浅拷贝别名：`output_shape` 不再修改 `batch_shape`。
- batched matmul reduction indices 显式使用 `AsPrimExpr(k)`，不把 `IterVarNode` 冒充 TIR expression。
- CI 的 CPU matrix 同时覆盖 Region Task DAG `OFF/ON`；`cpu` CTest label 包含 Core、Adaptive、Control、Runtime plan 与 dependency-free ONNX import contract。

## 3. 已执行的统一验证

CPU-only、LLVM/CUDA disabled：

```bash
cmake -S . -B out/build/foundation-integration -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=OFF \
  -DKXC_ENABLE_REGION_TASK_DAG=OFF \
  -DKXC_BUILD_RESNET18_IR_DUMP=OFF \
  -DKXC_BUILD_PASS_TESTS=ON \
  -DKXC_BUILD_CODEGEN_TESTS=OFF
cmake --build out/build/foundation-integration --parallel 2
ctest --test-dir out/build/foundation-integration \
  --output-on-failure --no-tests=error \
  --label-regex '(^|;)cpu(;|$)'
```

结果：**38/38 passed**。

包含：

- 3 Adaptive tests；
- 5 Control-flow tests；
- 3 Runtime-plan tests；
- Core/compiler/runtime tests；
- dependency-free ONNX import-spec contract；
- Relay operator contract；
- Pass contract；
- include-layer 和 public-header compile。

Shape tests 独立执行：**3/3 targets passed**。

NLP checker：**PASS**。

Region Task DAG `ON` 使用独立 build directory，完整 CPU build 与 **38/38 CPU CTest passed**。

机器可读合同：

- Relay operators：**19/19**。
- Passes：**20/20**；新增 `normalize_to_anf`。

## 4. 环境限制

本机没有 LLVM package，也没有 Python `onnx` / `numpy` / `pytest`：

- 未运行 LLVM codegen/numeric、artifact relocation numeric、exact attention runtime numeric。
- 未运行依赖 ONNX Python 包的 importer tests。
- 未下载或安装依赖。

CUDA 真实 device 证据沿用 NLP feature branch 的 elementwise/rejection 记录；本次统一 integration build 明确关闭 CUDA，因此没有新的 CUDA numeric、Compute Sanitizer 或 CUPTI 结论。

## 5. W2 必须完成的跨轨闭环

W1 的 mock/fake/DTO 存在不等于总计划完成。W2 至少需要：

1. **Shape production exact path：** 从真实 Relay/type/partition 生成不可变 GraphTemplate；exact profile 只重特化受 Shape 影响 unit；组装真实 static module/plan；RuntimeSession 仍只执行 frozen variant。
2. **Adaptive production exact path：** Core canonical key、production artifact pin、真实 compiler adapter、selected generation manifest、plan assembler 和 completion lease retention；不得在 Run 内 lookup/compile。
3. **Control production path：** unresolved Relay kernel 解析为 immutable artifact，ControlPlan 映射到 runtime-only task/control schema；单 stream branch/loop/Phi/backedge liveness 与双 oracle 验证。
4. **Runtime manifest/observability：** selected artifact identity/generation、fallback reason、task wait/launch/allocation/retire；打开默认 gate 前补 LLVM/CUDA numeric、pending CUDA retention 和 sanitizer。
5. **NLP vertical expansion：** Gather/embedding、mask/select、normalization、Slice/Concat 和 KV page/capacity/valid extent；逐项 frontend/type/lowering/backend/runtime/negative evidence。
6. **Target evidence：** LLVM-enabled CI 真实绿色记录；CUDA reduction/library path 和 device numeric；没有证据时 capability 继续 fail closed。

这些跨轨工作通过新的 W2 feature branches 继续，不直接在 `dev` 或 W1 baseline 上堆叠中间状态。
