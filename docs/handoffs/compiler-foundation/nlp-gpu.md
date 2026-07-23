# Compiler Foundation NLP/GPU 验证轨交接

> **历史文档：** 本页记录早期 `nlp-gpu` 基线，其中“Gather/Where 未实现”等语句
> 不再是 HEAD 能力声明。当前终审 contract 以
> [`nlp-transformer.md`](nlp-transformer.md)、[`../../ONNX_IMPORTER.md`](../../ONNX_IMPORTER.md)
> 和机器矩阵为准。
>
> **分支：** `feature/compiler-foundation-nlp-gpu`
> **基线：** `e295a73343f82a7852eeae4ba6d11e3d892adc52`
> **验证日期：** 2026-07-23
> **状态：** 仅完成 finite-logit exact-static CPU reference baseline 与 fail-closed contract gate；不代表完整 Transformer、LLVM/`RuntimeSession` 数值或 NLP/GPU 端到端 **Done**。
> **边界：** 本轨只建立验证资产、fail-closed gate 和最小算子/导入/后端切片，没有重塑 Compiler、`ExecutablePlan` 或 `RuntimeSession` 核心抽象。

## 1. 交付摘要

本轨完成了以下工作：

1. 建立不依赖 ONNX、NumPy、LLVM 或 CUDA 的确定性 CPU reference、fixture、workload manifest、能力矩阵和负向 gate。
2. Python ONNX protobuf importer 对 unknown rank 与 unresolved/symbolic dim 默认拒绝；只有调用方显式传入正数 `default_batch` 时才能绑定 axis 0，非 batch unresolved dim 永远不填 `1`。
3. C++ import-spec reifier 校验非负静态 shape，并在构图后用 Relay 类型推导核对每个声明 output 的 shape/dtype。
4. Relay softmax 改为 `max -> subtract -> exp -> sum -> divide`，LLVM 数值测试加入极大正负 logits 和非有限值检查。
5. `matmul` 扩展为 rank >= 2，并按 ONNX/NumPy 规则广播 leading batch dimensions；不兼容 K、batch 或 rank < 2 明确失败。
6. ONNX importer 增加静态 `MatMul`、`Softmax`、`Transpose` 映射；每个 `MatMul` 在 frontend fail-closed 校验两个可解析静态输入、rank >= 2、dtype/K、NumPy leading-batch broadcast 和声明 output contract，缺失 metadata 拒绝；`Softmax` 只接受 opset >= 13 的单 axis 语义，opset < 13 的 flatten-from-axis 语义明确拒绝。
7. 两条 lowering 路径在生成 TIR 前拒绝负 extent；Relay 仍可表示 unknown extent，但不能冒充 executable dynamic shape。
8. 增加 exact-static causal prefill 和 token=1 external-K/V decode 的 LLVM/`RuntimeSession` 数值测试代码。
9. 既有 add/relu CUDA device tests 通过；本轨 softmax/batched-matmul 的 CUDA 结果只有 `BindCudaThreads` reduction gate 的 fail-closed 拒绝，没有 NLP/attention GPU 数值证据。
10. KV cache 只提供 reference/mock append/page/capacity trace；没有向现有 runtime 注入错误的 alias、valid-extent 或 capacity 语义。

机器可读事实源：

- [`test/nlp_validation/transformer_capability_matrix.json`](../../../test/nlp_validation/transformer_capability_matrix.json)
- [`test/nlp_validation/deterministic_fixtures.json`](../../../test/nlp_validation/deterministic_fixtures.json)
- [`test/nlp_validation/workload_manifests.json`](../../../test/nlp_validation/workload_manifests.json)
- [`python/tools/check_nlp_gpu_validation.py`](../../../python/tools/check_nlp_gpu_validation.py)

## 2. 当前 support matrix

状态语义与计划一致：`unsupported < contracted < implemented < validated`。`validated` 只约束表中对应 layer 的证据；CPU reference fixture 通过不等于 Transformer、LLVM/`RuntimeSession` 或 GPU 执行通过。`implemented` 表示代码和测试入口存在，但本机可能因环境门禁未执行。

| 能力 | Frontend / Relay / lowering | CPU reference / LLVM | CUDA | 结论与 gate |
|---|---|---|---|---|
| stable softmax | ONNX opset >= 13 static axis、Relay 和 max-subtraction lowering 已实现；opset < 13 拒绝 | finite-logit exact-static CPU reference fixture validated；LLVM 极值测试已实现但本机未运行 | unsupported | 仅 unmasked static softmax；非有限/all-masked 表示没有执行语义，验收 gate 保持关闭 |
| masked/all-masked softmax | 无正式 mask/select 或 all-masked contract | checker 验证该 fixture 不得开放，不是 Relay 执行结果 | reduction 与 mask 均 unsupported | 不定义“全零”等临时替代语义，也不声称现有 Relay 会运行时拒绝 `-inf` 行 |
| batched matmul | ONNX frontend fail-closed 校验静态 rank/K/dtype/batch/output contract；Relay type check 和 lowering 也实现 rank >= 2 及 batch broadcast | LLVM batch-broadcast 数值测试已实现但本机未运行 | reduction/nested-loop unsupported | rank < 2、K/dtype 不匹配、batch 不可广播、缺失 metadata 或声明 output 不一致在 frontend/Relay gate fail closed |
| embedding/gather | contracted，未实现 | 未实现 | 未实现 | ONNX `Gather` 仍拒绝；OOB policy 未冻结 |
| mask/select | 未实现；exact slice 只把有限 additive mask 作为输入并复用 `add` | 仅 CPU causal finite-mask reference | 未实现 | 不等同于 `Where`、padding valid extent 或 all-masked 支持 |
| normalization | contracted，未实现正式 norm op | 未实现 | 未实现 | epsilon、axis、accumulation dtype 尚无完整证据 |
| slice/concat | contracted，未进入 Relay support matrix | 未实现 | 未实现 | 空 slice、bounds、alias/copy 未冻结 |
| exact prefill | Relay composition 已实现：batched matmul + finite additive mask + stable softmax + batched matmul | finite-logit exact-static CPU reference fixture only；LLVM/runtime 测试已实现但本机未运行 | unsupported | 不是完整 Transformer 执行；只覆盖固定静态 profile，不支持 bucket/tail/valid extent |
| decode external K/V | token=1、外部静态 K/V composition 已实现 | finite-logit external-K/V CPU reference fixture only；LLVM/runtime 测试已实现但本机未运行 | unsupported | 不是 KV-cache、runtime lifetime 或完整 Transformer 支持 |
| KV cache | 只有 append/page/capacity mock contract | reference/mock trace only | 未实现 | 没有 compiler/runtime KV-cache 能力；capacity 绝不冒充 valid context |
| dynamic batching | 未实现 | 未实现 | 未实现 | 控制面/dispatch key 未冻结；不进入 `RuntimeSession` |
| copy/event | contracted，未完成 NLP plan integration | CPU no-op 语义未作为本轨开放能力 | 通用异步 CUDA 基础已有既有测试，但无 KV task/event 证据 | 不宣称 decode copy/compute overlap |

### 2.1 ONNX importer 当前映射

| ONNX | Relay | 限制 |
|---|---|---|
| `Conv` | `nn_conv2d` | 既有静态 MVP |
| `Relu` | `nn_relu` | 既有静态 MVP |
| `MaxPool` | `nn_max_pool2d` | 既有静态 MVP |
| `Add` | `add` | 既有静态 MVP |
| `GlobalAveragePool` | `nn_global_avg_pool2d` | 既有静态 MVP |
| `Flatten` | `nn_flatten` | 既有静态 MVP |
| `Gemm` | `nn_gemm` | 既有静态 MVP |
| `MatMul` | `matmul` | frontend 要求两个可解析静态输入、rank >= 2、同 dtype/K、leading batch 静态广播及声明 output 一致；缺失 metadata 拒绝 |
| `Softmax`（opset >= 13） | `softmax` | 单 axis；缺省为 `-1`。opset < 13 flatten-from-axis 语义拒绝 |
| `Transpose` | `transpose` | 缺省 `perm` 使用 Relay 逆序规则 |

其余 Transformer 常见 ONNX op（例如 `Gather`、`Where`、`Slice`、`Concat`、norm decomposition 所需完整集合）仍由 importer 明确拒绝，不能通过 host fallback 或跳过节点获得绿灯。

### 2.2 Shape 与 negative gate

- Python protobuf `import_onnx(..., default_batch=None)` 是默认行为：缺失 shape field 的 ONNX unknown rank 以及 symbolic/unknown dim 立即报错；合法显式零维 shape 仍表示 scalar。
- 显式正数 `default_batch` 只允许绑定 unresolved axis 0；非 batch unresolved dim 即使提供 batch binding 仍拒绝。
- 每个 ONNX `MatMul` 在映射前解析两个输入的静态 TensorSpec（graph input/value_info、initializer 或此前推导的 MatMul output）；metadata 缺失/未解析、rank < 2、dtype/K/batch 不兼容以及声明 output 不一致都拒绝，且不推导无关算子。
- C++ `kxc.onnx_import.v1` JSON reifier 不能表示 ONNX unknown rank；input/output 必须提供非负整数静态 `shape`，缺失/负数/非整数维度拒绝，合法零维保留；`InferTypePass` 推导出的 output shape/dtype 必须与 JSON 声明一致。
- C++ API 中 `TensorType` 仍可表示负 extent，但 legacy whole-graph 与 production per-unit lowering 都在生成 TIR 前拒绝；这不是 dynamic shape 实现。
- all-masked fixture 只在独立 reference checker 中抛 unsupported，作用是阻止验收误开 gate；现有 unmasked Relay softmax 没有 mask 输入，也不宣称会拒绝全 `-inf` 数据。
- KV reference 分离 logical extent、physical capacity 和 valid extent；以 capacity 作为 valid context 的请求被拒绝。
- workload fingerprint 改动或不匹配由 checker 拒绝。
- CUDA softmax/matmul 在 Compiler reduction scheduling gate 拒绝；没有 silent CPU fallback。

## 3. 测试与证据

### 3.1 Phase A reference / manifest

```bash
python3 python/tools/check_nlp_gpu_validation.py --root .
cmake --build out/build/dev-ninja --target check_nlp_gpu_validation
```

结果：**PASS**。

覆盖：

- extreme-logit max-subtraction softmax；
- all-masked unsupported 负例；
- exact causal prefill CPU reference；
- token=1 external-K/V decode CPU reference；
- KV append/page/capacity trace；
- checker-level unknown extent、capacity-as-context、capacity overflow、CUDA capability 状态和 fingerprint mismatch 负例。

这些是 manifest/reference contract checks，不冒充 importer、Relay 或 CUDA 执行证据；实际 frontend/lowering/CUDA 路径分别由后续小节的 C++/device tests 覆盖。

### 3.2 Contract、架构与头文件

```bash
cmake --build out/build/dev-ninja --target \
  check_relay_op_contract check_pass_contract \
  check_include_layers check_public_headers
```

结果：

- Relay operator contract：**19/19 PASS**；
- Pass contract：**19/19 PASS**；
- include-layer：**209 files scanned, PASS**；
- public headers：**81 headers compiled, PASS**。

### 3.3 CPU/TIR/runtime regression

完整 CUDA-enabled、LLVM-disabled build 成功：

```bash
cmake --build --preset dev-ninja -j2
```

下列 20 个可执行测试通过：

- `object_test`
- `packed_func_test`
- `registry_test`
- `type_registration_test`
- `pass_pipeline_test`
- `device_info_test`
- `device_runtime_test`
- `executable_plan_test`
- `graph_partition_test`
- `infer_type_test`
- `onnx_import_spec_contract_test`
- `profile_bundle_test`
- `compiler_contract_test`
- `operator_compilation_test`
- `compiler_extension_contract_test`
- `kernel_signature_test`
- `compiled_module_test`
- `runtime_session_test`
- `cuda_schedule_test`
- `codegen_cuda_test`

`infer_type_test` 包含 batched matmul 普通 batch、leading-dimension broadcast、rank-2/rank-3 混合，以及 rank、K、batch 不兼容负例；batched lowering 到 TIR 通过。它还验证 softmax rank/dtype/axis contract，并验证负 extent 在 whole-graph 和 per-unit lowering 两条路径均被拒绝。

`onnx_import_spec_contract_test` 不依赖 Python ONNX 包，实际运行 C++ reifier，覆盖 batch-broadcast MatMul→Softmax→Transpose、负 input dimension，以及 `InferTypePass` 后声明 output shape/dtype mismatch。

### 3.4 LLVM

本机 CMake 配置结果：`LLVM_DIR=LLVM_DIR-NOTFOUND`，且无 `llvm-config`。因此：

- `op_numeric_llvm_test` target 未生成；
- 新增的 stable softmax 极值、batched matmul、exact prefill、external-K/V decode 数值测试**未在本机执行**；
- 只执行了无链接语法检查：

```bash
c++ -std=c++17 -Iinclude -Ithird_party/dlpack/include \
  -DKXC_USE_LLVM=0 -fsyntax-only test/op_numeric_llvm_test.cpp
```

结果：**PASS**。这不是 LLVM 数值通过证据，能力矩阵保持 `implemented` 而非 `validated`。

### 3.5 ONNX importer

依赖无关的 C++ import-spec contract test 已通过，见 3.3。本机缺少预装的 `onnx`、`numpy`、`pytest`，按约束没有安装或联网。因此 Python protobuf importer 部分：

- `onnx_importer_test` CMake target 未生成；
- `test/onnx_importer_py_test.py` 未执行；
- importer、CLI、测试文件执行 `python3 -m py_compile`：**PASS**；
- `kxc_frontend_obj` 编译：**PASS**。

新增 Python 测试代码覆盖 unknown input/output rank、symbolic batch 显式绑定、非 batch unresolved 拒绝、合法 scalar/零 extent、MatMul/Softmax/Transpose 映射、MatMul rank-1/K/batch/dtype/metadata/output 负例，以及 rank-2/rank-3 Softmax 的 opset < 13 拒绝与 opset 13 缺省 axis；这些 Python tests 在本机仍未运行，需要具备本地依赖的环境复验后才能升级为 `validated`。

### 3.6 CUDA

环境：

- GPU：NVIDIA GeForce GTX 1650；
- driver：580.159.03；
- compute capability：7.5；
- CUDA SDK/NVRTC：12.9；
- `nvcc` 不在 PATH，但本项目使用 NVRTC/Driver 路径，CMake 检测为 `CUDA support: enabled`。

```bash
./out/build/dev-ninja/cuda_schedule_test
./out/build/dev-ninja/codegen_cuda_test
```

结果：**PASS**。

`codegen_cuda_test` 的实际设备证据包括既有 add/relu/constant、异步 lifetime、batch module、byte-offset launch；本轨新增 `compiler_rejects_reduction_graphs`，验证 softmax 和 batched matmul 因 reduction/nested-loop schedule 未实现而明确失败。没有 CUDA softmax/matmul/attention 数值结果。

`cupti_smoke_test` 返回 1，诊断为：

```text
Manifest did not mark CUPTI as available
```

因此没有 CUPTI profile 证据，未把该失败伪装为通过或设备不可用 skip。

## 4. 提交

相对基线 `e295a73` 的主要实现/证据 Conventional Commits 如下。handoff 文档自身的 docs-only 修订不做不稳定的自引用；完整列表使用 `git log --oneline e295a73..HEAD` 查询。

| Commit | 内容 |
|---|---|
| `961a81c` | `test(nlp): add reference manifests and negative gates` |
| `4fa3c0c` | `fix(onnx): reject unresolved tensor dimensions` |
| `81499e4` | `fix(relay): stabilize softmax with max subtraction` |
| `cb0b75f` | `feat(nlp): add exact attention validation slice` |
| `386cf9d` | `test(nlp): gate unsupported mask semantics` |
| `0b68754` | `fix(nlp): enforce static execution capability gates` |
| `c95da8c` | `fix(onnx): validate static MatMul contracts` |

本次更新本文件的提交仅同步最终证据，不改变实现能力。

## 5. 环境与执行限制

| 项目 | 状态 | 影响 |
|---|---|---|
| OS | Linux 6.17.0-40-generic x86_64 | 无 |
| C++ | GCC 13.3.0，C++17 | build 通过 |
| CMake | 3.28.3 | configure/build 通过 |
| Python | 3.12.3 | stdlib reference/checker 通过 |
| LLVM | 未找到 `LLVMConfig.cmake` / `llvm-config` | LLVM 数值测试硬阻塞 |
| ONNX Python deps | `onnx`、`numpy`、`pytest` 均缺失 | importer Python/runtime fixture 测试硬阻塞 |
| CUDA | GTX 1650，driver 580.159.03，CUDA/NVRTC 12.9 | 既有 elementwise device tests 通过 |
| CUDA reduction schedule | 未实现 | softmax、matmul、attention CUDA 必须 gate off |
| CUPTI availability | smoke manifest 未标 available | 无 profiler artifact |
| 网络/依赖安装 | 未使用 | 无外部模型下载、无安装、无联网 |
| Git 操作 | 仅本分支本 worktree 原子提交 | 未 push、未 merge、未修改其他 worktree |

## 6. 跨轨门禁

| 所属轨 | 本轨等待的冻结能力 | 未满足时本轨行为 |
|---|---|---|
| 01 core contracts | production capability verifier、统一 target/backend fingerprint | 保持本地 manifest checker；不能将 importer 可构造等同于可执行 |
| 02 shape | symbolic binding、exact profile、logical/physical/valid extent；后续 bucket applicability | importer 对 unresolved rank/dim fail closed；Relay unknown extent 可表示但 lowering 拒绝；prefill 只开放 exact-static reference |
| 03 adaptive | dispatch key、immutable selected generation、singleflight/backpressure | dynamic batching 只保留 gate；不进入 `RuntimeSession`，不 fuzzy fallback |
| 04 control flow | 仅真实模型需要 If/loop 时消费结构化 control-flow contract | 当前纯 dataflow fixture 不声称 dynamic graph |
| 05 region/runtime plan | KV capacity/page、alias/effect、lifetime、copy/event task、valid-context metadata | KV cache 只运行 reference/mock trace；不改 `ValueSpec` 或复用 storage 冒充 cache |
| CUDA backend | reduction-safe scheduling、tail/mask correctness、target-matched device numeric tests | Compiler 明确拒绝 softmax/matmul；只保留已验证 add/relu CUDA 路径 |
| Frontend/operator | gather、mask/select、norm、slice/concat 完整 contract/type/lowering/backend tests | capability matrix 保持 unsupported/contracted；不跳节点、不 host fallback |

## 7. 剩余硬阻塞与下一步

当前只记录 finite-logit exact-static CPU reference baseline 和 fail-closed 验证资产；Transformer、LLVM/`RuntimeSession` 数值与 GPU 端到端完成门禁仍未满足。继续开放以下能力必须先解除对应硬门禁：

1. **LLVM 数值复验：** 提供本地 LLVM package 后运行 `run_op_numeric_llvm_test`，确认 stable softmax、batched matmul、prefill、decode external-K/V；通过后才把 LLVM 状态升级为 `validated`。
2. **ONNX importer 复验：** 在已有 `onnx/numpy/pytest` 的环境运行 Python importer tests 和 C++ generated fixture；不得由本轨安装依赖。
3. **真实 CUDA attention：** 需要 reduction/nested-loop CUDA schedule 或受支持 library region；随后补 target-matched softmax/matmul device numeric 和端到端 prefill/decode。当前明确拒绝是正确结果。
4. **真实 KV cache：** 等待 logical/physical/valid extent、page/capacity、alias/effect 和 dependency-aware lifetime 冻结；当前 mock 不能升级为 runtime 支持。
5. **prefill bucket/dynamic batching：** 等待 ShapeProfile/dispatch/PlanVariant；不得用“大 buffer 可运行”或 `cached >= query` 替代 applicability 证明。
6. **通用 Transformer frontend：** 逐个垂直补齐 gather、mask/select、normalization、slice/concat 的 schema/type/lowering/LLVM/CUDA/负例，不以新增 op 名数量替代验证。
7. **CUPTI/profile：** 先修复 profiling manifest 的 CUPTI availability，再采集设备 artifact；当前无性能结论。

以上阻塞均位于环境或其他 foundation 轨冻结契约之外。本轨没有通过修改核心 runtime/session 抽象绕过门禁。
