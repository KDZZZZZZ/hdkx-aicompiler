# Relay 算子支持矩阵

> **状态：Track01 仓内 closure 已实现，逐 target 批准仍进行中。** 本页中的
> contract/numeric 记录不能替代 `Compiler::Compile` per-unit executable proof 与
> backend CI。未列出或未获目标批准的算子必须 fail closed。

## 判定规则

`contracts/relay_op_contract.json` 是 MVP operator contract 的机器可读输入。一个算子只有
在 schema、type、lowering binding、FFI、per-unit compile、目标后端数值结果和 CI
均有可追溯证据时，才可由审查者标为 production supported。

`relay::LowerToTIR` 的单 whole-graph `PrimFunc` 测试只属于 compatibility/testing；它
不能证明 per-unit `Compiler::Compile`、artifact cache 或 `ExecutablePlan` 路径支持该
算子。

## 当前 contract 范围

下列 24 个算子已列入 Relay contract checker。此表只声明 checker 范围，**不是**
production support 的肯定结论；production approval 保持 pending，直到下方 closure
项目完成。

| 算子 | 分类 | Contract | Production approval |
|---|---|---|---|
| `add` | tensor.math | declared | pending |
| `subtract` | tensor.math | declared | pending |
| `mul` | tensor.math | declared | pending |
| `divide` | tensor.math | declared | pending |
| `where` | tensor.math | declared | pending |
| `sqrt` | tensor.math | declared | pending |
| `matmul` | tensor.math | declared | pending |
| `nn_dense` | nn | declared | pending |
| `nn_gemm` | nn | declared | pending |
| `nn_layer_norm` | nn | declared | pending |
| `nn_relu` | nn | declared | pending |
| `nn_conv2d` | nn | declared | pending |
| `nn_max_pool2d` | nn | declared | pending |
| `nn_avg_pool2d` | nn | declared | pending |
| `nn_global_avg_pool2d` | nn | declared | pending |
| `nn_flatten` | tensor.transform | declared | pending |
| `reshape` | tensor.transform | declared | pending |
| `transpose` | tensor.transform | declared | pending |
| `gather` | tensor.transform | declared | pending |
| `cast` | tensor.transform | declared | pending |
| `concatenate` | tensor.transform | declared | pending |
| `slice` | tensor.transform | declared | pending |
| `reduce_mean` | tensor.reduce | declared | pending |
| `softmax` | nn | declared | pending |

## 当前显式拒绝边界

- `nn_gemm` 的 `transA != 0`：type relation 可表达，但 production TE lowering 明确拒绝；capability 返回 `eligible_but_not_executable`，不得视为 executable。
- function tuple parameter：当前 value graph ABI 只接受 `TensorType` parameters；tuple outputs/flat multi-output 不等于 tuple parameter support。
- LLVM/CUDA backend 未编入当前构建：结构可 eligible，但 status 为 `eligible_but_not_executable`。
- CUDA Target 必须有可用 device、compute capability 与 launch limits；O3 reduction 不能通过当前保守 `bind_cuda_threads` schedule，因此在 backend 前返回 target-schedule rejection。Gather 的间接 `Load` 索引同样由通用 schedule gate 在 backend 前拒绝。
- custom lowering 的 tensor 数量、dtype、rank 或 shape 与 checked type 不一致时，由 production per-unit lowering 与 capability 同源拒绝。

## Required CI evidence

CPU CI configures with CUDA and LLVM disabled, then runs every CTest labelled
`cpu`: all built CPU/core C++ tests, Relay/pass contracts, include-layer checks,
and compiled public-header checks. It does not run conditional LLVM, CUDA, CUPTI,
or hardware tests.

LLVM CI explicitly builds and runs:

```bash
ctest --test-dir out/build/ci-llvm --output-on-failure \
  --tests-regex '^(operator_compilation_test|codegen_llvm_test|op_numeric_llvm_test|onnx_importer_test)$'
```

`operator_compilation_test` contains the LLVM numerical per-unit and primitive-cache
renumbered/symbol-relocation reuse assertions. `codegen_llvm_test` and
`op_numeric_llvm_test` provide the relevant LLVM codegen and operator numerical
coverage. `onnx_importer_test` conditionally includes a real protobuf exact-Transformer chain:
repository Python generates/validates a `ModelProto`, writes and reloads `.onnx`, runs the Python
importer/serializer, then the C++ test reifies, compiles nine LLVM units and executes
`RuntimeSession` numerically. This path has no skip switch in an LLVM+onnx/numpy build.

## Closure checklist

Track01 的仓内 static-exact contract 已进入集成验证，但各算子的 production
approval 仍按 target 单独判断。NLP 轨新增的实现边界同样不改变这一规则：

- `matmul` 的 type/TE contract 已扩展为 rank >= 2，并按 ONNX/NumPy 规则广播 leading batch dimensions；LLVM 数值测试源码存在，但本机尚无 LLVM 绿色记录，CUDA reduction/nested-loop schedule 仍拒绝。
- `softmax` 使用 max-subtraction 改善有限 logits 的数值稳定性；masked/all-masked 和非有限输入策略仍未支持，CUDA reduction schedule 仍拒绝。
- ONNX opset < 13 Softmax 的 trailing-flatten 语义不能直接映射为当前 Relay 单轴 softmax，因此 importer fail closed。
- 通用 Relay `gather` 支持 int32/int64 runtime indices，并保留 KXC-only OOB typed zero-fill 扩展。默认 ONNX 映射更窄：只接受 initializer-backed indices，Python importer 与 C++ reifier 都逐值验证 `[-extent, extent-1]`；动态或 OOB indices fail closed。LLVM 的 Relay 扩展 numeric source 保留，CUDA 通用间接-Load gate 保持拒绝。
- `where` 的静态 ONNX/Relay/TE vertical slice 已覆盖三输入 trailing-axis 广播：condition 必须为 `bool`，x/y 必须同 dtype，branch dtype 受限。LLVM numeric 源码存在但本机未运行。CUDA 状态仅为 `implemented/local-evidence`：仓内有非空 1-D Compiler/RuntimeSession source fixture，但没有提交的 GPU CI/local-run artifact，不能标 `validated`；多维 fixture 在 target schedule fail closed。`where` 不定义 masked-softmax 或 all-masked-row 行为。
- `slice` 是 exact-static positive-step vertical slice：实际提供的 starts/ends/axes/steps initializer 必须统一为 int32 或统一为 int64，之后 canonical attrs 全部为 int64；数组非空等长、axis 唯一、严格 `step == +1`。identity 与 empty 均 lower 为 fresh copy。LLVM numeric 源码覆盖正常、identity、empty。CUDA 仅有非空 1-D source/local evidence，状态为 implemented；多维在 schedule fail closed，empty-output 不声明。
- `concatenate` 是 exact-static binary vertical slice：显式 axis、同 rank/dtype、非 axis 维相等、checked axis sum 和零 extent 均有 contract。Python importer、canonical attrs/type registration/FFI、C++ contract 和 LLVM numeric 源码均有证据；本机未运行 LLVM。CUDA 仅有非空 1-D source/local evidence，状态为 implemented；多维在 schedule fail closed，empty-output 不声明。
- `nn_layer_norm` 是 exact-static affine LayerNorm：输入/Y 为 float32，epsilon 为有限正 float32 attr；sum、mean、variance、sqrt 与 affine 内部阶段固定 float64，最后 cast Y 回 float32。LLVM source 覆盖 `[FLT_MAX,FLT_MAX]`、巨大 offset、multi-axis、axis 0 和 prefix-zero。ONNX 接受 `[Y]`、`[Y,""]`、`[Y,"",""]`，serialized spec 只携 Y；非空 Mean/InvStdDev 请求拒绝。CUDA nested reduction 保持 fail closed。

- shared static TE-to-TIR contract 同时覆盖 whole-graph 和 production per-unit 路径：每个 data/reduction iter extent 必须 `<= INT32_MAX`；row-major element count/flatten max 和 byte count 使用 checked arithmetic，并同时受 `INT64_MAX` 与 `size_t` 限制；所有维度先验证，任一零维使 elements/bytes 为零但不会绕过非法 sibling extent。Slice/Concat/Flatten 均受同一 gate。

- [x] per-unit executable capability 正反例（含真实 lowering/schedule/backend proof）
- [x] normalized production pipeline、executable invariant 与 public static-exact transaction/pin adapter
- [x] 本地 CPU CTest、contract、include/public-header：38/38
- [x] LLVM workflow 选择 relocation/cache-reuse 与 numeric/codegen 测试
- [ ] LLVM-enabled builder 的实际绿色记录（workflow 已配置；本机/当前会话无记录）
- [ ] 每个 target 的 attrs/shape/schedule 限制逐项批准

TOPI helpers outside this table are not supported merely because declarations exist.
They must explicitly reject unsupported inputs rather than return an empty tensor,
placeholder result, or silent substitute.
