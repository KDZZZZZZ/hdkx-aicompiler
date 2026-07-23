# Relay 算子支持矩阵

> **状态：Track01 仓内 closure 已实现，逐 target 批准仍进行中。** 本页中的
> contract/numeric 记录不能替代 `Compiler::Compile` per-unit executable proof 与
> backend CI。未列出或未获目标批准的算子必须 fail closed。

## 判定规则

`test/relay_op_contract.json` 是 MVP operator contract 的机器可读输入。一个算子只有
在 schema、type、lowering binding、FFI、per-unit compile、目标后端数值结果和 CI
均有可追溯证据时，才可由审查者标为 production supported。

`relay::LowerToTIR` 的单 whole-graph `PrimFunc` 测试只属于 compatibility/testing；它
不能证明 per-unit `Compiler::Compile`、artifact cache 或 `ExecutablePlan` 路径支持该
算子。

## 当前 contract 范围

下列 22 个算子已列入 Relay contract checker。此表只声明 checker 范围，**不是**
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
| `reduce_mean` | tensor.reduce | declared | pending |
| `softmax` | nn | declared | pending |

## 当前显式拒绝边界

- `nn_gemm` 的 `transA != 0`：type relation 可表达，但 production TE lowering 明确拒绝；capability 返回 `eligible_but_not_executable`，不得标 supported。
- function tuple parameter：当前 value graph ABI 只接受 `TensorType` parameters；tuple outputs/flat multi-output 不等于 tuple parameter support。
- LLVM/CUDA backend 未编入当前构建：结构可 eligible，但 `supported=false`。
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
coverage. `onnx_importer_test` is the conditional ONNX fixture/import and LLVM
compile integration check.

## Closure checklist

Track01 的仓内 static-exact contract 已进入集成验证，但各算子的 production
approval 仍按 target 单独判断。NLP 轨新增的实现边界同样不改变这一规则：

- `matmul` 的 type/TE contract 已扩展为 rank >= 2，并按 ONNX/NumPy 规则广播 leading batch dimensions；LLVM 数值测试源码存在，但本机尚无 LLVM 绿色记录，CUDA reduction/nested-loop schedule 仍拒绝。
- `softmax` 使用 max-subtraction 改善有限 logits 的数值稳定性；masked/all-masked 和非有限输入策略仍未支持，CUDA reduction schedule 仍拒绝。
- ONNX opset < 13 Softmax 的 trailing-flatten 语义不能直接映射为当前 Relay 单轴 softmax，因此 importer fail closed。
- `gather` 的静态 type/TE contract 支持 int32/int64 indices 和 axis 归一化；ONNX 有效索引域为 `[-extent, extent - 1]`。KXC 对该域外的运行时索引作确定性的 typed zero-fill 扩展；LLVM numeric coverage 已接入但本机未运行，CUDA 通用间接-Load gate 保持拒绝。
- `where` 的静态 ONNX/Relay/TE vertical slice 已覆盖三输入 trailing-axis 广播：condition 必须为 `bool`，x/y 必须同 dtype，且 branch dtype 仅限 `{float32,float64,int32,int64,int8,uint8,bool}`。LLVM numeric 源码以 `uint8_t` 提供 byte-backed bool condition ABI；本机未运行 LLVM，CUDA 未支持且未验证。`where` 只是逐元素选择，**不定义 masked-softmax 或 all-masked-row 行为**。
- `nn_layer_norm` 是 exact-static affine LayerNorm：三个输入均为 float32，data rank >= 1 且非负静态，axis suffix 必须为正，scale/bias 必须严格等于该 suffix，epsilon 有限且 > 0，accumulation dtype 固定为 float32。ONNX 仅导入 opset >= 17 的单输出 `LayerNormalization`；CUDA nested reduction 由通用 schedule gate fail closed。

- [x] per-unit executable capability 正反例（含真实 lowering/schedule/backend proof）
- [x] normalized production pipeline、executable invariant 与 public static-exact transaction/pin adapter
- [x] 本地 CPU CTest、contract、include/public-header：27/27
- [x] LLVM workflow 选择 relocation/cache-reuse 与 numeric/codegen 测试
- [ ] LLVM-enabled builder 的实际绿色记录（workflow 已配置；本机/当前会话无记录）
- [ ] 每个 target 的 attrs/shape/schedule 限制逐项批准

TOPI helpers outside this table are not supported merely because declarations exist.
They must explicitly reject unsupported inputs rather than return an empty tensor,
placeholder result, or silent substitute.
