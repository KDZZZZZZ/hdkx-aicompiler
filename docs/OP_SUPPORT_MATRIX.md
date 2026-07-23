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

下列 19 个算子已列入 Relay contract checker。此表只声明 checker 范围，**不是**
production support 的肯定结论；production approval 保持 pending，直到下方 closure
项目完成。

| 算子 | 分类 | Contract | Production approval |
|---|---|---|---|
| `add` | tensor.math | declared | pending |
| `subtract` | tensor.math | declared | pending |
| `mul` | tensor.math | declared | pending |
| `divide` | tensor.math | declared | pending |
| `sqrt` | tensor.math | declared | pending |
| `matmul` | tensor.math | declared | pending |
| `nn_dense` | nn | declared | pending |
| `nn_gemm` | nn | declared | pending |
| `nn_relu` | nn | declared | pending |
| `nn_conv2d` | nn | declared | pending |
| `nn_max_pool2d` | nn | declared | pending |
| `nn_avg_pool2d` | nn | declared | pending |
| `nn_global_avg_pool2d` | nn | declared | pending |
| `nn_flatten` | tensor.transform | declared | pending |
| `reshape` | tensor.transform | declared | pending |
| `transpose` | tensor.transform | declared | pending |
| `cast` | tensor.transform | declared | pending |
| `reduce_mean` | tensor.reduce | declared | pending |
| `softmax` | nn | declared | pending |

## 当前显式拒绝边界

- `nn_gemm` 的 `transA != 0`：type relation 可表达，但 production TE lowering 明确拒绝；capability 返回 `eligible_but_not_executable`，不得标 supported。
- function tuple parameter：当前 value graph ABI 只接受 `TensorType` parameters；tuple outputs/flat multi-output 不等于 tuple parameter support。
- LLVM/CUDA backend 未编入当前构建：结构可 eligible，但 `supported=false`。
- CUDA Target 必须有可用 device、compute capability 与 launch limits；O3 reduction 不能通过当前保守 `bind_cuda_threads` schedule，因此在 backend 前返回 target-schedule rejection。
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

Track01 现为 **ready for supervisor re-review**，但各算子的 production approval
仍按 target 单独判断：

- [x] per-unit executable capability 正反例（含真实 lowering/schedule/backend proof）
- [x] normalized production pipeline、executable invariant 与 public static-exact transaction/pin adapter
- [x] 本地 CPU CTest、contract、include/public-header：27/27
- [x] LLVM workflow 选择 relocation/cache-reuse 与 numeric/codegen 测试
- [ ] LLVM-enabled builder 的实际绿色记录（workflow 已配置；本机/当前会话无记录）
- [ ] 每个 target 的 attrs/shape/schedule 限制逐项批准

TOPI helpers outside this table are not supported merely because declarations exist.
They must explicitly reject unsupported inputs rather than return an empty tensor,
placeholder result, or silent substitute.
