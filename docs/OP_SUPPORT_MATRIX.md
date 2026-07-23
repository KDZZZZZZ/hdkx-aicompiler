# Relay MVP 算子支持矩阵

本文档记录当前 TinyTVM MVP Relay 算子的实现状态。矩阵只覆盖 `test/relay_op_contract.json` 中声明的 MVP 算子；未进入矩阵的 TOPI/Relay helper 不视为已支持，调用时必须显式报错，不能返回空 `Tensor` 或错误替代实现。

## 支持判定

一个 MVP 算子只有同时满足以下条件，才能标记为 `tested`：

- schema：有唯一 canonical Relay op 注册，包含输入个数、参数说明和属性类型。
- type：注册 `FInferType`，能推导静态 shape/dtype，并对不支持形态立即失败。
- lowering：注册 `FRelayToTE`，生成定义完整的 TE tensor。
- ffi：有 canonical `_make.<op>` helper，不允许 alias helper。
- tir：有 `LowerToTIR` 测试引用。
- llvm：有 Relay -> TIR -> LLVM JIT -> 运行 -> 数值比对测试引用。
- ci：checker 和 CPU/LLVM 测试进入 CI。

## 检查命令

```bash
python python/tools/check_relay_op_contract.py --root .
cmake --build out/build/<llvm-build> --target run_op_numeric_llvm_test
cmake --build out/build/<llvm-build> --target run_onnx_importer_test
```

`run_op_numeric_llvm_test` 覆盖 19 个 MVP 算子的 per-op 数值测试，并包含 add 链、MLP、CNN 三个小模型级组合测试。`run_onnx_importer_test` 覆盖 ResNet18 ONNX 导入和 LLVM 编译路径；设置 `KXC_RUN_RESNET18_EXEC=1` 后会执行编译后的 ResNet18 kernel。

## MVP 矩阵

| 算子 | 分类 | 类型推断 | TE lowering | TIR 测试 | LLVM 数值测试 | 备注 |
|---|---|---:|---:|---:|---:|---|
| `add` | tensor.math | Y | Y | Y | Y | 支持 broadcast |
| `subtract` | tensor.math | Y | Y | Y | Y | 支持 broadcast |
| `mul` | tensor.math | Y | Y | Y | Y | Relay canonical 名称为 `mul` |
| `divide` | tensor.math | Y | Y | Y | Y | 支持 broadcast |
| `sqrt` | tensor.math | Y | Y | Y | Y | LLVM 走 intrinsic |
| `matmul` | tensor.math | Y | Y | Y | Y | rank >= 2，按 ONNX/NumPy 广播 leading batch dimensions；CUDA reduction/nested-loop scheduling 不支持 |
| `nn_dense` | nn | Y | Y | Y | Y | weight 形状为 `[N, K]` |
| `nn_gemm` | nn | Y | Y | Y | Y | 支持 `transB`，暂不支持 `transA=1` |
| `nn_relu` | nn | Y | Y | Y | Y | elementwise |
| `nn_conv2d` | nn | Y | Y | Y | Y | NCHW/OIHW，支持可选 bias |
| `nn_max_pool2d` | nn | Y | Y | Y | Y | NCHW |
| `nn_avg_pool2d` | nn | Y | Y | Y | Y | NCHW |
| `nn_global_avg_pool2d` | nn | Y | Y | Y | Y | NCHW |
| `nn_flatten` | tensor.transform | Y | Y | Y | Y | 静态 shape |
| `reshape` | tensor.transform | Y | Y | Y | Y | 静态 shape |
| `transpose` | tensor.transform | Y | Y | Y | Y | 支持负轴归一化 |
| `cast` | tensor.transform | Y | Y | Y | Y | LLVM 直接生成 cast 指令 |
| `reduce_mean` | tensor.reduce | Y | Y | Y | Y | 静态 reduction shape |
| `softmax` | nn | Y | Y | Y | Y | 使用 max-subtraction；LLVM 测试覆盖极大正负 logits。masked/all-masked 语义未支持 |

## 非 MVP helper 规则

`include/te/topi` 里的 helper 只有被 Relay MVP lowering 使用并进入矩阵后，才算已支持。其他 helper 必须遵守：

- 不允许 `return Tensor()` 作为失败路径。
- 不允许用其他 reducer 伪装实现，例如用 `sum` 代替 `prod`。
- 不允许保留 `TODO`、`placeholder`、`for now` 等占位标记。
- 当前不能实现的 helper 必须抛出明确异常。

当前状态：

- `topi::min` 已接入真实 `te::min` reducer。
- 未实现的 `topi::prod` 和 `topi::einsum` 公共 helper 已删除；调用方不能把“存在但只会抛错”误判为支持。
- `topi::concatenate` 的空输入路径显式失败。
