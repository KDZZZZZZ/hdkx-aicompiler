# 5. Frontend / Tests / CI API

本页描述 Relay op 的 public 构图入口、ONNX importer 和测试/CI 接入要求。

## FFI helper

位置：[src/relay/op/op_ffi.cc](../../src/relay/op/op_ffi.cc)

接口：

```cpp
KXC_REGISTER_GLOBAL("kxc.relay.op._make.<op>").set_body(ToPackedFunc(MakeXxx));
```

helper 规则：

| 项 | 要求 |
| --- | --- |
| 名称 | `_make.<op>` 必须等于 canonical op name |
| 输入 | `Call(GetOp("<op>"), {inputs})` 数量必须和 contract 一致 |
| attrs | 只构造 `XxxAttrs::Create(...)`；不在 helper 中做 type inference |
| alias | 不允许旧 helper 名转调新 op |
| unsupported | 不支持的前端形态要抛错，不要生成错误 Relay |

## ONNX importer

位置：[python/kxc_onnx/importer.py](../../python/kxc_onnx/importer.py)

当前映射：

```python
ONNX_TO_RELAY = {
    "Conv": "nn_conv2d",
    "Relu": "nn_relu",
    "MaxPool": "nn_max_pool2d",
    "Add": "add",
    "GlobalAveragePool": "nn_global_avg_pool2d",
    "Flatten": "nn_flatten",
    "Gemm": "nn_gemm",
}
```

新增 ONNX op 时必须同步：

| 步骤 | 文件 | 要求 |
| --- | --- | --- |
| 1 | `contracts/relay_op_contract.json` | 在目标 Relay op 的 `onnx_ops` 中写 ONNX op 名 |
| 2 | `python/kxc_onnx/importer.py` | `ONNX_TO_RELAY` 增加映射 |
| 3 | `python/kxc_onnx/importer.py` | `_convert_attrs` 把 ONNX attrs 转为 Relay attrs |
| 4 | tests | importer 测试覆盖新映射 |

当前 Python importer 只支持单输出 ONNX node。多输出 ONNX op 要先扩 importer spec、Relay `TupleType` 映射和测试。

## C++ 测试入口

| 测试 | 文件 | 覆盖 |
| --- | --- | --- |
| type inference | `test/infer_type_test.cpp` | `InferTypePass` 和 `LowerToTIR` 契约 |
| LLVM numeric | `test/op_numeric_llvm_test.cpp` | Relay -> TIR -> LLVM JIT -> runtime 数值 |
| ONNX importer | `test/onnx_importer_test.cpp`, `test/onnx_importer_py_test.py` | ONNX -> Relay spec / C++ 编译 |
| pass pipeline | `test/pass_pipeline_test.cpp` | pass 对 Relay IR 的行为 |
| codegen LLVM | `test/codegen_llvm_test.cpp` | 后端基础能力 |

新增可执行 Relay op 时，优先把 per-op numeric case 加到 `test/op_numeric_llvm_test.cpp`。只加 `LowerToTIR` 不够说明 backend 能运行。

## Checker 需要看到的测试信号

checker 静态扫描 `test` 目录：

| contract 状态 | 必须出现的信号 |
| --- | --- |
| `tests = true` | 测试中有 `"<op>"` 字符串引用 |
| `lowering = single` / `multi` | 同一个测试函数块里有 `"<op>"` 和 `LowerToTIR` |
| backend required | 同一个测试函数块里有 `"<op>"` 和 `Compiler::Compile` / `CompileAndRun` / LLVM 信号 |

为了让扫描准确，不要把一堆 op name 放在无关 helper 或全局注释里冒充测试覆盖。

## CI 入口

位置：[.github/workflows/ci.yml](../../.github/workflows/ci.yml)

CI 当前执行的验证命令：

| Job | 命令 |
| --- | --- |
| `relay-op-contract` | `python -m py_compile python/tools/check_relay_op_contract.py` |
| `relay-op-contract` | `python python/tools/check_relay_op_contract.py --root .` |
| `cpu-smoke` | `cmake -S . -B out/build/ci-cpu -G Ninja ... -DKXC_ENABLE_LLVM=OFF ...` |
| `cpu-smoke` | `cmake --build out/build/ci-cpu --parallel 2` |
| `cpu-smoke` | `cmake --build out/build/ci-cpu --target run_infer_type_test` |
| `cpu-smoke` | `cmake --build out/build/ci-cpu --target run_pass_pipeline_test` |
| `cpu-smoke` | `cmake --build out/build/ci-cpu --target run_profile_bundle_test` |
| `llvm-resnet18` | `python -m pip install --upgrade pip onnx numpy` |
| `llvm-resnet18` | `cmake -S . -B out/build/ci-llvm -G Ninja ... -DKXC_ENABLE_LLVM=ON ...` |
| `llvm-resnet18` | `cmake --build out/build/ci-llvm --target run_op_numeric_llvm_test --parallel 2` |
| `llvm-resnet18` | `cmake --build out/build/ci-llvm --target run_onnx_importer_test --parallel 2` |
| `python-onnx-importer` | `python -m pip install --upgrade pip pytest onnx numpy` |
| `python-onnx-importer` | `PYTHONPATH=python python -m pytest test/onnx_importer_py_test.py -q` |

本地最小验证：

```bash
python python/tools/check_relay_op_contract.py --root .
cmake --build out/build/<cpu-build> --target run_infer_type_test
cmake --build out/build/<cpu-build> --target run_pass_pipeline_test
cmake --build out/build/<cpu-build> --target run_profile_bundle_test
cmake --build out/build/<llvm-build> --target run_op_numeric_llvm_test
```

如果改了 ONNX importer、ONNX 映射或 ONNX attrs 转换，还要同时运行 C++ importer/LLVM 编译测试和 Python importer 测试：

```bash
cmake --build out/build/<llvm-build> --target run_onnx_importer_test
$env:PYTHONPATH = "python"  # PowerShell
python -m pytest test/onnx_importer_py_test.py -q
```

Linux/macOS 使用：

```bash
cmake --build out/build/<llvm-build> --target run_onnx_importer_test
PYTHONPATH=python python -m pytest test/onnx_importer_py_test.py -q
```

## 新 op PR checklist

| 项 | 检查 |
| --- | --- |
| contract | op 在 `contracts/relay_op_contract.json` 中声明，字段和实现一致 |
| Relay | 注册块完整，canonical name 唯一 |
| attrs | `TAttrs`、`XxxAttrs::Create`、FFI 参数一致 |
| type | `FInferType` 覆盖正常和失败路径 |
| TOPI/TE | helper 不假实现，生成后端支持的 TIR |
| lowering | `FRelayToTE` / `FRelayToTEMulti` 或 execution plan 路径正确 |
| FFI | `_make.<op>` 名称和 `GetOp("<op>")` 一致 |
| ONNX | contract、mapping、attrs conversion、tests 同步 |
| tests | type、lowering、backend numeric 按支持级别覆盖 |
| CI | checker 和相关 CMake target 本地通过 |
