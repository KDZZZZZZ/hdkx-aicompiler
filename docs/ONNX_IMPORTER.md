# ONNX Importer

本文档记录当前 ONNX 到 Relay 的正式导入路径。该 importer 面向静态 shape 的 MVP，目标是让 `resnet18.onnx` 可以导入为规范化 Relay Function 描述和真实参数张量绑定信息，不再把临时 C++ 构图脚本作为默认导入方式。

## Python API

入口位于 `python/kxc_onnx`：

```python
from kxc_onnx import import_onnx, save_imported_model

imported = import_onnx("resnet18.onnx", default_batch=1)
save_imported_model(
    imported,
    "out/resnet18.import.json",
    "out/resnet18.params.bin",
)
```

`import_onnx(model_path, default_batch=None)` 返回 `ImportedONNXModel`：

- `function`：结构化的 Relay function spec，包含 inputs、outputs 和 nodes。
- `params`：`dict[str, ParamTensor]`，保存每个 ONNX initializer 的 shape、dtype 和真实 tensor bytes。
- `param_order`：ONNX initializer 原始顺序，用于稳定序列化和 runtime 参数绑定。

如果需要对内存中的 ONNX `ModelProto` 做测试，可使用 `import_onnx_model(model, default_batch=None, base_dir=None)`。

## 支持的 MVP 算子

当前 ONNX op 到 Relay canonical op 的映射如下：

| ONNX op | Relay op |
|---|---|
| `Conv` | `nn_conv2d` |
| `Relu` | `nn_relu` |
| `MaxPool` | `nn_max_pool2d` |
| `Add` | `add` |
| `GlobalAveragePool` | `nn_global_avg_pool2d` |
| `Flatten` | `nn_flatten` |
| `Gemm` | `nn_gemm` |
| `MatMul` | `matmul` |
| `Softmax` | `softmax` |
| `Transpose` | `transpose` |

`Conv`、`MaxPool`、`Flatten`、`Gemm` 会转换必要 attrs；`Softmax` 转换 `axis`（未显式指定时 opset < 13 为 `1`，否则为 `-1`）；`Transpose` 转换 `perm`（缺失时写为空数组，由 Relay 使用逆序默认）；`Relu`、`Add`、`GlobalAveragePool`、`MatMul` 使用简单 attrs。

## Shape 与 dtype 行为

该 importer 按静态 shape MVP fail-closed 地处理 ONNX value info：

- 已知 `dim_value` 会按整数保留，包括 `0`。
- symbolic 或 unknown 维度默认会拒绝导入，并在错误中给出 tensor/value 名称、axis 和（适用时）`dim_param`。
- 仅当调用方显式传入正数 `default_batch`（CLI 为 `--batch N`）时，未解析的 axis 0 才会绑定为该 batch 值；所有非 batch 未解析维度仍会拒绝。
- C++ JSON reifier 同样只接受非负整数静态 shape，拒绝负数或非整数维度；它不支持 symbol runtime 语义。

当前支持的 tensor dtype：

- `float32`
- `float64`
- `int64`
- `int32`
- `int8`
- `uint8`
- `bool`

不支持的 ONNX tensor dtype 会抛出 `ValueError`。

## 参数序列化格式

`save_imported_model(imported, json_path, params_path)` 会写出两个文件：

- JSON：function spec、param metadata、`param_order`。
- BIN：所有 initializer bytes 按 `param_order` 顺序连续写入。

JSON 中每个参数记录包含：

```json
{
  "name": "fc.weight",
  "shape": [1000, 512],
  "dtype": "float32",
  "offset": 0,
  "nbytes": 2048000
}
```

runtime 或 C++ loader 通过 `offset` 和 `nbytes` 从 BIN 文件恢复 `runtime::NDArray`。

## C++ Reifier

C++ 入口位于 `include/frontend/onnx_importer.h`：

```cpp
#include "frontend/onnx_importer.h"

kxc::frontend::ImportedONNXModel imported =
    kxc::frontend::LoadONNXImportSpec(
        "out/resnet18.import.json",
        "out/resnet18.params.bin");
```

返回结果包含：

- `function`：真实 `kxc::Function`。
- `params`：`std::unordered_map<std::string, runtime::NDArray>`。
- `param_order`：稳定参数顺序。
- `input_names` / `output_names`：runtime binding 元数据。

C++ loader 会按 spec 构建 `Var`、`Constant(runtime::NDArray)` 和 `Call(Op::Get(...), args, attrs)`，并保留 initializer 真实 bytes。

## 错误信息

不支持的 ONNX op 会抛出 `UnsupportedONNXOpError`，消息包含 op type 和 node name，例如：

```text
Unsupported ONNX op 'Identity' in node 'bad_identity'
```

这便于定位模型中尚未纳入 MVP 的节点。

## 测试

相关测试：

```powershell
$env:PYTHONPATH="python"
python -m pytest test/onnx_importer_py_test.py -q

cmake --preset dev-mingw-cpu
cmake --build --preset dev-mingw-cpu --target onnx_importer_test
cmake --build out/build/dev-mingw-cpu --target run_onnx_importer_test
```

CMake 会在 build 目录自动生成 C++ 测试使用的 `resnet18.import.json` 和 `resnet18.params.bin` fixture。

## 当前限制

- 只覆盖静态 shape MVP，不承诺完整 ONNX opset；`MatMul` 支持 rank >= 2 的静态 batch broadcasting，动态维度仍不支持。
- 不引入 C++ ONNX/protobuf 依赖；ONNX protobuf 解析留在 Python 侧。
- 不提供动态 shape runtime 语义。
- 不做 ResNet18 数值执行验收；本阶段验收重点是导入 Relay Function、保留 params 数据、序列化 runtime binding 信息，以及清晰的 unsupported op 错误。
- `python/gen_resnet18_ir_dump_cpp.py` 仍保留为 debug codegen 工具，但它已经复用正式 importer spec；默认导入 API 是 `kxc_onnx.import_onnx` 和 C++ `LoadONNXImportSpec`。
