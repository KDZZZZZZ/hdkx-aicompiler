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
| `Softmax`（opset >= 13） | `softmax` |
| `Transpose` | `transpose` |
| `Gather` | `gather` |
| `Where` | `where` |
| `LayerNormalization`（opset >= 17） | `nn_layer_norm` |

`Conv`、`MaxPool`、`Flatten`、`Gemm` 会转换必要 attrs；opset >= 13 的 `Softmax` 转换单个 `axis`（缺省为 `-1`）。opset < 13 的 `Softmax` 会 fail closed：其从 `axis` 开始 flatten 后归一化的语义不能映射为 Relay 单 axis softmax，不能只改默认 axis 做半实现。`Transpose` 转换 `perm`（缺失时写为空数组，由 Relay 使用逆序默认）；`Gather` 转换 `axis`（缺省为 `0`）；`Relu`、`Add`、`GlobalAveragePool`、`Where` 使用简单 attrs。`LayerNormalization` 只接受 opset >= 17、三个非空输入和一个非空输出，写入强类型 `{axis, epsilon, accumulation_dtype:"float32"}` attrs；仅默认/`1` 的 `stash_type` 可导入。`MatMul`、`Gather`、`Where` 和 `LayerNormalization` 除 attrs 外还执行下述静态 contract gate。

## Shape 与 dtype 行为

Python protobuf importer 按静态 shape MVP fail-closed 地处理 ONNX value info：

- 已知 `dim_value` 会按整数保留，包括 `0`；显式但零维的 shape field 表示合法 scalar。
- 缺少 `tensor_type.shape` 的 unknown rank 会拒绝导入，并给出 tensor/value 名称和 rank context。
- symbolic 或 unknown 维度默认会拒绝导入，并在错误中给出 tensor/value 名称、axis 和（适用时）`dim_param`。
- 仅当调用方显式传入正数 `default_batch`（CLI 为 `--batch N`）时，未解析的 axis 0 才会绑定为该 batch 值；所有非 batch 未解析维度仍会拒绝。
- 每个 `MatMul` 在导入前必须能从 graph input/value_info、initializer 或此前推导的 `MatMul` output 解析两个静态 TensorSpec；缺失或未解析的 metadata 立即拒绝，且不会推导无关算子。
- `MatMul` 要求恰有两个 rank >= 2、同 dtype 的输入，K 相等且 leading batch dims 可按 NumPy 广播；frontend 推导 `[..., M, N]`，并要求任何 value_info/graph output 声明的 output shape/dtype 完全一致。
- `Gather` 要求恰有 data 和 int32/int64 indices 两个已解析静态输入，data rank >= 1，且 axis（允许负值）落在 data rank 内；frontend 推导 `data[:axis] + indices + data[axis + 1:]` 并验证所有声明 output 的 shape/dtype。ONNX 有效索引域为 `[-extent, extent - 1]`；KXC lowering 将域外运行时索引作为确定性的 typed zero-fill 扩展，而不是把该扩展误称为 ONNX 有效索引。
- `Where` 要求恰有 condition、x、y 三个已解析静态输入；condition 必须为 `bool`，x/y 必须同 dtype，且 branch dtype 严格限于 `{float32,float64,int32,int64,int8,uint8,bool}`。frontend 按 NumPy trailing-axis 规则对三个输入联合广播，并验证所有声明 output 的 shape/dtype。该逐元素选择映射**不定义 masked-softmax 或 all-masked-row 行为**。
- `LayerNormalization` 要求全部三个输入能从 graph input、initializer 或此前已推导输出解析为静态 `float32`；data rank >= 1、全部维度非负，axis 规范化后 suffix 全为正，且 scale/bias shape 必须严格等于 `data.shape[axis:]`。epsilon 必须有限且严格大于零，声明 output 必须与 data shape/dtype 完全一致。可选输入省略、mean/inv_std 附加输出、`stash_type != 1`、未解析前序值及任一 dtype/shape/axis/epsilon 不匹配均 fail closed。
- C++ `kxc.onnx_import.v1` reifier 消费已静态化的 JSON spec，每个 input/output 都必须包含由非负整数组成的 `shape` 数组。该格式不能表示或观察 ONNX unknown rank；缺失 `shape` 属于 malformed spec，而不是动态 rank 语义。

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

C++ loader 会按 spec 构建 `Var`、`Constant(runtime::NDArray)` 和 `Call(Op::Get(...), args, attrs)`，并保留 initializer 真实 bytes。构图后它会运行 Relay 类型推导，并要求每个推导出的输出 tensor shape/dtype 与 JSON 声明完全一致。

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

- 只覆盖静态 shape MVP，不承诺完整 ONNX opset；`MatMul` 仅接受 frontend 已验证的 rank >= 2 静态 K/batch/output contract，`Gather` 仅接受已验证的静态 data/indices/axis/output contract，`Where` 仅接受已验证的静态 bool condition、受限同 dtype branch 与联合广播/output contract，`LayerNormalization` 仅接受 opset >= 17 exact-static float32 affine subset；缺失 metadata 或动态维度仍不支持。
- `Softmax` 只接受 opset >= 13；opset < 13 的 flatten-from-axis 语义明确拒绝，不映射为 Relay 单 axis softmax。
- 不引入 C++ ONNX/protobuf 依赖；ONNX protobuf 解析留在 Python 侧。
- 不提供动态 shape runtime 语义。
- 不做 ResNet18 数值执行验收；本阶段验收重点是导入 Relay Function、保留 params 数据、序列化 runtime binding 信息，以及清晰的 unsupported op 错误。
- `python/gen_resnet18_ir_dump_cpp.py` 仍保留为 debug codegen 工具，但它已经复用正式 importer spec；默认导入 API 是 `kxc_onnx.import_onnx` 和 C++ `LoadONNXImportSpec`。
