# ONNX 导入器

本文档说明当前 ONNX 到 Relay 的正式导入路径。该导入器面向静态形状 MVP，把 `resnet18.onnx` 等模型转换为规范化的 Relay Function 描述和真实参数张量绑定信息。临时 C++ 构图脚本不再是默认导入方式。

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

- `function`：结构化 Relay Function 规格，包含输入、输出和节点。
- `params`：`dict[str, ParamTensor]`，保存每个 ONNX 初始化器的形状、数据类型和真实张量字节。
- `param_order`：ONNX 初始化器的原始顺序，用于稳定序列化和运行时参数绑定。

测试内存中的 ONNX `ModelProto` 时，使用 `import_onnx_model(model, default_batch=None, base_dir=None)`。

## 支持的 MVP 算子

当前 ONNX 算子到 Relay 规范算子的映射如下：

| ONNX 算子 | Relay 算子 |
|---|---|
| `Conv` | `nn_conv2d` |
| `Relu` | `nn_relu` |
| `MaxPool` | `nn_max_pool2d` |
| `Add` | `add` |
| `Concat` | `concatenate` |
| `Slice`（opset >= 10） | `slice` |
| `GlobalAveragePool` | `nn_global_avg_pool2d` |
| `Flatten` | `nn_flatten` |
| `Gemm` | `nn_gemm` |
| `MatMul` | `matmul` |
| `Softmax`（opset >= 13） | `softmax` |
| `Transpose` | `transpose` |
| `Gather` | `gather` |
| `Where` | `where` |
| `LayerNormalization`（opset >= 17） | `nn_layer_norm` |

`Conv`、`MaxPool`、`Flatten`、`Gemm` 会转换必要属性。`Relu`、`Add`、`GlobalAveragePool` 使用简单属性，`Where` 不接受属性。所有属性都按 ONNX Protobuf 的精确类型读取；重复属性名会被明确拒绝。

### Slice

`Slice` 只接受 opset >= 10 的输入形式：`data`、来自初始化器的非空一维 int32/int64 `starts`/`ends`，以及可选的同类 `axes`/`steps`。实际提供的控制初始化器必须全部为 int32 或全部为 int64，随后统一序列化为 int64 规范属性。缺少 `axes` 时写入 `[0..len(starts)-1]`；缺少 `steps` 时写入全 1。参数张量不会成为 Relay `Call` 输入，规范属性为 `{starts, ends, axes, steps}`。

### Softmax

opset >= 13 的 `Softmax` 转换单个 `axis`，默认值为 `-1`。opset < 13 的 `Softmax` 会被明确拒绝：其语义是从 `axis` 开始展平后归一化，不能映射为 Relay 的单轴 Softmax。

### Concat、Transpose 与 Gather

`Concat` 只支持二元、静态精确子集：必须显式给出唯一 `axis`，并转换为 `concatenate`。`Transpose` 转换 `perm`；缺少该属性时写为空数组，由 Relay 使用逆序默认值。`Gather` 转换 `axis`，默认值为 `0`，但只接受来自初始化器、且每个值都能证明处于 ONNX 有效范围内的索引。

### LayerNormalization

`LayerNormalization` 只接受 opset >= 17 和三个非空输入。合法输出列表为 `[Y]`、`[Y, ""]` 或 `[Y, "", ""]`；序列化规格仍只携带 Y，任何非空 Mean/InvStdDev 请求都会被拒绝。强类型属性为 `{axis, epsilon, accumulation_dtype:"float64"}`；只有默认值或 `1` 的 `stash_type` 可以导入。

## 形状与数据类型行为

Python Protobuf 导入器按静态形状 MVP 处理 ONNX 值信息；无法证明合法的输入会被明确拒绝：

- 已知 `dim_value` 按整数保留，包括 `0`；显式存在但长度为零的 `shape` 字段表示合法标量。
- 缺少 `tensor_type.shape` 表示秩未知，导入器会拒绝并给出张量或值名称及秩上下文。
- 默认拒绝符号维度和未知维度；错误会包含张量或值名称、轴，以及适用时的 `dim_param`。
- 只有调用方显式传入正数 `default_batch`（CLI 使用 `--batch N`）时，未解析的第 0 轴才会绑定到该批大小；其他未解析维度仍会被拒绝。
- 导入每个 `MatMul` 前，必须能从图输入、`value_info`、初始化器或此前推导的 `MatMul` 输出中解析出两个静态 `TensorSpec`。元数据缺失或无法解析时立即拒绝，不会借助无关算子推导。
- `MatMul` 要求恰好两个输入，秩均不小于 2、数据类型相同、K 维相等，前导批维可以按 NumPy 规则广播。前端推导 `[..., M, N]`，并要求 `value_info` 或图输出中声明的形状与数据类型完全一致。
- `Gather` 要求恰好包含 `data` 和来自初始化器的 int32/int64 `indices`；`data` 的秩不小于 1，`axis` 允许为负，但必须落在有效范围内。Python 导入器解码初始化器载荷，并逐值要求 `-extent <= index < extent`。动态索引、`INT64_MIN` 等越界值，以及零长度维度上的任何非空索引都会被拒绝；空索引合法。前端推导 `data[:axis] + indices + data[axis + 1:]` 并验证声明输出。通用 KXC Relay `gather` 仍保留越界时按类型补零的扩展，但默认 ONNX 映射不使用该扩展；C++ 重建器会再次要求常量载荷并复验每个值，手写 JSON 也不能绕过。
- `Slice` 只接受一个秩不小于 1、数据类型属于 `{float32,float64,int32,int64,int8,uint8,bool}`，且所有维度长度均为非负静态值的 `data`。`starts`/`ends`/`axes`/`steps` 必须非空且等长；轴归一化后必须唯一且位于有效范围内；步长必须严格为 `+1`。负端点仅在不小于 `-dim` 时加 `dim`，其他负值截断到 0，正值截断到 `dim`，并安全处理 `INT64_MIN`。输出长度为 `max(end-start,0)`；恒等切片和空切片都会生成新的复制计算，并验证声明输出。
- `Concat` 只接受两个非空、已解析的静态输入和一个非空输出，属性必须且只能包含显式 `axis`。输入的秩不小于 1，秩与数据类型必须相同，数据类型限于 `{float32,float64,int32,int64,int8,uint8,bool}`。`axis` 可以为负，但必须归一化到有效范围；非拼接轴维度必须严格相等；拼接轴长度相加不得溢出 int64。单侧零长度和零长度输出均合法；前端推导相加后的拼接轴长度，并要求所有声明输出的形状与数据类型完全一致。
- `Where` 要求恰好包含已解析的 `condition`、`x`、`y` 三个静态输入；`condition` 必须为 `bool`，`x` 与 `y` 的数据类型必须相同，且分支数据类型限于 `{float32,float64,int32,int64,int8,uint8,bool}`。前端按 NumPy 尾轴规则联合广播三个输入，并验证所有声明输出的形状与数据类型。该逐元素选择映射不定义掩码 Softmax 或全掩码行行为。
- `LayerNormalization` 要求三个输入都能解析为静态 `float32`。`data` 的秩不小于 1，所有维度非负，`axis` 后缀维度全为正，且 `scale`/`bias` 的形状严格等于 `data.shape[axis:]`。`epsilon` 必须是有限正 float32 属性值。内部求和、均值、中心化、方差、开方和仿射计算全部使用 float64，最终将 Y 转回 float32，因此规范属性 `accumulation_dtype` 固定为 `float64`。`[Y]`、`[Y, ""]` 与 `[Y, "", ""]` 等价，JSON 只记录 Y；非空 Mean/InvStdDev、`stash_type != 1`、未解析值或任何契约不匹配都会被拒绝。
- C++ `kxc.onnx_import.v1` 重建器读取已静态化的 JSON 规格。每个输入和输出都必须包含由非负整数构成的 `shape` 数组。该格式不能表示或观测 ONNX 的未知秩；缺少 `shape` 属于格式错误，不表示动态秩语义。

当前支持的张量数据类型：

- `float32`
- `float64`
- `int64`
- `int32`
- `int8`
- `uint8`
- `bool`

不支持的 ONNX 张量数据类型会抛出 `ValueError`。

## 参数序列化格式

`save_imported_model(imported, json_path, params_path)` 会写出两个文件：

- JSON：函数规格、参数元数据和 `param_order`。
- BIN：所有初始化器字节按 `param_order` 顺序连续写入。

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

运行时或 C++ 加载器通过 `offset` 和 `nbytes` 从 BIN 文件恢复 `runtime::NDArray`。

## C++ 重建器

C++ 入口位于 `include/kxc/frontend/onnx_importer.h`：

```cpp
#include "kxc/frontend/onnx_importer.h"

kxc::frontend::ImportedONNXModel imported =
    kxc::frontend::LoadONNXImportSpec(
        "out/resnet18.import.json",
        "out/resnet18.params.bin");
```

返回结果包含：

- `function`：真实 `kxc::Function`。
- `params`：`std::unordered_map<std::string, runtime::NDArray>`。
- `param_order`：稳定参数顺序。
- `input_names` / `output_names`：运行时绑定元数据。

C++ 加载器按规格构建 `Var`、`Constant(runtime::NDArray)` 和 `Call(Op::Get(...), args, attrs)`，并保留初始化器的真实字节。构图后，它会运行 Relay 类型推导，并要求每个推导输出的张量形状与数据类型和 JSON 声明完全一致。对于 Gather，重建器还会独立确认索引表达式是常量，读取 int32/int64 载荷，并逐值验证 ONNX 有效范围。

## 错误信息

不支持的 ONNX 算子会抛出 `UnsupportedONNXOpError`，消息包含算子类型和节点名称，例如：

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

CMake 在 Python、onnx 和 numpy 可用时生成 ResNet18 与精确 Transformer Fixture。`test/generate_onnx_transformer_fixture.py` 会先生成并校验真实 `ModelProto`、写入 `.onnx`，再通过 `onnx.load` 与 Python 导入器/序列化器生成 JSON/BIN。只有 `KXC_USE_LLVM=1` 时，`onnx_importer_test` CTest 才带 `llvm;protobuf;e2e` 标签，并继续执行 C++ 重建器、LLVM 编译器和 `RuntimeSession` 数值检查；非 LLVM 构建只运行导入与重建部分，并明确打印跳过信息。LLVM CI 命令会实际执行该链路，不能用文件存在性检查代替。

## 当前限制

- 只覆盖静态形状 MVP，不承诺完整 ONNX opset。`Slice` 仅支持统一整数宽度、来自初始化器的正步长子集；`Concat` 仅接受带显式轴的二元子集；`MatMul` 仅接受经过校验的静态 K 维、批维与输出契约；`Gather` 仅接受来自初始化器且逐值位于有效范围内的索引，动态索引不会导入；`Where` 仅接受静态布尔条件和数据类型相同的受限分支；`LayerNormalization` 仅接受 float32 输入输出、float64 内部累积、不请求统计输出的 opset >= 17 仿射子集。
- `Softmax` 只接受 opset >= 13；opset < 13 从指定轴开始展平的语义会被明确拒绝，不映射为 Relay 单轴 Softmax。
- 不引入 C++ ONNX/Protobuf 依赖；ONNX Protobuf 解析保留在 Python 侧。
- 不提供动态形状运行时语义。
- 不执行 ResNet18 数值验收。本阶段重点验证 Relay Function 导入、参数数据保留、运行时绑定信息序列化，以及清晰的“不支持算子”错误。
- `python/gen_resnet18_ir_dump_cpp.py` 保留为调试代码生成工具，但已经复用正式导入器规格。默认导入 API 是 `kxc_onnx.import_onnx` 和 C++ `LoadONNXImportSpec`。
