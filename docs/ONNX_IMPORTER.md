# ONNX 导入器

本文档说明当前 ONNX 到 Relay 的正式导入路径。默认入口把静态形状模型转换为 Relay Function 规格和真实参数绑定。另有显式的 shape-source 入口，保留形状控制链并交给 C++ restricted adapter 证明；该入口仍要求具体代表形状，不能直接编译未证明的 source 图。

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

`import_onnx(model_path, default_batch=None, fold_constants=True, preserve_shape_values=False)` 返回 `ImportedONNXModel`：

- `function`：结构化 Relay Function 规格，包含输入、输出和节点。
- `params`：`dict[str, ParamTensor]`，保存每个 ONNX 初始化器的形状、数据类型和真实张量字节。
- `param_order`：ONNX 初始化器的原始顺序，用于稳定序列化和运行时参数绑定。

测试内存中的 ONNX `ModelProto` 时，使用 `import_onnx_model`，额外支持 `base_dir`；其余选项相同。默认常量折叠只求值纯静态依赖，不把普通模型输入当作常量。

## 支持的 MVP 算子

当前 ONNX 算子到 Relay 规范算子的映射如下：

| ONNX 算子 | Relay 算子 |
|---|---|
| `Conv` | `nn_conv2d` |
| `Relu` | `nn_relu` |
| `MaxPool` | `nn_max_pool2d` |
| `Add` | `add` |
| `Concat` | `concatenate` |
| `Split`（静态常量分段，至少两路） | `split` |
| `Slice`（opset >= 10） | `slice` |
| `GlobalAveragePool` | `nn_global_avg_pool2d` |
| `Flatten` | `nn_flatten` |
| `Gemm` | `nn_gemm` |
| `MatMul` | `matmul` |
| `Softmax`（opset >= 13） | `softmax` |
| `Transpose` | `transpose` |
| `Gather` | `gather` |
| `Shape`（无属性、完整 rank） | `shape_of` |
| `ConstantOfShape`（显式 shape-source：int64 控制向量或有界 float32 填充） | `constant_of_shape` |
| `Trilu`（opset >= 14，float32，常量标量 k） | `trilu` |
| `Where` | `where` |
| `LayerNormalization`（opset >= 17） | `nn_layer_norm` |

`Conv`、`MaxPool`、`Flatten`、`Gemm` 会转换必要属性。`Relu`、`Add`、`GlobalAveragePool` 使用简单属性，`Where` 不接受属性。所有属性都按 ONNX Protobuf 的精确类型读取；重复属性名会被明确拒绝。

### Slice

`Slice` 只接受 opset >= 10 的输入形式：`data`、来自初始化器的非空一维 int32/int64 `starts`/`ends`，以及可选的同类 `axes`/`steps`。实际提供的控制初始化器必须全部为 int32 或全部为 int64，随后统一序列化为 int64 规范属性。缺少 `axes` 时写入 `[0..len(starts)-1]`；缺少 `steps` 时写入全 1。参数张量不会成为 Relay `Call` 输入，规范属性为 `{starts, ends, axes, steps}`。

### Softmax

opset >= 13 的 `Softmax` 转换单个 `axis`，默认值为 `-1`。opset < 13 的 `Softmax` 会被明确拒绝：其语义是从 `axis` 开始展平后归一化，不能映射为 Relay 的单轴 Softmax。

### Concat、Transpose 与 Gather

默认静态 `Concat` 接受 1..N 个输入，必须显式给出唯一 `axis`。三个及更多输入按原顺序展开为左结合二元 `concatenate`，静态与 shape-source 共用这条归一化；静态单输入用同 dtype/rank、拼接轴为零的零字节常量作第二操作数，复用二元复制语义并保留独立输出。未被常量折叠消去的单输入 shape-source 仍拒绝。真实图文模型的调用与边界见 [联合推理报告](implementation/M9_MINIMIND_V_JOINT_REPORT.md)。

生成的 Concat 值名和节点名避开原图全部名称，包括未解析输入和仅存在于 metadata 的值。Python 入口在常量折叠前拒绝非 `""`/`"ai.onnx"` domain，防止自定义同名节点被解释成标准 ONNX。

`Transpose` 转换 `perm`；缺少该属性时写为空数组，由 Relay 使用逆序默认值。`Gather` 转换 `axis`，默认值为 `0`；常量索引逐值检查范围，运行时索引走现有带越界保护的 Gather lowering。

### Split

静态 `Split` 只接受一个数据输入，或再加一个静态 int64 initializer 分段输入。`axis` 可为负并按数据 rank 归一化；分段长度必须是至少两个非负整数，且总和等于输入轴长度。输出必须有至少两个非空名称，且数量与分段长度一致；声明 shape/dtype 必须与推导结果一致。Python importer 将分段控制值规范化进 `SplitAttrs`，C++ spec 重建器只构造一个 Split Call，再按顺序绑定对应数量的 `TupleGetItem`。动态分段长度、少于两路以及输出数量不匹配明确拒绝；完整 LLVM 数值链见 [Split 基础报告](implementation/M4_SPLIT_REPORT.md)和[多路扩展报告](implementation/M4_SPLIT_VARIADIC_REPORT.md)。

`Shape` 只接无属性的完整形状查询，输入 rank 限于 1–8，结果为 `int64[rank]`；带 `start`/`end` 的切片形式仍拒绝。

### LayerNormalization

`LayerNormalization` 只接受 opset >= 17 和三个非空输入。合法输出列表为 `[Y]`、`[Y, ""]` 或 `[Y, "", ""]`；序列化规格仍只携带 Y，任何非空 Mean/InvStdDev 请求都会被拒绝。强类型属性为 `{axis, epsilon, accumulation_dtype:"float64"}`；只有默认值或 `1` 的 `stash_type` 可以导入。

## 形状与数据类型行为

Python Protobuf 导入器按静态形状 MVP 处理 ONNX 值信息；无法证明合法的输入会被明确拒绝：

- 已知 `dim_value` 按整数保留，包括 `0`；显式存在但长度为零的 `shape` 字段表示合法标量。
- 初始化器和常量折叠结果同样保留 rank 0；整理连续字节时不会把标量提升为 `[1]`。这保证 `Shape → Gather(标量索引) → Unsqueeze` 的秩语义正确。
- 缺少 `tensor_type.shape` 表示秩未知，导入器会拒绝并给出张量或值名称及秩上下文。
- 默认拒绝符号维度和未知维度；错误会包含张量或值名称、轴，以及适用时的 `dim_param`。
- 只有调用方显式传入正数 `default_batch`（CLI 使用 `--batch N`）时，未解析的第 0 轴才会绑定到该批大小；其他未解析维度仍会被拒绝。
- 导入每个 `MatMul` 前，必须能从图输入、`value_info`、初始化器或此前推导的 `MatMul` 输出中解析出两个静态 `TensorSpec`。元数据缺失或无法解析时立即拒绝，不会借助无关算子推导。
- `MatMul` 要求恰好两个输入，秩均不小于 2、数据类型相同、K 维相等，前导批维可以按 NumPy 规则广播。前端推导 `[..., M, N]`，并要求 `value_info` 或图输出中声明的形状与数据类型完全一致。
- `Gather` 要求恰好包含 `data` 和 int32/int64 `indices`；`data` 的秩不小于 1，`axis` 允许为负，但必须落在有效范围内。常量索引由 Python 和 C++ 独立逐值验证 `-extent <= index < extent`，空索引合法。运行时索引要求静态 shape/dtype/axis 和可寻址范围：合法负索引归一化，越界值按 KXC Gather 扩展返回零，且不会形成越界 Load。前端推导 `data[:axis] + indices + data[axis + 1:]` 并验证声明输出。M3 的形状值 Gather 仍只接受常量索引，不把普通张量内容当作 shape 来源。
- `Slice` 只接受一个秩不小于 1、数据类型属于 `{float32,float64,int32,int64,int8,uint8,bool}`，且所有维度长度均为非负静态值的 `data`。`starts`/`ends`/`axes`/`steps` 必须非空且等长；轴归一化后必须唯一且位于有效范围内；步长必须严格为 `+1`。负端点仅在不小于 `-dim` 时加 `dim`，其他负值截断到 0，正值截断到 `dim`，并安全处理 `INT64_MIN`。输出长度为 `max(end-start,0)`；恒等切片和空切片都会生成新的复制计算，并验证声明输出。
- 静态 `Concat` 接受至少一个非空名称、已解析的输入和一个非空输出，属性必须且只能包含显式 `axis`。输入的秩不小于 1，秩与数据类型必须相同，数据类型限于 `{float32,float64,int32,int64,int8,uint8,bool}`。`axis` 可以为负，但必须归一化到有效范围；非拼接轴维度必须严格相等；每次二元规范化阶段的拼接轴长度相加不得溢出 int64。空片段和零长度输出均合法；前端推导最终拼接轴长度，并要求所有声明输出的形状与数据类型完全一致。非法尾部操作数不能因展开为二元链而绕过检查。
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
- `declared_output_types`：JSON 声明的输出类型，shape-source 路径必须把它与 function 一同交给 producer 校验。

C++ 默认加载器按规格构建 `Var`、`Constant(runtime::NDArray)` 和 `Call(Op::Get(...), args, attrs)`，保留初始化器字节，运行 Relay 类型推导，并逐个校验声明输出的 shape/dtype。Gather 的常量载荷会再次逐值检查；非常量索引保留给带守卫的生产 lowering。

## 保留形状控制链的显式入口

输入文件必须先具有具体代表形状。例如真实 MiniMind 子图用 `[1,4,768]` 作为代表，B/S 的有限范围随后由 C++ 明确绑定：

```python
source = import_onnx(
    "out/fx_minimind_heads/heads_representative.onnx",
    preserve_shape_values=True,
)
save_imported_model(source, "out/heads.json", "out/heads.params")
```

该模式输出 `kxc.onnx_shape_source.v1`，不与默认的 `kxc.onnx_import.v1` 混用。Python 使用 ONNX 官方 shape inference 补齐代表元数据，保留 Shape 和原始 Reshape 控制输入；Unsqueeze 保留既有 axes 属性，多输入 Concat 确定性地转成二元链，生成名字避开原图值名。普通数据没有在 Python 中被执行。

```cpp
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"

namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
auto source = kxc::frontend::LoadONNXShapeSource("out/heads.json", "out/heads.params");
// config 是开启相应门禁的 CPU/LLVM CompileConfig。
auto prepared = Adapter::Prepare(source.function, config,
    {{0, 0, "B", 1, 3, 1}, {0, 1, "S", 1, 8, 1}}, source.declared_output_types);
auto compiled = kxc::api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
```

`LoadONNXShapeSource` 只重建尚未类型校验的 source Function；调用方必须传递 `declared_output_types`。adapter 解析受限形状来源、重建代表图并校验输出数量/shape/dtype，之后才进入已有编译准备。默认 loader 拒绝该格式，普通 Compile 也不能执行尚未解析的 ReshapeDynamic。

当前生产子集为 Shape → 常量索引 Gather → 标量 Unsqueeze → Concat（可混入 int64 字面向量）→ ReshapeDynamic，以及其上下游已经开放的 bounded 算子。数据 reshape 只接 `allowzero=0`；单个 `-1` 仅在下述固定正整数商证明成立时接受，仍拒绝动态推导结果和数据相关目标；形状向量的恒等展平单独按下述证明规则处理；形状标量只能在这条证明链内经 Unsqueeze 变成向量，不能直接作为标量结果逃逸。详情、四组 B/S 的真实子图结果和反例见 [M3 拆头报告](implementation/M3_ONNX_HEADS_REPORT.md)。

### GQA 的控制证明与 Expand

source 模式把 Expand 的两个输入保留为既有 `expand_dynamic`：opset >=13、无 attrs、float32 数据、同秩 int64 目标向量。ONNX shape inference 未能解析的中间 Expand extent 可以留给 C++ producer，前端仍检查已知 rank/dtype；代表图的输入和输出必须具体，C++ 最终核对全部声明输出。

ConstantOfShape 的控制向量形式接受显式单元素 int64 Tensor 填充值，整数绝对值不超过 2^53；序列化 attrs 为 `{dtype_code:2, value:整数}`，不接受调用方 target。producer 从控制输入证明目标长度是 0–16 的常量，再形成整数常量。默认静态入口仍要求这类节点先由既有纯常量折叠消解。

producer 可以折叠已证明的固定整数 Add/Mul/Div、恒等的形状向量 reshape、可证明的 Equal 和常量条件 Where。负标记留在常量载荷中，维度合同仍要求非负。符号算术、范围内真假不定的 Equal、任意运行时条件均不在这条控制折叠范围内。ShapeProgram/DimExpr 仍是唯一表达式权威，没有新增 Python shape VM。

实际 `[B,S,4,96] → [B,S,8,96]` 的 K/V 重复、四组 B/S LLVM 结果与失败边界见 [GQA 技术报告](implementation/M3_GQA_REPORT.md)。这项结果不代表一般动态数据 mask 或完整 attention 已完成。

### RoPE 的 Slice 控制与动态非操作轴

显式 shape-source 的 Slice 保留 `data/starts/ends/axes/steps` 五个输入且 attrs 为空，四个控制值必须是非空、长度至多 16 的 int64 向量。producer 证明它们为常量后，重写到既有单输入 `slice` 与 `SliceAttrs`。未准备的五输入形式不能通过普通 InferType；C++ source loader 仍能读取单输入/属性形式。`slice` schema v3 的输入数范围为 1..5；2 输入仅用于 producer 生成的动态前缀，3/4 输入仍不合法。调用方不能直接提交两输入 shape-source。

固定 head_dim 的 Shape/Gather/Div 及 int64→int64 Cast 因而可以成为实际旋转链的切片边界。Slice 仅允许静态操作轴和 +1 步长；C++ bounded producer 的 Concat 还允许一侧拼接轴为有界动态长度、另一侧为定长片段，并要求非拼接轴具有相同 DimExpr；P+C 输出和下游形状值的 LLVM 证据见 [KV 追加报告](implementation/M3_KV_APPEND_REPORT.md)。原始 decode 的 int64 Add/Sub 在显式 shape-source 中保留，由 C++ 证明、消去并选择 P:P+1 位置窗口；完整八层图的 17 个输出与执行证据见 [decode 报告](implementation/M3_FULL_DECODE_REPORT.md)。默认静态模式的 int64 Sub 仍拒绝，所有 Add/Sub 额外属性均拒绝。其他轴的 B/S 可保留动态 TE extent。ONNX 未推导出的中间长度不会被代表值替代，最终输出由 C++ producer 核对；未知长度也不能绕过 Cast 的 dtype 限制。

完整 RMSNorm/QKV/分头/RoPE 子图的四组 B/S、204 次 LLVM 调用及误差证据见 [RoPE 技术报告](implementation/M3_ROPE_REPORT.md)。图外提供的 cos/sin 表片段尚不等于全局位置表的图内动态切片已支持。


### 完整 prefill 的位置前缀与 embedding

静态 cos/sin 表可使用 source `Slice(table,[0],[S],[axis],[1])`。C++ 要求 end 是直接输入轴 Symbol，证明上界不超过表容量，再生成只读取 anchor 形状的两输入 Slice；静态控制数组为空，prefix_axis/extent_axis 进入 canonical attrs。其他动态切片端点仍拒绝。静态表 Gather 保留 token 索引的 B/S；有效负索引归一化，越界运行时值按既有 KXC 合同填零。

Slice 后未知的中间 extent 保留给 C++ 证明，Python 只检查已知 rank、dtype 和属性，最终声明输出仍由 producer 核对。完整八层 token→logits/16 KV、SwiGLU/残差及因果性数值证据见 [技术报告](implementation/M3_FULL_PREFILL_REPORT.md)。


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

- 默认静态入口不承诺完整 ONNX opset。`Slice` 仅支持统一整数宽度、来自初始化器的正步长子集；`Concat` 接受显式轴和静态输入，经上述规范化落到二元 Relay；`Split` 仅支持至少两路静态常量分段输出；`MatMul` 校验静态代表的 K、批维与输出契约；`Gather` 常量和运行时索引的不同边界见上文；`Where` 仅接受静态布尔条件和类型相同的受限分支；`LayerNormalization` 仅接受 float32 输入输出、float64 内部累积、不请求统计输出的 opset >= 17 仿射子集。
- `Softmax` 只接受 opset >= 13；opset < 13 从指定轴开始展平的语义会被明确拒绝，不映射为 Relay 单轴 Softmax。
- 不引入 C++ ONNX/Protobuf 依赖；ONNX Protobuf 解析保留在 Python 侧。
- shape-source 只衔接现有 M3 bounded 合同，仍要求具体代表。完整八层 prefill 的 B∈[1,3]、S∈[1,8] 已有同产物 LLVM 证据；变长 decode 和 state/extent 联合尚未完成，见 [完整 prefill 报告](implementation/M3_FULL_PREFILL_REPORT.md)。
- 不执行 ResNet18 数值验收。本阶段重点验证 Relay Function 导入、参数数据保留、运行时绑定信息序列化，以及清晰的“不支持算子”错误。
- `python/gen_resnet18_ir_dump_cpp.py` 保留为调试代码生成工具，但已经复用正式导入器规格。默认导入 API 是 `kxc_onnx.import_onnx` 和 C++ `LoadONNXImportSpec`。


## 有界因果 mask 与固定的推导维度

显式 shape-source 另允许 ConstantOfShape 的单元素 float32 Tensor 填充，含正/负无穷而拒绝 NaN。目标来自可证明的形状控制，rank 为 1–8，符号上界对应的输出不能超过共享的 256 MiB 上限。JSON 保存 `{dtype_code:0,value_bits:uint32位模式}`；C++ 解码后由 producer 产生具有数据来源和控制输入的已准备形式。调用方不能伪造 target/expr attrs，默认静态入口仍要求 ConstantOfShape 先折叠。

Trilu 映射为现有 Relay 层的普通单输入算子，支持 float32、rank≥2、upper=0/1、常量 int64 标量 k（省略为0）；在最后两轴选择三角区域。源 ONNX 的 k 控制在 importer 验证后进入 attrs，运行时 k、向量 k 和其他属性拒绝。静态和 shape-source 两种入口均可导入这个子集。

Reshape 源目标中的单个 -1 可以在消费处归一化，但只在约去相同且严格为正的直接符号后，剩余维度是固定正整数时接受。例如 `[B,S,8,96] → [B,S,-1]` 可证明为 `[B,S,768]`。结果仍含动态因子、多于一个 -1、非整数商及含零范围的取消都拒绝；负值不进入 DimExpr。共享源控制的不同 Reshape 分别归一化，避免相互覆盖。

ONNX 对中间 Reshape/Transpose/MatMul/Softmax 的未知 extent 留给 C++ 准备路径；rank、可检查 attrs 和具体代表输出仍有校验。真实第一层四组 B/S 和未来 token 扰动均已执行，证据与剩余整模型限制见 [因果 attention 报告](implementation/M3_CAUSAL_ATTENTION_REPORT.md)。
