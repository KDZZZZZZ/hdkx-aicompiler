# 新增 Relay 算子接入指南

本文说明如何把一个新 Relay 算子接入 KXC/TinyTVM。这里的“接入”不是只写 `KXC_REGISTER_OP`，而是让算子从前端构图、类型推导、TE/TOPI lowering、TIR、后端执行和测试链路上都处于明确状态。

## 0. 先填接入方案卡

新增算子 PR 先填这张方案卡。每个字段只能从本节给出的选项中选择，不要临时发明新分类、新命名或新链路。

| 字段 | 必选值 | 允许选项 |
| --- | --- | --- |
| `op_name` | canonical op name | `lower_snake_case`，或已有命名空间前缀 `nn_` / `device.` |
| `category` | 代码归属 | `tensor.math`、`tensor.reduce`、`tensor.transform`、`nn`、`device` |
| `source_file` | 注册文件 | 只能从“代码位置选择表”选择 |
| `num_inputs` | 输入数量 | 固定整数，或 `-1` 并同时填写 `min_inputs` / `max_inputs` |
| `attrs` | attrs 类型 | `null`、已有 attrs、或新增 attrs |
| `attrs_source` | 参数承载方式 | `none`、`attrs`、`tensor_input` |
| `type_rule` | 类型推导规则 | 只能从“类型推导选择表”选择 |
| `lowering` | lowering 路径 | `none`、`single`、`multi`、`exec_plan` |
| `topi` | TOPI/TE 位置 | `none`、已有 TOPI helper、新增 TOPI helper、op 文件内 `te::compute` |
| `ffi` | 是否需要 `_make` | `true` 或 `false` |
| `onnx_ops` | ONNX 来源 | 空数组，或明确 ONNX op 名列表 |
| `tests` | 测试集合 | 只能从“测试选择表”勾选 |

示例方案卡：

```yaml
op_name: negative
category: tensor.math
source_file: src/relay/op/tensor/math.cc
num_inputs: 1
attrs: null
attrs_source: none
type_rule: unary_same
lowering: single
topi: 新增 TOPI helper include/te/topi/elemwise.h::negative
ffi: true
onnx_ops: ["Neg"]
tests:
  - infer_type
  - lower_to_tir
  - onnx_importer
```

代码位置选择表：

| `category` | 注册文件 | TOPI 位置 | 说明 |
| --- | --- | --- | --- |
| `tensor.math` | [src/relay/op/tensor/math.cc](../src/relay/op/tensor/math.cc) | [include/te/topi/broadcast.h](../include/te/topi/broadcast.h) 或 [include/te/topi/elemwise.h](../include/te/topi/elemwise.h) | elementwise、broadcast、matmul 类 |
| `tensor.reduce` | [src/relay/op/tensor/reduce.cc](../src/relay/op/tensor/reduce.cc) | [include/te/topi/reduction.h](../include/te/topi/reduction.h) | reduce sum/mean/max/min 类 |
| `tensor.transform` | [src/relay/op/tensor/transform.cc](../src/relay/op/tensor/transform.cc) | [include/te/topi/transform.h](../include/te/topi/transform.h) | reshape、transpose、split、gather 类 |
| `nn` | [src/relay/op/nn](../src/relay/op/nn) 下已有子文件 | [include/te/topi/nn.h](../include/te/topi/nn.h) | conv、dense、pool、activation 类 |
| `device` | [src/relay/common_ops.cc](../src/relay/common_ops.cc) | 不使用 TOPI | 只用于 `device.` 通信 op，走 execution plan |

类型推导选择表：

| `type_rule` | 何时选择 | 允许实现 |
| --- | --- | --- |
| `identity` | 输出类型完全等于第一个输入 | 复用 `IdentityInferType` |
| `unary_same` | 单输入，shape/dtype 不变 | 复用 `UnarySameInferType` |
| `binary_broadcast` | 二元 elementwise broadcast | 复用现有二元广播规则，或新增一个共享规则后复用 |
| `matmul_like` | 矩阵乘、dense、gemm | 复用 `MatMulInferType` / `DenseInferType` / `GemmInferType`，或新增专用规则 |
| `attrs_shape` | 输出 shape 由 attrs 决定 | 新增专用 `XxxInferType` |
| `tuple_output` | 多输出 op | 新增专用 `XxxInferType`，返回 `TupleType` |
| `device_identity` | `device.` 通信 op | 复用 `IdentityInferType` |

attrs 选择表：

| 情况 | `attrs_source` | `attrs` | 规则 |
| --- | --- | --- | --- |
| 没有额外参数 | `none` | `null` | 注册时不写 `TAttrs` |
| axis、shape、layout、dtype、padding 等编译期参数 | `attrs` | 具体 attrs 类型 | 在 `include/relay/op.h` 定义，在 `src/relay/op_attrs.cc` 实现 `Create` |
| 参数本身是运行时 tensor | `tensor_input` | `null` 或仅保留必要 attrs | 参数必须计入 `num_inputs`，不要塞进 attrs |
| 设备通信属性 | `attrs` | `DeviceCopyAttrs` 或 `CollectiveAttrs` | 只用于 `device.` op，不进入 TE/TIR hook |

lowering 选择表：

| `lowering` | 何时选择 | 必须实现 | 禁止 |
| --- | --- | --- | --- |
| `none` | 只做注册/type，暂不能 lowering | matrix 写 `lowering = none` | 不得注册空 `FRelayToTE` 占位 |
| `single` | 输出是 `TensorType` | `FRelayToTE`，`LowerToTIR` 测试 | 不得返回空 `te::Tensor()` |
| `multi` | 输出是 `TupleType` | `FRelayToTEMulti`，`LowerToTIR` 多输出测试 | 不得重复返回同一个 tensor 冒充多输出 |
| `exec_plan` | `device.` 通信 op | `LowerRelayToExecPlanPass` 测试 | 不得注册 `FRelayToTE` / `FRelayToTEMulti` |

测试选择表：

| 测试 | 何时必须 | 推荐位置 |
| --- | --- | --- |
| `infer_type` | 所有 op | [test/infer_type_test.cpp](../test/infer_type_test.cpp) 或新增专用测试 |
| `lower_to_tir` | `lowering = single` / `multi` | lowering 专用测试文件 |
| `exec_plan` | `lowering = exec_plan` | pass / multi-device 专用测试 |
| `backend_compile` | `tir_executable` / `llvm_required` / `executable` 为 true | codegen 或 runtime 专用测试 |
| `runtime_numeric` | 声明 executable | runtime numeric 测试 |
| `onnx_importer` | `onnx_ops` 非空 | ONNX importer 测试 |
| `contract` | 所有 op | `check_relay_op_contract` |

## 1. 完成标准

新增算子必须先定义支持级别。

只完成注册时：

- 可以出现在 Relay IR 中。
- 必须有 canonical op name。
- 必须有 schema、输入数量、参数说明。
- support matrix 中必须标明 `lowering = none`。

完成类型推导时：

- 必须注册 `FInferType`。
- `InferTypePass` 能推导输出 `TensorType` 或 `TupleType`。
- 错误信息必须包含 canonical op name。

完成可执行 lowering 时：

- 普通单输出 Tensor/NN op 必须注册 `FRelayToTE`。
- 普通多输出 Tensor/NN op 必须注册 `FRelayToTEMulti`。
- 设备通信类 op 走 execution plan 路径，matrix 中写 `lowering = exec_plan`，不注册 `FRelayToTE` / `FRelayToTEMulti`。
- TOPI/TE helper 不能返回空 `te::Tensor()`。
- `single` / `multi` op 必须能通过 `LowerToTIR` 生成后端可接受的 TIR。
- `exec_plan` op 必须能通过 `LowerRelayToExecPlanPass` 生成 execution plan 节点。
- 至少有 `LowerToTIR` 或 `LowerRelayToExecPlanPass` 契约测试。

完成 runtime 支持时：

- LLVM/C 后端能编译生成的 TIR。
- 有 numeric runtime test。
- support matrix 中才能标为 executable。

## 2. 先定 canonical op name

新增算子第一步是确定 canonical name。规则见 [ISSUE_2_RELAY_OP_NAME_CANONICALIZATION.md](./ISSUE_2_RELAY_OP_NAME_CANONICALIZATION.md)。

命名要求：

- Relay IR 内部只使用 canonical name。
- public helper 也必须使用 canonical name。
- 不新增 metadata-only alias。
- ONNX importer 输出 canonical name。

示例：

| 语义 | canonical name | public helper |
| --- | --- | --- |
| subtraction | `subtract` | `_make.subtract` |
| multiplication | `mul` | `_make.mul` |
| softmax | `softmax` | `_make.softmax` |

## 3. 更新 support matrix

在实现代码前，先把算子加入机器可读 op support matrix：[test/relay_op_contract.json](../test/relay_op_contract.json)。

建议字段：

```json
{
  "negative": {
    "category": "tensor.math",
    "num_inputs": 1,
    "attrs": null,
    "lowering": "single",
    "ffi": true,
    "tests": true,
    "onnx_ops": ["Neg"]
  }
}
```

如果本次 PR 只做注册，不做 lowering，应写成：

```json
{
  "negative": {
    "category": "tensor.math",
    "num_inputs": 1,
    "attrs": null,
    "lowering": "none",
    "ffi": true,
    "tests": true,
    "onnx_ops": []
  }
}
```

matrix 的状态不能比实际实现更乐观。

字段填写规则：

- `category` 必须来自接入方案卡中的 `category` 选项。
- `num_inputs` 固定时写非负整数；可变输入只允许写 `-1`，并补 `min_inputs` / `max_inputs`。
- `attrs` 没有 attrs 时写 `null`；有 attrs 时写 C++ attrs 类型名，例如 `"ClipAttrs"`。
- `lowering` 只能写 `none`、`single`、`multi`、`exec_plan`。
- `ffi` 只表示是否需要 public `_make` helper；内部生成或 pass 专用 op 可以写 `false`。
- `tests` 对新增 op 默认写 `true`。如果写 `false`，PR 必须说明该 op 为什么不进入当前检查范围。
- `onnx_ops` 没有 ONNX 来源时写空数组；有来源时必须写 ONNX 原始 op 名。
- 只有已经有 backend compile/runtime 测试时，才能新增 `tir_executable`、`llvm_required`、`executable` 等乐观字段。

`lowering` 目前允许四类值：

- `none`：暂不支持 lowering，不能伪装成可执行。
- `single`：单输出 Tensor/NN op，必须注册 `FRelayToTE`，并有 `LowerToTIR` 契约测试。
- `multi`：多输出 Tuple op，必须注册 `FRelayToTEMulti`，并有 `LowerToTIR` 契约测试。
- `exec_plan`：设备通信类 op，不走 TE/TIR hook，必须能被 `LowerRelayToExecPlanPass` 转成 `CommExec` 或 execution plan 中的对应节点。

## 4. 定义 attrs

本章只处理方案卡里的 `attrs_source` 和 `attrs`。先按下面表格选分支，不要在实现时临时决定。

| `attrs_source` | 允许的 `attrs` | 要改的文件 | 注册时 `TAttrs` | 适用场景 |
| --- | --- | --- | --- | --- |
| `none` | `null` | 不新增 attrs | 不写 | 纯输入 tensor 决定语义 |
| `attrs` | `XxxAttrs` | [include/relay/op.h](../include/relay/op.h)、[src/relay/op_attrs.cc](../src/relay/op_attrs.cc) | 写 `"XxxAttrs"` | axis、layout、shape、dtype、padding、stride 等编译期参数 |
| `tensor_input` | `null` 或已有 attrs | 通常不新增 attrs | 通常不写 | 参数本身是运行时 tensor，例如 indices、condition、shape tensor |
| `device` | `DeviceCopyAttrs` / `CollectiveAttrs` | 复用已有 attrs | 写已有 attrs 名 | 只允许 `device.` 通信 op |

### 4.1 `attrs_source = none`

这种情况下不要定义 attrs，不要注册 `TAttrs`，也不要为了“以后可能用”创建空 attrs。

必须写法：

```json
{
  "negative": {
    "attrs": null,
    "lowering": "single"
  }
}
```

注册时：

```cpp
KXC_REGISTER_OP(negative)
    .describe(R"doc(Element-wise negation.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", NegativeCompute);
```

禁止：

- 新增 `NegativeAttrs` 空壳。
- 注册 `.set_attr<std::string>("TAttrs", "...")`。
- 在 FFI helper 里构造无意义 attrs。

### 4.2 `attrs_source = attrs`

使用固定三步：定义 attrs node、定义 attrs ref、实现 `Create`。

字段选择只能来自以下类型：

| 参数语义 | 字段类型 |
| --- | --- |
| 单个 axis、dtype code、group、channel 数 | `int` 或 `int64_t` |
| 多维 shape、axes、padding、stride、kernel size | `std::vector<int64_t>` |
| layout、dtype 字符串 | `std::string` |
| 开关参数 | `bool` |
| scale、epsilon 等浮点参数 | `float` 或 `double` |

`include/relay/op.h` 模板：

```cpp
class ClipAttrsNode : public BaseAttrsNode {
public:
    double a_min = 0.0;
    double a_max = 0.0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ClipAttrsNode)

class ClipAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ClipAttrs, ClipAttrsNode)

public:
    static ClipAttrs Create(double a_min, double a_max);
};
```

[src/relay/op_attrs.cc](../src/relay/op_attrs.cc) 模板：

```cpp
ClipAttrs ClipAttrs::Create(double a_min, double a_max) {
    auto* node = new ClipAttrsNode();
    node->a_min = a_min;
    node->a_max = a_max;
    return InternalCreate(node);
}
```

注册时必须绑定同名 `TAttrs`：

```cpp
.set_attr<std::string>("TAttrs", "ClipAttrs")
```

FFI helper 参数顺序必须和 `Create` 参数顺序一致：

```cpp
Call MakeClip(Expr data, double a_min, double a_max) {
    auto attrs = ClipAttrs::Create(a_min, a_max);
    return Call(GetOp("clip"), {data}, attrs);
}
```

ONNX importer 生成的字段名必须和 `Create` 参数语义一致。例如 `kernel_shape` 可以转成 `kernel_size`，但不能在 C++ attrs、FFI、ONNX 三处使用三套含义不同的字段名。

### 4.3 `attrs_source = tensor_input`

如果参数是运行时 tensor，必须作为 `Call` 输入，不允许塞进 attrs。

示例：`where(condition, x, y)`：

```json
{
  "where": {
    "num_inputs": 3,
    "attrs": null,
    "lowering": "single"
  }
}
```

注册时每个 tensor 输入都要有 `add_argument`：

```cpp
KXC_REGISTER_OP(where)
    .describe(R"doc(Select values from x or y by condition.)doc")
    .set_num_inputs(3)
    .add_argument("condition", "Tensor", "The condition tensor.")
    .add_argument("x", "Tensor", "The true branch tensor.")
    .add_argument("y", "Tensor", "The false branch tensor.")
    .set_attr<FInferType>("FInferType", WhereInferType)
    .set_attr<FRelayToTE>("FRelayToTE", WhereCompute);
```

禁止：

- 把 shape tensor、indices tensor、condition tensor 转成 attrs。
- `num_inputs` 少写，然后在 helper 里偷偷创建常量 tensor。

### 4.4 `attrs_source = device`

只允许 `device.` 通信 op 使用。当前只能选：

- `device.copy`：`DeviceCopyAttrs`
- `device.allreduce`、`device.broadcast_from_worker0`、`device.scatter_from_worker0`、`device.gather_to_worker0`、`device.send_to_worker`、`device.recv_from_worker`：`CollectiveAttrs`

device op 不写 `FRelayToTE` / `FRelayToTEMulti`，只进入 execution plan。

### 4.5 attrs 检查清单

- [ ] matrix 的 `attrs` 与注册的 `TAttrs` 完全一致。
- [ ] `Create` 参数顺序与 FFI helper 参数顺序一致。
- [ ] type inference 和 lowering 使用同一个 attrs 类型。
- [ ] 默认值在 attrs node 或 `Create` 中明确。
- [ ] 错误路径检查 attrs 是否为空或类型不匹配。
- [ ] `tensor_input` 参数没有被塞进 attrs。

## 5. 增加类型推导规则

本章只处理方案卡里的 `type_rule`。类型推导函数声明放在 [include/relay/type_infer.h](../include/relay/type_infer.h)，实现放在 [src/relay/type_infer.cc](../src/relay/type_infer.cc)。

新增规则时优先复用 `type_infer.cc` 现有内部 helper：`RequireArity`、`RequireTensor`、`RequireSameDType`、`ShapeVector`、`MakeTensorType`、`NormalizeAxis`、`BinaryBroadcastInferType`。

### 5.1 按 `type_rule` 选模板

| `type_rule` | 直接可用 | 模板 |
| --- | --- | --- |
| `identity` | `IdentityInferType` | 不新增函数 |
| `unary_same` | `UnarySameInferType` | 不新增函数，除非要检查 attrs |
| `binary_broadcast` | 内部 `BinaryBroadcastInferType` | 新增薄封装函数 |
| `matmul_like` | 已有专用规则 | 优先复用 `MatMulInferType` / `DenseInferType` / `GemmInferType` |
| `attrs_shape` | 无 | 新增专用 `XxxInferType` |
| `tuple_output` | 无 | 新增专用 `XxxInferType`，返回 `TupleType` |
| `device_identity` | `IdentityInferType` | 只给 `device.` op |

`identity` / `unary_same` 注册模板：

```cpp
KXC_REGISTER_OP(negative)
    .set_attr<FInferType>("FInferType", UnarySameInferType);
```

`binary_broadcast` 模板：

```cpp
Type MaximumInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return BinaryBroadcastInferType("maximum", input_types);
}
```

`attrs_shape` 模板：

```cpp
Type ClipInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("clip", input_types, 1);
    const auto* data = RequireTensor("clip", input_types[0], "data");
    const auto* clip_attrs = attrs.As<ClipAttrsNode>();
    if (!clip_attrs) {
        throw std::runtime_error("clip expects ClipAttrs");
    }
    return MakeTensorType(ShapeVector(data), data->dtype);
}
```

`tuple_output` 模板：

```cpp
Type SplitLikeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    RequireArity("split_like", input_types, 1);
    const auto* data = RequireTensor("split_like", input_types[0], "data");
    const auto* split_attrs = attrs.As<SplitLikeAttrsNode>();
    if (!split_attrs) {
        throw std::runtime_error("split_like expects SplitLikeAttrs");
    }

    Array<Type> fields;
    fields.push_back(MakeTensorType({data->shape[0]}, data->dtype));
    fields.push_back(MakeTensorType({data->shape[0]}, data->dtype));
    return TupleType(fields);
}
```

### 5.2 必须检查的内容

每个新增专用 infer rule 按这个顺序写：

1. `RequireArity(op_name, input_types, expected)`。
2. 对每个 tensor 输入调用 `RequireTensor(op_name, input_types[i], input_name)`。
3. 检查 dtype：同 dtype 用 `RequireSameDType`；不同 dtype 必须明确输出 dtype。
4. 检查 rank：例如 `if (data->shape.size() != 4) throw ...`。
5. 检查 axis/layout/padding/shape 参数合法。
6. 返回 `MakeTensorType(shape, dtype)` 或 `TupleType(fields)`。
7. 错误信息必须包含 canonical op name。

### 5.3 不能复用已有规则的情况

出现以下任一情况，必须新增专用 `XxxInferType`：

- 输出 shape 依赖 attrs，例如 axis、newshape、kernel、stride、padding。
- 输出 dtype 不是输入 dtype。
- 输入可以是 tuple 或输出是 tuple。
- 需要检查 layout、rank、group、broadcast 以外的特殊语义。
- 报错需要区分多个输入角色，例如 data、weight、bias。

### 5.4 类型推导检查清单

- [ ] `include/relay/type_infer.h` 声明了新增规则，或注册处复用了已有规则。
- [ ] `src/relay/type_infer.cc` 实现与声明一致。
- [ ] arity、TensorType、dtype、rank、attrs 都有检查。
- [ ] 输出类型和 `FRelayToTE` / `FRelayToTEMulti` 生成的 tensor 数量、shape、dtype 一致。
- [ ] 错误信息包含 canonical op name。

示例最小 unary 规则：

```cpp
Type NegativeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return UnarySameInferType(attrs, input_types);
}
```

## 6. 实现 TE/TOPI compute

本章只处理方案卡里的 `topi`。先选 TOPI 位置，再写 compute。

| `topi` | 何时选择 | 写在哪里 | 是否允许 `lowering` |
| --- | --- | --- | --- |
| `none` | `lowering = none` 或 `exec_plan` | 不写 TOPI | 只允许 `none` / `exec_plan` |
| 已有 TOPI helper | 语义完全匹配已有 helper | 不新增 helper | `single` / `multi` |
| 新增 TOPI helper | 逻辑可复用或索引复杂 | 对应 TOPI 头文件 | `single` / `multi` |
| op 文件内 `te::compute` | 一次性简单 compute | 对应 `src/relay/op/**.cc` | `single` / `multi` |

TOPI 文件只能从这里选：

- elementwise: [include/te/topi/elemwise.h](../include/te/topi/elemwise.h)
- broadcast: [include/te/topi/broadcast.h](../include/te/topi/broadcast.h)
- nn: [include/te/topi/nn.h](../include/te/topi/nn.h)
- transform: [include/te/topi/transform.h](../include/te/topi/transform.h)
- reduction: [include/te/topi/reduction.h](../include/te/topi/reduction.h)

### 6.1 `topi = none`

只允许：

- `lowering = none`：当前 PR 不支持 lowering。
- `lowering = exec_plan`：`device.` op 走 execution plan。

禁止为了通过编译返回 `te::Tensor()`。

### 6.2 使用已有 TOPI helper

只有语义完全一致时才能复用。检查项：

- shape 规则一致。
- dtype 规则一致。
- axis/layout/padding/stride 语义一致。
- 不支持场景的错误行为一致。

`FRelayToTE` 中直接调用：

```cpp
return te::topi::add(inputs[0], inputs[1], "T_add");
```

### 6.3 新增 TOPI helper

新增 helper 使用固定签名风格：

```cpp
inline Tensor negative(const Tensor& x,
                       std::string name = "negative",
                       std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return 0 - x(indices);
        },
        name,
        tag);
}
```

带参数 helper 模板：

```cpp
inline Tensor clip(const Tensor& x,
                   tir::PrimExpr a_min,
                   tir::PrimExpr a_max,
                   std::string name = "clip",
                   std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            auto value = x(indices);
            return tir::Min(tir::Max(value, a_min), a_max);
        },
        name,
        tag);
}
```

TOPI helper 不接收 Relay `Attrs`，只接收 TE tensor 和已经解析好的普通参数。attrs 解析只放在 `FRelayToTE`。

### 6.4 op 文件内 `te::compute`

只允许用于不可复用、非常短的 compute。模板：

```cpp
te::Tensor NegativeCompute(const Attrs& attrs,
                           const Array<te::Tensor>& inputs,
                           const kxc::Type& out_type) {
    (void)attrs;
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("negative expects exactly 1 input");
    }
    return te::compute(inputs[0]->shape,
                       [x = inputs[0]](const Array<kxc::tir::Var>& indices) {
                           return 0 - x(indices);
                       },
                       "T_negative");
}
```

如果 compute 超过一个表达式、需要 axis normalize、需要 layout/padding 处理，改用新增 TOPI helper。

### 6.5 TOPI/TE 检查清单

- 返回 defined `te::Tensor`。
- shape/dtype 和 type inference 一致。
- 不支持的参数组合必须抛错。
- 不能返回空 tensor。
- 不能用错误实现占位。
- 不能在 TOPI helper 中读取 Relay attrs。
- 生成的 TIR 表达式不能超出当前 `LowerToTIR` 和 C/LLVM codegen 支持范围；超出时必须同步补后端。

## 7. 写 `FRelayToTE`

本章只处理方案卡里的 `lowering`。`FRelayToTE` / `FRelayToTEMulti` 是 Relay op 到 TE compute 的桥。通常写在对应注册文件里：

- tensor math: [src/relay/op/tensor/math.cc](../src/relay/op/tensor/math.cc)
- tensor transform: [src/relay/op/tensor/transform.cc](../src/relay/op/tensor/transform.cc)
- nn: [src/relay/op/nn](../src/relay/op/nn)

### 7.1 `lowering = none`

不写 `FRelayToTE`，不写 `FRelayToTEMulti`，不写空 compute。matrix 必须保留：

```json
{
  "lowering": "none"
}
```

### 7.2 `lowering = single`，无 attrs

固定检查顺序：输入数量、输出类型、调用 TOPI/TE、检查返回 tensor。

```cpp
te::Tensor NegativeCompute(const Attrs& attrs,
                           const Array<te::Tensor>& inputs,
                           const kxc::Type& out_type) {
    (void)attrs;
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("negative expects exactly 1 input");
    }
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error("negative expects TensorType output");
    }
    te::Tensor out = te::topi::negative(inputs[0], "T_negative");
    if (!out.defined()) {
        throw std::runtime_error("negative lowering returned undefined tensor");
    }
    return out;
}
```

### 7.3 `lowering = single`，有 attrs

固定检查顺序：输入数量、attrs 类型、输出类型、解析 attrs、调用 TOPI/TE、检查返回 tensor。

```cpp
te::Tensor ClipCompute(const Attrs& attrs,
                       const Array<te::Tensor>& inputs,
                       const kxc::Type& out_type) {
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("clip expects exactly 1 input");
    }
    const auto* clip_attrs = attrs.As<ClipAttrsNode>();
    if (!clip_attrs) {
        throw std::runtime_error("clip expects ClipAttrs");
    }
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error("clip expects TensorType output");
    }
    te::Tensor out = te::topi::clip(
        inputs[0],
        tir::FloatImm(clip_attrs->a_min, inputs[0]->dtype),
        tir::FloatImm(clip_attrs->a_max, inputs[0]->dtype),
        "T_clip");
    if (!out.defined()) {
        throw std::runtime_error("clip lowering returned undefined tensor");
    }
    return out;
}
```

### 7.4 `lowering = multi`

固定检查顺序：输入数量、attrs 类型、`TupleType` 输出、生成每个输出、检查数量、检查 undefined、检查重复 tensor。

多输出 compute 必须把每个输出 tensor 的 shape 和索引映射写清楚。下面是 split-like 算子的模板：输出 shape 来自 `TupleType::fields`，输入索引在 split axis 上增加当前输出的 offset。

```cpp
Array<kxc::tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type,
                                              const std::string& op_name) {
    if (!type) {
        throw std::runtime_error(op_name + " output field must be TensorType");
    }
    Array<kxc::tir::PrimExpr> shape;
    for (int64_t dim : type->shape) {
        if (dim < 0) {
            throw std::runtime_error(op_name + " lowering requires static output shape");
        }
        shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    return shape;
}

Array<te::Tensor> SplitLikeCompute(const Attrs& attrs,
                                   const Array<te::Tensor>& inputs,
                                   const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("split_like expects exactly 1 input");
    }
    const auto* split_attrs = attrs.As<SplitLikeAttrsNode>();
    if (!split_attrs) {
        throw std::runtime_error("split_like expects SplitLikeAttrs");
    }
    const auto* tuple_type = out_type.As<TupleTypeNode>();
    if (!tuple_type) {
        throw std::runtime_error("split_like expects TupleType output");
    }
    const int axis = NormalizeSplitAxis(split_attrs->axis,
                                        static_cast<int>(inputs[0]->shape.size()));
    const std::vector<int64_t> offsets = SplitOffsets(split_attrs, tuple_type);
    if (offsets.size() != tuple_type->fields.size()) {
        throw std::runtime_error("split_like output count mismatch");
    }

    Array<te::Tensor> outputs;
    for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        const auto* field_type = tuple_type->fields[i].As<TensorTypeNode>();
        Array<kxc::tir::PrimExpr> out_shape = ShapeFromTensorType(field_type, "split_like");
        const int64_t offset = offsets[i];
        outputs.push_back(te::compute(
            out_shape,
            [input = inputs[0], axis, offset](const Array<kxc::tir::Var>& indices) {
                Array<kxc::tir::PrimExpr> input_indices;
                for (const auto& index : indices) {
                    input_indices.push_back(index);
                }
                input_indices[static_cast<size_t>(axis)] =
                    input_indices[static_cast<size_t>(axis)] +
                    kxc::tir::IntImm(offset, kxc::tir::DataType::Int(64));
                return input(input_indices);
            },
            "T_split_like_" + std::to_string(i)));
    }

    if (outputs.size() != tuple_type->fields.size()) {
        throw std::runtime_error("split_like output count mismatch");
    }
    for (const auto& out : outputs) {
        if (!out.defined()) {
            throw std::runtime_error("split_like returned undefined tensor");
        }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        for (size_t j = i + 1; j < outputs.size(); ++j) {
            if (outputs[i] == outputs[j]) {
                throw std::runtime_error("split_like returned duplicate tensor outputs");
            }
        }
    }
    return outputs;
}
```

`NormalizeSplitAxis`、`SplitOffsets` 这类 helper 必须是当前算子的真实语义实现，并有类型推导和 lowering 测试覆盖；不能返回固定 offset 或固定 shape。

- 输出类型必须是 `TupleType`。
- 返回 tensor 数量必须等于 `TupleType::fields.size()`。
- 不允许用同一个 tensor 重复冒充多个输出。

### 7.5 `lowering = exec_plan`

不写 `FRelayToTE`，不写 `FRelayToTEMulti`。实现重点转到：

- 注册 `device.` op。
- attrs 使用 `DeviceCopyAttrs` 或 `CollectiveAttrs`。
- `LowerRelayToExecPlanPass` 能生成 execution plan。
- 测试里检查 `CommExec` 或序列化后的 execution plan 节点。

### 7.6 lowering 检查清单

- [ ] `none` 没有空 hook。
- [ ] `single` 返回 defined `te::Tensor`。
- [ ] `multi` 返回数量等于 `TupleType::fields.size()`。
- [ ] `multi` 不重复返回同一个 tensor。
- [ ] attrs 类型检查在 compute 函数开头完成。
- [ ] `out_type` 类型检查在 compute 函数开头完成。
- [ ] 报错信息包含 canonical op name。
- [ ] `single` / `multi` 有同函数块 `LowerToTIR` 测试。
- [ ] `exec_plan` 有同函数块 `LowerRelayToExecPlanPass` 测试。

## 8. 注册 Relay op

使用 `KXC_REGISTER_OP` 注册普通 op。注册点应放在按类别划分的源文件中，不要随意塞进 `common_ops.cc`；只有 `device.` 通信 op 使用 `common_ops.cc` 中的 `OpRegEntry(Op::Get(...))` 形式。

单输出 unary 示例：

```cpp
KXC_REGISTER_OP(negative)
    .describe(R"doc(Element-wise negation.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", NegativeCompute);
```

带 attrs 示例：

```cpp
KXC_REGISTER_OP(clip)
    .describe(R"doc(Clip tensor values into [a_min, a_max].)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ClipAttrs")
    .set_attr<FInferType>("FInferType", ClipInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ClipCompute);
```

注册要求：

- `set_num_inputs` 和实际 helper/lowering 一致。
- 每个输入都用 `add_argument` 描述。
- 有 attrs 的 op 必须设置 `TAttrs`。
- 可 type inference 的 op 必须设置 `FInferType`。
- `single` op 必须设置 `FRelayToTE`。
- `multi` op 必须设置 `FRelayToTEMulti`。
- `exec_plan` op 不设置 TE lowering hook。
- 不要注册历史 alias。

注册模板只能从下面四种选。

`lowering = none`：

```cpp
KXC_REGISTER_OP(op_name)
    .describe(R"doc(One sentence description.)doc")
    .set_num_inputs(N)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", OpNameInferType);
```

`lowering = single`，无 attrs：

```cpp
KXC_REGISTER_OP(op_name)
    .describe(R"doc(One sentence description.)doc")
    .set_num_inputs(N)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", OpNameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", OpNameCompute);
```

`lowering = single`，有 attrs：

```cpp
KXC_REGISTER_OP(op_name)
    .describe(R"doc(One sentence description.)doc")
    .set_num_inputs(N)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "OpNameAttrs")
    .set_attr<FInferType>("FInferType", OpNameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", OpNameCompute);
```

`lowering = multi`：

```cpp
KXC_REGISTER_OP(op_name)
    .describe(R"doc(One sentence description.)doc")
    .set_num_inputs(N)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "OpNameAttrs")
    .set_attr<FInferType>("FInferType", OpNameInferType)
    .set_attr<FRelayToTEMulti>("FRelayToTEMulti", OpNameCompute);
```

`lowering = exec_plan`，只给 `device.` 通信 op 使用：

```cpp
static OpRegEntry __make_OpEntry_device_xxx__ =
    OpRegEntry(Op::Get("device.xxx"))
        .describe("Device communication op description")
        .set_num_inputs(1)
        .add_argument("data", "Tensor", "The input tensor.")
        .set_attr<FInferType>("FInferType", IdentityInferType)
        .set_attr<std::string>("TAttrs", "CollectiveAttrs");
```

注意：`device.xxx` 不能使用 `KXC_REGISTER_OP(device.xxx)`，因为宏参数不能包含 `.`；只能使用 `OpRegEntry(Op::Get("device.xxx"))`。

## 9. 增加 C++ `_make` helper

如果算子需要从 Python/FFI 构图，在 [src/relay/op/op_ffi.cc](../src/relay/op/op_ffi.cc) 中增加 helper。

示例：

```cpp
Call MakeNegative(Expr data) {
    return Call(GetOp("negative"), {data});
}

KXC_REGISTER_GLOBAL("kxc.relay.op._make.negative")
    .set_body(ToPackedFunc(MakeNegative));
```

带 attrs helper 模板：

```cpp
Call MakeClip(Expr data, double a_min, double a_max) {
    auto attrs = ClipAttrs::Create(a_min, a_max);
    return Call(GetOp("clip"), {data}, attrs);
}

KXC_REGISTER_GLOBAL("kxc.relay.op._make.clip")
    .set_body(ToPackedFunc(MakeClip));
```

多输入 helper 模板：

```cpp
Call MakeWhere(Expr condition, Expr x, Expr y) {
    return Call(GetOp("where"), {condition, x, y});
}

KXC_REGISTER_GLOBAL("kxc.relay.op._make.where")
    .set_body(ToPackedFunc(MakeWhere));
```

helper 要求：

- public helper 名必须是 canonical name，例如 `kxc.relay.op._make.subtract`，不能写 `_make.sub`。
- helper 内部必须 `GetOp(canonical_name)`。
- helper 必须构造正确 attrs。
- helper 不能返回历史 alias op。
- helper 的行为要被 registry/contract test 覆盖。

## 10. 接入 ONNX importer

如果该算子来自 ONNX，更新 [python/kxc_onnx/importer.py](../python/kxc_onnx/importer.py)：

- `ONNX_TO_RELAY` 输出 canonical op name。
- `_convert_attrs` 生成对应 attrs 字段。
- 多输出 ONNX op 要明确是否支持；不支持时抛清晰错误。

示例：

```python
ONNX_TO_RELAY = {
    "Neg": "negative",
}
```

ONNX importer 测试需要断言 `RelayNodeSpec.op_name` 是 canonical name。

## 11. 检查 TIR 和后端

新增 op 的 lowering 通过后，要检查 TIR 是否能被后端消费。

必须检查：

- `LowerToTIR(func)` 成功。
- TIR 里没有后端不支持的 stmt/expr。
- intrinsic 名称能被 C/LLVM codegen 识别。
- dtype cast 能被后端正确生成。

LLVM 后端只消费 TIR，不认识 Relay op。只有新算子生成了新的 TIR 节点、intrinsic、cast 语义或 ABI 需求时，才需要改 LLVM codegen。

常见判断：

- 只是 loop + load/store + add/sub/mul/div：通常不用改 LLVM。
- 生成 `Call("cast")`、`Call("exp")`、新 intrinsic：需要检查 LLVM 的 `GenCall`。
- 生成 `Block`、`AttrStmt`、vectorize/parallel 结构：需要补 `GenStmt`。

## 12. 添加测试

最低测试集：

- Type inference test：放在 `test/infer_type_test.cpp` 或新增专门测试。
- Lowering test：Relay -> `LowerToTIR` 成功。
- TOPI/TE test：TOPI helper 返回 defined tensor，关键 shape/index 正确。
- Runtime numeric test：如果 matrix 标记 executable，必须有 LLVM 或 C backend numeric test。
- ONNX importer test：如果接入 ONNX。
- Contract test：support matrix、canonical name、hook 覆盖。

按 `lowering` 选择测试：

| `lowering` | 必须测试 | 测试里必须同时出现 |
| --- | --- | --- |
| `none` | type inference 或明确 unsupported 行为 | `Op::Get("op_name")` |
| `single` | type inference、`LowerToTIR` | `Op::Get("op_name")` 和 `LowerToTIR` 在同一个测试函数块 |
| `multi` | type inference、`LowerToTIR`、输出数量和 shape | `Op::Get("op_name")` 和 `LowerToTIR` 在同一个测试函数块 |
| `exec_plan` | type inference、`LowerRelayToExecPlanPass`、`CommExec` / execution plan 节点 | `Op::Get("device.xxx")` 和 `LowerRelayToExecPlanPass` 在同一个测试函数块 |

如果 matrix 写了 `llvm_required`、`tir_executable` 或 `executable`，还必须新增 backend compile/runtime 测试，并让测试函数中同时出现 op name 和 `Compiler::Compile`、`CompileConfig`、`CodeGenLLVM` 或 `LLVMJIT`。

示例 type inference test：

```cpp
bool TestNegativeInferType() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Call neg(kxc::relay::Op::Get("negative"), {x});
    kxc::Function func({x}, neg);
    kxc::relay::InferTypePass(func);
    return CheckTensor(neg.checked_type(), {2, 3}, "float32");
}
```

示例 lowering test：

```cpp
bool TestNegativeLowerToTIR() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Call neg(kxc::relay::Op::Get("negative"), {x});
    kxc::Function func({x}, neg);
    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    return lowered.defined();
}
```

示例 execution-plan test：

```cpp
bool TestDeviceCopyExecPlan() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Call copy(kxc::relay::Op::Get("device.copy"), {x},
                   kxc::relay::DeviceCopyAttrs::Create(
                       kxc::VirtualDevice(), kxc::VirtualDevice()));
    kxc::Function func({x}, copy);
    kxc::ExecutionPlan plan = kxc::relay::LowerRelayToExecPlanPass(func);
    return plan.defined();
}
```

示例 runtime test：

```cpp
auto config = kxc::api::CompileConfig::AOT(kxc::BuildTarget(kxc::kCPU), 2);
auto module = kxc::api::Compiler::Compile(func, config);
module.Run({input_data, output_data});
```

## 13. 更新 CMake 和 CI

新增测试文件时，在 [CMakeLists.txt](../CMakeLists.txt) 中加入 target。

示例：

```cmake
kxc_add_runtime_exe(relay_negative_test test/relay_negative_test.cpp)

add_custom_target(run_relay_negative_test
  COMMAND "$<TARGET_FILE:relay_negative_test>"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
  DEPENDS relay_negative_test
  USES_TERMINAL
  COMMENT "Run relay_negative_test"
)
```

如果该算子属于 MVP required op，必须进入 `run_cpu_required_tests` 或等价 CI 聚合 target。

提交 PR 前还必须运行算子契约检查：

```bash
python python/tools/check_relay_op_contract.py --root .
```

该命令默认硬失败。它会对照 [test/relay_op_contract.json](../test/relay_op_contract.json) 检查所有 Relay 算子的统一接入模板，包括：

- 是否只使用 canonical op name。
- 是否存在重复注册或未声明 op。
- 是否有完整 schema、`set_num_inputs`、`add_argument`。
- 是否注册 `FInferType`。
- `single` op 是否注册 `FRelayToTE`，`multi` op 是否注册 `FRelayToTEMulti`，`exec_plan` op 是否避免注册 TE lowering hook。
- 是否有 canonical `_make` helper，不允许 `_make.sub`、`_make.conv2d` 这类 alias。
- ONNX importer 是否输出 canonical op name。
- 是否有测试引用。
- 可 lowering / executable op 是否有 TIR 和后端契约测试覆盖。
- 源码里是否残留 `TODO`、`FIXME`、`placeholder`、`for now`、`skip` 这类占位实现标记。

检查器逻辑如下：

1. 读取 [test/relay_op_contract.json](../test/relay_op_contract.json) 作为唯一规范来源。`operators` 定义允许存在的 canonical op、输入数、attrs、lowering 类型、是否需要 FFI、ONNX 映射和测试；`rules` 定义 forbidden op/helper name 和占位实现关键字。
2. 静态扫描 `src` 下的 C++ 源码，识别 `KXC_REGISTER_OP(name)` 和 `OpRegEntry(Op::Get("name"))`。每个注册块会提取 `describe`、`set_num_inputs`、`add_argument` 数量、`TAttrs`、`FInferType`、`FRelayToTE`、`FRelayToTEMulti`。
3. 静态扫描 [src/relay/op/op_ffi.cc](../src/relay/op/op_ffi.cc)，识别 `KXC_REGISTER_GLOBAL("kxc.relay.op._make.xxx")` 绑定到的 `MakeXxx` 函数，再从函数体里提取 `GetOp("name")` 和 `Call(..., {inputs})` 的输入个数。
4. 解析 [python/kxc_onnx/importer.py](../python/kxc_onnx/importer.py) 中的 `ONNX_TO_RELAY`，反向生成 `relay op -> ONNX op` 映射，用来确认 importer 只输出 canonical name。
5. 扫描 `test` 目录中对 op name 字符串的引用，作为最低限度的测试覆盖信号。普通引用按文件统计；`LowerToTIR`、backend compile/runtime、`LowerRelayToExecPlanPass` 覆盖按测试函数块统计，避免同一个测试文件里无关 op 被误算成已覆盖。
6. 对所有来源取并集生成检查对象：matrix 中声明的 op、源码注册的 op、FFI helper 指向的 op、ONNX importer 输出的 op 都会进入报告。因此未声明 op、历史 alias、孤立 helper 都会被发现。
7. 对每个 op 做规范判定：必须在 matrix 中声明，不能是 forbidden alias，必须且只能注册一次，schema 必须完整，`TAttrs` 必须和 matrix 一致，必须有 `FInferType`。`single` 必须有 `FRelayToTE`，`multi` 必须有 `FRelayToTEMulti`，`exec_plan` 不允许注册 TE lowering hook。需要 FFI 时必须有同名 canonical `_make` helper；声明的 ONNX 映射必须存在；声明需要测试时必须有测试引用。
8. 阶段按最远完成点推导：无注册为 `missing`，schema 不完整为 `registered`，缺 type 为 `schema`，缺 lowering 为 `typed`，缺 FFI 为 `lowered`，缺测试为 `ffi`，全部满足为 `tested`。对 `exec_plan` op，注册和 type 之后的 lowering 完成度由 execution-plan 路径表达，不由 `FRelayToTE` 表达。阶段只是进度展示，任何规范问题都会让检查失败。
9. 全局源码扫描会额外检查 op 链路相关文件中的占位关键字和 `return te::Tensor()` 空 tensor 返回。命中后记入 `Global issues`。
10. 默认模式下只要存在任意 operator issue 或 global issue 就返回非零退出码；`--report-only` 只改变退出码，不改变报告内容；`--format json` 输出机器可消费报告，便于 CI 或后续工具读取。
11. TIR/LLVM 是否真的支持某个 op 生成的 stmt、expr 或 intrinsic，不能只靠静态扫描判断。正确做法是把它拆成两层：checker 强制 matrix 中 `single` / `multi` op 必须有 `LowerToTIR` 契约测试，`exec_plan` op 必须有 `LowerRelayToExecPlanPass` 契约测试，可执行 op 必须有 C/LLVM compile 或 runtime numeric 测试；CI 实际运行这些测试。如果 TE/TOPI 生成了 TIR 不支持的节点，`LowerToTIR` 测试失败；如果生成了 LLVM codegen 不支持的 intrinsic、stmt 或 dtype 组合，LLVM compile/run 测试失败。
12. 新 op 只有在对应的 TIR/LLVM 契约测试进入 CMake/CI 后，才能把 matrix 中的 `tir_executable`、`llvm_required` 或 executable 状态标为已支持。否则即使静态字段齐全，也只能算“lowering 已注册但后端未证明”。

只想查看完整状态报告时可以运行：

```bash
python python/tools/check_relay_op_contract.py --root . --report-only
```

CMake 侧提供同名 target：

```bash
cmake --build --preset dev-mingw-cpu --target check_relay_op_contract
```

CI 应直接运行不带 `--report-only` 的版本；发现 alias、占位实现或半拉链路时立刻失败。

## 14. PR 检查清单

提交 PR 前检查：

- [ ] canonical op name 已确定。
- [ ] support matrix 已更新。
- [ ] attrs 已定义并实现 `Create`。
- [ ] `FInferType` 已注册或 matrix 明确标为未支持。
- [ ] `single` 已注册 `FRelayToTE`，`multi` 已注册 `FRelayToTEMulti`，`exec_plan` 已覆盖 `LowerRelayToExecPlanPass`。
- [ ] TOPI/TE helper 不返回空 tensor。
- [ ] ONNX importer 输出 canonical name。
- [ ] `_make` helper 名称和内部 `Op::Get` 都使用 canonical name。
- [ ] `single` / `multi` op 的 `LowerToTIR` 测试通过。
- [ ] `exec_plan` op 的 execution-plan 测试通过。
- [ ] 后端 numeric test 覆盖 executable op。
- [ ] 没有新增 metadata-only alias。
- [ ] 错误信息包含 canonical op name。
- [ ] CMake/CI target 已更新。
- [ ] `check_relay_op_contract` 已通过，或本 PR 明确只是在暴露既有缺口并附带报告。

## 15. 常见错误

只写 `KXC_REGISTER_OP`：

- 结果：Relay IR 能构造，但 `InferTypePass` 或 `LowerToTIR` 失败。
- 修复：补 `FInferType`，并在 matrix 中真实标注 lowering 状态。

只写 `FInferType` 不写 `FRelayToTE`：

- 结果：类型推导通过，但 lowering 报 `No FRelayToTE registered`。
- 修复：补 TE/TOPI compute 和 lowering test，或 matrix 标为 `lowering = none`。

TOPI 返回空 tensor：

- 结果：lowering 后续阶段报错，错误位置远离真实问题。
- 修复：不支持就抛错，支持就返回完整 compute。

helper 使用历史 alias：

- 结果：pass、type inference、lowering 可能匹配不到完整 metadata。
- 修复：helper 内部只使用 canonical op name。

TIR 使用后端不支持的 intrinsic：

- 结果：LLVM/C codegen 把它当外部函数或直接报 unsupported。
- 修复：统一 intrinsic name，并补 codegen 映射或专用生成逻辑。
