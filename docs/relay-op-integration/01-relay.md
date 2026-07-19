# 1. Relay 注册 API

本页描述 Relay 层可用的注册和构图接口。新增 op 时只在本页列出的接口中选择，不新增旁路注册方式。

## Op 注册

位置：

- 宏定义：[include/relay/op_macros.h](../../include/relay/op_macros.h)
- 注册实现文件：按 [00-contract.md](00-contract.md) 的 `category` 选择

接口：

```cpp
KXC_REGISTER_OP(op_name)
    .describe("...")
    .set_num_inputs(n)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "XxxAttrs")
    .set_attr<FInferType>("FInferType", XxxInferType)
    .set_attr<FRelayToTE>("FRelayToTE", XxxCompute);
```

| API | 必填 | 参数 | 约束 |
| --- | --- | --- | --- |
| `KXC_REGISTER_OP(op_name)` | 是 | canonical op name | 宏参数不能带 `.`；`device.` op 用 `OpRegEntry(Op::Get("device.xxx"))` |
| `.describe(text)` | 是 | 英文描述 | checker 只检查存在性，内容要能说明语义 |
| `.set_num_inputs(n)` | 是 | 固定数量或 `-1` | 必须和 contract 一致 |
| `.add_argument(name, type, desc)` | 是 | 每个输入一条 | 至少覆盖固定输入数或 `min_inputs` |
| `.set_attr<std::string>("TAttrs", "...")` | 有 attrs 时必填 | attrs 类名 | 必须和 contract 的 `attrs` 一致 |
| `.set_attr<FInferType>("FInferType", fn)` | 是 | type rule | 所有 op 都要有 |
| `.set_attr<FRelayToTE>("FRelayToTE", fn)` | `single` 必填 | 单输出 lowering | 输出必须是 defined tensor |
| `.set_attr<FRelayToTEMulti>("FRelayToTEMulti", fn)` | `multi` 必填 | 多输出 lowering | 返回数量必须等于 `TupleType.fields` |

## Attrs 定义

位置：

- 定义：[include/relay/op.h](../../include/relay/op.h)
- `Create` 实现：[src/relay/op_attrs.cc](../../src/relay/op_attrs.cc)

选择：

| 场景 | 用法 |
| --- | --- |
| 无额外参数 | 不写 `TAttrs`，构图时 `Call(op, inputs)` |
| 只有开关或无字段标记 | 用 `KXC_DEFINE_SIMPLE_ATTRS(XxxAttrs)` |
| 有 axis、shape、layout、dtype 等编译期参数 | 定义 `XxxAttrsNode` 字段、`XxxAttrs` ref、`XxxAttrs::Create(...)` |
| 参数是运行时 tensor | 放进 `Call` inputs，不塞进 attrs |

模板：

```cpp
class XxxAttrsNode : public BaseAttrsNode {
public:
    int axis = 0;
    Array<int64_t> shape;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(XxxAttrsNode)

class XxxAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(XxxAttrs, XxxAttrsNode)

public:
    static XxxAttrs Create(int axis, Array<int64_t> shape);
};
```

```cpp
XxxAttrs XxxAttrs::Create(int axis, Array<int64_t> shape) {
    auto* node = new XxxAttrsNode();
    node->axis = axis;
    node->shape = std::move(shape);
    return InternalCreate(node);
}
```

## FFI `_make` helper

位置：[src/relay/op/op_ffi.cc](../../src/relay/op/op_ffi.cc)

接口：

```cpp
Call MakeXxx(Expr data, int axis) {
    return Call(GetOp("xxx"), {data}, XxxAttrs::Create(axis));
}

KXC_REGISTER_GLOBAL("kxc.relay.op._make.xxx").set_body(ToPackedFunc(MakeXxx));
```

规则：

| 项 | 规则 |
| --- | --- |
| helper 名 | 必须是 `kxc.relay.op._make.<canonical_op_name>` |
| `GetOp` | 必须引用同一个 canonical op |
| 输入数量 | `Call(..., {inputs})` 数量必须和 contract 一致 |
| attrs | 只传编译期参数；runtime tensor 参数必须在 inputs 中 |
| alias | 不允许 `_make.div` 调到 `divide` 这类兼容 helper |

## 可直接使用的 Relay 构造件

| 构造件 | 位置 | 用途 |
| --- | --- | --- |
| `Op::Get("name")` | `include/relay/op.h` | 取 canonical op |
| `Call(op, args, attrs)` | `include/relay/relay.h` | 构造 Relay call |
| `Var(name, TensorType(...))` | `include/relay/relay.h` | 构造函数参数 |
| `TensorType(shape, dtype)` | `include/relay/relay.h` | 单输出 tensor 类型 |
| `TupleType(fields)` | `include/relay/relay.h` | 多输出类型 |
| `Tuple(fields)` / `TupleGetItem(tuple, i)` | `include/relay/relay.h` | 多输出表达式和投影 |

## 最小单输出示例

```cpp
te::Tensor SqrtCompute(const Attrs& attrs,
                       const Array<te::Tensor>& inputs,
                       const kxc::Type& out_type) {
    (void)attrs;
    if (inputs.size() != 1) {
        throw std::runtime_error("sqrt expects exactly 1 input");
    }
    if (!out_type.As<TensorTypeNode>()) {
        throw std::runtime_error("sqrt expects TensorType output");
    }
    return te::topi::sqrt(inputs[0], "T_sqrt");
}

KXC_REGISTER_OP(sqrt)
    .describe(R"doc(Square root of elements.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SqrtCompute);
```
