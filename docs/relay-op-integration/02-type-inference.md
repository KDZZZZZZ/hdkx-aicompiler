# 2. Type Inference API

`FInferType` 是 Relay op 到 typed Relay 的唯一入口。新增 op 不允许依赖 TOPI 或 lowering 来补猜 shape；type rule 必须在 `InferTypePass` 阶段给出确定的 `TensorType` 或 `TupleType`，不支持的形态立即抛错。

## 接口

位置：

- 声明：[include/relay/op_attr_types.h](../../include/relay/op_attr_types.h)
- 公共规则声明：[include/relay/type_infer.h](../../include/relay/type_infer.h)
- 实现：[src/relay/type_infer.cc](../../src/relay/type_infer.cc)

签名：

```cpp
using FInferType = std::function<Type(const Attrs&, const Array<Type>&)>;
```

注册：

```cpp
.set_attr<FInferType>("FInferType", XxxInferType)
```

## 可选 type rule

| 规则 | 直接可用函数 | 适用场景 |
| --- | --- | --- |
| identity | `IdentityInferType` | 输出类型完全等于第一个输入 |
| unary same | `UnarySameInferType` | 单输入，shape/dtype 不变，例如 `sqrt`, `nn_relu` |
| binary broadcast | `AddInferType`, `SubtractInferType`, `MultiplyInferType`, `DivideInferType` | 二元 broadcast，输出 dtype 等于输入 |
| bool broadcast | `EqualInferType`, `GreaterInferType` | 二元比较，输出 dtype 是 `bool` |
| cast | `CastInferType` | 输出 shape 不变，dtype 来自 `CastAttrs` |
| matmul-like | `MatMulInferType`, `DenseInferType`, `GemmInferType` | rank-2 矩阵类 |
| conv/pool | `Conv2DInferType`, `Pool2DInferType`, `GlobalAvgPool2DInferType` | NCHW/OIHW MVP |
| transform | `FlattenInferType`, `ReshapeInferType`, `TransposeInferType` | shape 由 attrs 或输入 rank 决定 |
| reduce | `ReduceMeanInferType` | axis/keepdims reduce |
| softmax | `SoftmaxInferType` | shape/dtype 不变，但检查 axis |
| tuple output | `SplitInferType` | 返回 `TupleType(fields)` |
| indexed output | `GatherInferType` | 输出 shape 由 data 和 indices 组合 |
| conditional | `WhereInferType` | condition/x/y broadcast |

如果没有合适规则，新增 `XxxInferType`。不要在 op 注册里复用一个语义不相同的近似规则。

## 输入和错误约束

| 检查项 | 要求 |
| --- | --- |
| 输入数量 | 第一行检查 arity；错误信息包含 op name |
| 输入类型 | 需要 tensor 时必须检查 `TensorTypeNode` |
| dtype | 明确检查相等、目标 dtype 或 bool 条件 |
| axis | 负轴要归一化；越界立即失败 |
| shape | 静态 MVP 中未知维度用 `-1` 表示；不能执行的动态 shape 在 lowering 前失败 |
| tuple | 多输出返回 `TupleType`；`TupleGetItem` 依赖 `InferTypePass` 取 field 类型 |

## 可直接使用的类型构造件

| 构造件 | 用法 |
| --- | --- |
| `TensorType(Array<int64_t>, dtype)` | 构造单 tensor 输出类型 |
| `TupleType(Array<Type>)` | 构造多输出类型 |
| `TypeToString(type)` | 错误信息中打印类型 |
| `attrs.As<XxxAttrsNode>()` | 读取 attrs |
| `input_types[i].As<TensorTypeNode>()` | 读取输入 tensor shape/dtype |

## 新增规则模板

```cpp
Type XxxInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() != 1) {
        throw std::runtime_error("xxx expects exactly 1 input");
    }
    const auto* data = input_types[0].As<TensorTypeNode>();
    if (!data) {
        throw std::runtime_error("xxx expects TensorType for data, got " +
                                 TypeToString(input_types[0]));
    }
    const auto* p = attrs.As<XxxAttrsNode>();
    if (!p) {
        throw std::runtime_error("xxx expects XxxAttrs");
    }

    Array<int64_t> out_shape;
    for (int64_t dim : data->shape) {
        out_shape.push_back(dim);
    }
    return TensorType(out_shape, data->dtype);
}
```

## 多输出模板

```cpp
Type SplitLikeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    if (input_types.size() != 1) {
        throw std::runtime_error("split_like expects exactly 1 input");
    }
    const auto* data = input_types[0].As<TensorTypeNode>();
    if (!data) {
        throw std::runtime_error("split_like expects TensorType input");
    }

    Array<Type> fields;
    fields.push_back(TensorType({data->shape[0], 1}, data->dtype));
    fields.push_back(TensorType({data->shape[0], data->shape[1] - 1}, data->dtype));
    return TupleType(fields);
}
```

`FInferType` 只负责类型和形状，不创建 TE tensor，不读取 runtime data，不做数值计算。
