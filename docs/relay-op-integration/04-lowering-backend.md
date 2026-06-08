# 4. Lowering / TIR / LLVM API

本页描述 Relay op 如何进入 TE/TIR，以及生成的 TIR 必须落在哪些后端能力范围内。

## Relay lowering hook

位置：[include/relay/op_attr_types.h](../../include/relay/op_attr_types.h)

```cpp
using FRelayToTE =
    std::function<te::Tensor(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;

using FRelayToTEMulti =
    std::function<Array<te::Tensor>(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;
```

| hook | contract `lowering` | 输出类型 | 返回 |
| --- | --- | --- | --- |
| `FRelayToTE` | `single` | `TensorType` | 一个 defined `te::Tensor` |
| `FRelayToTEMulti` | `multi` | `TupleType` | 与 `TupleType.fields` 等长的 defined tensor 数组 |
| 不注册 TE hook | `exec_plan` | 设备通信语义 | 由 `LowerRelayToExecPlanPass` 处理 |

## `LowerToTIR` 当前能力

位置：[src/relay/backend/lower.cc](../../src/relay/backend/lower.cc)

支持：

| Relay 节点 | 支持情况 |
| --- | --- |
| `Var` | 从 `TensorType` 参数创建 TE placeholder |
| `Constant` | 当前以 constant placeholder 进入 TIR 参数 |
| `Call` + `TensorType` | 调 `FRelayToTE` |
| `Call` + `TupleType` | 调 `FRelayToTEMulti` |
| `Tuple` | 展平为多个 TE tensor；不支持嵌套 tuple field |
| `TupleGetItem` | 从多输出数组取对应 tensor |
| `device.` op | 不走 `LowerToTIR`，要求先走 execution plan |

输出 ABI：

| 顺序 | PrimFunc 参数 |
| --- | --- |
| 1 | Relay function inputs |
| 2 | Relay constants |
| 3 | output buffers，数量等于最终输出 tensor 数 |

`PrimFunc.attrs` 会写入 `kxc.input_count`、`kxc.constant_count`、`kxc.output_count`、`kxc.output_param_start`。

## TIR expr 支持表

| TIR 节点 | LowerToTIR | LLVM codegen | 备注 |
| --- | --- | --- | --- |
| `IntImm`, `FloatImm`, `Var` | Y | Y | 标量 |
| `Add`, `Sub`, `Mul`, `Div`, `Mod` | Y | Y | float/int 路径 |
| `Min`, `Max` | Y | Y | LLVM 用 compare + select |
| `EQ`, `LT`, `And`, `Or`, `Not` | Y | Y | bool/compare |
| `Load`, `Store` | Y | Y | 线性化 buffer index |
| `Call("cast")` | Y | Y | LLVM 直接生成 cast |
| `Call("exp")`, `log`, `sqrt`, `floor`, `ceil` | Y | Y | LLVM float intrinsic |
| `Select` | Y | Y | LLVM basic block + phi/select |
| `te::Reduce` | Y | Y | 只支持单 source，reduce type 为 sum/max/min |

如果新增 TOPI helper 生成了表外节点，必须先扩 `LowerToTIR`、TIR pass、C/LLVM codegen 和测试。

## TIR stmt 支持表

| TIR stmt | LLVM codegen | 备注 |
| --- | --- | --- |
| `For` | Y | 当前 lowering 生成 serial loop |
| `Store` | Y | 输出和中间 buffer 写入 |
| `SeqStmt` | Y | 多 compute 顺序执行 |
| `Allocate` | Y | 中间 tensor buffer |
| `IfThenElse` | Y | codegen 支持；当前 TOPI 多使用 `Select` |
| `LetStmt`, `AttrStmt`, `Block`, `Evaluate` | 部分后端/打印支持不等同于 op 可执行 | 新 op 不应依赖这些节点，除非先补 LLVM numeric test |

## 单输出 hook 模板

```cpp
te::Tensor XxxCompute(const Attrs& attrs,
                      const Array<te::Tensor>& inputs,
                      const kxc::Type& out_type) {
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error("xxx expects TensorType output");
    }
    te::Tensor out = te::topi::xxx(inputs[0], "T_xxx");
    if (!out.defined()) {
        throw std::runtime_error("xxx lowering returned undefined tensor");
    }
    return out;
}
```

## 多输出 hook 模板

```cpp
Array<te::Tensor> XxxCompute(const Attrs& attrs,
                             const Array<te::Tensor>& inputs,
                             const kxc::Type& out_type) {
    const auto* tuple_type = out_type.As<TupleTypeNode>();
    if (!tuple_type) {
        throw std::runtime_error("xxx expects TupleType output");
    }

    Array<te::Tensor> outputs;
    for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        const auto* field = tuple_type->fields[i].As<TensorTypeNode>();
        if (!field) {
            throw std::runtime_error("xxx expects TensorType tuple fields");
        }
        outputs.push_back(te::compute(
            /* shape from field */,
            [input = inputs[0], i](const Array<tir::Var>& indices) {
                return input(indices);
            },
            "T_xxx_" + std::to_string(i)));
    }
    return outputs;
}
```

多输出 hook 必须额外检查：

| 检查项 | 要求 |
| --- | --- |
| `TupleType` | `out_type` 必须是 `TupleTypeNode` |
| 输出数量 | 等于 `tuple_type->fields.size()` |
| 输出定义 | 每个 tensor 都 `defined()` |
| 输出唯一 | 不重复返回同一个 tensor |
| shape/dtype | 每个 tensor 和对应 field 一致 |

## Execution plan op

`device.` op 不注册 `FRelayToTE` 或 `FRelayToTEMulti`。它们通过：

```cpp
kxc::relay::LowerRelayToExecPlanPass(func)
```

生成 execution plan。测试需要在同一个测试函数块里引用 op name 和 `LowerRelayToExecPlanPass`，checker 才会认为 execution plan 覆盖存在。

## 后端能力扩展规则

| 新需求 | 需要改动 |
| --- | --- |
| 新 intrinsic | `include/te/topi/elemwise.h` 构造 `tir::Call`，`src/codegen/codegen_llvm.cc` 增加 intrinsic 映射，补 numeric test |
| 新 dtype cast | `DTypeFromCastCode`、type inference、LLVM `CastValue` 路径和 numeric test |
| 新 TIR expr | `include/tir/expr.h`、`src/tir/expr.cc`、TIR pass visitor、print、C/LLVM codegen |
| 新 stmt | `include/tir/stmt.h`、lowering、pass visitor、print、C/LLVM codegen |
| 新 reduce 类型 | `te::ReduceType`、identity、lowering update、TOPI helper、numeric test |

只改 Relay 注册但不扩后端，会在 `LowerToTIR`、LLVM compile 或 runtime numeric test 中失败。
