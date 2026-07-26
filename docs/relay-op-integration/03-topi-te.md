# 3. TOPI / TE API

TOPI 是构造 TE compute 的 helper 集合。它是“可用于搭算子”的内部构造件，不等价于 Relay op 已支持。一个 TOPI helper 只有被 Relay lowering 使用、进入 contract，并通过 type/lowering/backend/tests 后，才算对外支持。

## 入口和命名空间

| 内容 | 位置 |
| --- | --- |
| 汇总头文件 | `include/te/topi.h` |
| broadcast | `include/te/topi/broadcast.h` |
| elemwise | `include/te/topi/elemwise.h` |
| reduction | `include/te/topi/reduction.h` |
| transform | `include/te/topi/transform.h` |
| nn | `include/te/topi/nn.h` |
| TE 基础 API | `include/te/te.h` |

所有 TOPI helper 在命名空间 `kxc::te::topi` 下。

## TE 构造件

| API | 返回 | 用途 | 约束 |
| --- | --- | --- | --- |
| `te::placeholder(shape, dtype, name)` | `te::Tensor` | 构造外部输入或常量占位 | lowering 自动为 Relay 参数和常量创建 |
| `te::compute(shape, lambda, name, tag)` | `te::Tensor` | 构造 compute tensor | lambda 返回一个 `tir::PrimExpr` |
| `te::reduce_axis(min, extent, name)` | `te::IterVar` | 构造 reduce 轴 | 配合 `te::sum/max/min` |
| `te::sum(expr, axes)` | `tir::PrimExpr` | sum reduce | 支持单 source reduce |
| `te::max(expr, axes)` | `tir::PrimExpr` | max reduce | 支持单 source reduce |
| `te::min(expr, axes)` | `tir::PrimExpr` | min reduce | 支持单 source reduce |
| `tensor(indices...)` | `tir::PrimExpr` | 读取 tensor 元素 | indices rank 必须匹配 tensor rank |

## TOPI helper 表

### Broadcast

| API | Header | 输入 | 输出 | 当前限制 |
| --- | --- | --- | --- | --- |
| `broadcast_to(t, output_shape)` | `broadcast.h` | tensor, 输出 shape | 指定 shape tensor | shape 必须可被后端线性化 |
| `add(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | dtype 应由 Relay type rule 先保证兼容 |
| `subtract(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | 同上 |
| `multiply(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | Relay canonical 名是 `mul` |
| `divide(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | 同上 |
| `maximum(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | 生成 `tir::Max` |
| `minimum(A, B)` | `broadcast.h` | 两个 tensor | broadcast 后 shape | 生成 `tir::Min` |

### Elemwise

| API | Header | 输入 | 输出 | 当前限制 |
| --- | --- | --- | --- | --- |
| `exp(x)` | `elemwise.h` | tensor | shape 不变 | LLVM 支持 float intrinsic |
| `log(x)` | `elemwise.h` | tensor | shape 不变 | LLVM 支持 float intrinsic |
| `sqrt(x)` | `elemwise.h` | tensor | shape 不变 | LLVM 支持 float intrinsic |
| `floor(x)` | `elemwise.h` | tensor | shape 不变 | LLVM 支持 float intrinsic |
| `ceil(x)` | `elemwise.h` | tensor | shape 不变 | LLVM 支持 float intrinsic |
| `sigmoid(x)` | `elemwise.h` | tensor | shape 不变 | 只适合 float 路径 |
| `identity(x)` | `elemwise.h` | tensor | shape 不变 | 用于显式拷贝 |
| `negative(x)` | `elemwise.h` | tensor | shape 不变 | 生成 `0 - x` |
| `clip(x, min, max)` | `elemwise.h` | tensor, 标量边界 | shape 不变 | 生成 `Min/Max` |
| `cast(x, dtype)` | `elemwise.h` | tensor, `tir::DataType` | shape 不变，dtype 改变 | LLVM 通过 `Call("cast")` 支持 |

### Reduction

| API | Header | 输入 | 输出 | 当前限制 |
| --- | --- | --- | --- | --- |
| `sum(data, axis, keepdims)` | `reduction.h` | tensor, axes | reduce shape | 支持单 source reduce |
| `max(data, axis, keepdims)` | `reduction.h` | tensor, axes | reduce shape | 支持单 source reduce |
| `min(data, axis, keepdims)` | `reduction.h` | tensor, axes | reduce shape | 支持单 source reduce |
| `prod(data, axis, keepdims)` | `reduction.h` | tensor, axes | 无 | 当前显式 unsupported |

### Transform

| API | Header | 输入 | 输出 | 当前限制 |
| --- | --- | --- | --- | --- |
| `transpose(x, axes)` | `transform.h` | tensor, axes | permuted shape | axes 必须合法且无重复 |
| `expand_dims(x, axis, num_newaxis)` | `transform.h` | tensor | 插入 1 维 | 仅作为内部构造件 |
| `squeeze(x, axes)` | `transform.h` | tensor | 删除 1 维 | 仅作为内部构造件 |
| `concatenate(inputs, axis)` | `transform.h` | tensor array | 拼接后 shape | 空输入显式失败；Relay concat 当前不在 MVP contract |

### NN

| API | Header | 输入 | 输出 | 当前限制 |
| --- | --- | --- | --- | --- |
| `relu(x)` | `nn.h` | tensor | shape 不变 | 生成 `Max(x, 0)` |
| `leaky_relu(x, alpha)` | `nn.h` | tensor | shape 不变 | 仅作为内部构造件 |
| `dense(A, B, bias)` | `nn.h` | `[M,K]`, `[N,K]`, optional `[N]` | `[M,N]` | weight 语义是转置布局 `[N,K]` |
| `matmul(A, B)` | `nn.h` | `[M,K]`, `[K,N]` | `[M,N]` | rank-2 MVP |
| `conv2d_nchw(data, kernel, strides, padding, dilation)` | `nn.h` | NCHW, OIHW | NCHW | `strides` / `dilation` 为 `AxisPair2D`，`padding` 为 `Padding2D`；groups/layout 由 Relay type rule 限制 |
| `pool2d(data, kernel, stride, padding, dilation, type, ceil)` | `nn.h` | NCHW | NCHW | kernel/stride/dilation 为 `AxisPair2D`；`pool_type` 只支持 `max` / `avg` |
| `global_avg_pool2d(data)` | `nn.h` | NCHW | `[N,C,1,1]` | NCHW |

> **属性规范化必须走共享 helper。** `strides`、`dilation`、`pool_size` 用
> `te::topi::ExpandPair2D`，`padding` 用 `te::topi::ExpandPadding2D`，两者都在
> `kxc/te/topi/window.h`。输出尺寸公式用同一头文件的 `WindowOutputExtent`。
>
> 不要在 lowering 里自己展开这些属性。类型推导和 TE compute 必须对同一属性得出
> 同一组值，否则推导出的 `TensorType` 与 compute 出的 shape 不一致，会在
> `LowerCompilationUnit` 的边界校验处报 `Unit TE output shape mismatch`——错误
> 指向编译器内部不变量，而不是真正出错的算子属性。单元素写法（如 `{2}`）和
> 非对称 `padding` 是最容易出分歧的两处。

旧的 TOPI 调用签名仍作为兼容重载保留：`conv2d_nchw` 的两个 padding 整数继续按
高、宽方向对称展开；`pool2d` 的 `Array<int>` 形式继续默认 `dilation={1,1}`。
此前使用 `Padding2D` 但仍分别传递 int/`Array<int>` stride、dilation 的过渡签名也
保留为转发重载。
新代码应先用上述共享 helper 规范化属性，再调用 `AxisPair2D` / `Padding2D` 主接口，
从而保留 Relay 的 `int64_t` 属性精度并支持非对称 padding 与显式 dilation。

## 选择规则

| 情况 | 选择 |
| --- | --- |
| 现有 TOPI helper 语义完全匹配 | 在 `FRelayToTE` 中直接调用 |
| 只服务一个 op，逻辑很短 | 可以在 op 源文件内写局部 `te::compute` |
| 多个 op 可复用，或索引逻辑复杂 | 新增 TOPI helper |
| 后端不支持生成的 TIR 节点或 intrinsic | 先扩 TIR/LLVM 并加测试，再把 Relay op 标为 supported |
| 当前不能真实实现 | 显式抛出 unsupported；不要返回空 tensor 或假结果 |

## `FRelayToTE` 中调用 TOPI

```cpp
te::Tensor XxxCompute(const Attrs& attrs,
                      const Array<te::Tensor>& inputs,
                      const kxc::Type& out_type) {
    (void)attrs;
    if (inputs.size() != 2) {
        throw std::runtime_error("xxx expects exactly 2 inputs");
    }
    if (!out_type.As<TensorTypeNode>()) {
        throw std::runtime_error("xxx expects TensorType output");
    }
    te::Tensor out = te::topi::add(inputs[0], inputs[1], "T_xxx");
    if (!out.defined()) {
        throw std::runtime_error("xxx lowering returned undefined tensor");
    }
    return out;
}
```

## 新增 TOPI helper 模板

```cpp
inline Tensor xxx(const Tensor& x,
                  std::string name = "xxx",
                  std::string tag = kElementWise) {
    return compute(
        x->shape,
        [x](const Array<tir::Var>& indices) {
            return x(indices) + make_const(x->dtype, 1);
        },
        name,
        tag);
}
```

新增 helper 后必须检查：

| 检查项 | 要求 |
| --- | --- |
| header | 放到对应 `include/te/topi/*.h` |
| namespace | `kxc::te::topi` |
| 输入校验 | 不支持形态立即抛错 |
| dtype | 常量用 `make_const(dtype, value)` |
| shape | 输出 shape 写清楚，必要时由 Relay `out_type` 控制 |
| 后端 | 生成的 TIR expr/stmt 在 [04-lowering-backend.md](04-lowering-backend.md) 支持表内 |
| 测试 | Relay op 进入 contract 后必须有 lowering/backend numeric 覆盖 |

## 不允许的写法

```cpp
return te::Tensor();
```

```cpp
// 用 sum 冒充 prod
return sum(data, axis, keepdims, name);
```

```cpp
// 注册 Relay op 但 TOPI helper 只是先凑一个结果
```

这些都会让 checker 或 CI 失败；即使 checker 没扫描到，也不能作为 PR 合入。
