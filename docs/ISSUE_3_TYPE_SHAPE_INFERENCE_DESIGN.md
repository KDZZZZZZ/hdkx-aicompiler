# Issue #3 Relay Type/Shape Inference 实现设计

状态：Proposed
日期：2026-06-06
关联 issue：https://github.com/KDZZZZZZ/hdkx-aicompiler/issues/3

## 1. 范围

Issue #3 的目标是为 TinyTVM MVP 实现 Relay 静态 type/shape inference。它不是完整 TVM Relay 类型系统，而是让当前编译闭环具备一个可靠的类型契约：

```text
Relay Function
  -> InferTypePass
  -> Relay optimization passes
  -> InferTypePass
  -> LowerToTIR
  -> TIR optimization/codegen/runtime
```

MVP 只支持静态 tensor shape。用户只需要给模型输入和常量提供类型信息，中间 `Call` 节点的 `checked_type_` 由推断 pass 填充。rank、dtype、attrs 或 shape 组合非法时，必须在 Relay type inference 阶段报错，而不是拖到 TE/TIR lowering 或 codegen 阶段。

Issue #3 覆盖的算子族：

- Elementwise broadcast：`add`、`subtract`、`mul`、`divide`、`sqrt`、`cast`
- Matrix：`matmul`、`nn_dense`、`nn_gemm`
- Convolution：`nn_conv2d`，限定 NCHW / OIHW
- Pooling：`nn_max_pool2d`、`nn_avg_pool2d`、`nn_global_avg_pool2d`
- Transform：`nn_flatten`、`reshape`、`transpose`
- Reduction：`reduce_mean`
- Normalization：`softmax`

不在 issue #3 范围内：

- 完整 dynamic shape
- symbolic shape expression
- 完整 ONNX opset
- 完整 tuple/multi-output lowering
- schedule 级 shape specialization
- 全量修复 issue #2 的 op 命名规范问题

但 issue #3 的实现不能加重 issue #2 的负担。类型规则应优先使用 canonical op name；必要的历史 alias 只能作为临时兼容路径。

## 2. 现状

### 2.1 已有类型元数据，但不是编译契约

当前仓库已经有最小类型结构：

- [include/base/expr.h](../include/base/expr.h) 定义了 `TypeNode`、`Type` 和 `ExprNode::checked_type_`。
- [include/relay/relay.h](../include/relay/relay.h) 定义了 `TensorTypeNode`，字段为 `shape: Array<int64_t>` 和 `dtype: std::string`。
- `VarNode` 有 `type_annotation`，调用方可以写 `Var("x", TensorType(...))`。

缺口是：

- 没有统一的 `SetCheckedType` / `GetCheckedTypeOrThrow` API。
- 没有 pass 会系统性填充 `checked_type_`。
- 现有 Relay pass 重建节点时只保留 `virtual_device_`，不保留 `checked_type_`。
- `checked_type_` 现在只是一个可能为空的字段，不是 pipeline invariant。

### 2.2 LowerToTIR 已经隐式依赖 typed Relay

[src/relay/backend/lower.cc](../src/relay/backend/lower.cc) 里有两处关键依赖：

- `LowerToTIR` 要求 function params 有 `TensorType` annotation。
- `RelayToTEConverter::VisitCall` 会读取 `ref.checked_type()`，然后传给 op 的 `FRelayToTE`。

目前一些 `FRelayToTE` callback 不使用 `out_type`，而是在 TE/TOPI 里根据输入 tensor shape 现算输出 shape。这让简单图可以侥幸 lower，但带来两个问题：

- shape 错误晚于 Relay 阶段才暴露。
- TE helper 可能接受不合法 shape，例如 broadcast 维度不匹配时静默选择较大维度。

Issue #3 后，`LowerToTIR` 应只接收已经被验证的 typed Relay function。

### 2.3 Op registry 有 lowering hook，没有 type hook

[include/relay/op_attr_types.h](../include/relay/op_attr_types.h) 目前只有：

```cpp
using FRelayToTE =
    std::function<te::Tensor(const Attrs&, const Array<te::Tensor>&, const Type&)>;
```

缺少 `FInferType`。虽然 [docs/OPERATOR_REGISTRATION_GUIDE.md](OPERATOR_REGISTRATION_GUIDE.md) 提到了 `FInferShape` / `FInferType`，但代码里没有对应实现。

当前 op 注册状态也不均匀：

- 有些 op 已有 `FRelayToTE`，例如 `add`、`nn_relu`、`nn_conv2d`、pool、dense、gemm、flatten。
- 有些 issue #3 op 只有 schema/attrs，还没有 lowering hook，例如 `matmul`、`mul`、`subtract`、`divide`、`sqrt`、`cast`、`reshape`、`transpose`、`reduce_mean`、`softmax`。
- 有些 op name 不一致，例如 `softmax` / `nn_softmax`、`mul` / `multiply`、`sub` / `subtract`。

Issue #3 可以先实现 type rule，但测试必须区分“type inference 支持”和“lowering 支持”。

### 2.4 Shape 逻辑散落在 TE/TOPI

当前 shape 公式主要在 TE/TOPI helper 里：

- [include/te/topi/broadcast.h](../include/te/topi/broadcast.h)：broadcast shape
- [include/te/topi/nn.h](../include/te/topi/nn.h)：dense、matmul、conv2d、pool
- [include/te/topi/transform.h](../include/te/topi/transform.h)：transpose、squeeze、concat 等
- [include/te/topi/reduction.h](../include/te/topi/reduction.h)：reduce

这些代码负责构造 TE compute，不应该承担 Relay 类型校验职责。Relay inference 必须成为更早、更严格的一层。

### 2.5 Pipeline 和 Compiler 没有运行 type inference

[src/relay/transforms/pipeline.cc](../src/relay/transforms/pipeline.cc) 没有 `infer_type` pass entry，`optimize_default` 也不会运行类型推断。

[src/api/compiler.cc](../src/api/compiler.cc) 当前流程是：

```text
RunRelayPassPipeline
LowerToTIR
RunTIRPassPipeline
Codegen
```

缺少优化前后的类型验证。Adaptive runtime 通过 `api::Compiler::Compile` 做同步和后台编译，因此只要 compiler 入口接入 inference，冷热自适应路径也会被覆盖。

### 2.6 Debug 输出不显示 checked type

[src/relay/pass/print_ir.cc](../src/relay/pass/print_ir.cc) 打印 Var、Constant、Call、Tuple、Let、Function，但不打印 `checked_type_`。这会让 typed Relay 难以调试，也会降低 profiling IR artifact 的价值。

## 3. 必要性

Type/shape inference 是 TinyTVM 从“能手写 smoke test”走向“能编译模型图”的基础设施。

必须做的原因：

1. **给 lowering 一个稳定前置条件。**
   `LowerToTIR` 不应该自己猜 shape，也不应该在 TE 阶段才发现 rank/dtype 错误。

2. **统一算子扩展链路。**
   新增 op 时必须同时提供 schema、attrs、`FInferType`、`FRelayToTE` 和测试。否则同一个 shape 规则会散落在 FFI、TOPI、lowering 和测试里。

3. **让错误更早、更可读。**
   Conv layout 错误、reduce axis 越界、broadcast shape 不兼容、dtype 不匹配，应在 Relay inference 阶段报出 op name 和具体参数。

4. **减少构图端手动传播类型。**
   ONNX 或 C++ graph builder 只需要标注输入和常量，中间节点由 pass 推断。

5. **支撑后续 runtime specialization。**
   adaptive runtime 会对 hot shape 重复编译同一个 Relay function。typed Relay 是 shape specialization、kernel cache 和 profiling 的共同基线。

## 4. 实现目标模型

### 4.1 类型模型

MVP 类型系统保持小而明确：

```text
Type
  TensorType(shape: Array<int64_t>, dtype: string)
  TupleType(fields: Array<Type>)              建议实现
  FuncType(params: Array<Type>, ret: Type)    可选
```

Issue #3 必须支持：

- `TensorType`
- `TensorType` equality
- `TensorType` string formatting
- dtype normalization / comparison
- shape helper：rank、product、axis normalize、broadcast、conv/pool output formula

建议同 PR 或紧随其后支持：

- `TupleType`：IR 已经有 `Tuple` 和 `TupleGetItem`，现有 pass 测试也使用 tuple。
- `FuncType`：不是 lowering 必需，但可以让 `FunctionNode::checked_type_` 更完整。

静态 shape 规则：

- `TensorType.shape` 只能包含 concrete non-negative `int64_t`。
- scalar tensor 用空 shape `[]`。
- `-1` 和 `0` 只允许作为 `ReshapeAttrs::newshape` 的语法，解析后的 `TensorType` 不应包含 `-1`。
- dynamic/symbolic dim 不进入 issue #3。

dtype 规则：

- canonical dtype string：`float32`、`float64`、`int32`、`int64`、`int8`、`uint8`、`bool`。
- MVP 默认不做隐式 dtype promotion。
- 除 `cast` 外，二元 numeric op 要求 dtype 相同。

### 4.2 InferTypePass 模型

新增 Relay pass：

```cpp
Function InferTypePass(const Function& func);
```

核心流程：

1. 从 function params 构建 `Var -> Type` 环境。
2. 后序遍历 Relay 表达式。
3. 先推断子表达式类型。
4. 根据节点类型和 op type relation 推断当前节点类型。
5. 写入 `ExprNode::checked_type_`。
6. 返回一个 typed `Function`。

节点行为：

- `Var`：使用 `type_annotation`。function param 缺少 `TensorType` 时直接报错。
- `Constant`：从 `runtime::NDArray` 的 shape 和 dtype 生成 `TensorType`。
- `Call`：检查 `Call.op`、arity、attrs 和 input types；调用 op 的 `FInferType`。
- `Let`：先推断 value；如果 let var 有 annotation，需要和 value type 一致；body 中 var 绑定为 value type。
- `Tuple`：如果实现 `TupleType`，fields 类型组成 tuple type。
- `TupleGetItem`：要求输入是 `TupleType`，index 合法。
- `If`：MVP 可以支持基本校验，但不要求 lowerable；cond 应为 bool scalar，两个分支类型一致。
- `Function`：body type 作为 return type；如果实现 `FuncType`，params 和 body 组成 function type。

推荐新增 helper，避免到处 `const_cast`：

```cpp
void SetCheckedType(const Expr& expr, Type type);
Type GetCheckedTypeOrThrow(const Expr& expr, const std::string& context);
```

由于现有 Relay pass 可能重建节点并丢失类型，compiler 入口至少要在 Relay optimize 后再次运行 `InferTypePass`。

推荐 compiler 顺序：

```text
typed_input = InferTypePass(func)
optimized = RunRelayPassPipeline(typed_input, relay_passes)
typed_optimized = InferTypePass(optimized)
prim_func = LowerToTIR(typed_optimized)
```

### 4.3 Op type relation 模型

在 `FRelayToTE` 旁边新增：

```cpp
using FInferType =
    std::function<Type(const Attrs& attrs, const Array<Type>& input_types)>;
```

op 注册形态：

```cpp
KXC_REGISTER_OP(add)
    .set_num_inputs(2)
    .set_attr<FInferType>("FInferType", BroadcastBinaryInferType)
    .set_attr<FRelayToTE>("FRelayToTE", AddCompute);
```

职责划分：

- `InferTypePass` 负责通用流程：查 op、查 arity、查 `FInferType`、记录错误上下文。
- `FInferType` 负责具体 op 规则：rank、dtype、attrs、shape 公式和输出类型。

这样算子扩展链路会变成：

```text
op schema
  -> attrs
  -> FInferType
  -> FRelayToTE
  -> pass/lowering/codegen tests
```

### 4.4 错误模型

MVP 用 deterministic `std::runtime_error` 即可。错误消息必须包含：

- op name
- argument index 或 attr name
- 期望条件
- 实际类型或实际值

示例：

```text
InferType(nn_conv2d): expected data rank 4 NCHW, got TensorType(shape=[1, 224, 224], dtype=float32)
InferType(add): broadcast dim mismatch at axis -1: lhs=7, rhs=5
InferType(reduce_mean): axis 4 out of range for rank 4
```

### 4.5 Lowering 契约

Issue #3 后：

- `Compiler::Compile` 负责在 lowering 前运行 `InferTypePass`。
- `LowerToTIR` 可以假设输入是 typed Relay，但仍要 guard：
  - params 必须有 `TensorType`
  - function body 必须有 defined `checked_type_`
  - MVP 单输出 call 的 `checked_type_` 必须是 `TensorType`
- `FRelayToTE` 不再负责修复非法 shape；它只消费已经验证过的类型。

## 5. MVP 算子规则

### 5.1 Elementwise Broadcast

算子：

- `add`
- `subtract`
- `mul`
- `divide`

规则：

- 输入必须是 `TensorType`。
- dtype 必须相同。
- 输出 dtype 等于输入 dtype。
- shape 使用严格 NumPy-style right-aligned broadcast：
  - `a == b` 可兼容
  - `a == 1` 可 broadcast 到 `b`
  - `b == 1` 可 broadcast 到 `a`
  - 其它情况报错

临时 alias：

- `multiply` 可临时映射到 `mul`。
- `sub` 可临时映射到 `subtract`。
- `div` 只作为 frontend helper alias 映射到 `divide`，不建议新增 canonical op。

### 5.2 Unary / Cast

`sqrt`：

- 输入必须是 `TensorType`。
- 输出 shape 等于输入 shape。
- 输出 dtype 等于输入 dtype。
- MVP 建议只接受 float dtype。

`cast`：

- 输入必须是 `TensorType`。
- 输出 shape 等于输入 shape。
- 输出 dtype 来自 `CastAttrs`。
- 当前 `CastAttrs::to` 是整数编码，需要定义唯一映射，或后续改成 dtype string。

### 5.3 Matmul / Dense / Gemm

`matmul`：

- A rank-2 `[M, K]`
- B rank-2 `[K, N]`
- dtype 相同
- 输出 `[M, N]`

`nn_dense`：

- data `[M, K]`
- weight `[N, K]`
- dtype 相同
- `DenseAttrs::units` 如果大于 0，必须等于 `N`
- `out_dtype` 非空时作为输出 dtype，否则等于 data dtype
- 输出 `[M, N]`

`nn_gemm`：

- A、B 必须 rank-2，C 必须能 broadcast 到 `[M, N]`
- 当前 lowering 不支持 `transA=1`，inference 应直接报错
- `transB=1` 时 B 视为 `[N, K]`
- `transB=0` 时 B 视为 `[K, N]`
- A/B/C dtype MVP 要求一致
- 输出 `[M, N]`

### 5.4 Conv2D

`nn_conv2d`：

- data layout：`NCHW` 或空字符串视为 `NCHW`
- kernel layout：`OIHW` 或空字符串视为 `OIHW`
- data rank-4 `[N, C_in, H, W]`
- weight rank-4 `[C_out, C_weight, KH, KW]`
- MVP 只支持 `groups == 1`
- `C_in == C_weight`
- `channels` 如果大于 0，必须等于 `C_out`
- `kernel_size` 如果非空，必须等于 `[KH, KW]`
- `strides`、`padding`、`dilation` 归一化为 2D/4D 参数

输出公式：

```text
eff_kh = (KH - 1) * dilation_h + 1
eff_kw = (KW - 1) * dilation_w + 1
OH = floor((H + pad_top + pad_bottom - eff_kh) / stride_h) + 1
OW = floor((W + pad_left + pad_right - eff_kw) / stride_w) + 1
```

输出类型：

```text
TensorType([N, C_out, OH, OW], out_dtype_or_data_dtype)
```

`OH <= 0` 或 `OW <= 0` 时必须报错。

备注：当前 `Conv2DCompute` 支持可选 bias，但 op registration 是 2 输入。bias 支持应延后到算子规范链路统一后处理。

### 5.5 Pooling

`nn_max_pool2d` / `nn_avg_pool2d`：

- input rank-4 NCHW `[N, C, H, W]`
- layout 为空或 `NCHW`
- `pool_size`、`strides`、`padding` 归一化
- 当前 TOPI pool 忽略 dilation，因此 MVP 要求 dilation 为空或全 1

输出公式：

```text
numerator_h = H + pad_top + pad_bottom - KH
numerator_w = W + pad_left + pad_right - KW
OH = ceil_mode ? floor((numerator_h + stride_h - 1) / stride_h) + 1
               : floor(numerator_h / stride_h) + 1
OW = ceil_mode ? floor((numerator_w + stride_w - 1) / stride_w) + 1
               : floor(numerator_w / stride_w) + 1
```

输出 `[N, C, OH, OW]`，dtype 等于输入 dtype。`OH <= 0` 或 `OW <= 0` 时必须报错。

`nn_global_avg_pool2d`：

- input rank-4 NCHW `[N, C, H, W]`
- 输出 `[N, C, 1, 1]`
- dtype 等于输入 dtype

### 5.6 Transform / Reduce / Softmax

`nn_flatten`：

- input rank >= 1
- `axis` 归一化到 `[0, rank]`
- 输出 `[product(shape[0:axis]), product(shape[axis:rank])]`

`reshape`：

- MVP 使用静态 `ReshapeAttrs::newshape`
- `newshape` 最多一个 `-1`
- `allowzero == 0` 时，`0` 表示复制对应输入维度
- `allowzero != 0` 时，`0` 表示真实 0 维
- 元素数量必须匹配

当前不一致：

- `transform.cc` 注册 `reshape` 为 2 输入。
- `op_ffi.cc::MakeReshape` 创建 1 输入 + `ReshapeAttrs`。

建议 issue #3 将 MVP reshape 统一为 1 输入 + static `ReshapeAttrs`，更贴合当前 FFI helper。

`transpose`：

- 空 `perm` 表示 reverse dimensions
- 非空 `perm` 必须是 `[0, rank)` 的排列
- 输出 shape 为 `input_shape[perm[i]]`

`reduce_mean`：

- 空 axes 表示 reduce all axes
- negative axes 需要 normalize
- duplicate axes 报错
- axis 越界报错
- `keepdims != 0` 保留 dim=1
- `keepdims == 0` 删除被 reduce 维度

`softmax`：

- axis 归一化到 `[0, rank)`
- 输出 shape/dtype 等于输入

当前不一致：

- `softmax.cc` 注册 `softmax`
- `common_ops.cc` 注册 `nn_softmax`
- `op_ffi.cc::MakeSoftmax` 调用 `nn_softmax`

建议 issue #3 以 `softmax` 为 canonical rule，`nn_softmax` 只作为 issue #2 解决前的临时 alias。

## 6. 需要改动的地方

### 6.1 类型工具

修改：

- [include/base/expr.h](../include/base/expr.h)
- [src/base/expr.cc](../src/base/expr.cc)
- [include/relay/relay.h](../include/relay/relay.h)
- [src/relay/relay.cc](../src/relay/relay.cc)

新增能力：

- `SetCheckedType`
- `GetCheckedTypeOrThrow`
- `TensorTypeToString`
- `TypeEqual`
- `CopyExprMetadata`，同时保留 `virtual_device_` 和 `checked_type_`
- 可选：`TupleTypeNode`
- 可选：`FuncTypeNode`

约束：不要在 issue #3 中扩展成完整类型代数。MVP 只需要静态 tensor typing。

### 6.2 Op registry type hook

修改：

- [include/relay/op_attr_types.h](../include/relay/op_attr_types.h)

新增：

```cpp
using FInferType =
    std::function<Type(const Attrs&, const Array<Type>&)>;
```

`OpRegEntry::set_attr` 已经基于 `std::any`，不需要大改 registry。

### 6.3 InferTypePass

新增：

- `include/relay/transforms/infer_type.h`
- `src/relay/transforms/infer_type.cc`

实现：

- `Function InferTypePass(const Function& func)`
- Relay visitor
- `Var -> Type` 环境
- Constant type extraction
- Call arity validation
- `FInferType` lookup
- checked type 写入
- op-specific error context

建议 helper：

```text
AsTensorType(type, context)
MakeTensorType(shape, dtype)
BroadcastShape(lhs, rhs, op_name)
NormalizeAxis(axis, rank, context)
NormalizeAxes(axes, rank, context)
Product(dims)
Normalize2D(values, default_value, attr_name)
NormalizePadding(values)
```

### 6.4 Op type rules

修改：

- [src/relay/op/tensor/math.cc](../src/relay/op/tensor/math.cc)
- [src/relay/op/tensor/transform.cc](../src/relay/op/tensor/transform.cc)
- [src/relay/op/tensor/reduce.cc](../src/relay/op/tensor/reduce.cc)
- [src/relay/op/nn/dense.cc](../src/relay/op/nn/dense.cc)
- [src/relay/op/nn/convolution.cc](../src/relay/op/nn/convolution.cc)
- [src/relay/op/nn/pooling.cc](../src/relay/op/nn/pooling.cc)
- [src/relay/op/nn/softmax.cc](../src/relay/op/nn/softmax.cc)
- [src/relay/op/nn/activation.cc](../src/relay/op/nn/activation.cc)

为 issue #3 算子注册 `FInferType`。

原则：

- `InferTypePass` 负责 orchestration。
- op 文件负责 op-specific type relation。
- 不要把所有规则塞进一个巨大 pass 文件。

### 6.5 Pipeline / Compiler / Lowering

修改：

- [include/relay/transforms/pipeline.h](../include/relay/transforms/pipeline.h)
- [src/relay/transforms/pipeline.cc](../src/relay/transforms/pipeline.cc)
- [src/api/compiler.cc](../src/api/compiler.cc)
- [src/relay/backend/lower.cc](../src/relay/backend/lower.cc)
- [CMakeLists.txt](../CMakeLists.txt)

Pipeline：

- 注册 `"infer_type"` pass。
- 注册 global function：`kxc.relay.transform.infer_type`。
- `optimize_default` 末尾运行 `infer_type`。

Compiler：

- optimize 前运行一次 inference，提前发现输入图错误。
- optimize 后再运行一次 inference，修复 pass 重建节点后丢失的 `checked_type_`。

Lowering：

- 检查 function body 是否 typed。
- 检查每个 lowerable call 的 output type 是否为 `TensorType`。
- 直接调用未 typed Relay 时，报清楚：

```text
LowerToTIR expects typed Relay. Run InferTypePass before lowering.
```

### 6.6 Debug / 文档

修改：

- [src/relay/pass/print_ir.cc](../src/relay/pass/print_ir.cc)
- [docs/OPERATOR_REGISTRATION_GUIDE.md](OPERATOR_REGISTRATION_GUIDE.md)
- [docs/PASS_IMPLEMENTATION_SUMMARY.md](PASS_IMPLEMENTATION_SUMMARY.md)

IR printer 增加 checked type 输出，例如：

```text
Call(op=add, args=2) : TensorType(shape=[1, 3, 224, 224], dtype=float32)
```

算子扩展文档增加：

- `FInferType` 是 MVP executable op 的必填项。
- 新 op 必须有 success/failure type inference tests。
- shape 错误必须在 Relay inference 阶段暴露。

### 6.7 测试

建议新增：

- `test/infer_type_test.cpp`
- CMake executable：`infer_type_test`
- custom target：`run_infer_type_test`

最小覆盖：

- elementwise broadcast 成功/失败
- `sqrt` dtype 成功/失败
- `cast` dtype 输出
- `matmul` 成功/失败
- `nn_dense` 成功/失败
- `nn_gemm` 成功/失败，包括 `transA=1` 当前不支持
- `nn_conv2d` rank/layout/channel 成功/失败
- pool/global pool 成功/失败
- flatten axis 成功/失败
- reshape element count、`-1`、`0/allowzero` 成功/失败
- transpose perm 成功/失败
- reduce axes 成功/失败
- softmax axis 成功/失败
- pipeline 能通过 `"infer_type"` 调用
- compiler 会在 lowering 前运行 inference
- 直接 `LowerToTIR` 未 typed graph 时有清晰错误

## 7. 建议实现顺序

1. 增加 type helper 和 `FInferType` typedef。
2. 增加 `InferTypePass` skeleton，先支持 Var/Constant/Call 框架。
3. 实现 elementwise、unary、cast。
4. 接入 pipeline，并加基础测试。
5. 实现 transform/reduce。
6. 实现 matmul/dense/gemm。
7. 实现 conv/pool/softmax。
8. 接入 compiler 和 lowering guard。
9. 增加 IR printer checked type 输出。
10. 更新算子扩展文档和 pass summary。

这个顺序能先验证 pass 框架，再逐步扩展 op family，避免一口气改完整条链路导致 review 不可控。

## 8. 验收标准映射

Issue #3 要求：用户只需要 annotate inputs/constants，中间 `Call` 自动推断。

设计响应：

- `InferTypePass` 从 `Var::type_annotation` 和 `NDArray` 生成初始类型。
- 所有支持的 intermediate `Call` 写入 `checked_type_`。

Issue #3 要求：非法 rank/dtype/attrs/shape 在 inference 阶段失败。

设计响应：

- 每个 MVP op family 都有 `FInferType` 规则和 failure tests。

Issue #3 要求：`LowerToTIR` 可以假设 validated typed Relay。

设计响应：

- `Compiler::Compile` 在 lowering 前运行 inference。
- `LowerToTIR` 增加 typed Relay precondition guard。

Issue #3 要求：每个 op family 有成功和失败测试。

设计响应：

- 新增 `infer_type_test`，覆盖 issue #3 列出的所有算子族。

## 9. 风险和依赖

### 9.1 依赖 issue #2 的 op 命名规范

Type rule lookup 依赖 op name。当前存在 `softmax` / `nn_softmax`、`sub` / `subtract` 等不一致。

处理方式：

- issue #3 使用 canonical name 注册规则。
- 对已有调用路径保留最小 alias。
- 在测试里明确 alias 是临时兼容。
- 后续 issue #2 完成后删除 alias。

### 9.2 checked_type_ 原地写入

原地写 metadata 简单，且和现有 `virtual_device_` 处理一致，但可能让 IR immutability 语义变弱。

处理方式：

- 通过 `SetCheckedType` 集中写入。
- pass 后重新 inference，避免 stale type。
- 不把 checked type 纳入 structural hash/equality，除非后续明确需要。

### 9.3 Relay 和 TOPI shape 公式分叉

如果 Relay inference 和 TOPI compute 公式不一致，会出现类型正确但 lowering buffer 不一致的问题。

处理方式：

- shape 公式尽量抽公共 helper。
- 增加 infer + lower 的组合 smoke tests。
- TE/TOPI 不再作为非法 shape 的兜底层。

### 9.4 attrs 语义不统一

`reshape`、`softmax` 和若干 FFI helper 当前存在 arity/name/attrs 不一致。

处理方式：

- issue #3 明确 MVP 使用方式。
- 对 reshape 优先统一为 static attrs-based。
- 更大范围 schema 规范化放到 issue #2。

## 10. 关键决策

1. `LowerToTIR` 是否自动运行 `InferTypePass`？

   建议：不自动运行。`Compiler::Compile` 负责跑 inference；`LowerToTIR` 只验证 precondition。

2. 是否支持 dtype promotion？

   建议：MVP 不支持。除 `cast` 外要求 dtype 完全一致。

3. 是否在 `TensorType` 中用 `-1` 表示动态维？

   建议：不支持。`TensorType` 保持 fully static。`-1` 仅作为 reshape attr 解析前的语法。

4. `TupleType` 是否放进 issue #3？

   建议：放入。IR 已有 Tuple/TupleGetItem，支持 `TupleType` 能让 pass 更完整；但 LowerToTIR 仍可只支持 single Tensor output。
