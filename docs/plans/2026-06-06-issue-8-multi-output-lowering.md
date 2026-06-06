# Issue 8 多输出 Lowering 实现计划

> **给执行代理:** 实现本计划时，按任务逐项执行，并在每个任务后做验证和提交。

**目标:** 扩展 Relay 到 TIR 的 lowering，让它从只支持单个 tensor 输出，升级到支持 Relay tuple、多输出结果和 `split -> tuple_get_item`，同时保持现有单输出路径稳定。

**架构:** lowering 阶段把每个 Relay 表达式表示为 `Array<te::Tensor>`。单输出表达式返回一个 TE tensor，`Tuple` 返回多个 TE tensor，`TupleGetItem` 从数组中选择一个 tensor，多输出算子直接返回多个 TE tensor。最终 `PrimFunc` 的参数顺序固定为输入、常量、输出，runtime 仍使用当前 `void** packed_args` ABI，调用方按顺序传入多个输出 buffer 指针。

**技术栈:** C++17、Relay IR、TE tensor、TIR `PrimFunc`、LLVM packed `void**` kernel ABI、CMake 测试。

---

## 背景

- issue 地址：https://github.com/KDZZZZZZ/hdkx-aicompiler/issues/8
- 依赖 issue #3 的类型推断能力：`TupleType`、`Tuple`、`TupleGetItem` 和 `split` 的类型推断必须已经存在。
- 当前本地工作区可能仍处于“把 PR diff 展开给 IDE 看”的状态。真正实现 issue #8 前，建议从合入后的 `main` 新建干净分支。

## 当前缺口

- `src/relay/backend/lower.cc` 里显式拒绝 `outputs.size() != 1`。
- `RelayToTEConverter::VisitCall` 要求每个 call 的输出类型必须是 `TensorType`。
- `RelayToTEConverter` 还没有实现 `VisitTuple` 和 `VisitTupleGetItem`。
- `include/relay/op_attr_types.h` 里的 `FRelayToTE` 只能返回一个 `te::Tensor`。
- `split` 已有 `FInferType`，但没有 lowering hook。
- `LowerToTIR` 只创建一个输出 buffer 参数。
- `PrimFunc` 和 module metadata 没有明确记录输出数量和输出参数区间。

## 范围

本 issue 要做：

- 显式 Relay tuple 函数体：`Function(params, Tuple({expr0, expr1, ...}))`。
- tuple 投影：`TupleGetItem(tuple_expr, index)`。
- `split` lowering：把一个多输出 Relay op lowering 成多个单输出 TE compute tensor。
- 多输出 `PrimFunc` ABI：参数顺序为输入、常量、输出。
- 对不支持的嵌套 tuple、缺失多输出 lowering hook、TE 多 body compute 给出清晰错误信息。

本 issue 不做：

- 完整嵌套 tuple ABI。
- 让 `CompiledModule::Run` 直接返回 tuple 对象；调用方仍然传入输出 buffer。
- 真正的多 body TE `ComputeOp` statement lowering；遇到 `ComputeOpNode::body.size() != 1` 时继续报清晰错误。
- `concatenate` 的完整 lowering；它需要 tuple input compute 支持，建议单独拆 issue。

## 输出 ABI

保持稳定参数顺序：

```text
params = function inputs + constant placeholders + output buffers
```

给 `PrimFunc.attrs` 增加 metadata：

```text
kxc.input_count        = IntImm(函数输入数量)
kxc.constant_count     = IntImm(常量 placeholder 数量)
kxc.output_count       = IntImm(lowering 后输出数量)
kxc.output_param_start = IntImm(input_count + constant_count)
```

单输出图保持现有调用方式：

```text
{input0, ..., output0}
```

多输出图使用：

```text
{input0, ..., output0, output1, ...}
```

## 任务 1：先补失败用例

**文件:**

- 新增：`test/lower_multi_output_test.cpp`
- 修改：`CMakeLists.txt`

**步骤 1：新增专用测试目标**

在 `test/lower_multi_output_test.cpp` 里添加本地 helper：

```cpp
bool HasAttrInt(const kxc::tir::PrimFunc& func, const std::string& key, int64_t expected);
bool CheckParamCount(const kxc::tir::PrimFunc& func, size_t expected);
std::string TIRText(const kxc::tir::PrimFunc& func);
```

**步骤 2：添加显式 tuple output 测试**

构造：

```cpp
Var x("x", TensorType({4}, "float32"));
Var y("y", TensorType({4}, "float32"));
Call add(Op::Get("add"), {x, y});
Call mul(Op::Get("multiply"), {x, y});
Function func({x, y}, Tuple({add, mul}));
tir::PrimFunc lowered = relay::LowerToTIR(func);
```

实现前预期失败：报 `LowerToTIR currently supports single output only` 或不支持 tuple 节点。

实现后预期：

- `lowered->params.size() == 4`
- `kxc.input_count == 2`
- `kxc.constant_count == 0`
- `kxc.output_count == 2`
- `kxc.output_param_start == 2`
- TIR 文本里能看到两个输出 buffer 的 store。

**步骤 3：添加直接 tuple projection 测试**

构造：

```cpp
Function func({x, y}, TupleGetItem(Tuple({add, mul}), 1));
```

实现后预期：

- lowering 成单输出函数。
- `output_count == 1`。
- 只给被选中的字段创建外部输出；未选中的 tuple 字段不应该变成 output param。

**步骤 4：添加 `split -> tuple_get_item` lowering 测试**

构造：

```cpp
Var x("x", TensorType({2, 6}, "float32"));
Call split(Op::Get("split"), {x}, SplitAttrs::Create({3}, 1));
Function func({x}, TupleGetItem(split, 1));
```

实现后预期：

- lowering 成一个输出 buffer。
- 输出 buffer shape 为 `{2, 2}`。
- TIR 文本能体现从输入 axis=1 的偏移位置读取。

**步骤 5：添加 split tuple output 测试**

构造：

```cpp
Function func({x}, split);
```

实现后预期：

- lowering 成三个输出 buffer。
- `output_count == 3`。
- `params.size() == 4`。

**步骤 6：添加诊断测试**

覆盖：

- `Tuple({split})` 或嵌套 tuple 字段应该报清晰的 nested tuple unsupported。
- tuple 类型 call 缺失 `FRelayToTEMulti` 时，错误里要包含 op 名。
- 真正的 TE 多 body `ComputeOp` 仍然失败，但错误里要包含 tensor/op 名和 body count。

**步骤 7：注册 CMake target**

增加：

```cmake
kxc_add_runtime_exe(lower_multi_output_test test/lower_multi_output_test.cpp)
add_custom_target(run_lower_multi_output_test
  COMMAND "$<TARGET_FILE:lower_multi_output_test>"
  DEPENDS lower_multi_output_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
```

运行：

```powershell
cmake --build --preset dev-mingw-cpu --target lower_multi_output_test --parallel 4
cmake --build out/build/dev-mingw-cpu --target run_lower_multi_output_test
```

实现前预期：测试按当前缺口失败。

提交：

```bash
git add CMakeLists.txt test/lower_multi_output_test.cpp
git commit -m "test: cover multi-output Relay lowering gaps"
```

## 任务 2：引入多输出 lowering hook

**文件:**

- 修改：`include/relay/op_attr_types.h`
- 修改：`src/relay/backend/lower.cc`

**步骤 1：新增类型别名**

在 `FRelayToTE` 旁边新增：

```cpp
using FRelayToTEMulti =
    std::function<Array<te::Tensor>(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;
```

**步骤 2：在 lowering 中新增 hook 调用 helper**

在 `lower.cc` 里增加：

```cpp
std::string RelayNodeKind(const Expr& expr);
Array<te::Tensor> InvokeRelayToTE(const OpNode* op_node,
                                  const Attrs& attrs,
                                  const Array<te::Tensor>& inputs,
                                  const Type& out_type);
```

规则：

- 如果 `out_type` 是 `TensorType`，优先使用现有 `FRelayToTE`，并把返回的单个 tensor 包成数组。
- 如果 `out_type` 是 `TupleType`，要求 op 注册 `FRelayToTEMulti`。
- 如果 multi hook 返回数量和 `TupleType::fields.size()` 不一致，错误里要包含 op 名、期望数量和实际数量。
- 如果 hook 缺失，错误里要包含 op 名和输出类型。

**步骤 3：保持单输出路径不变**

本任务不应该要求现有单输出 op 文件跟着改。

运行：

```powershell
cmake --build --preset dev-mingw-cpu --target lower_multi_output_test infer_type_test --parallel 4
```

预期：现有单输出测试仍能编译；多输出测试继续失败，直到 tuple lowering 和 split hook 实现。

提交：

```bash
git add include/relay/op_attr_types.h src/relay/backend/lower.cc
git commit -m "feat: add multi-output Relay lowering hook"
```

## 任务 3：把 Relay Tuple 和 TupleGetItem lowering 到 TE 值

**文件:**

- 修改：`src/relay/backend/lower.cc`

**步骤 1：实现 `VisitTuple`**

行为：

- 依次 visit 每个 field。
- 每个 field 必须正好产生一个 `te::Tensor`。
- 按 field 顺序拼接 tensor。
- 如果某个 field 产生 0 个或多个 tensor，报：

```text
LowerToTIR does not support nested tuple field <i>; field produced <n> tensors
```

**步骤 2：实现 `VisitTupleGetItem`**

行为：

- visit `op->tuple`。
- 检查 index 范围。
- 返回只包含被选中 tensor 的数组。

**步骤 3：增强 `VisitDefault` 诊断**

错误里带上 Relay 节点类型：

```text
Unsupported Relay node in LowerToTIR: TupleGetItem
```

运行：

```powershell
cmake --build out/build/dev-mingw-cpu --target run_lower_multi_output_test
```

预期：显式 tuple 和 tuple-get-item 测试开始通过或推进到后续缺口；split 测试仍失败。

提交：

```bash
git add src/relay/backend/lower.cc
git commit -m "feat: lower Relay tuple values to TE tensors"
```

## 任务 4：让 LowerToTIR 生成多个输出 buffer

**文件:**

- 修改：`src/relay/backend/lower.cc`

**步骤 1：移除 `outputs.size() != 1` 硬限制**

替换为校验：

- output 数组非空。
- 每个 output tensor 已定义。
- 每个 output tensor 都来自 `ComputeOpNode`。

如果不是 compute tensor，错误里包含 output index 和 tensor name。

**步骤 2：从所有 outputs 做 DFS 收集**

使用同一个 `visited_ops`，对每个 output tensor 调用 `CollectOpsDFS`。

**步骤 3：每个输出 tensor 创建一个外部 output buffer**

对每个 output tensor：

```cpp
tir::Var out_var(MakeUniqueOutputName(out_tensor, output_index), out_tensor->dtype);
tir::Buffer out_buf(out_var, out_tensor->dtype, out_tensor->shape, {}, tir::IntImm(0),
                    out_var->name_hint, 0, 0);
params.push_back(out_var);
buffer_map.Set(out_var, out_buf);
buffer_var_by_tensor[out_tensor.get()] = out_var;
```

命名保持确定性：

```text
<tensor_name>_out
<tensor_name>_out_1
```

**步骤 4：只把非输出 compute tensor 分配为中间 buffer**

构造 output tensor object 指针集合，分配 intermediates 时跳过所有 output tensor。

**步骤 5：附加 metadata attrs**

设置：

```cpp
attrs.Set(String("kxc.input_count"), tir::IntImm(input_count, tir::DataType::Int(64)));
attrs.Set(String("kxc.constant_count"), tir::IntImm(constant_count, tir::DataType::Int(64)));
attrs.Set(String("kxc.output_count"), tir::IntImm(outputs.size(), tir::DataType::Int(64)));
attrs.Set(String("kxc.output_param_start"), tir::IntImm(input_count + constant_count,
                                                        tir::DataType::Int(64)));
```

**步骤 6：保留 TE 多 body compute 不支持，但诊断更清晰**

把 `LowerComputeStmt` 的错误从：

```text
Only single-output compute is supported
```

改成：

```text
LowerComputeStmt does not support multi-body TE compute for tensor '<name>': body_count=<n>
```

运行：

```powershell
cmake --build out/build/dev-mingw-cpu --target run_lower_multi_output_test
cmake --build out/build/dev-mingw-cpu --target run_infer_type_test
```

预期：显式 tuple output 和直接 tuple-get-item 测试通过。

提交：

```bash
git add src/relay/backend/lower.cc
git commit -m "feat: emit multi-output PrimFunc buffers"
```

## 任务 5：实现 split lowering

**文件:**

- 修改：`src/relay/op/tensor/transform.cc`

**步骤 1：添加静态 shape helper**

本地 helper：

```cpp
Array<tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type);
int NormalizeSplitAxis(int axis, int ndim);
std::vector<int64_t> SplitOffsets(const SplitAttrsNode* attrs, int64_t axis_dim);
```

lowering 阶段要求 split axis 维度是静态已知的：

```text
split lowering requires static axis dimension
```

类型推断可以允许未知维度，但 lowering 在引入符号 shape 前应要求具体维度。

**步骤 2：添加 `SplitCompute`**

签名：

```cpp
Array<te::Tensor> SplitCompute(const Attrs& attrs,
                               const Array<te::Tensor>& inputs,
                               const kxc::Type& out_type);
```

行为：

- 要求一个输入 tensor。
- 要求 `out_type` 是 `TupleType`。
- 每个 tuple field 必须是 `TensorType`。
- 每个输出 `i` 创建一个 `te::compute`，shape 使用对应 field 的 shape。
- compute lambda 中复制输出 indices 到输入 indices，并在 split axis 上加预计算 offset。
- 输出命名确定：`T_split_0`、`T_split_1`、……

**步骤 3：注册 multi hook**

```cpp
.set_attr<FRelayToTEMulti>("FRelayToTEMulti", SplitCompute)
```

`FInferType` 保持不变。

运行：

```powershell
cmake --build --preset dev-mingw-cpu --target lower_multi_output_test --parallel 4
cmake --build out/build/dev-mingw-cpu --target run_lower_multi_output_test
```

预期：split projection 和 split tuple output lowering 测试通过。

提交：

```bash
git add src/relay/op/tensor/transform.cc
git commit -m "feat: lower split as multi-output TE tensors"
```

## 任务 6：添加 LLVM runtime 多输出 ABI smoke test

**文件:**

- 修改：`test/codegen_llvm_test.cpp`

**步骤 1：添加 `TestRelaySplitMultiOutputRuntime`**

放在 `#ifdef KXC_USE_LLVM` 下。

构造：

```cpp
Var x("x", TensorType({2, 6}, "float32"));
Call split(Op::Get("split"), {x}, SplitAttrs::Create({3}, 1));
Function func({x}, split);
auto module = api::Compiler::Compile(func, api::CompileConfig::AOT(BuildTarget(kCPU), 2));
```

运行：

```cpp
float x_data[12] = {...};
float out0[4] = {0};
float out1[4] = {0};
float out2[4] = {0};
module.Run({x_data, out0, out1, out2});
```

验证：

- `out0` 包含列 `[0, 1]`
- `out1` 包含列 `[2, 3]`
- `out2` 包含列 `[4, 5]`

**步骤 2：通过 `GetPrimFunc()` 验证 metadata**

检查：

```cpp
module.GetPrimFunc()->attrs.at(String("kxc.output_count"))
```

预期值：`3`。

运行：

```powershell
cmake --build --preset dev-mingw-cpu --target codegen_llvm_test --parallel 4
cmake --build out/build/dev-mingw-cpu --target run_codegen_llvm_test
```

预期：LLVM 开启时通过；LLVM 未开启时保持现有 skip 行为。

提交：

```bash
git add test/codegen_llvm_test.cpp
git commit -m "test: verify multi-output LLVM packed ABI"
```

## 任务 7：补文档和诊断说明

**文件:**

- 修改：`docs/REPO_IMPLEMENTATION_OVERVIEW.md`
- 修改：`docs/TINYTVM_REQUIREMENTS.md`
- 可选修改：`docs/ISSUE_3_TYPE_SHAPE_INFERENCE_DESIGN.md`

**步骤 1：记录 ABI**

补一节：

```text
LowerToTIR output buffer ABI:
params = inputs + constants + outputs
PrimFunc attrs kxc.output_count and kxc.output_param_start describe output range.
```

**步骤 2：记录仍不支持的情况**

列出：

- 嵌套 tuple output ABI
- 真正的 multi-body TE compute lowering
- `concatenate` 的 tuple input compute

**步骤 3：如果 issue #3 边界文档仍存在，更新边界**

说明 issue #8 负责 multi-output lowering/runtime ABI。

运行：

```powershell
git diff --check
```

提交：

```bash
git add docs/REPO_IMPLEMENTATION_OVERVIEW.md docs/TINYTVM_REQUIREMENTS.md docs/ISSUE_3_TYPE_SHAPE_INFERENCE_DESIGN.md
git commit -m "docs: describe multi-output lowering ABI"
```

## 任务 8：最终验证和 PR

**步骤 1：运行聚焦测试**

```powershell
cmake --preset dev-mingw-cpu
cmake --build --preset dev-mingw-cpu --target lower_multi_output_test infer_type_test pass_pipeline_test --parallel 4
cmake --build out/build/dev-mingw-cpu --target run_lower_multi_output_test
cmake --build out/build/dev-mingw-cpu --target run_infer_type_test
cmake --build out/build/dev-mingw-cpu --target run_pass_pipeline_test
```

**步骤 2：运行集成测试**

```powershell
cmake --build out/build/dev-mingw-cpu --target run_profile_bundle_test
cmake --build out/build/dev-mingw-cpu --target run_resnet18_ir_dump
cmake --build --preset dev-mingw-cpu --parallel 4
```

如果 LLVM 可用：

```powershell
cmake --build out/build/dev-mingw-cpu --target run_codegen_llvm_test
```

**步骤 3：静态检查**

```powershell
git diff --check
git status --short
```

**步骤 4：开 PR**

PR 标题：

```text
feat: support multi-output Relay lowering
```

PR 描述必须包含：

- closes issue #8
- 单输出 lowering 路径保持稳定
- 支持 `split -> tuple_get_item`
- multi-output ABI 是 `inputs + constants + outputs`
- 不支持的 nested tuple 和 true multi-body TE compute 会给出清晰诊断

## 风险说明

- 本 issue 不改 `CompiledModule::Run` 签名；现有 packed `void**` ABI 已经能传多个输出指针。
- 不要递归 flatten `Tuple`，除非另起 ABI 设计。
- 不要把 `concatenate` lowering 混进本 issue，除非先单独解决 tuple input compute。
- `split` lowering 先只支持静态 shape；符号 split offset 留给后续 shape polymorphism issue。
