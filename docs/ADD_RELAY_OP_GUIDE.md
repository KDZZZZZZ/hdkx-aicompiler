# 新增 Relay 算子接入指南

本文说明如何把一个新 Relay 算子接入 KXC/TinyTVM。这里的“接入”不是只写 `KXC_REGISTER_OP`，而是让算子从前端构图、类型推导、TE/TOPI lowering、TIR、后端执行和测试链路上都处于明确状态。

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
- `LowerToTIR` 能生成后端可接受的 TIR。
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
    "category": "tensor",
    "required": true,
    "infer_type": true,
    "lowering": "single",
    "topi": "required",
    "tir_executable": true,
    "llvm_required": true,
    "public_helpers": ["kxc.relay.op._make.negative"],
    "onnx_ops": []
  }
}
```

如果本次 PR 只做注册，不做 lowering，应写成：

```json
{
  "negative": {
    "category": "tensor",
    "required": false,
    "infer_type": false,
    "lowering": "none",
    "topi": "none",
    "tir_executable": false,
    "llvm_required": false,
    "public_helpers": [],
    "onnx_ops": []
  }
}
```

matrix 的状态不能比实际实现更乐观。

`lowering` 目前允许四类值：

- `none`：暂不支持 lowering，不能伪装成可执行。
- `single`：单输出 Tensor/NN op，必须注册 `FRelayToTE`，并有 `LowerToTIR` 契约测试。
- `multi`：多输出 Tuple op，必须注册 `FRelayToTEMulti`，并有 `LowerToTIR` 契约测试。
- `exec_plan`：设备通信类 op，不走 TE/TIR hook，必须能被 `LowerRelayToExecPlanPass` 转成 `CommExec` 或 execution plan 中的对应节点。

## 4. 定义 attrs

没有额外参数的算子可以复用简单 attrs，或不设置 `TAttrs`。

有额外参数时，在 [include/relay/op.h](../include/relay/op.h) 中定义 attrs node 和引用类型，在 [src/relay/op_attrs.cc](../src/relay/op_attrs.cc) 中实现 `Create`。

示例：

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

`Create` 实现：

```cpp
ClipAttrs ClipAttrs::Create(double a_min, double a_max) {
    auto* node = new ClipAttrsNode();
    node->a_min = a_min;
    node->a_max = a_max;
    return InternalCreate(node);
}
```

attrs 要求：

- 字段名要和 ONNX/importer/FFI helper 语义一致。
- 默认值必须明确。
- type inference 和 lowering 都要使用同一个 attrs 类型。
- 错误路径要检查 attrs 是否为空或类型不匹配。

## 5. 增加类型推导规则

类型推导函数声明放在 [include/relay/type_infer.h](../include/relay/type_infer.h)，实现放在 [src/relay/type_infer.cc](../src/relay/type_infer.cc)。

常见模式：

- 输入输出 shape/dtype 相同：复用 `UnarySameInferType`。
- 二元广播：复用或新增类似 `BinaryBroadcastInferType` 的规则。
- 输出 shape 由 attrs 决定：新增专用 infer rule。
- 多输出：返回 `TupleType`。

示例：

```cpp
Type NegativeInferType(const Attrs& attrs, const Array<Type>& input_types) {
    (void)attrs;
    return UnarySameInferType(attrs, input_types);
}
```

如果复用已有函数，注册时可以直接挂 `UnarySameInferType`。

类型推导要求：

- 检查 arity。
- 检查输入必须是 `TensorType`。
- 检查 dtype、rank、axis、layout、broadcast 是否合法。
- 输出类型必须和 lowering 产物一致。
- 报错包含 op name，例如 `negative expects 1 input`。

## 6. 实现 TE/TOPI compute

如果算子 compute 可复用，优先放在 TOPI helper 中，例如：

- elementwise: [include/te/topi/elemwise.h](../include/te/topi/elemwise.h)
- broadcast: [include/te/topi/broadcast.h](../include/te/topi/broadcast.h)
- nn: [include/te/topi/nn.h](../include/te/topi/nn.h)
- transform: [include/te/topi/transform.h](../include/te/topi/transform.h)
- reduction: [include/te/topi/reduction.h](../include/te/topi/reduction.h)

简单 unary op 示例：

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

TOPI 要求：

- 返回 defined `te::Tensor`。
- shape/dtype 和 type inference 一致。
- 不支持的参数组合必须抛错。
- 不能返回空 tensor。
- 不能用错误实现占位。

## 7. 写 `FRelayToTE`

`FRelayToTE` 是 Relay op 到 TE compute 的桥。通常写在对应注册文件里：

- tensor math: [src/relay/op/tensor/math.cc](../src/relay/op/tensor/math.cc)
- tensor transform: [src/relay/op/tensor/transform.cc](../src/relay/op/tensor/transform.cc)
- nn: [src/relay/op/nn](../src/relay/op/nn)

单输出示例：

```cpp
te::Tensor NegativeCompute(const Attrs& attrs,
                           const Array<te::Tensor>& inputs,
                           const kxc::Type& out_type) {
    (void)attrs;
    (void)out_type;
    if (inputs.size() != 1) {
        throw std::runtime_error("negative expects exactly 1 input");
    }
    return te::topi::negative(inputs[0], "T_negative");
}
```

带 attrs 示例：

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
    return te::topi::clip(inputs[0],
                          tir::FloatImm(clip_attrs->a_min, inputs[0]->dtype),
                          tir::FloatImm(clip_attrs->a_max, inputs[0]->dtype),
                          "T_clip");
}
```

多输出 op 使用 `FRelayToTEMulti`，参考 `split` 的设计：

- 输出类型必须是 `TupleType`。
- 返回 tensor 数量必须等于 `TupleType::fields.size()`。
- 不允许用同一个 tensor 重复冒充多个输出。

## 8. 注册 Relay op

使用 `KXC_REGISTER_OP` 注册 op。注册点应放在按类别划分的源文件中，不要随意塞进 `common_ops.cc`。

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
- 可 lowering 的 op 必须设置 `FRelayToTE` 或 `FRelayToTEMulti`。
- 不要注册历史 alias。

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

helper 要求：

- public helper 名可以短，但内部必须 `GetOp(canonical_name)`。
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
kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
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
- [ ] `FRelayToTE` / `FRelayToTEMulti` 已注册或 matrix 明确标为未支持。
- [ ] TOPI/TE helper 不返回空 tensor。
- [ ] ONNX importer 输出 canonical name。
- [ ] `_make` helper 使用 canonical `Op::Get`。
- [ ] `LowerToTIR` 测试通过。
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
