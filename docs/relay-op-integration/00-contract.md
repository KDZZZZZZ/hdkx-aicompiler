# 0. 算子契约 API

`test/relay_op_contract.json` 是新增 Relay op 的第一处修改。它描述“这个 op 允许存在到什么阶段”，checker 会按它反查 C++ 注册、FFI、ONNX 映射和测试引用。

## Contract 字段

| 字段 | 类型 | 允许值 | 说明 |
| --- | --- | --- | --- |
| `category` | string | `tensor.math`, `tensor.reduce`, `tensor.transform`, `nn`, `device` | 决定注册文件和 TOPI 文件 |
| `num_inputs` | int | 固定输入数，或 `-1` | 固定输入直接写数量；可变输入必须同时写 `min_inputs` / `max_inputs` |
| `min_inputs` | int | 正整数 | 仅 `num_inputs = -1` 时使用 |
| `max_inputs` | int | 正整数 | 仅 `num_inputs = -1` 时使用 |
| `attrs` | string/null | `null` 或 `XxxAttrs` | 必须和注册块里的 `TAttrs` 一致 |
| `lowering` | string | `single`, `multi`, `exec_plan`, `none` | `single` 用 `FRelayToTE`；`multi` 用 `FRelayToTEMulti`；`exec_plan` 不进 TE |
| `ffi` | bool | `true`, `false` | public 构图入口是否需要 `kxc.relay.op._make.<op>` |
| `tests` | bool | `true`, `false` | 是否要求测试引用 |
| `onnx_ops` | array | ONNX op 名称列表 | 非空时 importer 必须有映射 |

## Canonical name

| 场景 | 规则 |
| --- | --- |
| 普通 tensor op | 使用小写 snake case，例如 `reduce_mean` |
| NN op | 使用 `nn_` 前缀，例如 `nn_relu` |
| 设备通信 op | 使用 `device.` 前缀，例如 `device.copy` |
| 历史别名 | 不新增，不保留 metadata-only alias |
| FFI helper | 名称必须和 op 完全一致，例如 `_make.reduce_mean` |

`rules.forbidden_op_names` 和 `rules.forbidden_helper_names` 中的名称不能重新引入。常见禁止名包括 `div`、`sub`、`multiply`、`concat`、`flatten`、`nn_softmax`。

## Category 选择

| `category` | Relay 注册文件 | TOPI/TE 文件 | 适用范围 |
| --- | --- | --- | --- |
| `tensor.math` | `src/relay/op/tensor/math.cc` | `include/te/topi/broadcast.h`, `elemwise.h`, `nn.h` | elementwise、broadcast、matmul |
| `tensor.reduce` | `src/relay/op/tensor/reduce.cc` | `include/te/topi/reduction.h` | sum/mean/max/min 类 reduce |
| `tensor.transform` | `src/relay/op/tensor/transform.cc` | `include/te/topi/transform.h` | reshape、transpose、flatten、cast |
| `nn` | `src/relay/op/nn/*.cc` | `include/te/topi/nn.h` | conv、dense、pool、activation、softmax |
| `device` | 新增时创建专用 `src/relay/op/device/*.cc` | 不使用 TOPI | execution plan 通信 op；当前不在 19 算子 MVP 矩阵中 |

## Lowering 选择

| `lowering` | 输出类型 | 必须注册 | 禁止 |
| --- | --- | --- | --- |
| `single` | `TensorType` | `FRelayToTE` | 返回空 `te::Tensor()` |
| `multi` | `TupleType` | `FRelayToTEMulti` | 返回数量和 `TupleType.fields` 不一致；重复返回同一个 tensor 冒充多个输出 |
| `exec_plan` | 通常是通信语义 | execution plan lowering | `FRelayToTE` / `FRelayToTEMulti` |
| `none` | 尚不对外支持 lowering | 完整 schema 和 type 状态说明 | 空 lowering hook、假 TOPI helper |

## 示例

```json
"softmax": {
  "category": "nn",
  "num_inputs": 1,
  "attrs": "SoftmaxAttrs",
  "lowering": "single",
  "ffi": true,
  "tests": true,
  "onnx_ops": []
}
```

## Checker 判定

checker 会检查：

| 检查项 | 失败示例 |
| --- | --- |
| op 是否只注册一次 | 同名 `KXC_REGISTER_OP` 出现两次 |
| schema 是否完整 | 缺 `describe()`、`set_num_inputs()`、`add_argument()` |
| `TAttrs` 是否一致 | contract 写 `CastAttrs`，注册块未写或写错 |
| type hook | 缺 `FInferType` |
| lowering hook | `single` 缺 `FRelayToTE`，`multi` 缺 `FRelayToTEMulti` |
| FFI helper | 缺 `_make.<op>` 或 helper 名 alias 到别的 op |
| ONNX 映射 | contract 声明 ONNX op，但 importer 未映射 |
| 测试引用 | 没有同函数块 `LowerToTIR`、backend 或 execution plan 覆盖 |
| 禁止占位 | op/TOPI/importer 源码中出现占位标记或 `return te::Tensor()` |

checker 只能证明结构完整；TIR/LLVM 是否真的能承接，必须靠测试实际编译和运行。
