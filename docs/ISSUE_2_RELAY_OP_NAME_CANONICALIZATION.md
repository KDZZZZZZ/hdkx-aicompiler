# Issue #2: Relay 算子命名统一规范

## 1. 背景

Issue #2 要解决的是 Relay op name 在注册、前端 helper、ONNX importer、pass 和 lowering 之间不一致的问题。

当前已经看到的典型漂移：

- `mul` / `multiply` 同时存在。
- `subtract` / `sub` 同时存在。
- `softmax` / `nn_softmax` 同时存在。
- 部分 pass 为了兼容历史名称写了多分支判断，例如 `mul || multiply`、`subtract || sub`。
- `common_ops.cc` 和 `src/relay/op/**` 中存在重复注册，容易出现一个名字只有 metadata、另一个名字才有完整 type/lowering hook 的情况。

这类问题会导致一个算子看起来已经注册，但在 `_make`、ONNX 导入、类型推导、优化 pass 或 lowering 阶段失败。

## 2. 目标

本规范定义 issue #2 的实现目标：

1. 建立 MVP 范围内的 canonical Relay op name 表。
2. 统一内部 IR、类型推导、pass、lowering 和 ONNX importer 使用 canonical name。
3. 明确允许保留的 public helper alias，并要求 alias 必须映射到 canonical name。
4. 禁止 metadata-only alias 注册，避免别名指向缺失 `FInferType` / `FRelayToTE` 的 op。
5. 增加测试，确保公开 helper、ONNX importer 输出和 pass 逻辑不会再产生非 canonical name。

新增算子的完整接入步骤见 [ADD_RELAY_OP_GUIDE.md](./ADD_RELAY_OP_GUIDE.md)。本文负责约束命名和 gate，新增算子指南负责说明从 Relay 注册到 TE/TOPI/TIR/后端测试的具体链路。

## 3. 命名原则

### 3.1 Canonical name 是内部唯一语义名

Relay `CallNode::op.As<OpNode>()->name` 应使用 canonical name。以下路径都应读取或生成 canonical name：

- C++ `Call(Op::Get(...))`
- C++ `_make` helper
- Python / ONNX importer
- Relay type inference
- Relay pass pattern matching
- Relay-to-TE/TIR lowering
- IR dump、profile、execution plan 中记录的 kernel op name

### 3.2 Public helper 可以保留短别名

用户接口可以保留符合习惯的短 helper，例如：

- `_make.sub(...)`
- `_make.mul(...)`
- `_make.div(...)`

但这些 helper 创建的 Relay `Call` 必须使用 canonical op：

- `_make.sub` -> `Op::Get("subtract")`
- `_make.mul` -> `Op::Get("mul")`
- `_make.div` -> `Op::Get("divide")`

public helper 名称是 API 兼容层，不等于 Relay IR 内部 op name。

### 3.3 Alias 必须集中管理

如果短期必须兼容历史 IR 中的非 canonical op name，不能在各个 pass 里散落 `if (op == "mul" || op == "multiply")`。

推荐做法：

- 新增集中 alias 表，例如 `relay/op_names.h` / `relay/op_names.cc`。
- 提供 `CanonicalizeOpName(name)`。
- 提供 `IsCanonicalOpName(name)`。
- 在构造 `Call` 或专门的 early pass 中完成 op name canonicalization。
- pass、type inference、lowering 只匹配 canonical name。

### 3.4 不允许 metadata-only alias

历史别名如果确实需要注册，也必须满足以下条件：

- 明确标记为 alias。
- alias 不能拥有独立、不完整的 schema。
- alias 解析后必须复用 canonical op 的完整 attrs / type rule / lowering hook。
- alias 必须有测试覆盖。

如果当前注册系统无法表达“别名指向 canonical op”，则 issue #2 的第一阶段应直接删除或停止生成这些别名，而不是注册一个缺失 hook 的同名 op。

## 4. MVP Canonical Op Name 表

### 4.1 Tensor / Elementwise

| 语义 | Canonical name | 允许的 public helper alias | 历史/禁止内部名 | 备注 |
| --- | --- | --- | --- | --- |
| add | `add` | `_make.add` | 无 | 保持现状。 |
| subtract | `subtract` | `_make.sub` | `sub` | Relay IR 不应出现 `sub`。 |
| multiply | `mul` | `_make.mul` | `multiply` | Relay IR 不应出现 `multiply`。 |
| divide | `divide` | `_make.div` | `div` | Relay IR 不应出现 `div`。 |
| pow | `pow` | `_make.pow` | 无 | 保持现状。 |
| sqrt | `sqrt` | `_make.sqrt` | 无 | 保持现状。 |
| erf | `erf` | `_make.erf` | 无 | 保持现状。 |
| equal | `equal` | `_make.equal` | 无 | 保持现状。 |
| greater | `greater` | `_make.greater` | 无 | 保持现状。 |

### 4.2 Matrix / NN

| 语义 | Canonical name | 允许的 public helper alias | 历史/禁止内部名 | 备注 |
| --- | --- | --- | --- | --- |
| matmul | `matmul` | `_make.matmul` | 无 | Tensor op，不加 `nn_` 前缀。 |
| dense | `nn_dense` | `_make.dense` | 无 | 保持现状。 |
| gemm | `nn_gemm` | 无或 `_make.gemm` | 无 | 如果暴露 helper，应返回 `nn_gemm`。 |
| conv2d | `nn_conv2d` | `_make.conv2d` | 无 | 保持现状。 |
| relu | `nn_relu` | `_make.relu` | 无 | 保持现状。 |
| max_pool2d | `nn_max_pool2d` | `_make.max_pool2d` | 无 | 保持现状。 |
| avg_pool2d | `nn_avg_pool2d` | `_make.avg_pool2d` | 无 | 若新增 helper，应返回 `nn_avg_pool2d`。 |
| global_avg_pool2d | `nn_global_avg_pool2d` | `_make.global_avg_pool2d` | 无 | 当前 ONNX importer 已输出 canonical name。 |
| softmax | `softmax` | `_make.softmax` | `nn_softmax` | `softmax` 作为 canonical，`nn_softmax` 不应进入 Relay IR。 |

`softmax` 选择不加 `nn_` 的原因：当前 `src/relay/op/nn/softmax.cc` 已注册 `softmax`，`op_ffi.cc::MakeSoftmax` 也返回 `softmax`；issue #3 的类型推导设计也以 `softmax` 为 canonical。

### 4.3 Transform / Shape / Reduce

| 语义 | Canonical name | 允许的 public helper alias | 历史/禁止内部名 | 备注 |
| --- | --- | --- | --- | --- |
| concatenate | `concatenate` | `_make.concatenate` | `concat` | Relay IR 不应出现 `concat`，除非显式 alias 后立即 canonicalize。 |
| flatten | `nn_flatten` | `_make.flatten` | `flatten` | 当前 ONNX importer 已输出 `nn_flatten`。 |
| reshape | `reshape` | `_make.reshape` | 无 | 保持现状。 |
| shape | `shape` | `_make.shape` | 无 | 保持现状。 |
| slice | `slice` | `_make.slice` | 无 | helper 仍需按实现状态补齐。 |
| split | `split` | `_make.split` | 无 | 多输出 lowering 使用该 canonical name。 |
| squeeze | `squeeze` | `_make.squeeze` | 无 | helper 仍需按实现状态补齐。 |
| transpose | `transpose` | `_make.transpose` | 无 | 保持现状。 |
| unsqueeze | `unsqueeze` | `_make.unsqueeze` | 无 | helper 仍需按实现状态补齐。 |
| where | `where` | `_make.where` | 无 | 保持现状。 |
| gather | `gather` | `_make.gather` | 无 | 保持现状。 |
| cast | `cast` | `_make.cast` | 无 | 保持现状。 |
| reduce_mean | `reduce_mean` | `_make.reduce_mean` | 无 | 保持现状。 |
| constant_of_shape | `constant_of_shape` | `_make.constant_of_shape` | 无 | 保持现状。 |
| expand_dims | `expand_dims` | `_make.expand_dims` | 无 | 保持现状。 |

### 4.4 Device / Communication

通信 op 使用 `device.` 前缀，不参与 Tensor/NN 命名 alias：

- `device.copy`
- `device.allreduce`
- `device.broadcast_from_worker0`
- `device.scatter_from_worker0`
- `device.gather_to_worker0`
- `device.send_to_worker`
- `device.recv_from_worker`

这些名字已经作为 execution plan 和 runtime executor 的语义字段使用，不应新增短别名。

## 5. 需要改动的地方

### 5.1 Op 注册

目标：

- 每个 canonical op 只保留一个权威注册点。
- 权威注册点必须包含完整 schema、`TAttrs`、`FInferType`、`FRelayToTE` / `FRelayToTEMulti` 等 hook。
- `common_ops.cc` 不应继续注册 `multiply`、`nn_softmax` 这类历史名字。

建议：

1. 以 `src/relay/op/tensor/*.cc` 和 `src/relay/op/nn/*.cc` 为 canonical 注册位置。
2. `common_ops.cc` 只保留确实没有分类文件承载的公共注册，例如 device communication op。
3. 如果需要 alias，先实现 alias 基建，再用 alias API 表达，不要用 `KXC_REGISTER_OP(alias)` 注册一份独立 metadata。

### 5.2 C++ `_make` helper

目标：

- public helper 名可兼容历史 API。
- helper 内部必须 `Op::Get(canonical_name)`。

必须确认：

- `_make.sub` 返回 `subtract`。
- `_make.mul` 返回 `mul`。
- `_make.div` 返回 `divide`。
- `_make.softmax` 返回 `softmax`。
- 新增 `_make.gemm`、`_make.avg_pool2d`、`_make.global_avg_pool2d` 时也必须返回 canonical name。

### 5.3 ONNX importer

目标：

- `python/kxc_onnx/importer.py::ONNX_TO_RELAY` 只输出 canonical name。
- legacy 脚本如果仍保留，例如 `python/onnx_to_cpp.py`，也必须输出 canonical name，或明确标记为废弃并从测试路径移除。

ONNX 映射建议：

| ONNX op | Relay canonical name |
| --- | --- |
| `Conv` | `nn_conv2d` |
| `Relu` | `nn_relu` |
| `MaxPool` | `nn_max_pool2d` |
| `AveragePool` | `nn_avg_pool2d` |
| `GlobalAveragePool` | `nn_global_avg_pool2d` |
| `Flatten` | `nn_flatten` |
| `Gemm` | `nn_gemm` |
| `MatMul` | `matmul` |
| `Add` | `add` |
| `Sub` | `subtract` |
| `Mul` | `mul` |
| `Div` | `divide` |
| `Pow` | `pow` |
| `Sqrt` | `sqrt` |
| `Erf` | `erf` |
| `Softmax` | `softmax` |
| `Concat` | `concatenate` |
| `Reshape` | `reshape` |
| `Transpose` | `transpose` |
| `ReduceMean` | `reduce_mean` |
| `Gather` | `gather` |
| `Cast` | `cast` |
| `Where` | `where` |
| `Split` | `split` |
| `Squeeze` | `squeeze` |
| `Unsqueeze` | `unsqueeze` |

### 5.4 Type inference

目标：

- `FInferType` 挂在 canonical op 上。
- 类型推导错误信息使用 canonical op name。
- 不为了历史 alias 复制一份规则。

实现时应检查：

- `MultiplyInferType` 中错误信息使用 `mul`。
- `SoftmaxInferType` 中错误信息使用 `softmax`。
- `common_ops.cc` 删除历史 alias 后，测试仍能通过。

### 5.5 Relay pass

目标：

- pass 只匹配 canonical name。
- `fold_constant` 和 `simplify_expr` 不应长期保留 `mul || multiply`、`subtract || sub`、`divide || div` 这种分散 alias 逻辑。

推荐迁移顺序：

1. 先保证所有构造路径生成 canonical name。
2. 增加 op name canonicalization 测试。
3. 移除 pass 中的 alias 分支。

如果为了兼容旧 IR 暂时保留 alias 分支，必须用 TODO 标明对应 issue #2，并在测试中明确这是临时兼容路径。

### 5.6 Lowering

目标：

- `FRelayToTE` / `FRelayToTEMulti` 只注册在 canonical op 上。
- lowering 报错时输出 canonical op name。
- 不允许历史 alias 进入 lowering 后才失败。

必须覆盖：

- `mul` 能 lower。
- `multiply` 不应作为新 IR 进入 lowering。
- `softmax` 能找到 type/lowering hook；`nn_softmax` 不应作为新 IR 进入 lowering。

### 5.7 TE / TOPI / TIR 适配

Issue #2 的命名规范只是算子扩展链路的入口。一个 Relay op 只有同时满足 TE/TOPI/TIR 适配要求，才能在 support matrix 中标记为 executable。

#### 5.7.1 Relay 到 TE 的边界

每个可执行 Relay op 必须明确一个 lowering 路径：

- 单输出 Tensor/NN op：matrix 中 `lowering = single`，注册 `FRelayToTE`。
- 多输出 Tuple op：matrix 中 `lowering = multi`，注册 `FRelayToTEMulti`。
- 设备通信语义目前不属于 Compiler 的可执行 lowering；在 ExecutionPlan 能启动真实 `CompiledModule` 并有数值集成测试前，不得将其标记为 supported。
- 暂不支持 lowering 的 op：不得伪装为 supported；matrix 中 `lowering` 必须标为 `none`。

`FRelayToTE` / `FRelayToTEMulti` 的职责：

- 校验输入 tensor 数量。
- 校验 attrs 类型和必要字段。
- 使用 `out_type` 中的 `TensorType` / `TupleType` 作为输出 shape、dtype 的权威来源。
- 调用 TOPI/TE helper 生成 `te::Tensor`。
- 抛出可诊断错误，错误信息必须包含 canonical op name。

禁止行为：

- 返回未定义的 `te::Tensor()`。
- 忽略 attrs 中影响 shape/index 的字段。
- 在 hook 中重新发明一套与 type inference 不一致的 shape 计算。
- 对多输出 op 只返回第一个输出，或用重复 tensor 冒充多个输出。

#### 5.7.2 TOPI helper 要求

TOPI 是可复用 TE compute 的位置。新增或接入 TOPI helper 时必须满足：

- 返回定义完整的 `te::Tensor`，包括 shape、dtype、compute body 和稳定 name。
- 不支持的模式必须 `throw std::runtime_error`，不能返回空 tensor。
- 不能用错误实现占位，例如用 `sum` 代替 `min` / `prod`。
- index 计算必须显式处理 broadcast、axis normalize、negative axis、layout 和 padding 等规则。
- helper 的行为必须和 Relay type inference 的 shape/dtype 规则一致。

如果算子逻辑很简单，可以直接在 `FRelayToTE` 中写 `te::compute`；但一旦该 compute 可复用或涉及复杂索引，应沉到 TOPI helper。

#### 5.7.3 TIR lowering 合约

Relay op 支持 lowering 不等于支持执行。`LowerToTIR` 生成的 TIR 必须落在后端支持范围内。

新增 op 时需要检查 TIR 产物：

- 语句节点是否只使用当前后端支持的节点，例如 `For`、`Store`、`Allocate`、`IfThenElse`、`LetStmt`、`SeqStmt`、`Evaluate`。
- 表达式节点是否只使用当前后端支持的节点，例如 `IntImm`、`FloatImm`、`Var`、二元表达式、`Load`、`Call`、`Select`、`Not`。
- 是否引入新的 intrinsic，例如 `exp`、`sqrt`、`cast`、`floor`、`ceil`。
- 是否引入新的 dtype 转换语义。
- 是否需要真实 reduction init/update 语义，而不是只靠普通 loop 偶然表达。

如果 TIR 产物使用后端尚未支持的节点或 intrinsic，必须同步补 codegen，或在 matrix 中把 executable 状态标为 false。

#### 5.7.4 LLVM codegen 适配要求

LLVM 后端消费的是 TIR，不直接认识 Relay op。新增 Relay op 时只有在 TIR 产物超出现有 LLVM codegen 支持范围时，才需要改 LLVM。

需要改 LLVM 的典型情况：

- 新增 TIR 表达式节点或语句节点。
- 新增 intrinsic name，例如 `tir.exp` / `exp` 命名不统一。
- 新增 dtype cast，需要生成 LLVM cast 指令，而不是当成外部函数。
- 新增向量化、并行、block、attr 等 TIR 结构。
- 新增 runtime ABI 需求，例如动态 shape、额外 metadata、workspace 分配。

不需要改 LLVM 的典型情况：

- 新算子最终只是嵌套 loop、load/store、加减乘除、比较和 select。
- 新算子的复杂度只体现在 TE/TOPI index 计算，生成的 TIR 仍在支持子集内。

#### 5.7.5 算子扩展完成定义

一个 MVP Relay op 只有满足以下条件，才能标记为完整支持：

- canonical op name 已进入 matrix。
- op 注册含完整 schema、attrs、`FInferType`。
- `FRelayToTE` / `FRelayToTEMulti` 已注册，或 matrix 明确标为不支持 lowering。
- TOPI/TE compute 不返回空 tensor，不使用错误占位实现。
- `LowerToTIR` 成功并生成后端可接受的 TIR。
- LLVM/C 后端能处理该 TIR，或 matrix 明确标记该后端不支持。
- 有 type inference test。
- 有 lowering test。
- 对 executable op，有 numeric runtime test。

## 6. 测试要求

Issue #2 至少应增加以下测试。

### 6.1 Op registry / helper 测试

新增轻量 C++ 测试，例如 `test/relay_op_name_test.cpp`：

- 遍历公开 `_make` helper 的代表路径，确认生成的 `Call` op name 为 canonical。
- 直接检查 canonical op 能通过 `Op::Get(name)` 取得，并带有必要 metadata。
- 检查禁止内部名不会由 helper 生成。

必测项：

- `_make.sub` -> `subtract`
- `_make.mul` -> `mul`
- `_make.div` -> `divide`
- `_make.softmax` -> `softmax`
- `_make.flatten` -> `nn_flatten`
- `_make.conv2d` -> `nn_conv2d`

### 6.2 ONNX importer 测试

扩展 `test/onnx_importer_py_test.py`：

- 构造包含 `Sub`、`Mul`、`Div`、`Softmax`、`Concat` 的小 ONNX 图。
- 断言 importer 输出的 `node.op_name` 均为 canonical name。
- 断言不会输出 `sub`、`multiply`、`div`、`nn_softmax`、`concat`。

### 6.3 Pass 测试

扩展 `pass_pipeline_test` 或新增专门测试：

- canonical `mul` 能被 `simplify_expr` 和 `fold_constant` 处理。
- canonical `subtract` 能被处理。
- canonical `divide` 能被处理。
- 测试名称中明确不再依赖历史 alias。

### 6.4 Lowering / type inference 测试

扩展 `infer_type_test` 和 lowering 相关测试：

- `mul`、`subtract`、`divide`、`softmax` 均使用 canonical name。
- 如果输入历史 alias，预期行为必须明确：要么前置 canonicalization 后成功，要么报出“不支持历史 alias”的清晰错误。

### 6.5 TE / TOPI 测试

新增或扩展 TOPI/TE 测试，覆盖每个 matrix 中 `topi = required` 的 op：

- helper 返回 defined `te::Tensor`。
- 输出 shape、dtype 和 Relay type inference 一致。
- 关键 index 计算可通过 TIR 文本或小 shape numeric test 验证。
- 不支持的 attrs/layout/axis 组合会抛出清晰错误。
- 禁止空 tensor、错误占位实现和 silent fallback。

优先覆盖：

- elementwise/broadcast：`add`、`subtract`、`mul`、`divide`、`sqrt`、`cast`。
- transform：`reshape`、`transpose`、`nn_flatten`、`split`。
- reduction：`reduce_mean`。
- NN：`nn_dense`、`nn_gemm`、`nn_conv2d`、`nn_max_pool2d`、`nn_avg_pool2d`、`nn_global_avg_pool2d`、`softmax`。

### 6.6 TIR / LLVM 后端适配测试

对 matrix 中 `tir_executable = true` 的 op，需要至少有 lowering test：

- Relay -> InferType -> LowerToTIR 成功。
- TIR 不包含后端不支持的节点或 intrinsic。
- 错误路径包含 canonical op name。

对 matrix 中 `llvm_required = true` 的 op，需要 numeric LLVM runtime test：

- Relay -> Compile(LLVM) -> Run。
- 输出和手写参考、numpy 或 ONNX Runtime 结果比较。
- 覆盖至少一个代表性小 shape。
- 如果本地配置关闭 LLVM，CI 中必须有单独 LLVM job 或明确记录为 optional backend gap。

## 7. 编译与 CI 固化计划

Issue #2 不能只靠人工 review。规范落地后必须有构建期和 CI 期的硬失败检查，用来自动发现“半拉实现”。

### 7.1 半拉实现的定义

以下任一情况都应让检查失败：

- 新增 `KXC_REGISTER_OP(...)`，但该 op 没有进入 op support matrix。
- 新增或保留历史内部名，例如 `KXC_REGISTER_OP(multiply)`、`KXC_REGISTER_OP(nn_softmax)`、`Op::Get("sub")`、`Op::Get("div")`、`Op::Get("concat")`。
- support matrix 声明某 op 需要 `FInferType`，但注册表里没有挂 `FInferType`。
- support matrix 声明某 op 需要 lowering，单输出 op 没有 `FRelayToTE`，多输出 op 没有 `FRelayToTEMulti`。
- support matrix 声明某 op 需要 TOPI，但没有 TOPI/TE helper 测试覆盖。
- support matrix 声明某 op 可执行，但 LowerToTIR 生成了后端不支持的 TIR 节点或 intrinsic。
- support matrix 声明某 op 需要 LLVM runtime，但没有 numeric LLVM test。
- public `_make` helper 创建出的 `Call` 不是 canonical name。
- ONNX importer 输出非 canonical name。
- pass 中继续散落 alias 判断，例如 `mul || multiply`、`subtract || sub`、`divide || div`，但没有集中 canonicalization 或明确临时豁免。
- TOPI helper 返回空 `Tensor()` 或错误替代实现，但 matrix 将依赖它的 Relay op 标记为 supported。

### 7.2 机器可读 support matrix

建议新增机器可读文件作为单一事实源，例如：

- `docs/OP_SUPPORT_MATRIX.md`：面向人阅读。
- `contracts/relay_op_contract.json`：面向检查脚本和 C++ 测试。

`relay_op_contract.json` 至少包含：

```json
{
  "canonical_ops": {
    "add": {
      "category": "tensor",
      "required": true,
      "infer_type": true,
      "lowering": "single",
      "topi": "required",
      "tir_executable": true,
      "llvm_required": true,
      "public_helpers": ["kxc.relay.op._make.add"],
      "onnx_ops": ["Add"]
    },
    "split": {
      "category": "transform",
      "required": true,
      "infer_type": true,
      "lowering": "multi",
      "topi": "required",
      "tir_executable": true,
      "llvm_required": true,
      "public_helpers": ["kxc.relay.op._make.split"],
      "onnx_ops": ["Split"]
    }
  },
  "forbidden_internal_names": ["sub", "multiply", "div", "nn_softmax", "concat"],
  "temporary_alias_allowlist": []
}
```

规则：

- 新增 Relay op 必须先进入 matrix，再实现注册、type、lowering、测试。
- matrix 里的状态不能比实际能力更乐观。
- 如果某 op 只完成注册和类型推导，`lowering` 必须显式标为 `none`，不能伪装成 supported lowering。
- 如果某 op 依赖 TOPI helper，`topi` 必须标明 `required`，并由 TOPI 测试覆盖。
- 如果某 op 生成的 TIR 暂时不能被后端执行，`tir_executable` 必须为 `false`。
- 如果某 op 需要进入 LLVM runtime 测试，`llvm_required` 必须为 `true`；否则必须说明后端限制。
- `temporary_alias_allowlist` 默认应为空；需要临时兼容时必须写原因、过期 issue 和测试。

### 7.3 C++ registry contract test

新增 `test/relay_op_contract_test.cpp`，加入 `KXC_BUILD_PASS_TESTS`。

它负责运行时检查：

- matrix 中每个 canonical op 能在注册表中找到。
- canonical op 的 `num_inputs`、`TAttrs`、`FInferType`、`FRelayToTE` / `FRelayToTEMulti` 与 matrix 声明一致。
- matrix 中声明的 forbidden internal name 不应作为独立 op 注册。
- public helper 生成的 `Call` op name 等于 canonical name。

为避免 `Op::Get(name)` 当前会自动创建空 op，issue #2 实现时应同步补注册表 introspection API：

- `Op::IsRegistered(name)`：只查询，不创建。
- `Op::ListRegisteredNames()`：返回已注册 op 名列表。

在这两个 API 落地前，C++ contract test 不能依赖 `Op::Get` 判断“是否注册”，只能检查已经明确构造出的 canonical op metadata。

### 7.4 源码扫描 gate

新增脚本，例如 `python/tools/check_relay_op_contract.py`，用于静态扫描源码。

它负责检查：

- `KXC_REGISTER_OP(...)` 只能注册 matrix 中的 canonical name，或显式允许的临时 alias。
- `Op::Get("...")` 不能使用 forbidden internal name。
- `ONNX_TO_RELAY` 只能输出 canonical name。
- pass 源码中不能出现分散 alias 逻辑，除非命中临时 allowlist。
- legacy 生成脚本不能继续输出 forbidden internal name。

源码扫描 gate 可以在 C++ introspection API 完成前先落地；因此它是 issue #2 的第一道硬 gate。

### 7.5 CMake target

新增以下 target：

- `relay_op_contract_test`
- `run_relay_op_contract_test`
- `check_relay_op_contract`
- `run_cpu_required_tests`

建议聚合关系：

```cmake
add_custom_target(check_relay_op_contract
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/tools/check_relay_op_contract.py"
          --root "${CMAKE_CURRENT_SOURCE_DIR}"
          --matrix "${CMAKE_CURRENT_SOURCE_DIR}/contracts/relay_op_contract.json"
)

add_custom_target(run_cpu_required_tests
  DEPENDS
    check_relay_op_contract
    run_relay_op_contract_test
    run_infer_type_test
    run_pass_pipeline_test
    run_lower_multi_output_test
    run_profile_bundle_test
)
```

如果 ONNX Python 依赖可用，`run_cpu_required_tests` 还应包含：

- `run_onnx_importer_test`

### 7.6 GitHub Actions CI

仓库目前只有 PR template，没有实际 `.github/workflows`。issue #2 或 issue #13 应新增 CPU-only CI：

```yaml
name: cpu-required

on:
  pull_request:
  push:
    branches: [main]

jobs:
  cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: lukka/get-cmake@latest
      - name: Install deps
        run: python -m pip install numpy onnx
      - name: Configure
        run: cmake --preset dev-ninja-cpu -DKXC_ENABLE_LLVM=OFF
      - name: Build
        run: cmake --build --preset dev-ninja-cpu --parallel 2
      - name: Required tests
        run: cmake --build out/build/dev-ninja-cpu --target run_cpu_required_tests
```

CI 必须默认跑硬 gate。LLVM/CUDA/NCCL 相关测试可以单独作为 optional workflow，不阻塞 CPU-only MVP。

### 7.7 失败策略

用户已经接受“立刻失败无所谓”，因此 gate 不需要为了历史债务做软失败。

推荐策略：

1. 第一版 gate 直接检查并失败。
2. PR 描述里列出当前失败项。
3. 后续实现提交逐项消除失败。
4. 不引入长期 allowlist；临时 allowlist 必须带 issue 编号和删除条件。

## 8. 相关 issue 合并范围

这条链路的核心是“新增算子时，从名称、注册、类型、TOPI、lowering、测试、CI 都能自动对齐”。可以按以下范围合并推进。

### 8.1 建议一起完成

这些 issue 与 issue #2 的硬 gate 直接相关，适合放在同一组 PR 或连续 PR 中完成：

- #2 `Canonicalize Relay op names across registration, helpers, importer, and passes`
  - 当前文档和后续硬 gate 的主 issue。
- #13 `Add op support matrix, per-op numeric tests, model tests, and CPU-only CI`
  - support matrix 和 CPU-only CI 是 issue #2 gate 的承载物。
- #4 `Complete FRelayToTE coverage for MVP Relay ops`
  - matrix 如果声明 lowering supported，就必须由 #4 补齐 `FRelayToTE` / `FRelayToTEMulti`。
- #6 `Remove placeholder and incorrect TOPI implementations before treating ops as supported`
  - 防止 Relay op 通过了注册/type/lowering 检查，但底层 TOPI 返回空 tensor 或错误 compute。

建议拆分方式：

1. PR A：新增 matrix、源码扫描 gate、CMake target、CPU CI。允许先失败。
2. PR B：完成 issue #2 命名清理，让 gate 中的 forbidden name 检查通过。
3. PR C：补 issue #4/#6 的 MVP op lowering 和 TOPI 正确性，让 matrix 中 required op 全部通过。
4. PR D：补 #13 的 per-op numeric tests 和 model-level smoke tests。

### 8.2 可以同一阶段联动，但不建议塞进同一个 PR

- #20 `Write TinyTVM quickstart, extension guides, and troubleshooting docs`
  - 与算子扩展链路强相关，但文档量大，建议在 gate 和实现稳定后补“如何新增 op”的用户向教程。
- #9 `Expand LLVM codegen coverage for MVP TIR and add numeric tests`
  - 是 lowering 后的下游。可以复用 matrix，但不应阻塞 issue #2 的命名 gate。
- #7 `Define constant and parameter binding through lowering, codegen, and runtime`
  - ONNX 模型执行会需要，但不是 op name canonicalization 的前置。
- #16 `Introduce structured pass manager with metadata, opt levels, dependencies, and instrumentation`
  - pass 规范化会受益于 canonical op name，但不是本 issue 的必要范围。

### 8.3 不建议一起做

以下 issue 关联较远，容易扩大 PR blast radius：

- #10 C backend AOT。
- #11 用户态 Runtime Module API。
- #12 Disco real kernel execution。
- #14 热/冷自适应编译设计。
- #15 TE schedule primitives。
- #17 CUDA codegen。
- #18 Device/Target semantics。
- #19 profiling 覆盖。

这些可以依赖 op support matrix 的结果，但不应和 issue #2 的命名/CI gate 混在同一个实现 PR。

## 9. 实现顺序建议

1. 新增 op name 常量与 alias 表。
2. 修改 `_make` helper，确保所有 helper 输出 canonical name。
3. 修改 ONNX importer 和 legacy 生成脚本的 op 映射。
4. 清理 `common_ops.cc` 中的历史 alias 注册和重复注册。
5. 将 pass 中的 alias 分支迁移到集中 canonicalization。
6. 补齐 registry/helper、importer、pass、type inference/lowering 测试。
7. 运行完整验证并删除临时兼容 TODO。

## 10. 验收标准

Issue #2 完成时必须满足：

- Relay IR 中新生成的 op name 均来自本规范的 canonical 表。
- `sub/subtract`、`mul/multiply`、`softmax/nn_softmax` 已被一致处理。
- ONNX importer 输出 canonical name。
- `_make` helper 不调用未注册或 metadata-only op name。
- `fold_constant`、`simplify_expr` 等 pass 不依赖分散 alias 判断。
- lowering 失败不再由“别名指向缺失 hook 的 op”造成。
- 新增测试覆盖 canonical helper、ONNX importer 输出和关键 pass 行为。

## 11. 非目标

Issue #2 不负责：

- 新增尚未支持的算子语义。
- 补齐所有算子的 TE/TOPI 实现。
- 重新设计 attrs 系统。
- 改变 execution plan 中 `device.*` 通信 op 的命名。
- 解决多输出 lowering ABI；该部分属于 issue #8。
