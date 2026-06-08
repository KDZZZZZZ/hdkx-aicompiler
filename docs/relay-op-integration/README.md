# Relay 算子接入文档包

本文档包是新增或修改 Relay 算子的权威接入说明。写 PR 前先读本页，再按子文档逐项完成。旧的单篇教程只作为背景材料；如果旧文档和本目录冲突，以本目录、`test/relay_op_contract.json` 和 `python/tools/check_relay_op_contract.py` 为准。

## 接入顺序

| 顺序 | 文档 | 产物 | 通过标准 |
| --- | --- | --- | --- |
| 0 | [00-contract.md](00-contract.md) | support matrix 条目和方案卡 | op name、category、attrs、lowering、测试范围都只从允许选项中选择 |
| 1 | [01-relay.md](01-relay.md) | Relay schema、attrs、`_make` helper | canonical op 注册完整，无 alias，无半拉 helper |
| 2 | [02-type-inference.md](02-type-inference.md) | `FInferType` | shape/dtype 可推断，错误立即失败 |
| 3 | [03-topi-te.md](03-topi-te.md) | TOPI helper 或本地 `te::compute` | 不返回空 tensor，不写假实现，能生成后端可承接的 TE |
| 4 | [04-lowering-backend.md](04-lowering-backend.md) | `FRelayToTE` / `FRelayToTEMulti` / execution plan | `LowerToTIR` 或 `LowerRelayToExecPlanPass` 通过 |
| 5 | [05-frontend-tests-ci.md](05-frontend-tests-ci.md) | FFI、ONNX、测试、CI | checker、type、lowering、LLVM numeric tests 通过 |
| 6 | [DRIFT.md](DRIFT.md) | 漂移核对记录 | 确认旧 guide 没有继续指导错误实现 |

## 当前 source of truth

| 内容 | 文件 |
| --- | --- |
| 机器可读算子契约 | [test/relay_op_contract.json](../../test/relay_op_contract.json) |
| 契约检查器 | [python/tools/check_relay_op_contract.py](../../python/tools/check_relay_op_contract.py) |
| 当前支持矩阵说明 | [docs/OP_SUPPORT_MATRIX.md](../OP_SUPPORT_MATRIX.md) |
| Relay op 注册 API | [include/relay/op_macros.h](../../include/relay/op_macros.h), [include/relay/op_attr_types.h](../../include/relay/op_attr_types.h) |
| attrs 定义和实现 | [include/relay/op.h](../../include/relay/op.h), [src/relay/op_attrs.cc](../../src/relay/op_attrs.cc) |
| type inference 规则 | [include/relay/type_infer.h](../../include/relay/type_infer.h), [src/relay/type_infer.cc](../../src/relay/type_infer.cc) |
| Relay -> TE/TIR lowering | [src/relay/backend/lower.cc](../../src/relay/backend/lower.cc) |
| TOPI/TE helper | [include/te/topi](../../include/te/topi) |
| LLVM numeric coverage | [test/op_numeric_llvm_test.cpp](../../test/op_numeric_llvm_test.cpp) |

## 一条硬规则

一个 Relay op 只有在 contract、schema、type、lowering、FFI、测试和后端能力都一致时，才算对外支持。`include/te/topi` 里存在 helper 不等于 Relay op 已支持；旧名字、alias helper、空 `te::Tensor()`、占位实现和静默降级都不允许。

## PR 前最小命令

```bash
python python/tools/check_relay_op_contract.py --root .
cmake --build out/build/<build-dir> --target run_infer_type_test
cmake --build out/build/<llvm-build-dir> --target run_op_numeric_llvm_test
```

如果新增 ONNX 映射，还要运行：

```bash
cmake --build out/build/<llvm-build-dir> --target run_onnx_importer_test
```
