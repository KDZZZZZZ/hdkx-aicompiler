# 新增 Relay 算子接入指南

本页是入口页。具体接入规范已经拆到 [relay-op-integration](relay-op-integration/README.md) 文档包，新增或修改算子时以该目录为准。

## 推荐阅读顺序

| 顺序 | 文档 | 内容 |
| --- | --- | --- |
| 0 | [relay-op-integration/00-contract.md](relay-op-integration/00-contract.md) | support matrix、canonical name、category、lowering 选择 |
| 1 | [relay-op-integration/01-relay.md](relay-op-integration/01-relay.md) | Relay schema、attrs、FFI `_make` helper |
| 2 | [relay-op-integration/02-type-inference.md](relay-op-integration/02-type-inference.md) | `FInferType` 可用规则和新增模板 |
| 3 | [relay-op-integration/03-topi-te.md](relay-op-integration/03-topi-te.md) | TOPI/TE 可用构造件和选择规则 |
| 4 | [relay-op-integration/04-lowering-backend.md](relay-op-integration/04-lowering-backend.md) | `FRelayToTE`、`FRelayToTEMulti`、TIR/LLVM 支持边界 |
| 5 | [relay-op-integration/05-frontend-tests-ci.md](relay-op-integration/05-frontend-tests-ci.md) | FFI、ONNX、测试、CI |
| 6 | [relay-op-integration/DRIFT.md](relay-op-integration/DRIFT.md) | 旧 GUIDE 与当前实现的漂移记录 |

## 当前硬规则

新增 Relay op 不能只注册 metadata。一个 op 只有同时满足 contract、schema、type inference、lowering、FFI、测试和后端能力，才算对外支持。

不允许：

- metadata-only alias。
- `_make` alias helper。
- 空 `FRelayToTE` / `FRelayToTEMulti`。
- `return te::Tensor()`。
- 用近似实现冒充真实语义。
- TOPI helper 存在但 Relay contract、测试、CI 没跟上。

## PR 前最小检查

```bash
python python/tools/check_relay_op_contract.py --root .
cmake --build out/build/<cpu-build> --target run_infer_type_test
cmake --build out/build/<llvm-build> --target run_op_numeric_llvm_test
```

如果新增 ONNX 映射，还要运行：

```bash
cmake --build out/build/<llvm-build> --target run_onnx_importer_test
```
