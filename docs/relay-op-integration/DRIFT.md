# 旧文档漂移核对

本页记录把现有 GUIDE 对照当前代码后的漂移点。后续新增算子时不要再按这些漂移内容实现。

## 已确认漂移

| 旧文档 | 漂移点 | 当前做法 |
| --- | --- | --- |
| `ADD_RELAY_OP_GUIDE.md` | 单篇内容过长，把 contract、Relay、TOPI、lowering、CI 混在一起 | 拆到 `docs/relay-op-integration/`，入口只保留导航 |
| `TOPI_COMPOSITION_GUIDE.md` | 可读性好，但没有区分“TOPI helper 可用”和“Relay op 已支持” | 新 `03-topi-te.md` 保留 API 表，同时明确支持边界 |
| `OPERATOR_REGISTRATION_GUIDE.md` | 偏旧式注册背景，不覆盖当前 `FInferType`、`FRelayToTE`、checker contract | 新 op 注册以 `01-relay.md` 为准 |
| `RELAY_OP_END_TO_END_TUTORIAL.md` | 部分路径和 API 与当前实现不一致 | 只作历史教程，不作 PR 接入依据 |
| `TYPE_REGISTRATION_GUIDE.md` | 讲对象类型注册，不是 Relay op 接入规范 | 只作对象系统背景 |

## 当前实现事实

| 领域 | 当前事实 |
| --- | --- |
| TOPI 路径 | helper 分散在 `include/te/topi/*.h`，汇总头是 `include/te/topi.h` |
| Relay 注册 | 普通 op 使用 `KXC_REGISTER_OP(name)`；`device.` op 不能用宏参数表达点号 |
| Type hook | 所有 contract op 必须有 `FInferType` |
| Lowering hook | 单输出用 `FRelayToTE`，多输出用 `FRelayToTEMulti` |
| 多输出 lowering | `LowerToTIR` 已支持 `Tuple`、`TupleGetItem`、多个 output buffer；具体 multi op 仍要逐个注册和测试 |
| Checker | 以 `contracts/relay_op_contract.json` 为规范源，静态扫描注册、FFI、ONNX 和测试引用 |
| LLVM | 是否支持不能靠文档声明，必须通过 `run_op_numeric_llvm_test` 或等价 numeric test |

## 保留旧文档的用途

旧文档可以帮助理解背景，但不能替代 contract 和新文档包。新增 op 的 PR 描述应引用本目录中的具体子文档，而不是引用旧单篇教程。
