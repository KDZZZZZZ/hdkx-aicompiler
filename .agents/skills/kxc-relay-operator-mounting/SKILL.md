---
name: kxc-relay-operator-mounting
description: Add or migrate KXC Relay operators and ONNX importer mappings through the declaration-driven operator contract. Use for operator schema, attrs, InferType, FRelayToTE/FRelayToTEMulti, generated registration, FFI helpers, NLP operators, and operator compilation or numeric tests.
compatibility: KXC repository with contracts/relay_op_contract.json and generated Relay registration support.
---

# KXC Relay 算子挂载

先完整读取 [总则](../kxc-capability-mounting/SKILL.md)。本 skill 只负责 Relay 声明、类型和 TE 边界；schedule/backend 继续读取 `kxc-te-backend-mounting`。

## 现有 owner

```text
contracts/relay_op_contract.json                     schema 唯一声明源
python/tools/generate_relay_op_contract.py           metadata/registration 生成器
python/tools/check_relay_op_contract.py              freshness/单注册/符号/测试检查
src/relay/generated/relay_op_contract.inc            generated metadata
src/relay/generated/relay_op_registration.cc         generated registration TU
src/ffi/builtin_registry.cc                          static archive anchor
src/relay/op/**                                      attrs 与 TE callback 实现
src/relay/type_infer.cc                              InferType 实现
```

`registration` 当前已支持 fieldless、固定 arity、单输出、`FInferType + FRelayToTE`。碰到 attrs、variadic 或 multi-output 时，不要宣布被生成器阻塞：为当前真实算子最小扩展 generator/checker，并用一个端到端算子证明；不要先造通用 ODS。

## 挂载步骤

1. **先定义语义**
   - canonical op 名，不建同义 alias。
   - 明确输入 arity、attrs、输出叶子数、effect、determinism、alias contract。
   - dtype/rank/broadcast/reduction/axis 的失败规则先写清楚。
2. **更新 JSON**
   - 保持 operators 字典排序和字段稳定顺序。
   - `schema_version` 只在可观察 schema/语义改变时递增。
   - ONNX 名只写进 `onnx_ops`，Relay canonical 名不随 importer 改变。
3. **选择注册路径**
   - 符合 generated subset：增加 `registration`，只保留 callback 定义；删除手写 `KXC_REGISTER_OP`。
   - 不符合 subset：优先按当前算子扩展生成器。迁移完成前，只有**没有** `registration` 的 entry 才可暂时手写注册。
   - 任何 entry 都不能 generated + manual 双注册。
4. **实现 InferType**
   - 使用 checked attrs 和现有 type helpers。
   - 对 arity、dtype、rank、axis、broadcast 明确拒绝；不要让 TE/backend 代替类型检查。
5. **实现 TE callback**
   - 只表达计算 DAG，不在 callback 中选择 Target、分配 runtime storage 或生成 launch metadata。
   - 单输出用 `FRelayToTE`；多输出用既有 `FRelayToTEMulti` 契约。
   - 返回 shape/dtype 必须与 InferType 完全一致，生产 lowering 会再次校验。
6. **接 importer/FFI（若需要）**
   - importer 只映射已注册 canonical op，并正确构造 attrs。
   - FFI helper 名和输入数必须与 JSON/checker 一致。
7. **重新生成并检查**

```bash
python3 python/tools/generate_relay_op_contract.py \
  --matrix contracts/relay_op_contract.json \
  --output src/relay/generated/relay_op_contract.inc \
  --registration-output src/relay/generated/relay_op_registration.cc
python3 python/tools/check_relay_op_contract.py --root .
```

生成文件应由 generator 改写，禁止手改。

## Identity 规则

- canonical op 名、schema version、attrs、类型/常量语义进入既有 graph/unit identity。
- callback C++ symbol 是 build-local 细节，不放进 `OperatorSpec` 或 artifact identity。
- 不用新 kernel naming 体系规避现有 `PrimitiveUnit` identity。
- 若 lowering 行为改变但 Relay 语义不变，schedule/pipeline/backend identity 必须在对应 owner 变化。

## 最小测试矩阵

- Registry：lookup 只有一个 owner，静态 archive anchor 可达。
- InferType：合法 shape/dtype 与至少一个边界负例。
- Lowering：真实 `LowerPrimitiveUnit` 产生 TIR，不只直接调用 callback。
- CPU/LLVM：至少一个数值正例。
- Importer：有 ONNX mapping 时验证 attrs 和 canonical op。
- Generated：freshness 和 checker 必须通过。

```bash
ctest --test-dir <build> --output-on-failure \
  -R 'relay_op_contract|registry_test|infer_type_test|operator_compilation_test|op_numeric_llvm_test'
```

## 禁止

- 在 runtime 解析 operator JSON。
- 为一个 op 建第二 registry/factory。
- generated entry 保留手写 macro。
- InferType 接受、TE callback 再偷偷修正 shape。
- 为了 CUDA 不可用而跳过 CPU/LLVM 数值证据。
- 一次批量加入许多只有 schema、没有 consumer 的 op。

完成标准：JSON、注册、InferType、TE、生产 lowering、identity 和数值测试形成一条纵向链。
