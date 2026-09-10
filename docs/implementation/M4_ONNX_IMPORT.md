# M4：把 MiniMind 模型入口接通，并支持多输出

第一波 B 线已经完成 Constant、Cast、Div、Mul、Sub、Sqrt、ReduceMean、Reshape 的受限静态导入，并由 G1 证明 Equal→Where 的 ONNX 组合可以进入 LLVM RuntimeSession。目标已经从旧的匿名 Encoder 快照改为 MiniMindForCausalLM：实际导出包含 prefill 和 decode 两张图，decode 有 8 层 past/present，静态非原地 mask 的常量折叠统计与 dynamic_axes 原始统计不同。当前仍不能把 importer 的名称交集写成“MiniMind 可运行”，因为 Concat 输入数、Gather 索引、ReduceMean 中间轴和动态 shape 控制值都有语义边界；Split 已在静态常量两路以上的子集形成完整纵向证据。

本模块的下一段工作是以 M9 锁定的导出为唯一输入，按实际节点和属性补齐 L1a prefill，再接通 decode 的 past/present 输出顺序。静态常量控制输入可以在 importer 中证明；Shape/Expand/ConstantOfShape 等运行时 shape 值交给 M3，KV 所有权和 extent 交给 M2。不能在 Python 中偷算普通模型数据来绕过 Relay/TE/Runtime。

> 状态（2026-09-10）：静态 MiniMind prefill/decode 与多图输出已完成，见 [G2](G2_RECORD.md)和 [M2 报告](M2_MINIMIND_STATE_REPORT.md)。显式 shape-source 已接通真实拆头、Q/K 归一化、RoPE 组合链与 K/V GQA，见 [拆头报告](M3_ONNX_HEADS_REPORT.md)、[GQA 报告](M3_GQA_REPORT.md)和 [RoPE 报告](M3_ROPE_REPORT.md)。ConstantOfShape 与 Expand 控制输入保留到 producer 证明，内部暂未确定的 Expand extent 不在 Python 中冻结。float32 ConstantOfShape、静态对角 Trilu 与可证明固定商的 Reshape -1 已联合到实际第一层 attention，见 [因果 attention 报告](M3_CAUSAL_ATTENTION_REPORT.md)。完整八层变长 prefill 已通过，含 embedding、图内位置前缀、FFN/残差和全部 17 个输出，见 [完整 prefill 报告](M3_FULL_PREFILL_REPORT.md)。完整八层变长 decode 已从原始 source 图导入，并验证不同 B/P 与连续 greedy，见 [完整 decode 报告](M3_FULL_DECODE_REPORT.md)；state/extent 联合已通过，见 [状态报告](M2_BOUNDED_STATE_REPORT.md)。M4-D Split 已完成限定范围的 ONNX → Relay → LLVM → RuntimeSession 静态常量两路以上输出验证，详见 [基础 Split 技术报告](M4_SPLIT_REPORT.md)和[多路扩展报告](M4_SPLIT_VARIADIC_REPORT.md)。下文保留分阶段计划，实际入口边界以 [ONNX importer](../ONNX_IMPORTER.md) 为准。

## 本模块要做的模块

| 模块 | 当前情况 | 计划结果 |
|---|---|---|
| M4-A 真实图审计 | M9 有 exporter/inventory；raw 图和折叠图统计口径不同 | 每个 MiniMind stage 有 opset、输入/输出顺序、attrs 和拒绝原因报告 |
| M4-B L1a prefill | 第一波静态子集已接通，仍缺实际 MiniMind 算子 | 真实 prefill protobuf → Relay → LLVM → RuntimeSession 数值证据 |
| M4-C decode 多输出 | 8 层 past/present 名称存在于导出包装器 | 输出顺序、Tuple/叶子绑定和 ABI 与 M2 state 合同一致 |
| M4-D 后续 Split | 已完成静态常量 Split 的 canonical 纵向证据 | 已有双路和三路 fixture、下游/图输出顺序、负例和 LLVM 数值报告；动态 split lengths、少于两路和输出数不匹配仍关闭 |

## 当前需要区分的三个表面

| 表面 | 现状 | 这次如何处理 |
|---|---|---|
| Relay 声明和数学实现 | 第一波已补齐一批静态算术/归约/变形和 Equal；MiniMind 仍有实际缺口 | 复用现有 canonical op；新语义按 M5 独立完成 LLVM 数值 |
| Python ONNX 解析和序列化 | 已能导入第一波静态子集，M9 inventory 中的动态控制节点和部分属性仍未开放 | 以真实 MiniMind opset 17 图逐节点校验；不把名称交集当成能力 |
| C++ spec 重建 | 第一波已同步 Constant 与算术 attrs；多输出和 past/present 顺序仍需证明 | 同步重建同一 canonical op；手写错误 spec、输出数量和名字顺序都拒绝 |

当前 `Gemm` 映射到 `nn_gemm`，不是 `nn_dense`。Split 使用独立的 canonical `split` 条目和既有 Tuple/多输出 ABI；该条目开放静态常量、至少两路的分段子集。

## S1：第一波静态范围（已完成）

首批 fixture 固定使用 ONNX opset 17，只声明实际实现并测试的 dtype/shape 子集。不能把某个版本的通过写成所有 opset 都支持。

| ONNX 名称 | 入口处理与复用对象 | 首版必须写明的边界 |
|---|---|---|
| Constant | 进入已有 ParamTensor/Relay Constant 路径，不新增数学 op 或运行时 kernel | 先支持 dense Tensor 的 value，明确 dtype；多属性、sparse 或不支持格式拒绝 |
| Mul、Sub、Div | 复用 mul/subtract/divide | 先 float32，固定 rank/shape，合法广播；不顺带宣称整数除法 |
| Sqrt | 复用 sqrt | float32；数据的 NaN/负值结果按数学/backend 合同处理，不伪装成 shape 校验错误 |
| Cast | 复用 cast attrs | 首批覆盖模型需要的 int32/int64 到 float32；其他转换明确说明是否支持 |
| ReduceMean | 复用 reduce_mean attrs | opset 17 的静态 axes、keepdims；opset 18 的输入形式不自动开放 |
| Reshape | 常量 shape 输入解析为已证明的目标形状，再用 reshape | 先 allowzero=0；shape 仅来自 initializer/Constant；总元素数相同 |

Constant 的属性互斥规则、Reshape 的 0/-1 及 allowzero 行为、ReduceMean 的版本区别分别以 [ONNX Constant](https://onnx.ai/onnx/operators/onnx__Constant.html)、[Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html)、[ReduceMean](https://onnx.ai/onnx/operators/onnx__ReduceMean.html)为依据。实现中选择受限子集，并保留其它组合的明确拒绝。

### S1 实施顺序

1. 先收集所有会消费新节点结果的位置：后续节点 shape 推断、常量控制输入、序列化参数偏移、C++ MakeAttrs、图输出声明。
2. 在按拓扑处理节点时将 Constant 规范化到现有常量表示。保留 dtype、标量/零尺寸语义、名字唯一性和真实内容，不能与 initializer 冲突，也不能引用尚未产生的值。
3. 接通 Mul/Sub/Div/Sqrt 的映射与静态输出信息传播，验证 `Constant → Mul → Sub → Div → Sqrt` 的一段安全数值小图。
4. 接通 Cast、ReduceMean 和 Reshape。处理负 axis、空 axes、keepdims、0/-1 推导及整数溢出；无法证明唯一目标 shape 时拒绝。
5. C++ 重建器同步构造 attrs 并校验手写 spec。Python 已经校验不能成为 C++ 信任非法 spec 的理由。
6. 更新现有 op 条目的 onnx_ops 元数据，重新生成合同；从真实 protobuf 导入后走 Compiler/RuntimeSession，与独立参考逐元素比较。
7. 接收 [M5](M5_ELEMENTWISE_OPS.md) 的 Equal 后，由本模块独占 importer 接线，增加 `Equal → Where` 组合 fixture。

第一波已经完成 S1 的受限小图和 Equal→Where 组合；这些证据证明导入链可用，不等于 MiniMind 全图、动态控制值或多输出已经开放。后续 S2/S3 以 M9 的真实节点清单为准。

## S2：已有名称的模型子集补齐

S1 后单独增加静态 Squeeze/Unsqueeze 规范化和多输入 Concat。axes 固定、结果 rank 可证明时优先转换为已有 reshape；三/四输入 Concat 优先确定性地展开为二元 concatenate 链，不为一个模型需要重做 variadic 基座。

记录展开后的稳定顺序和 identity，验证结果与原 ONNX 语义一致。普通运行时 Gather 索引需要明确范围检查，作为后续切片与 M3/目标后端一起推进，不能因为“名字已映射”就取消当前常量索引限制。

## S3：多输出与 Split

1. 审计当前序列化 spec 的 outputs 数组、版本合同，以及 C++ 单输出断言。必要时最小提升格式版本，旧单输出规格仍按原规则处理。
2. 新增 Split 的 canonical op、attrs、输出叶子数量与 InferType；支持至少两个输出、静态 axis、常量 split sizes，按所选 opset 验证。
3. 复用既有 FRelayToTEMulti 和扁平多输出 lowering；generated 注册对 attrs/multi-output 不足的部分随这个真实算子最小扩展，不能新建第二 registry。
4. C++ 只构造一次 Call，将各输出名按顺序绑定到对应 TupleGetItem，防止重复计算或把所有名字都指向同一个结果。
5. 校验结果数量、名字唯一性、声明 shape/dtype、叶子顺序及 kernel 输出 ABI。已有多输出计划若已足够就直接复用，不预先重做整个 ABI。
6. 用两个结果被不同下游消费的 ONNX 图验证；一个输出作为图结果，另一个参与后续算术，再比较两者数值。

ONNX Split 的输入 sizes 与 num_outputs 规则有版本差异，按[官方 Split 文档](https://onnx.ai/onnx/operators/onnx__Split.html)固定测试版本和支持子集。LayerNormalization 已支持的空 optional output 槽位仍按其原合同处理，不能因 Split 开放而放宽全部可选输出。

## 代码落点

- [Python importer](../../python/kxc_onnx/importer.py)、[Python 包](../../python/kxc_onnx)：解析、模型值表和参数序列化。
- [C++ importer](../../src/frontend/onnx_importer.cc)：attrs、常量和结果名重建。
- [Relay contract](../../contracts/relay_op_contract.json)、[生成器](../../python/tools/generate_relay_op_contract.py)：映射同步以及后续 Split 注册。
- [resolved_relay_call](../../src/compiler/analysis/resolved_relay_call.cc)、[lowered_graph](../../src/compiler/lowering/lowered_graph.cc)：复用并验证现有多输出通道。
- [Python 导入测试](../../test/onnx_importer_py_test.py)、[spec 测试](../../test/onnx_import_spec_contract_test.cpp)、[端到端测试](../../test/onnx_importer_test.cpp)。

## 验证与退出条件

```bash
PYTHONPATH=python python3 -m pytest test/onnx_importer_py_test.py
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'onnx_importer_test|onnx_import_spec_contract_test|relay_op_contract|infer_type_test|operator_compilation_test|op_numeric_llvm_test'
python3 python/tools/check_relay_op_contract.py --root .
```

依赖缺失时不能把 pytest 未执行或 LLVM 条件测试跳过写成通过；记录缺口并完成现有可用层。随后执行第一波公共检查和 LLVM 全量回归。

- [x] S1 八个新名称均有 protobuf → Relay → LLVM → RuntimeSession 数值证据（第一波 G1）。
- [ ] 控制输入缺失、属性类型错误、shape/dtype 声明不一致均得到包含节点名的诊断。
- [ ] Constant 不被错误抹成 float32，不新增多余运行 kernel；常量字节参与既有 identity。
- [ ] Python 和 C++ 手写 spec 的负例都覆盖，未依赖 Python 作为唯一校验边界。
- [x] 与 C 线联合完成 Equal ONNX 入口，合同映射和生成文件一致（第一波 G1）。
- [x] S3 单独证明至少两个输出的顺序、类型、后续消费和数值，旧单输出/optional output 不回归；详见 [基础 Split 技术报告](M4_SPLIT_REPORT.md)和[多路扩展报告](M4_SPLIT_VARIADIC_REPORT.md)。

## 身份与交接

原有 op 的正确接线一般不改变其 kernel ABI；其 attrs、常量和形状仍由现有 semantic key 覆盖。新增 Split 的输出结构、规范 attrs 和必要格式版本必须进入既有身份规则。向 M3 交接清楚哪些控制输入仍必须是常量，不能在 importer 中暗自接受任意动态 tensor 后再让 backend 猜测。

## 第二波 D 线实施记录（2026-09-07）

S2 的 Unsqueeze 部分已实现：axes 已知、结果 rank 可证明时 Unsqueeze 在
Python importer 内规范化为既有 `reshape`（不新增 canonical op），axes 必须
来自 initializer/Constant、按 ONNX-13 在 `rank(data)+len(axes)` 上规范化负轴，
重复/越界拒绝；动态 axes 输入不开放。本波同时接线 `Neg`、`Sigmoid`、`Pow`、
`Expand`（opset >= 13 边界、float32 子集、C++ reifier 双侧负例）。

`Expand` 的目标 shape 是常量控制输入：导入期解析为 canonical `expand` 算子的
`ExpandAttrs`（与 Reshape 常量 shape 输入规范化到 attrs 的既有方式一致），使
类型推导能证明唯一输出 shape；动态 shape 输入被拒绝并交接 M3 形状值切片。
真实 protobuf 的五算子组合 fixture（`test/generate_onnx_m4m5_ops_fixture.py`
→ `onnx_importer_test` 的 `TestRunM4M5OpsProtobufLLVM`）以独立手写参考逐元素
比较（max_abs_error=0）。
