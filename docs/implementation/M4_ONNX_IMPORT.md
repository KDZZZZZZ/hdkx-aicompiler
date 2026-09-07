# M4：把模型入口接通，并支持一个节点的多个结果

现在 Relay 中已有 24 个算子契约，但 Python ONNX importer 只有 15 个映射，其中只有 8 个名字出现在目标 Encoder 快照里。模型即使只需要已经能算的乘法、除法或变形，也可能在导入时被拒绝。Python 导入器和 C++ 重建器还都限制一个节点只能产生一个结果，因此 Split 不能直接到达现有多输出 lowering。

本模块分两段工作。第一波先把已有静态计算接到 ONNX，让一段常量、算术、归约和变形小图能够真正编译执行；后续再把一个节点的多个输出按顺序交给 Relay 和运行时，并用 Split 证明。动态 shape 值属于 M3，不能在这里用 Python 临时求值普通模型数据来绕过。

> 状态：待实施。第一波 B 线，依赖 [M0](M0_BASELINE.md)。当前入口限制见 [ONNX 导入器](../ONNX_IMPORTER.md)和[架构总览](../ARCHITECTURE.md)。

## 当前需要区分的三个表面

| 表面 | 现状 | 这次如何处理 |
|---|---|---|
| Relay 声明和数学实现 | 已有 cast/divide/mul/reduce_mean/reshape/sqrt/subtract | 复用，不另写算术内核 |
| Python ONNX 解析和序列化 | 缺少上述映射和 Constant 节点处理 | 增加明确的 opset、属性、常量控制输入校验 |
| C++ spec 重建 | MakeAttrs 和节点结果绑定不自动跟随 Python 映射增加 | 同步重建同一 canonical op；对手写错误 spec 同样拒绝 |

当前 `Gemm` 映射到 `nn_gemm`，不是 `nn_dense`。当前 Relay 契约也没有 `split` 条目；已有的是 Tuple/多输出基础能力，不等于 Split 已经实现。

## S1：第一波静态范围

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

第一波只完成 S1。新增名称交集不是完整模型验收，特别是动态控制值和多输出仍未开放。

## S2：已有名称的模型子集补齐

S1 后单独增加静态 Squeeze/Unsqueeze 规范化和多输入 Concat。axes 固定、结果 rank 可证明时优先转换为已有 reshape；三/四输入 Concat 优先确定性地展开为二元 concatenate 链，不为一个模型需要重做 variadic 基座。

记录展开后的稳定顺序和 identity，验证结果与原 ONNX 语义一致。普通运行时 Gather 索引需要明确范围检查，作为后续切片与 M3/目标后端一起推进，不能因为“名字已映射”就取消当前常量索引限制。

## S3：多输出与 Split

1. 审计当前序列化 spec 的 outputs 数组、版本合同，以及 C++ 单输出断言。必要时最小提升格式版本，旧单输出规格仍按原规则处理。
2. 新增 Split 的 canonical op、attrs、输出叶子数量与 InferType；首例固定两个输出、静态 axis、常量 split sizes，按所选 opset 验证。
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

- [ ] S1 八个新名称均有 protobuf → Relay → LLVM → RuntimeSession 数值证据。
- [ ] 控制输入缺失、属性类型错误、shape/dtype 声明不一致均得到包含节点名的诊断。
- [ ] Constant 不被错误抹成 float32，不新增多余运行 kernel；常量字节参与既有 identity。
- [ ] Python 和 C++ 手写 spec 的负例都覆盖，未依赖 Python 作为唯一校验边界。
- [ ] 与 C 线联合完成 Equal ONNX 入口，合同映射和生成文件一致。
- [ ] S3 单独证明两个输出的顺序、类型、后续消费和数值，旧单输出/optional output 不回归。

## 身份与交接

原有 op 的正确接线一般不改变其 kernel ABI；其 attrs、常量和形状仍由现有 semantic key 覆盖。新增 Split 的输出结构、规范 attrs 和必要格式版本必须进入既有身份规则。向 M3 交接清楚哪些控制输入仍必须是常量，不能在 importer 中暗自接受任意动态 tensor 后再让 backend 猜测。
