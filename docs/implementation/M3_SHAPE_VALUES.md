# M3：形状作为值与有界 Transformer 计算

模型不只计算浮点数据，也会计算“下一步张量应该有多长”。例如先读取输入的序列长度，再拼出一个目标形状，最后做 Reshape 或 Expand。现在已有形状代数和精确 profile 路由，但 ONNX 里的这条形状计算链还没有接通；有界动态分支也只证明了窄的逐元素执行。

本模块要让编译器理解由输入维度和常量推导的形状，并在已声明的范围内执行。它还必须把有界能力扩展到实际需要的注意力算子。否则即使 Shape 节点能导入，MatMul 和 Softmax 仍会在动态维度上被拒绝，变长 Transformer 依然无法运行。

> 状态：待实施，G0 后推进。与 [M2](M2_KV_STATE.md)、[M4](M4_ONNX_IMPORT.md)协同。现有边界见[架构总览](../ARCHITECTURE.md)，返回[模块总览](README.md)。

## 当前可以复用什么

- [ShapeProgram/DimExpr](../../include/kxc/shape/shape.h)：符号、范围、整除约束、求值和 canonical 内容。
- [精确 profile 控制](../../src/compiler/shape/shape_control.cc)：只发布/查询 caller 已准备的产物，miss 没有编译副作用。
- [受限 symbolic adapter](../../src/compiler/shape/restricted_symbolic_shape.cc)和 M0 集成后的动态 unit shape contract：把已证明的形状关系传到 lowering。
- [ModuleInvocationContract](../../include/kxc/runtime/compiled_module.h)：运行时消费的形状/容量及 extent 参数合同。

ShapeProgram 是编译控制面的形状权威；runtime 使用既有 ModuleInvocationContract 表达和验证已经 lower 的合同。不能让 RuntimeSession 直接依赖 compiler 的 ShapeProgram，也不能在 Python importer、router 中各复制一套形状解释器。

## 允许处理的形状来源

首版只接受固定 rank、有限范围、来自输入维度和常量的可证明表达式。一个 shape tensor 的长度必须可知，它表示的每个维度有界。由普通张量内容决定的任意输出大小，以及未知 rank，继续拒绝。

例如输入为 `[B,S,H]`，B、S 有明确上下限，H 固定。编译器可以证明由输入维度组成的目标形状；运行时在 guard 通过后只求值并分配。它不能遇到一个没见过的 S 就编译新模型。

## 分阶段实施

### S1：最小 shape 值纵向链

1. 在现有 Relay 值和类型边界中表达 shape 结果及其维度来源。给 `Shape` 定义明确 dtype、固定向量长度和可支持的切片/axis 子集。
2. 静态已知部分在编译准备中解析成常量；动态部分保留 ShapeProgram 表达式，并复用既有 canonical 编码。
3. 选择 `Shape(input) → Reshape` 的一个固定秩例子贯通到生产运行。同一产物用两组合法输入形状执行，输出 shape 和数值都与参考一致。
4. 如果 shape 值本身是图输出，也要物化为真实结果，不能只更新调试 metadata；如果只作为形状控制输入，lower 到既有 invocation/extent 合同。
5. shape-to-module 的翻译验证范围、溢出、参数顺序和 producer/consumer 对应，运行时不增加第二个任意 shape VM。

S1 第一项定义和最后一项消费者必须同一纵向切片交付，不能先合入一个没有 lowerer 的 shape tensor API。

### S2：模型中的形状算子

| 算子/链路 | 最小实现范围 | 验证重点 |
|---|---|---|
| Shape | 固定 rank，来源为输入维度 | 输出内容、dtype、固定长度 |
| Gather/Concat 组成的形状表达式 | 常量索引、静态已知的向量长度 | 保持唯一表达式来源，索引越界拒绝 |
| Reshape | 元素总数可证明相同，控制输入来自受限形状表达式 | 0、-1、元素数和溢出，opset 行为一致 |
| Expand | 受限形状表达式确定目标，各轴广播合法 | 多向广播、零维度、输出容量 |
| ConstantOfShape | 有界目标形状和明确填充值类型 | 标量值、空结果、字节上限 |
| Squeeze/Unsqueeze | axes 已知且结果 rank 可证明固定 | 重复/越界 axis；Squeeze 所移除维度须证明为 1 |

静态 Squeeze/Unsqueeze 优先由 M4 规范化为既有 reshape。动态维度并不自动意味着任意 rank 可变化；不能证明固定输出 rank 的省略 axes 用法继续拒绝。

### S3：有界注意力算子

先逐项审计当前 bounded adapter 的 allowlist 和实际 lowering，不以静态 InferType 已支持为依据。按一个微型注意力图所需的顺序扩展 MatMul、Transpose、Softmax/归约、mask、Slice/Concat，以及必要的 normalization。

每次只放开已实现的动态维度组合。例如 head_dim/归约容量先保持固定，batch/序列长度有界；需要动态归约时，明确 init/update、loop extent、有效区和尾部访问的证明。通用 Gather 的运行时索引还需独立范围检查，不能沿用 ONNX 当前“常量索引已证明安全”的结论。

与 M2 联合完成一个同会话 prefill + decode 的有界图；各形状的输出都与显式 exact 编译或独立参考比较。若还缺某个动态算子，则保留该项 unsupported，不把 S1/S2 结果替代 S3。

## 实现落点与分工

| 位置 | 负责的事实 |
|---|---|
| [shape.h](../../include/kxc/shape/shape.h)、[shape.cc](../../src/shape/shape.cc) | 共享代数、约束、规范化；仅按实际需要最小扩展 |
| `src/compiler/shape/`、dynamic shape contract | Relay shape 来源、求值结果与 unit 边界的转换 |
| [type_infer.cc](../../src/relay/type_infer.cc)、Relay/TE callbacks | 结果类型和计算语义 |
| [lowered_graph.cc](../../src/compiler/lowering/lowered_graph.cc)、[te_to_tir.cc](../../src/compiler/lowering/te_to_tir.cc) | 受限动态循环、ABI extent 顺序和实际计算 |
| 模块调用合同和 runtime | 已证明表达式的求值、容量校验和分配；不做 profile 编译 |
| M4 的 Python/C++ importer | 把 ONNX 节点送入上述唯一链路，不在 Python 偷算普通数据 |

exact profile 与 bounded applicability 保持不同含义。共用类型/evaluator/builder 不等于强行把所有形状模块合成一个类；整理只围绕真实重复发生，不在本模块开展大规模文件搬迁。

## 身份、失败和测试

shape 表达式、bounds、控制输入来源以及有序 extent ABI 都必须进入相应版本化合同。复用已有 graph/profile/plan builder；相同结构仅临时变量名不同应保持身份，相同 dtype 但不同有效范围不能误复用不兼容产物。

```bash
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error \
  -R 'shape_.*test|bounded_attention_llvm_test|restricted_symbolic_shape_test|bounded_dynamic_graph_llvm_test|infer_type_test|te_schedule_test|compiled_module.*test|compiler_identity_test|onnx_importer_test'
```

S1/S2 的生产 fixture 拟放入新 `test/shape_value_llvm_test.cpp`，S3 拟放入新 `test/bounded_attention_llvm_test.cpp`。将同名目标注册并用 `ctest -N` 确认后，运行公共检查和完整 LLVM 回归。必须证明：

- [ ] 同一产物在两个以上合法 shape 上返回正确数值和实际输出形状，runtime 期间编译/cache 统计不变。
- [ ] 缺失 binding、重复 symbol 不一致、越界、整除失败、非法广播和溢出均在 launch 前拒绝。
- [ ] 数据相关任意 shape、未知 rank 和无法证明的 axis 行为仍拒绝。
- [ ] route miss 不编译，不偷偷扩大 bucket，不选择近似 shape。
- [ ] shape 值本身被输出或消费时，结果来自真实执行链，非仅 metadata。
- [ ] S3 动态算子和 M2 状态的联合证据独立列出，不能由 elementwise 例子推断。

## 下一步交接

给 M4 一份受支持的 shape 子图、opset/axes 限制和报错规则；给 M2 一份 prefill/decode 可共用的具体 shape/extent ABI；给集成负责人逐格证据。请求级动态合批另行立项，不能把本模块的 batch 维可变写成 continuous batching 已实现。
