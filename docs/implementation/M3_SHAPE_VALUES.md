# M3：形状作为值与有界 Transformer 计算

MiniMind-L1 的目标是 prefill + decode，MiniMind-O 的 Thinker 复用这条文本 backbone。第一波 G0 已证明一份 CPU/LLVM fresh-output 产物可以服务多个合法 shape，但它明确拒绝 state、alias 和动态有效长度；第一波 G1 也只接通了静态 ONNX 子集。MiniMind 的 dynamic_axes 原始导出会出现 Shape、Gather、Concat、Range、ConstantOfShape 和大量 Unsqueeze，静态/常量折叠后又会消失一部分，因此“图里出现 Shape”不等于必须照搬导出器的 shape 链。

本模块要把真正需要的形状值接入唯一的 ShapeProgram/ModuleInvocationContract：先验证 MiniMind 导出的 batch、prefill sequence、past 和 total 关系，再在固定 rank、有界范围和整除约束内执行。它还要把有界能力扩展到实际 attention；否则 Shape 虽可计算，MatMul/Softmax/KV 仍不能在变长输入上运行。未知 rank、数据相关任意形状和运行时隐式编译继续拒绝。

> 状态：待实施，第二波 C 线。先过 [M9](M9_MINIMIND_TARGET.md) E0 的导出审计，与 [M2](M2_KV_STATE.md) 的 extent 合同、[M4](M4_ONNX_IMPORT.md) 的节点重建协同；M10 的控制流 runtime 当前拒绝运行时生成的 extent，因此若控制图要参与 decode，必须消费本模块发布的受限 shape/extent ABI，不能在控制流 executor 里复制 ShapeProgram。第一波证据见 [G0](G0_BASELINE.md)/[G1](G1_RECORD.md)。

## 本模块要做的模块

| 模块 | 要解决的问题 | 首个证据 |
|---|---|---|
| 导出 shape 审计 | 真实 past/total/batch/seq 轴和输出顺序未锁定 | M9 receipt + 轴/范围报告 |
| Shape-as-value | Shape 等控制张量不能稳定进入 Reshape/Expand | Shape→Reshape 真实生产链，结果不是 metadata |
| 有界 attention | dynamic batch/sequence 下 attention 仍会被拒绝 | 两个合法 shape 的同一产物 LLVM 结果 |
| 交接 M2 | KV 的 cursor/valid extent 不能由普通数据 shape 猜出 | prefill→decode 的显式 extent ABI |
| 失败边界 | route miss、越界、未知 rank 可能触发隐式编译 | launch 前零调用/零编译负例 |

## 当前可以复用什么

- [ShapeProgram/DimExpr](../../include/kxc/shape/shape.h)：符号、范围、整除约束、求值和 canonical 内容。
- [精确 profile 控制](../../src/compiler/shape/shape_control.cc)：只发布/查询 caller 已准备的产物，miss 没有编译副作用。
- [受限 symbolic adapter](../../src/compiler/shape/restricted_symbolic_shape.cc)和 M0 集成后的动态 unit shape contract：把已证明的形状关系传到 lowering。
- [ModuleInvocationContract](../../include/kxc/runtime/compiled_module.h)：运行时消费的形状/容量及 extent 参数合同。

ShapeProgram 是编译控制面的形状权威；runtime 使用既有 ModuleInvocationContract 表达和验证已经 lower 的合同。不能让 RuntimeSession 直接依赖 compiler 的 ShapeProgram，也不能在 Python importer、router 中各复制一套形状解释器。

## 允许处理的形状来源

首版只接受固定 rank、有限范围、来自输入维度和常量的可证明表达式。一个 shape tensor 的长度必须可知，它表示的每个维度有界。由普通张量内容决定的任意输出大小，以及未知 rank，继续拒绝。

MiniMind 首轮只绑定这些维度：batch、prefill sequence、decode current sequence（首版为 1）、past length 和 total length；hidden、heads、kv_heads、head_dim、vocab 和层数固定。`total = past + current`、GQA 的 repeat factor、RoPE 的 head 维度必须在导出 receipt 和 invocation contract 中各出现一次，不能由 importer、router、RuntimeSession 分别推导。

例如输入为 `[B,S,H]`，B、S 有明确上下限，H 固定。编译器可以证明由输入维度组成的目标形状；运行时在 guard 通过后只求值并分配。它不能遇到一个没见过的 S 就编译新模型。

## 分阶段实施

### S1：最小 shape 值纵向链

1. 在现有 Relay 值和类型边界中表达 shape 结果及其维度来源。给 `Shape` 定义明确 dtype、固定向量长度和可支持的切片/axis 子集。
2. 静态已知部分在编译准备中解析成常量；动态部分保留 ShapeProgram 表达式，并复用既有 canonical 编码。
3. 选择 `Shape(input) → Reshape` 的一个固定秩例子贯通到生产运行。同一产物用两组合法输入形状执行，输出 shape 和数值都与参考一致。
4. 如果 shape 值本身是图输出，也要物化为真实结果，不能只更新调试 metadata；如果只作为形状控制输入，lower 到既有 invocation/extent 合同。
5. shape-to-module 的翻译验证范围、溢出、参数顺序和 producer/consumer 对应，运行时不增加第二个任意 shape VM。

S1 第一项定义和最后一项消费者必须同一纵向切片交付，不能先合入一个没有 lowerer 的 shape tensor API。若 M9 的 dynamic decoder 导出失败，S1 先用静态/受控 shape fixture 验证合同，同时把真实动态路径留在失败清单。

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

MiniMind 的第一组 attention 证据固定 head_dim=96、kv_heads=4、current sequence=1，先只放开 batch 和 past/total 的有限区间。不要把把 static prefill 图上 650 个实算节点的通过，写成 dynamic attention 已通过；动态轴导出中额外的 Shape/Range/ConstantOfShape 只有在真实图仍保留且被执行时才进入能力矩阵。

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
