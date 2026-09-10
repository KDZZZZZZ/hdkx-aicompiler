# M3 技术报告：同一 LLVM 产物执行不同长度的注意力

日期：2026-09-08。范围：M3 S3 的通用 attention 计算核心，以及 S1/S2 状态回写。

## 结果与用途

普通 Relay 图现在可以通过 `RestrictedSymbolicShapeAdapter` → `Compiler::CompileBounded` → `RuntimeSession`，用一份编译产物执行不同 batch、query 长度和 KV 长度的注意力。测试使用 4 个头、head_dim=96，执行四组输入，最大绝对误差 **5.43662e-7**，期间全部 primitive cache 统计不变。

此前这条入口只接通逐元素运算和受限形状值链。虽然静态 MatMul/Softmax 和专用 KV attention 已经可用，普通 bounded attention 仍被拒绝。本次补的是普通编译路径；没有增加 attention 专用算子、模型专用 executor 或第二套形状求值器。

这为变长 prefill/decode 的计算部分提供了基础。它尚未把真实 MiniMind 的整个动态导出图接入，也没有把 bounded fresh-output plan 与 KV 状态计划合并。真实模型当前仍使用 [M2 报告](M2_MINIMIND_STATE_REPORT.md) 中的静态 prefill 与容量型 decode。

## 方法

### 1. 复用已有五个 Relay 调用

测试图为：

```text
Q[B,H,Q,D]                         K[B,H,T,D]
    |                                  |
    |                           Transpose(0,1,3,2)
    +---------------- MatMul ----------+
                         |
               Add(mask[B,H,Q,T])
                         |
                    Softmax(-1)
                         |
                  MatMul(V[B,H,T,D])
                         |
                    O[B,H,Q,D]
```

测试输入 Q 已显式乘以 `1/sqrt(D)`。mask 是调用方提供的有限加性因果 mask，每一行至少保留一个有效位置。五个调用各编译为一个 LLVM kernel，复用现有 TE callbacks。

采用的不变量与 [TVM v0.20 Softmax](https://github.com/apache/tvm/blob/v0.20.0/python/tvm/topi/nn/softmax.py) 一致：先求最大值，再对减去最大值的输入求指数、求和、归一化；归约轴来自实际输入 shape。本次 MatMul 先取比 [ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) 更窄的同秩、同 batch 轴子集，避免把未证明的广播带入动态执行。

### 2. 由唯一形状解析器证明轴关系

[shape_value_resolver.cc](../../src/compiler/shape/shape_value_resolver.cc) 新增三个已有算子的受限形状传播：

- MatMul：两侧 rank 相同且至少为 2；batch 轴和 K 归约轴必须是相同的常量或同一个符号表达式；输出继承左侧 M 和右侧 N。
- Transpose：固定 rank 下的合法轴排列；支持已有默认反转轴和负轴规范化。
- Softmax：输出形状不变，归约轴必须有严格大于零的下界；不能执行空归约域。

代表样例中的相等长度不能用来证明两个符号相等。因此输入 overlay 从“按静态 shape 查表”改为“按参数身份绑定”。测试特意让 Q、K、V 的代表 shape 完全相同，再让运行时 Q 和 T 不同；另有反例证明独立的 K1/K2 即使代表值都为 4，也不能作为相等归约轴放行。

范围、整除和输入间相等关系仍由 ShapeProgram 声明，再投影到既有 graph guards 和 ModuleInvocationContract。runtime 不求解新的符号代数，也不在 miss 时编译。

### 3. 在既有 TE→TIR 中执行动态归约和临时分配

[te_to_tir.cc](../../src/compiler/lowering/te_to_tir.cc) 的 bounded serial policy 现在允许归约。数据循环先初始化输出，归约循环再逐项更新；循环仍保持串行、未拆分。动态归约长度必须是 ABI 中已登记的直接 extent load。

Softmax 的最大值、偏移值、指数值和分母使用已有 TE 中间张量。lowering 从同一 `DynamicUnitShapeContract` 取得 extent 上界，构造最大形状，复用静态张量大小检查证明 int32 循环域、int64 地址/字节数和 size_t 分配均可表示。缺失上界或最大临时分配溢出时，在 backend/cache 获取前拒绝。

生成的 `tir::Allocate` 仍使用实际长度，通过既有 LLVM scoped `malloc/free` 分配和释放。没有按最大长度遍历尾部，也没有新增 runtime scratch registry。

对于该 float32 Softmax，四块临时张量共需 `8 × B × H × Q × (T+1)` 字节。测试合同最大 B=3、H=4、Q=T=8 时为 **6912 字节**；实际分配随当前 shape 缩小。这是内部临时存储的计算量，不是 profiler 已测出的 malloc 总量；现有 alloc 事件不覆盖 kernel 内部 malloc。

### 4. 身份和观测

- bounded applicability 从 v2 升为 **v3**；schedule policy 从 `bounded-dynamic-serial-v1` 升为 **v2**，进入既有 pipeline/artifact identity。
- shape bounds、符号来源和有序 extent ABI 仍进入既有 unit/invocation/plan 身份；测试确认更改长度上界会改变 unit 合同。无需修改公开 kernel 参数角色或模块 ABI 版本。
- 使用一个公共 `ProfileContext`/`ActivationScope` 覆盖准备、编译和运行。多个独立 context 写同一目录会在析构时互相覆盖，因此驱动显式复用已有上下文。
- 成功 run 及其事件附带 `stage`、`batch`、`query_length`、`total_length`、实际 `plan_abi`。观测字段只用于关联，不参与 ABI 校验。

## 验证与效果

生产测试：[bounded_attention_llvm_test.cpp](../../test/bounded_attention_llvm_test.cpp)。独立参考使用 double 累加与稳定 Softmax，不调用 KXC 的 TE 或 LLVM 实现。

| B | Q | T | H / D | 验证行为 |
|---:|---:|---:|---|---|
| 1 | 1 | 3 | 4 / 96 | 单 token query，对已有 KV 做 attention |
| 2 | 3 | 5 | 4 / 96 | 多 batch、非方阵、多 query |
| 3 | 6 | 6 | 4 / 96 | Q/T 数值相等时的方阵 |
| 1 | 7 | 8 | 4 / 96 | 非整齐长度和上界 T |

上述四次运行共 **20 次 LLVM kernel 调用**，实际输出形状均为 `[B,4,Q,96]`，最大绝对误差 **5.43662e-7**；所有 primitive cache 计数在运行前后完全相同。

另一个 Softmax 测试在中间轴 `axis=1` 归约，执行 `[1,1,3]`、`[2,3,3]`、`[3,8,3]`，输入含 `±10000`。结果均有限，最大绝对误差 **1.0386e-8**，无重新编译。

九组运行时负例覆盖 V 的 T 不一致、mask 的 Q 不一致、B/Q 越界、T=0、固定 D 错误、rank 错误、dtype 错误和缺输入。profile 证明这些调用没有增加 kernel submit 或 runtime allocation，cache 统计不变。准备/lowering 负例另覆盖重复 transpose 轴、空 Softmax 归约域、未证明的 MatMul batch 广播、不同 K 符号、缺少 scratch 上界和 scratch 字节溢出。

attention bundle 包含 **4 个成功 run、9 个失败 run、20 对 kernel submit/exec**。四个成功 run 关联的 **120 个事件**均有 shape 和 ABI 关联字段。文件在 CTest 工作目录的 `out/bounded_attention_profile/` 下。

复现命令（要求 bounded、dynamic ABI、restricted/exact shape gates 和 LLVM）：

```bash
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
```

本轮实际执行结果：

| 检查 | 结果 |
|---|---|
| 默认 CPU/LLVM 全量 CTest | **48/48** |
| bounded + dynamic ABI + control gate-on 全量 CTest | **53/53**，含新增 attention 目标 |
| Python `pytest -q test/ python/` | **233/233** |
| Relay/Pass 生成物 freshness、契约、include layers、public headers | 通过；公共检查包含在两套全量 CTest 中 |
| NLP 矩阵检查及删除报告、错误提升 CUDA/请求合批的反例 | 通过；反例均被拒绝 |
| 文档索引/本地链接、`git diff --check` | 通过 |
| bounded 静态库重复 strong symbol 审计 | 输出为空 |

清理 serial 校验器中已无调用者的“禁用归约”参数后，重新构建两套产物并复查受影响的调度、LLVM 数值、bounded attention/shape/KV 测试。MiniMind 外部大模型 fixture 本轮未重新执行；本报告不把 CTest 的环境变量跳过路径记为新的模型证据。

## 三轮设计复核

1. **复用检查**：复用已注册 MatMul/Transpose/Softmax、ShapeProgram、extent ABI、TE reduction 和 LLVM Allocate；未添加算子、Pass、IR 类型或 evaluator。
2. **权威与身份检查**：符号关系来自解析器；上界只从同一个 invocation 合同投影；policy/applicability 有版本变化。相等样例不再充当形状关系证明。
3. **消费者与失败检查**：通过普通编译和 RuntimeSession 跑真实 kernel，覆盖非方阵、非末轴归约、shape/dtype 负例及临时存储溢出。原有静态、形状值和 stateful KV 路径纳入回归。

后续 [加权投影报告](M3_WEIGHTED_PROJECTION_REPORT.md) 已补齐静态权重、相同表达式或常量 1 的广播、固定归约轴 RMSNorm 和实际 Q/K/V 投影子图；MatMul 两侧 rank 可不同但须至少为 2。下列整图和状态任务仍未完成。

## 仍需完成的部分

- 真实 MiniMind 变长 ONNX 图的导入与全部算子传播，包括 GQA、RoPE、Slice/Concat 等；本次没有宣称模型整体动态化。
- 同一输入内部重复同一个符号，以及 `total = past + current` 的表达式传播。共享 Call DAG 与明确的 tuple 输出已由后续 [图结构报告](M3_GRAPH_STRUCTURE_REPORT.md) 补齐；该后续切片将 applicability 升至 v4。
- 通用 bounded attention 与 M2 状态的同计划交接。当前测试输入为紧凑有效 K/V，不能把它直接等同于容量 buffer 的 valid extent。
- 全 mask 行的专门语义、无法静态证明的动态广播和 CUDA 归约；CPU 通过不提升这些格子。
- 请求级动态批处理、跨变体热替换和性能优化。这里仅证明合法 shape 在同一已编译变体内变化；没有进行速度提升或自动调优声明。
