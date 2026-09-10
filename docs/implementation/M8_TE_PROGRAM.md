# M8：te::Program 与第一个跨算子融合

M8 已接通静态 TE Program 与首个跨算子融合候选，方法、数值、缓存及运行观测见 [技术报告](M8_TE_PROGRAM_REPORT.md)。MiniMind 的 attention、RMSNorm 和 SwiGLU 扩展继续以真实模型 profile 为依据。

本模块保留一个最小、可验证的性能切片：静态精确、同设备、纯逐元素 sqrt(add(x,y))。L1a 稳定后，再用 M1 bundle 判断是否值得扩展到 MiniMind attention/FFN。te::Program 不是新的 agent IR，也不支持动态形状、KV state 或自动候选搜索。

> 状态：T1–T7 已完成，默认/adaptive/bounded 三套完整门禁通过（2026-09-09）。CPU:0、优化等级 3 选择相邻静态同形 float32/float64 `add → sqrt`；默认等级 2 保留独立单元。当前安排见 [WAVE_2](WAVE_2.md)，完整边界见 [TE_PROGRAM_IR.md](../TE_PROGRAM_IR.md)。

## 必须保持的 owner

- Relay 继续定义数学语义、attrs、类型和边界。
- TE Compute DAG 提供 producer/consumer、索引和归约事实。
- `te::Program` 只描述一个已验证的算法/调度候选。
- `Target`、PassContext、primitive cache、ExecutablePlan 和 RuntimeSession 继续由现有模块拥有。
- CUDA thread/launch 继续由 `tir::BindCudaThreads` 拥有；本切片不扩大 CUDA 范围。

不能新建 fusion registry、Math registry、图级执行计划或第二套 identity。

## T1–T7 单一纵向切片

1. **T1 数据合同**：在 `include/kxc/te/` 定义最小不可变 Program，包含有序输入/常量/输出边界、DAG、stage 循环事实、目标快照和 canonical bytes。构造非法边界、重复顺序和缺字段的负例。
2. **T2 DAG 捕获**：从已验证的两个相邻 PrimitiveUnit 读取 `add → sqrt` 事实，验证同设备、纯度、类型、无 alias/side effect，并冻结 region ABI。
3. **T3 确定 lowering**：Program 加显式 Target、规范 pipeline，调用现有 TE→TIR 入口。两次构造产生相同 canonical bytes、TIR 语义和 identity；不能依赖 map 顺序或对象地址。
4. **T4 cache 身份**：把完整 Program contract 纳入 `PrimitiveArtifactKey`；提升必要的 schema/ABI 版本。仅 graph-local 名称或临时 storage id 不得改变等价性；真实调度改变必须改变身份。
5. **T5 单 Call 回归**：让未融合单 Call 也经过 Program 路径，保持已有算子结果、计划 ABI 和 fail-closed 行为。没有这个回归不要直接启用 region。
6. **T6 region 单元**：扩展 `PrimitiveUnit`、partition、analysis 和 lowered_graph，只接受连通、静态、纯、同设备的 region；不符合条件的图保留原独立单元。
7. **T7 数值切片**：编译和运行 `sqrt(add(x,y))`，与未融合双 kernel 参考比较输出、shape、dtype、错误和 ABI。比较 primitive cache hit/miss 及 artifact 生命周期。

T1–T4 的字段必须立刻被 T5–T7 消费；禁止先合入一组没有 lowering consumer 的空类型。Program 只表示一个候选，不能在 RuntimeSession 中择优或重编译。

## 后端边界

首切片采用当前已验证的静态 CPU/LLVM 计算。LLVM 若未发现，完成 P0/P1 结构和负例，但不标记 P2/P3。现有 `GenFor`、allocate 和 CUDA 保守映射不因融合切片获得新的性能承诺。只有真实 TIR 结构、backend 编译和数值测试通过后，才更新对应后端格子。

## 代码落点

| 位置 | 工作 |
|---|---|
| [TE public headers](../../include/kxc/te)、`src/te/` | Program 不可变数据、canonicalization、P0 验证 |
| [PrimitiveUnit](../../src/compiler/internal/primitive_unit.h)、analysis | region 边界、纯度、类型、alias、设备和稳定顺序 |
| [partition](../../src/compiler/graph/partition.cc) | 选择已证明 region，保留独立单元 |
| [lowered_graph](../../src/compiler/lowering/lowered_graph.cc)、[te_to_tir](../../src/compiler/lowering/te_to_tir.cc) | DAG/Program 到 TIR 的确定 lower |
| [primitive compiler](../../src/compiler/primitive/primitive_compiler.cc)、identity | cache 请求、artifact key 和版本隔离 |
| `test/te_program_test.cpp`、`test/graph_partition_test.cpp`、`test/compiler_identity_test.cpp`、`test/op_numeric_llvm_test.cpp` | P0–P3、确定性、融合前后数值和拒绝路径 |

## 验证与完成条件

```bash
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'te_program|te_schedule_test|graph_partition_test|operator_compilation_test|compiler_identity_test|codegen_llvm_test|op_numeric_llvm_test|runtime_session_test'
```

实际测试目标名称以 `ctest -N` 为准；新增测试必须注册。随后执行公共契约、include、public header、docs 和 LLVM 检查。

- [x] 非法 Program、非规范顺序、无法证明的依赖和目标不匹配在 TIR/cache 前拒绝。
- [x] 融合前后 `sqrt(add(x,y))` 数值一致，输出 ABI 不变；内部 kernel 数量和边界有实际证据。
- [x] 单 Call 路径回归；cache 对 Program contract 正确隔离并保活 artifact。
- [x] 重复构造的 TIR/身份稳定，图编号、对象地址和链接符号不参与等价性。
- [x] 不声称已实现任意多 Call fusion、broadcast/reduction/dynamic/CUDA 并行或 autotuning。

M8 后续是否扩展到归约、布局或任务级调度，取决于 M1 的真实执行数据和实际性能问题。没有 profile 证据时不扩大融合范围。
