# M8 技术报告：静态 TE Program 与首个跨算子融合

M8 已将既有 TE Compute DAG 与调度冻结成不可变 `te::Program`，并接到生产 lowering、primitive cache 和 LLVM 执行。现有 `CompileConfig` 的 **优化等级 3** 在 CPU:0 上选择已证明的静态 `sqrt(add(x,y))` region；默认等级 2 保留独立单元。六组真实执行中，kernel 提交及完成数均由 **2 降到 1**，运行时 value 数由 **4 降到 3**，输出与未融合参考逐位一致。

本模块落实的是已经存在的 [TE Program 方案](../TE_PROGRAM_IR.md)与 [M8 计划](M8_TE_PROGRAM.md)，属于项目目标中的编译实现与性能切片。新 agent IR、parser 和 Python 编译入口仍在原延后范围。

## 采用的方法

**冻结完整候选。** [Program](../../include/kxc/te/program.h) 保存有序输入、常量和输出边界、元数据专用输入、TE producer/consumer DAG、表达式 dtype/索引/归约、stage 轴域与顺序、split relation 和执行标注，以及显式 Target 的稳定字节与规范 TIR pipeline。边界采用连续张量、只读输入/常量和 fresh output；常量 payload 继续由既有 compiler/module 持有，Program 记录其 ABI 顺序与类型。

[捕获实现](../../src/te/program.cc) 深拷贝 DAG，通过既有 `Stage::split/reorder/vectorize/unroll/parallel` 重放调度，并核对原候选的轴关系、域、变量唯一性与拓扑。它拒绝自由变量、循环、未知表达式、非整数索引、非规范 stage/metadata 顺序、重复边界、未声明依赖和不支持的 TE attrs。现有静态整数求值器移至 TIR 的私有实现，Program 与 lowering 共用其溢出检查；静态域仍受既有 int32 迭代边界约束，表达式嵌套上限为 256。

Program 公共接口不返回可修改的 TE 句柄。lowering 再取得独立工作副本，防止返回的可修改 TIR 泄露候选内部状态。测试修改原 TE body、shape、schedule 以及已返回 TIR 的 buffer shape，再次 lowering 仍得到原结果。审查发现现有 TIR printer 的 map 顺序与对象地址有关；现在 buffer 描述和 attrs 名称稳定排序，使诊断文本和 IR hash 可复现。

**继续使用现有编译 owner。** [lowered_graph](../../src/compiler/lowering/lowered_graph.cc) 从已解析 Relay callback 构造 TE，选择既有目标调度，再构造 Program。[LowerProgramToTIR](../../src/compiler/lowering/te_to_tir.cc) 核对 Target、pipeline 和常量 ABI，调用原 `LowerTensorGraphToTIR`；轴物化、tail predicate、分配和后端证明仍由该路径完成。普通静态单 Call 也经过 Program。受限 bounded/stateful lowering 保留其显式 extent 合同，Program v1 不承载动态 extent。

`shape_of/shape_expr/constant_of_shape`、受限 reshape/expand 的控制输入和 Slice shape anchor 按既有语义明确标注为 metadata-only。它们保留物理 ABI，不被误当成缺失的 payload 依赖；形状来源证明继续由现有 shape resolver 拥有。

**一种候选身份进入缓存。** 静态路径的 `kxc.te.schedule_contract` 槽现在保存完整 `kxc.te.program.v1` 字节，替代原 schedule-only 身份。`PrimitiveArtifactKey` 升至 **v4**，候选字段为 `te_candidate`；compiler execution contract 升至 **v5**，包含 Program lowering 与选定 partition 策略。已有 bounded/stateful schedule schema 仍通过同一候选槽消费。Kernel callable ABI 未因此升级。

规范编码以 boundary、producer、stage 和 axis 序号表示关系，忽略 graph-local 名称、对象地址、value/storage id 和链接符号。完整字节决定等价性，digest 只用于索引。真正的 body、shape、schedule、Target 或 pipeline 改变会改变候选身份；输入长度变化不能误命中同一静态产物。

**先证明 region，再合并边界。** [PrimitiveUnit](../../src/compiler/analysis/primitive_unit.cc) 最小扩展为一个 root Call 加一个内部 producer；[partition](../../src/compiler/graph/partition.cc) 只合并相邻的纯、确定、无 alias、同设备、相同静态 float32/float64 类型的 `add → sqrt`。add 结果必须只有该消费者，且不是图输出。共享、可观察、广播、非 add producer 等图继续形成独立单元；非法显式候选会报错。

两个数学定义仍分别由原 Relay lowering callback 产生。ValueGraph 保留源图事实，ExecutablePlan 去掉 region 内部 value，并保持一个 unit 对应一个 artifact 和 KernelCall。profile 的算子身份包含两者的名称和 schema。exact-profile 的模板与核验复用同一静态计划 owner，排除内部 value，并使用生产规范 pipeline 核对 artifact；普通与 exact 编译的融合计划 ABI 已实测一致。

## 达到的效果

以下每组都通过 `Compiler::Compile → CompiledModule → RuntimeSession` 运行，而非仅检查生成文本。基准为优化等级 2 的两 kernel 计划，融合为等级 3。两者外部输入/输出类型及顺序一致，内部计划结构有所变化。

| dtype | shape | 实际 kernel 完成数 | 运行时 value 数 | 数值结果 |
|---|---|---|---|---|
| float32 | `[10]` | 2 → 1 | 4 → 3 | 逐位相等，并与标量 sqrt 参考对齐 |
| float32 | `[2,5]` | 2 → 1 | 4 → 3 | 同上 |
| float32 | `[0]` | 2 → 1 | 4 → 3 | 两者均为空输出 |
| float32 | `[]` | 2 → 1 | 4 → 3 | 标量结果逐位相等 |
| float32 | `[1]` | 2 → 1 | 4 → 3 | 逐位相等 |
| float64 | `[10]` | 2 → 1 | 4 → 3 | 逐位相等，并与标量 sqrt 参考对齐 |

[te_program_test.cpp](../../test/te_program_test.cpp) 还证明：

- 相同候选首次编译产生 1 次 miss，重命名后的等价图产生 1 次 hit；长度改为 11 产生新 miss。
- 改变 graph-local value/constant id 后仍复用同一 fused artifact；不同常量 payload 正确绑定，清空缓存及释放编译图后仍得到各自结果。
- 重复实参 `sqrt(add(x,x))` 保留数学上的两次使用，只需要一个输入 ABI 槽，并通过真实 LLVM 数值验证。
- 同步执行与清空缓存后的异步执行都保持 artifact 生命周期；运行期 cache 统计不变。
- 输入 arity/shape 错误在任何 kernel 提交前拒绝。
- split tail 谓词、确定性 TIR、非法轴关系、目标/pipeline 不匹配和外部共享边界均有独立验证。

每组数值用例通过现有 M1 adapter 写出 Profile Bundle，测试用 LLVM JSON reader 核验 `kernel_submit`、完成的 `kernel_exec` 和 `runtime_session_run.submit_count`，包括错误输入的零执行证据。默认 CTest 的产物位于 `out/build/dev-ninja-cpu/out/te_program/<dtype>_n<元素数>_rank<秩>/events.jsonl`。

## 回归与复现

最终核对日期：2026-09-09。三套构建均完成重编译与全量回归；模型 fixture 使用显式绝对路径，未以跳过替代模型执行。

| 检查 | 结果 |
|---|---|
| 默认 CPU/LLVM 全量 | 50/50，44.59 秒 |
| adaptive gate 全量及真实 MiniMind fixture | 50/50，116.45 秒；真实模型测试执行成功 |
| bounded gate 全量及八组显式 fixture | 64/64，262.69 秒；实际执行完整八层 prefill/decode 与 KV/greedy fixture |
| 新 Program、exact-profile 与形状值定向检查 | 3/3；普通和 exact 编译的融合 artifact/计划 ABI 相同，LLVM 结果一致 |
| Relay / Pass 合同及生成文件 freshness | 37/37、20/20；freshness 通过 |
| 公共头文件与 include 方向 | 90 个安装头、10 个实验头均可独立编译；284 个文件的 include 检查通过 |
| 文档、差分和强全局符号 | 57 篇文档通过；差分无空白错误；三个 archive 均无重复强 C++ 定义 |

adaptive 的模型回归实际运行八层 MiniMind prefill：替换前后各 650 次 LLVM kernel，替换 unit 649，17 个输出逐位相等，对 ONNX 参考最大绝对误差 `8.82149e-06`。本次候选身份升级后，已有 route/plan ABI 仍为 `828297404` 字节。该测试核验 M6 兼容性，未对 MiniMind 启用本次融合。

M8 的最短复现命令为：

```bash
cmake --build out/build/dev-ninja-cpu -j2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'te_program_test|te_schedule_test|graph_partition_test|op_numeric_llvm_test'
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
```

完整模型回归须显式提供既有 fixture。adaptive 使用 `KXC_MINIMIND_ADAPTIVE_PREFILL_DIR=out/fx_decode_stateful` 的绝对路径；bounded 同时设置 `KXC_MINIMIND_PROJECTION_DIR`、`KXC_MINIMIND_HEADS_DIR`、`KXC_MINIMIND_ROPE_DIR`、`KXC_MINIMIND_GQA_DIR`、`KXC_MINIMIND_ATTENTION_DIR`、`KXC_MINIMIND_BOUNDED_PREFILL_DIR`、`KXC_MINIMIND_BOUNDED_DECODE_DIR` 与 `KXC_MINIMIND_DECODE_LOOP_DIR`。对应导出/复现入口见 [M6 报告](M6_RUNTIME_REPORT.md)、[prefill 报告](M3_FULL_PREFILL_REPORT.md)、[decode 报告](M3_FULL_DECODE_REPORT.md)和 [KV state 报告](M2_BOUNDED_STATE_REPORT.md)。未设置 fixture 的跳过不能作为模型证据。

三轮 QA 分别检查了：①快照隔离、规范身份与确定性 TIR；②非法调度/依赖、region 外部消费者、常量重绑定及寿命；③默认与两套 gate 的完整生产路径，并补齐 exact-profile 和 metadata-only 控制输入的接入。

## 当前边界

这次证明的是运行时 kernel 边界减少。原 TE→TIR 仍生成内部 `Allocate` 和各 stage 循环；没有证明消除了中间分配，也没有给出吞吐或延迟加速比。此前 MiniMind profile 中的 650 次调用包含 73 次 add、33 次 sqrt，但其中能否组成值得优化的 region，仍需逐图证明和测量。

任意多 Call、广播/归约融合、动态形状、KV state、控制流 region、CUDA 融合/并行、autotuning 和模型级 attention/FFN 优化均不在本次结果中。普通单 Call 的既有归约和静态算子继续通过生产回归。内部计划 ABI 随融合变化，本报告没有证明把旧双 kernel 计划直接热替换成单 kernel 计划。
