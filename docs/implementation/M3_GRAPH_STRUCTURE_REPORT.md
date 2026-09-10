# M3 技术报告：有界图保留共享计算与多个输出

日期：2026-09-08。范围：受限 shape 适配层的共享 Call DAG、重复实参、明确的 tuple 结果，以及真实 MiniMind 动态图结构审计。

## 结果与用途

一份普通 CPU/LLVM 有界产物现在可以在不同输入长度下执行共享计算，并返回函数明确指定的多个结果。结果可以包含仍被后续计算使用的中间张量，也可以包含输入张量；嵌套 tuple 按既有执行器规则展开为有序张量列表。

测试图在长度 **1、3、7、8** 下各执行四个 kernel，返回四个结果，所选输入的数值与独立参考完全一致。共享生产者只执行一次；`mul(shared, shared)` 保留两个逻辑参数，物理 ABI 只传一份输入。七类非法输入在 runtime 分配和启动前被拒绝，运行期间 primitive cache 的全部统计不变。

这解决了真实 MiniMind 动态图的一类结构障碍。完整模型的动态 ONNX 导入、GQA/RoPE 和状态联合尚未接通；权重常量与第一层 RMSNorm/QKV 后续已由[加权投影报告](M3_WEIGHTED_PROJECTION_REPORT.md)补齐。本报告不把结构测试等同于整模型执行。

## 真实导出发现了什么

本次从本地 MiniMind 源码重新导出了完整八层模型，使用已有导出器和工作区中的非原地 mask 修改。配置为 hidden=768、attention heads=8、KV heads=4、head_dim=96、vocab=6400、seed=0、opset=17。RoPE 最大位置本次取 **128**；这是新的动态图结构审计样本，不替代先前 E0 或静态模型数值 receipt。

| 图 | 原始节点 | 纯静态折叠后节点 | 被至少两个不同节点消费的计算值 | 输出数 | 仍被图内消费的输出 |
|---|---:|---:|---:|---:|---:|
| prefill | 2074 | 1338 | 220 | 17 | 16 |
| decode | 2184 | 1423 | 229 | 17 | 16 |

最后一列全部是 `present_k_*` / `present_v_*`。例如某层 K 张量既需要供本层 attention 使用，也需要作为模型结果返回。按“未被任何节点消费”寻找输出会漏掉这些结果，按树处理调用也不能表达其共享关系。

审计复用现有 `fold_static_subgraph`，只统计剩余 ONNX 图和输入/输出签名，没有执行普通模型数据计算。实际调用静态 importer 后，两份动态模型仍在 `input_ids` 的 `batch` 轴报 unresolved dimension；没有放宽静态导入 API。

生成文件位于 `out/minimind_onnx_bounded/`：`export_metadata.json`、两份 ONNX 和 `graph_structure_audit.json`。源码基准为 `6fc918beb68a0d8c40452338df6319fe168014ba`，同时使用上述本地 mask 修改；源码 commit 本身不能替代导出文件身份。

| 文件 | SHA-256 |
|---|---|
| `minimind_prefill.onnx` | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| `minimind_decode.onnx` | `c0733d61304970a37fff14fec8d56336baf945576398a0f5e0d14d8acd1d5fb2` |

导出命令：

```bash
out/venv/bin/python python/tools/export_minimind_onnx.py \
  --src out/minimind --out out/minimind_onnx_bounded --max-pos 128
```

## 使用的方法

### 保留原图，不从物理边界重建逻辑调用

既有 [ValueGraph](../../src/compiler/graph/value_graph.cc) 已区分 `argument_value_ids` 和去重后的输入边界；[RelaySnapshotCloner](../../src/compiler/analysis/relay_snapshot.cc) 已按节点身份记忆化并保留共享。受限适配层此前却只接受调用树，并试图从去重的 unit 输入重建 Call。对 `mul(x,x)`，两个逻辑参数会被误认为只有一个。

本次删除这段重建逻辑，复用 `CloneRelaySnapshot`，只在私有快照上根据 ShapeProgram 求值结果重新标注参数类型，再由既有 InferType 重新推导所有结果。bounded 和 exact 两条物化路径共用这一方法，Call 的参数次序、重复引用、共享节点、attrs 和 tuple 结构由原始快照持有。

按表达式身份记忆化、按 tuple 字段顺序重写的做法也见于 [TVM v0.14 ExprMutator](https://github.com/apache/tvm/blob/v0.14.0/src/relay/ir/expr_functor.cc)。本次复用 KXC 已有实现，不引入新的 IR 或通用 mutator。

### 让 ValueGraph 独占值编号

[形状解析器](../../src/compiler/shape/shape_value_resolver.cc) 的第二阶段现在对每个共享调用只收集一次证明。它返回与有序调用平行的输出维度，不再自己推算 `value.<id>`。

准备阶段从既有 frozen unit 读取输出值名，再把对应的维度证明挂到 ShapeProgram。这样 tuple 本身不占一个张量编号，共享节点也不会重复占位。后续 bounded preparation 仍校验代表图、逻辑图、unit 接线、形状和 dtype 一致。

### 明确验证结果顺序和语义身份

删除了“图输出等于无消费者节点”的推测。物化函数保留明确的结果结构，exact 产物验证使用既有 `FlattenLogicalTensorTypes` 检查每个结果的 shape/dtype，并继续绑定整个物化函数的 semantic key。

仅比较边界不足以发现同形状结果的调换，因此测试在相同四个算子、相同输出形状下交换两个结果，验证两份错误产物均被拒绝。重复输出同一个张量、空 tuple 和把 tuple 传给张量算子的行为仍在准备阶段拒绝，与当前 ExecutablePlan 边界一致。

bounded applicability 从 **v3 升到 v4**，沿既有 identity builder 进入编译与计划身份。kernel ABI、调度 policy、RuntimeSession 和存储分配规则均复用既有实现。

## 验证与效果

生产测试：[bounded_graph_structure_llvm_test.cpp](../../test/bounded_graph_structure_llvm_test.cpp)，已注册为只在完整 bounded/LLVM 门禁下构建的 CTest。

| 用例 | 实际执行 | 结果 |
|---|---|---|
| 共享数据 DAG | `add → mul(shared,shared) → sqrt`，另有共享输入的 transpose；返回 root/shared/输入/transpose | N=1/3/7/8，四个结果顺序和形状正确，共 16 次 LLVM 调用 |
| 输出生命周期 | 保留第一轮所有结果，继续运行另外三种长度 | 第一轮结果仍正确，中间输出未被后续调用覆盖 |
| exact 物化 | 显式编译 N=2 和 N=6 两个 profile | 两者运行正确，通过 producer 验证；同形状结果调换的两份产物被拒绝 |
| 数据与形状值共享 | `relu` 同时供 `Shape→Gather→ReshapeDynamic` 和结果使用 | N=1/5/8；折叠为 shape_expr 的目标、float32 数据和实际 int64 shape 结果均正确 |
| 运行时负例 | N=0、N=9、两输入 N 不同、固定轴错误、rank 错误、dtype 错误、缺输入 | 七次均不新增 runtime allocation 或 kernel submit |
| 准备阶段负例 | 重复结果、空嵌套 tuple、tuple 作为张量实参 | 全部拒绝，cache 统计不变 |

数值参考直接计算加法、绝对值、转置及 ReLU，不调用 KXC TE。输入取能精确表示的二进制分数，因此这些用例要求逐元素完全相等；这不代表任意浮点归约都没有舍入误差。

主 DAG 的 bundle 为 `out/build/bounded-llvm/out/bounded_graph_structure_profile/`。实际读取的记录包含 **4 个成功 run、7 个失败 run、16 对 submit/exec、32 个 runtime allocation**。属于成功运行的 **68 个事件**全部带有 stage、length 和实际 plan ABI；七次失败没有增加提交或分配。

复现命令：

```bash
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error \
  -R 'bounded_graph_structure_llvm_test|restricted_symbolic_shape_test|shape_value_llvm_test|bounded_attention_llvm_test'
PYTHONPATH=python out/venv/bin/python -m pytest -q test/ python/
```

本轮完整回归：

| 检查 | 结果 |
|---|---|
| 默认 CPU/LLVM 全量 CTest | **48/48** |
| bounded + dynamic ABI + control gate-on 全量 CTest | **54/54**，含本模块新目标 |
| Python | **233/233**；命令需设置 `PYTHONPATH=python`，未设置时会在收集阶段找不到本地包 |
| Relay/Pass 生成物 freshness、契约、include layers、public headers | 全部通过 |
| NLP capability checker | 通过；本模块没有提升尚未重新审计的模型能力格子 |
| 文档索引、本地链接、`git diff --check` | 通过，文档检查覆盖 45 篇 Markdown |
| bounded 静态库重复 strong symbol 审计 | 输出为空 |

实际 MiniMind 大模型数值 fixture 本轮未重新运行，动态导出只作为结构审计证据。

## 三轮设计复核

1. **复用**：找到已有 ValueGraph、Relay 快照、InferType 和 tuple flatten，删除适配层的重复 Call 重建与值编号逻辑；没有新增算子、Pass、IR 或 evaluator。
2. **权威和身份**：形状仍来自 ShapeProgram，接线与结果来自 Relay 快照，值编号来自 ValueGraph，物化产物以既有 semantic key 绑定；applicability 明确版本变化。
3. **消费者和失败**：真实 LLVM 执行同时覆盖 bounded/exact、共享数据/形状值、被消费的输出和保留结果的生命周期；有同形状错序产物及零启动反例。

后续 [加权投影报告](M3_WEIGHTED_PROJECTION_REPORT.md) 已补齐静态数据常量、可证明广播和固定归约轴 RMSNorm，并执行实际 MiniMind 第一层子图；applicability 升至 v5。

## 下一步

数据常量和可证明的广播/normalization 已由后续切片接通。继续处理真实导出 shape 链、GQA/RoPE、`total=past+current` 与 M2 的有效长度交接。当前仍不支持数据相关任意 shape、未知 rank、tuple 型算子实参、控制流及请求级合批。
