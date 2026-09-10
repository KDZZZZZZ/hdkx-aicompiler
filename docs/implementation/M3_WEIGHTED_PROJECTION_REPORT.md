# M3 技术报告：常量权重与变长 RMSNorm/QKV 投影

日期：2026-09-08。范围：普通 bounded CPU/LLVM 编译中的静态权重、可证明广播、固定归约轴 RMSNorm，以及真实 MiniMind 第一层投影子图。

后续补记：下一模块修复了 ONNX 标量被误提升为 `[1]` 的问题，因此用当前生成器重建 fixture 时 SHA 会改变。本报告保留当时的 receipt；修复后的独立目录、SHA、投影回归与新增拆头证据见 [M3/M4 拆头报告](M3_ONNX_HEADS_REPORT.md)。

## 结果与用途

一份编译产物现在可以接收不同 batch 和序列长度的激活，使用冻结权重完成 RMSNorm 和 Q/K/V 投影。测试从实际 MiniMind 动态 ONNX 导出中抽取第一层子图，保留原来的 **11 个调用、7 个常量**，在四组输入上执行 **44 次 LLVM kernel 调用**，与 ONNX ReferenceEvaluator 的最大绝对误差为 **2.74181e-6**。运行期间 primitive cache 的全部统计不变。

此前 bounded 入口拒绝数据常量和广播，无法接入投影矩阵、RMSNorm 权重或标量 epsilon。现在 `[B,S,768]` 可与 `[768,768]` / `[768,384]` 的静态权重配合，B 和 S 由同一有界合同管理。这里的 B 可变表示张量 batch 维可变，没有实现请求排队或连续合批。

这是一段真实模型子图的计算证据。输入是随机生成的 embedding 后激活，未执行 embedding、RoPE、GQA、完整 attention 或全八层模型；也没有把 fresh-output plan 与 KV 状态计划合并。

## 使用的方法

### 复用静态常量池和现有算子

常量继续由 Relay `Constant`、`CloneRelaySnapshot`、ValueGraph 和 CompiledModule 管理。ShapeProgram 保留其静态维度，只有函数参数可以接受 B/S overlay；运行时调用方只传激活，模块按 KernelSignature 注入常量。

RMSNorm 沿原有 Cast/Pow/ReduceMean/Add/Sqrt/Div/Mul 的 TE callbacks 编译，Q/K/V 沿原有 MatMul 编译。没有增加 RMSNorm 专用执行器、权重注册表、新算子或新 IR。

合成测试另保留一个只处理常量权重的 Transpose。它使用既有静态默认调度，并与有动态 extent 的调用共存于同一计划；每次 Run 确实执行这次静态调用，没有暗中编译或假称已经将它折叠。没有 runtime extent 的 unit 合同因此成为合法形式。

### 在编译准备阶段证明广播和归约

广播沿用 [ONNX 多向广播](https://onnx.ai/onnx/repo-docs/Broadcasting.html) 的右对齐原则，但只接受能够静态证明的关系：两个维度表达式完全相同，或其中一个是常量 1。缺少的前导轴按 1 处理。独立符号即使在代表样例中数值相等，也不能据此通过。

[MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) 接受两侧 rank 至少为 2，batch 前缀按上述规则广播，K 归约轴必须为同一个表达式。没有开放向量 MatMul 或需运行时择一判断的任意广播。

ReduceMean 只开放**静态、正的归约轴长度**，axes 规范化后必须唯一，keepdims 为 0 或 1，归约元素数检查 int64 溢出。现有 TE 的分母是常量，因此 B/S 可以位于保留轴，不能位于变化的归约轴。RMSNorm 的 hidden=768 固定，正好落在这个范围内；动态 Softmax 继续使用前一模块已经验证的实现。

### 让代表图和有界图使用相同的生产流水线

加入常量后发现了原准备链中的值编号错位：ANF 会先提升嵌套调用，再在消费者处引用原子的常量。仅对原始 DAG 做 InferType，其 ValueGraph 编号可能与已经经过 ANF 的静态模板不同。典型触发为 `Mul(constant_weight, normalized)`，两边的 `value.1` 分别可能代表常量和更早的计算结果。

bounded preparation 现在通过既有 `PrepareRelayProgram`，用 CompileConfig 声明的同一 Relay pipeline 处理代表图和逻辑边界图，再建立并比较 ValueGraph。Pass 顺序、Target 和验证仍来自现有 resolver/executor；没有插入未登记的规范化 Pass。适配层仍在私有 Relay 快照上重注参数类型，原始实参顺序、共享和结果顺序不变。

身份沿既有 builder 更新：bounded applicability **v4→v5**、preparation **v2→v3**、DynamicUnitShapeContract **v3→v4**。这些版本进入 bounded pipeline/artifact identity；公开 kernel ABI 和模块常量接口无需改动。

## 真实模型证据如何生成

生成器 [make_minimind_projection_fixture.py](../../python/tools/make_minimind_projection_fixture.py) 从实际导出中找到第一层 input_layernorm 和 q/k/v_proj，按依赖反向截取；常量复用已有 `fold_static_subgraph` 的结果。它检查算子计数和权重形状，导出结构漂移就报错，不手写一份外观相似的 ONNX 图。

源模型是八层 MiniMind，hidden=768、heads=8、KV heads=4、head_dim=96、vocab=6400、模型 seed=0、max_pos=128、opset=17。源码基准及本地 mask 修改见 [动态图结构报告](M3_GRAPH_STRUCTURE_REPORT.md)。

| 产物 | SHA-256 |
|---|---|
| 实际完整 `minimind_prefill.onnx` | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| 抽取的动态 `projection_dynamic.onnx` | `fb1fb422c906ab7065c7a2685b1ef4f76fc0bfdc0ec0cc3115f34461bfb893b9` |
| 显式代表 `projection_representative.onnx` | `65138364ca99ff995e42629479bbd5bc35adab3050aed1407c60f6d6857f29ee` |

静态 Python importer 接收 `[1,4,768]` 的明确代表；C++ restricted adapter 再给参数绑定 B∈[1,3]、S∈[1,8]。参考结果由**动态子图**上的 ONNX ReferenceEvaluator 按每个实际输入 shape 生成。此过程没有放宽静态 importer 的 unresolved-dimension 拒绝规则，也没有新增 Python 编译 API。

fixture 位于 `out/fx_minimind_projection/`，`receipt.json` 保存源节点名、输入/输出顺序、版本及所有产物 SHA；输入 RNG seed=20260908，ONNX=1.22.0，NumPy=2.5.3。

| B | S | 输入 | 输出 Q / K / V |
|---:|---:|---|---|
| 1 | 1 | `[1,1,768]` | `[1,1,768]` / 两个 `[1,1,384]` |
| 1 | 4 | `[1,4,768]` | `[1,4,768]` / 两个 `[1,4,384]` |
| 2 | 3 | `[2,3,768]` | `[2,3,768]` / 两个 `[2,3,384]` |
| 3 | 8 | `[3,8,768]` | `[3,8,768]` / 两个 `[3,8,384]` |

## 验证与效果

生产测试为 [bounded_projection_llvm_test.cpp](../../test/bounded_projection_llvm_test.cpp)，只在完整 bounded/LLVM 门禁下注册。真实模型部分需要显式 fixture 环境变量；缺失时会打印 SKIP，不能将这一分支当成真实模型通过。

| 验证 | 结果 |
|---|---|
| 合成加权投影 | hidden=8，四组 B/S，12 个调用含静态权重 Transpose，共 48 次 LLVM 调用；double RMSNorm/MatMul 参考最坏误差 **1.90407e-7**，阈值 1e-5 |
| 实际 MiniMind 子图 | 四组 B/S，11 个调用，共 44 次 LLVM 调用；ONNX 参考最坏误差 **2.74181e-6**，阈值 1e-4 |
| 常量生命周期 | Prepare 后将调用方权重清零、再将 module.constants() 返回的副本清零；实际结果仍与原权重一致。用新权重重新 Prepare 会改变图身份 |
| exact 分支 | 显式物化并编译 B=2/S=5，通过 producer 验证且数值正确 |
| 运行时反例 | B/S 下界和上界、固定 hidden、rank、dtype、额外权重参数，共八次拒绝；无新 runtime allocation 或 kernel submit |
| 准备反例 | 无法证明的广播、不等 K、动态 ReduceMean 归约轴、重复 axes、非法 keepdims、直接返回常量；全部拒绝且 cache 不变 |
| 运行时编译检查 | 成功和失败 Run 均比较全部 primitive cache 统计，包括命中、未命中、条目、字节、淘汰、失败及 pins；均不变 |

profile bundle 在 `out/build/bounded-llvm/out/bounded_projection_profile/`。实际读取结果如下：

| 运行组 | 成功 run | submit/exec 对数 | runtime alloc | 带关联字段的事件 |
|---|---:|---:|---:|---:|
| 合成有界投影 | 4 | 48 | 136 | 236 |
| 实际 MiniMind 子图 | 4 | 44 | 132 | 224 |

实际子图的 224 个事件全部带动态导出 SHA、stage、B/S 和 builder 生成的 plan ABI；合成事件带 stage、B/S、ABI。bundle 另含一次 exact 成功运行和八次非法输入失败记录。观测字段只关联证据，不替代形状或 ABI 校验；runtime alloc 包含输出和 extent 参数存储，kernel 内部的 scoped malloc 仍不计入这些事件。

normalization 能力矩阵同时审计静态 `nn_layer_norm` 和本次 RMSNorm 组合链：静态 LayerNorm 的极大数、微小方差、多轴和空前缀用例由现有 `op_numeric_llvm_test` 执行。静态 LayerNorm 使用 float64 内部累加，本次实际 RMSNorm 保留模型导出的 float32 运算；两者的数值范围不能互相代替。CUDA 格子保持关闭，profile 证据仅覆盖明确记录的 RMSNorm/QKV 运行。

复现：

```bash
out/venv/bin/python python/tools/make_minimind_projection_fixture.py \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx \
  --out out/fx_minimind_projection
cmake --build out/build/bounded-llvm -j2
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure \
  --no-tests=error -R '^bounded_projection_llvm_test$'
```

本模块完整回归结果：

| 检查 | 结果 |
|---|---|
| 默认 CPU/LLVM 全量 CTest | **48/48** |
| bounded/dynamic ABI/restricted/exact/control gate-on 全量 CTest | **55/55**，本轮显式提供真实投影 fixture，未跳过该模型子图 |
| Python：`PYTHONPATH=python out/venv/bin/python -m pytest -q test/ python/` | **233/233** |
| Relay/Pass 生成物 freshness、契约、include layers、public headers | 全部通过 |
| NLP capability checker | 通过；另确认删掉 normalization 报告证据、提升 CUDA、宣称请求合批的三个反例均拒绝 |
| 文档索引、本地链接、`git diff --check` | 通过；覆盖 46 篇 Markdown |
| bounded 静态库重复 strong symbol 审计 | 输出为空 |

本轮没有重新执行完整八层静态模型 fixture；它此前的证据仍见 [M2 模型报告](M2_MINIMIND_STATE_REPORT.md)。normalization 是本轮逐格刷新的一行，其他尚未审计的能力不会随全量测试通过而整体升级。

## 三轮设计复核

1. **复用**：保留现有常量池、Relay/TE 算子、ShapeProgram、生产 Relay pipeline 和 RuntimeSession。没有独立权重执行器、形状 evaluator 或新 IR。
2. **权威与身份**：参数 overlay 不修改常量；ValueGraph 只在相同规范化阶段比较；直接常量结果拒绝，避免暴露模块私有负载。版本进入既有编译身份。
3. **消费者与失败**：实际 LLVM 同时执行常量与动态数据路径，覆盖逻辑参数顺序不同于 ABI 顺序、静态/动态调用共存、权重不可变及零启动反例；外部 fixture 缺失明确区分 SKIP。

## 后续工作

继续接通实际动态导出的 shape 值链、GQA/RoPE、embedding 和 `total=past+current`，再与 M2 的状态/有效长度合同交接。动态 ReduceMean 分母、native bounded LayerNorm、完整模型变长、请求级合批、CUDA 数值和性能优化不由本子图结果自动获得。本次没有测速或速度提升声明。
