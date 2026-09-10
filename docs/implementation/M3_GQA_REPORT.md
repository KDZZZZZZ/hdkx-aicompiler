# M3/M4 技术报告：变长 GQA 与可证明的形状控制

日期：2026-09-08。范围：真实 MiniMind 第一层 K/V 的 grouped-query attention 准备链，以及 bounded CPU/LLVM 的 Squeeze/Unsqueeze。

## 结果与用途

一份编译产物现在可以把不同 B/S 的 K/V 从 `[B,S,4,96]` 变成 `[B,S,8,96]`。每个 head 连续重复两次，顺序为 `h0,h0,h1,h1,…`，供后续 attention 使用。实际 ONNX 子图在四组输入上执行 **40 次 LLVM kernel 调用**，与动态 ONNX ReferenceEvaluator 的结果逐元素完全一致，**最大绝对误差为 0**。运行期间 primitive cache 的所有统计不变。

这次完成的是实际 GQA 重复链。输入是按真实接口生成的外部 K/V 张量，未从 M2 的持久状态直接读取；RoPE、attention 分数/mask/Softmax、全八层变长模型和 state/valid extent 联合仍需接通。这里的 B 可变是张量 batch 维可变，不表示请求级动态合批。没有测速或速度提升声明。

## 使用的方法

### 保留真实导出，再证明控制链

实际导出没有直接把目标 shape 交给 Expand，而是保留了下列链：

```text
Shape(K/V) → Gather/Unsqueeze/Concat → [B,S,heads,2,head_dim]
  → Reshape([-1]) → Shape → ConstantOfShape(1) → Mul(-1)
  → Equal(target, marker) → Where(condition, ones, target)
  → Expand → Reshape([B,S,heads*2,head_dim])
```

前端沿上一模块的显式 `kxc.onnx_shape_source.v1` 保留这些节点。新增 `ConstantOfShape → constant_of_shape` 映射，复用已有 Relay op 和 attrs；source 模式的 Expand 保留两个输入，映射到既有 `expand_dynamic`。Add/Mul/Div 可以保留 int64 source 操作数，后续证明和 lowering 仍由 C++ producer 负责。

ONNX 官方 shape inference 不会把这段 Where 的控制值传播成 Expand 的具体长度。source importer 因此只核对这个中间结果已知的 rank/dtype，把尚未解析的 extent 留给 producer；图输入和图输出仍要求具体代表，最终输出数量/shape/dtype 仍在编译/cache 前校验。没有用代表数据在 Python 中运行 K/V 计算，也没有把 B/S 冻结为代表值。

### 在唯一形状 owner 内消解固定事实

restricted resolver 复用 ShapeProgram/DimExpr，新增的证明都发生在编译准备阶段：

- 形状向量的长度已知，因此该向量自身的 Shape 是常量。对已是一维的形状向量，`Reshape([-1])`、`Reshape([0])` 或 `Reshape([原长度])` 只在能证明恒等时消解；普通数据 reshape 的 `-1` 推断仍拒绝。
- ConstantOfShape 只在控制长度为静态已知、输出为至多 16 个元素的 int64 向量时形成常量；填充值必须是能由现有 double attrs 精确表示的整数，范围为 ±2^53，调用方不能填写目标 attrs 来跳过控制来源证明。
- 只有已经证明为整数常量的形状算术才会折叠。本次覆盖乘法、非负加法和非负数除以正整数；溢出由既有 DimExpr 算术检查。负号保留在整数常量载荷中，**不放宽维度必须非负的合同**；INT64_MIN 的幅值不在本子集内。
- Equal 只在表达式相同、常量可比较，或常量小于符号的已声明下界时得到固定结果。`B∈[1,3]` 时，`B == -1` 可证明为假，`B == 2` 不能根据代表 B=1 折为假，必须拒绝。
- Where 要求能证明为常量的布尔标量/向量，按实际条件选择形状元素，并保持可追溯的共同来源。普通运行时条件不会进入这条控制折叠路径。

这些边界遵循 [ConstantOfShape](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) 的 shape/填充值分工及 [Where](https://onnx.ai/onnx/operators/onnx__Where.html) 的选择规则。GQA 的目标维度均非负，Equal 对负标记恒为假，因此 Where 可以被证明为原目标向量。测试还实际执行恒为真的条件，避免实现退化为“总选同一边”。

解析后的 source 只留下已有的 shape_expr、Unsqueeze、ExpandDynamic 和 ReshapeDynamic，沿原来的 unit 合同、extent ABI、TE/TIR、CompiledModule 和 RuntimeSession 执行。applicability 从 **v6 升至 v7**，进入既有身份 builder；preparation v3、unit shape contract v4、公开 kernel ABI 保持原合同。没有增加新 IR、运行时形状解释器或隐式编译入口。

### 修复动态轴编辑的真实 lowering

首轮 LLVM 执行暴露了已有 Squeeze/Unsqueeze 的缺口：producer 已允许动态 B/S，但 TE callback 试图从 TensorType 的 `-1` 重建静态输出 shape，因而拒绝执行。

现在动态路径按 axes 保留输入 TE extent，仅插入常量 1 或删除已证明为 1 的轴；已有线性索引复制体负责数据计算。两个 sibling callback 共用这段轴映射，静态 shape 路径沿用原逻辑。另用多轴 `Unsqueeze({0,-2}) → Squeeze({0,4})` 跑两个 B/S，验证 rank、数据、fresh storage 和 cache 不变。

[Expand](https://onnx.ai/onnx/operators/onnx__Expand.html) 的生产准入继续采用受限同秩合同：各轴须为相同表达式，或输入轴为常量 1。GQA 插入的单例轴恰好可扩展为重复倍数 2；没有借此开放任意广播或未知 rank。

## 实际模型与产物

[make_minimind_projection_fixture.py](../../python/tools/make_minimind_projection_fixture.py) 增加 `--gqa`，从实际 MiniMind 导出按输出依赖抽取第一层 `Reshape_4` / `Reshape_6`，以 `present_k_0`、`present_v_0` 为输入边界。它检查实际节点计数和模型权重形状，没有手写替代 ONNX。源码版本、导出补丁和完整配置沿用[拆头报告](M3_ONNX_HEADS_REPORT.md)及[加权投影报告](M3_WEIGHTED_PROJECTION_REPORT.md)。

| 阶段 | 规模 |
|---|---|
| 实际 ONNX 子图 | 56 个节点，34 个控制常量；含 10 Shape、8 Gather、18 Unsqueeze、4 Concat、4 Reshape、2 ConstantOfShape、4 Mul、2 Equal、2 Where、2 Expand |
| source 规格 | 66 个 Relay 调用；差额来自四处多输入 Concat 的有序二元展开 |
| 编译后的计划 | 10 个调用，无模块常量池：每个 K/V 分支各有一个 Unsqueeze、两个 shape_expr、一个 ExpandDynamic、一个 ReshapeDynamic |

控制链消解减少了实际调用数，但仍是各算子独立执行，没有做 kernel 融合或原地重复。

| 文件 | SHA-256 |
|---|---|
| 完整动态 `minimind_prefill.onnx` | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| `gqa_dynamic.onnx` | `86481ff45cb7c15cdf09c87cf7fea714e782aa1aafaa3e90c0503daeb711375b` |
| `gqa_representative.onnx` | `33e89fc3f7f9cf083b9daf11aa71105942b6f44251631fae0b2c4ac97ade1f8e` |
| `gqa.json` | `4a875cfb72f9b4b3ca91a422525ec93ebf372c6b151fd95c7ab5402e2a5fb31d` |

fixture 位于 `out/fx_minimind_gqa/`，`receipt.json` 保存完整节点名、参数/结果顺序和全部文件 SHA。ONNX=1.22.0、NumPy=2.5.3，输入 RNG seed=20260908。两个代表输入均为 `[1,4,4,96]`，绑定共享的 B∈[1,3]、S∈[1,8]。

| B | S | 每个 K/V 输入 | 每个重复后的输出 |
|---:|---:|---|---|
| 1 | 1 | `[1,1,4,96]` | `[1,1,8,96]` |
| 1 | 4 | `[1,4,4,96]` | `[1,4,8,96]` |
| 2 | 3 | `[2,3,4,96]` | `[2,3,8,96]` |
| 3 | 8 | `[3,8,4,96]` | `[3,8,8,96]` |

## 验证与效果

生产测试为 [bounded_gqa_llvm_test.cpp](../../test/bounded_gqa_llvm_test.cpp)，只在完整 bounded/LLVM 门禁下注册。无下载依赖的合成图固定 heads=2、head_dim=3，用独立下标公式验证复制次序；实际模型分支显式加载上述 source，并同时比较动态 ONNX 参考和下标公式。未提供模型目录会打印 SKIP，本次实际提供了目录。

| 验证 | 结果 |
|---|---|
| 合成 GQA | 四组 B/S、40 次 LLVM 调用，数据精确一致；恒真 Where 分支另运行一次、10 次调用，并覆盖固定整数 Add/Div |
| 实际 K/V GQA | 四组 B/S、40 次 LLVM 调用，最大绝对误差 **0**，两个输出均为 fresh storage |
| 动态轴编辑 | 两组 B/S、4 次 LLVM 调用，多轴 Unsqueeze/Squeeze 往返精确一致，未固化 B/S |
| 准备阶段反例 | 11 次拒绝：范围内无法确定的 Equal、符号乘法、乘法溢出、INT64_MIN 幅值、除零、非恒等形状向量 reshape、动态 fill 长度、超过 16 的长度、伪造 target attrs、错误 fill dtype、运行时 Where 条件；cache 统计不变 |
| 运行时反例 | B/S 上下界、固定 heads/head_dim、K/V 的 B/S 不一致，共 8 次拒绝；无新增 allocation/submit，cache 不变 |
| 前端反例 | int64 fill 秩/类型/精度边界、重复属性、Expand 属性；保留未知中间 extent 的 source 正例也通过 |

实际读取 `out/build/bounded-llvm/out/bounded_gqa_profile/events.jsonl`：

| stage | 成功 run | submit/exec 对数 | runtime alloc | 带关联字段的事件 |
|---|---:|---:|---:|---:|
| `gqa_synthetic` | 4 | 40 | 120 | 204 |
| `gqa_true_branch` | 1 | 10 | 30 | 51 |
| `gqa_axis_roundtrip` | 2 | 4 | 12 | 22 |
| `minimind_gqa` | 4 | 40 | 120 | 204 |

所有表内事件均有 stage、B/S 和 builder 生成的 plan ABI；实际子图另带动态导出 SHA。bundle 还有 8 次非法输入的失败记录。runtime alloc 包括结果和 extent 参数存储，不计 kernel 内部 scoped malloc；metadata 只用于关联，不替代合同校验。

复现：

```bash
out/venv/bin/python python/tools/make_minimind_projection_fixture.py --gqa \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_gqa
cmake --build out/build/bounded-llvm -j2
KXC_MINIMIND_GQA_DIR="$PWD/out/fx_minimind_gqa" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure --no-tests=error \
  -R '^bounded_gqa_llvm_test$'
```

完整回归：默认 CPU/LLVM **48/48**，bounded gate-on **57/57**，Python **245/245**。本轮同时显式提供 GQA、heads、projection 三个实际子图 fixture；旧投影/拆头的误差仍分别为 2.74181e-6 / 4.29153e-6。Relay/Pass 生成物、契约、include layers、public headers、NLP checker 和静态库重复 strong symbol 审计均通过，后者输出为空。文档索引、本地链接和 `git diff --check` 通过，覆盖 **48** 篇 Markdown。真实 GQA 计划另补充断言无常量池，并重新执行该 CTest 通过。

本轮未重跑完整八层静态模型及容量 decode fixture，其历史证据见 [M2 报告](M2_MINIMIND_STATE_REPORT.md)。GQA 形状控制的 Equal/Where 被消解，不能当作动态数据 mask、通用 Slice/Concat、CUDA 或完整 attention 的新证据；本轮没有据此升级能力矩阵的其他格子。

## 三轮设计复核与下一步

1. **复用**：已有 source 格式、Relay 算子、ShapeProgram/DimExpr、TE callback 和 RuntimeSession；没有新表示层或运行时 evaluator。
2. **权威**：只依据整个声明范围内成立的事实折叠，负标记不进入维度合同，输出声明仍由 producer 核对，applicability 版本覆盖新增行为。
3. **消费者**：真实 ONNX、强制合成 LLVM、恒真/恒假选择、动态轴编辑与准备/运行时反例一同验证；外部模型缺失不会被当作执行成功。

下一步接 RoPE 的固定 head 切片、旋转和位置表输入，再组合完整 attention 与 M2 的 KV/valid extent。一般符号算术、数据 reshape 的 `-1` 推断、任意运行时条件和状态合并不在本次完成范围内。
