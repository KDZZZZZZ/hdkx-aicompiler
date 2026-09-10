# M3/M4 技术报告：ONNX 形状控制链与变长拆头

日期：2026-09-08。范围：CPU/LLVM 下保留实际 ONNX 形状控制输入，执行 MiniMind 第一层 RMSNorm、Q/K/V 投影、拆头和 Q/K RMSNorm。

## 结果与用途

实际 MiniMind 子图现在用**一份编译产物**处理四组 B/S。输出 Q 为 `[B,S,8,96]`，K/V 为 `[B,S,4,96]`；Q/K 已完成模型中的 head RMSNorm。四次运行共 **132 次 LLVM kernel 调用**，与动态 ONNX 子图上的 ReferenceEvaluator 相比，最大绝对误差 **4.29153e-6**，阈值为 1e-4。运行期间 primitive cache 的全部统计不变。

上一模块只执行 `[B,S,768]` 到 Q/K/V 投影。继续接实际拆头时，导出中的 `Shape → Gather(标量索引) → Unsqueeze → Concat → Reshape` 不能沿静态 importer 的“把目标 shape 固化成 attrs”路线处理。本次保留控制链，由既有 restricted producer 在有界合同内证明和 lower。

这里验证同一 bounded 变体适应不同输入长度。跨变体热替换仍由 M6 验收。本次未执行完整八层变长模型、embedding、RoPE、GQA、完整 attention 或 KV 状态联合，没有性能提升声明。

## 使用的方法

### 显式保留 source，交给现有 producer 证明

Python importer 增加 `preserve_shape_values=True`，序列化格式为 `kxc.onnx_shape_source.v1`。输入仍必须有具体代表形状，未绑定的 B/S 会被拒绝。ONNX 官方 shape inference 补齐代表元数据，原始 Shape 和 Reshape 控制输入留在图中，普通输入数据没有被执行。

多输入 Concat 按原实参顺序展开为已有二元 `concatenate`；中间值名避开原图所有输入、初始化器和节点结果。Unsqueeze 使用已有算子与 axes attrs；Reshape 使用已有双输入 `reshape_dynamic`，不填写未经 producer 证明的目标 attrs。唯一新增的 ONNX 映射元数据是 `Shape → shape_of`，已写回声明契约并重新生成。

C++ `LoadONNXShapeSource` 复用默认 importer 的 JSON、常量和 Relay 构图代码，返回尚未通过类型校验的 source Function。调用方必须把 `function` 和 `declared_output_types` **一起**交给 `RestrictedSymbolicShapeAdapter::Prepare`。producer 解析控制链后运行已有 InferType，核对输出数量/shape/dtype，再进入编译准备。默认 loader 拒绝 source 格式，普通 Compile 拒绝未解析的 ReshapeDynamic。这是现有 Relay source 的显式序列化入口，没有设计新 IR 或 Python 编译 API。

### 修复标量秩，保留实际 ONNX 语义

原常量折叠出口和 initializer 导入无条件调用 NumPy `ascontiguousarray`，会把零维数组变成长度为 1 的数组。标量索引因此被写成 `[1]`，Gather 结果变成向量，Unsqueeze 又提升为二维，最终 Concat 秩不匹配。

现在两处都只在 `ndim > 0` 时整理连续存储，标量保留 `shape=[]`。Constant、折叠结果和原始 initializer 的含义一致。Python 测试用 ONNX ReferenceEvaluator 比较折叠前后的实际输出，并检查 int32/int64 标量初始化器的秩和载荷。

秩规则依据 [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html) 与 [Unsqueeze](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html)。形状 Gather 仅接受常量 int32/int64 索引，可为标量或向量；合法负索引归一化，越界拒绝。标量结果在 producer 内带来源标记，只允许经 axis=0/-1 的 Unsqueeze 变成形状向量，不能直接作为标量图结果或普通数据逃逸。

### 复用形状合同与执行链

resolver 在 shape 控制位置识别 int64 字面向量，把动态 B/S 与固定 heads/head_dim 组合。它要求可证明的共同来源，不让任意输入张量的内容成为 shape。int32 索引按 32 位读取后扩大为 int64，避免按错误的字节宽度读取。

ShapeProgram/DimExpr 仍是唯一形状表达式权威。解析结果落到已有 `shape_expr` 和 `reshape_dynamic`，再走原 unit 合同、extent ABI、TE/TIR 和 RuntimeSession。shape 输出是真实 int64 张量，随实际 B/S 改变；reshape 保持 fresh-output 复制语义。

bounded applicability 从 **v5 升至 v6**，进入既有身份 builder；preparation v3、unit shape contract v4 和公开 kernel ABI 保持原合同。未增加运行时 shape VM、隐式编译或候选路由。

## 真实模型证据

[make_minimind_projection_fixture.py](../../python/tools/make_minimind_projection_fixture.py) 增加 `--heads`：从实际 ONNX 导出按输出依赖抽取子图，沿用已有纯常量折叠结果，验证节点计数和权重形状。模型配置、源码基准和本地导出补丁见 [加权投影报告](M3_WEIGHTED_PROJECTION_REPORT.md)及[图结构报告](M3_GRAPH_STRUCTURE_REPORT.md)。

| 阶段 | 结构 |
|---|---|
| 抽取的实际子图 | 43 个 ONNX 计算/控制节点，29 个常量 |
| Python source | 49 个 Relay 调用；差额来自三处多输入 Concat 的二元展开 |
| producer 解析并编译 | 33 个调用：8 个输入 RMSNorm、3 个投影、3 个 shape_expr、3 个 reshape_dynamic、16 个 Q/K RMSNorm |

控制链缩短属于形状来源解析；33 个调用仍各自执行，未进行 TE kernel 融合。

| 文件 | SHA-256 |
|---|---|
| 完整动态 `minimind_prefill.onnx` | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| `heads_dynamic.onnx` | `2db9216a7d2ff51cd344d5d91e707422d63411e962c463b1b9b19fb1bf61b9c9` |
| `heads_representative.onnx` | `d6151bb43622eb51243e62f916ee961245c4a9a20a2dc6a02240360dffeb4800` |
| `heads.json` | `2e2d1067b2216f7aa85f19bfffb043501398dc9355474ee3f5dba04a6aea2fe4` |

fixture 位于 `out/fx_minimind_heads/`。`receipt.json` 保存源节点、输出顺序、全部文件 SHA、ONNX=1.22.0、NumPy=2.5.3、RNG seed=20260908。代表输入为 `[1,4,768]`，由 C++ 明确绑定 B∈[1,3]、S∈[1,8]。参考输入是 embedding 后的随机激活，并非 token 经 embedding 得到的结果。

| B | S | Q | K 与 V |
|---:|---:|---|---|
| 1 | 1 | `[1,1,8,96]` | `[1,1,4,96]` |
| 1 | 4 | `[1,4,8,96]` | `[1,4,4,96]` |
| 2 | 3 | `[2,3,8,96]` | `[2,3,4,96]` |
| 3 | 8 | `[3,8,8,96]` | `[3,8,4,96]` |

修复标量秩后，另在 `out/fx_minimind_projection_scalar/` 重新生成默认投影 fixture，仍为 11 个调用、44 次 LLVM 执行、最大误差 **2.74181e-6**。新的动态/代表 SHA 分别为 `87732b4169914714ebbf1a85c2bde6fe1be1e15f65213a3c03d6aa91d0eb6838` / `25bfaeb6e8fad90d8abdbb437ee5fdfab381a186ca4c9061ace0cc515f0efa33`。原报告的 fixture 和历史 receipt 未覆盖。

## 验证与观测效果

| 验证 | 实际结果 |
|---|---|
| 无模型依赖的 source → producer → LLVM | [onnx_shape_source_llvm_test.cpp](../../test/onnx_shape_source_llvm_test.cpp)：四组 B/S、2 个调用、8 次 LLVM 执行；数据与实际 shape 结果精确一致，reshape 输出不别名输入 |
| source/准备反例 | 9 次拒绝：格式混用、未解析 source 直接 Compile、声明输出 shape/dtype/数量错误、形状标量直接返回、非法 Unsqueeze 轴、标量直接 Concat、普通 shape 参数；cache 不变 |
| runtime 反例 | B/S 各自越下界/上界，共 4 次拒绝，零新增 runtime allocation 和 kernel submit，cache 不变 |
| Python 前端 | 折叠前后语义、标量 rank、四输入 Concat 次序、名字碰撞、Shape 属性、Reshape allowzero、未绑定代表维度、模式布尔类型均覆盖 |
| 元素数证明 | 旧反例误把追加长度 1 的轴当作元素数不等，过去实际依赖“不支持混合 Concat”而失败；现改成长度 2 的真实反例，并确认长度 1 合法。`-1` 推断和非法 Expand 仍拒绝 |
| 实际模型子图 | [bounded_projection_llvm_test.cpp](../../test/bounded_projection_llvm_test.cpp) 显式提供 heads 和重新生成的 projection fixture，两者均执行 |

本轮实际读取 profile bundle，结果如下。runtime alloc 包括输出和 extent 参数存储，不包含 kernel 内部 scoped malloc。

| stage | 成功 run | submit/exec 对数 | runtime alloc | 带完整关联字段的事件 |
|---|---:|---:|---:|---:|
| `onnx_heads` 小型链 | 4 | 8 | 24 | 44 |
| `minimind_heads` 实际子图 | 4 | 132 | 396 | 664 |
| `minimind_projection` 回归 | 4 | 44 | 132 | 224 |

小型链事件均有 stage、B/S、builder 生成的 plan ABI；实际子图另带动态导出 SHA。bundle 分别位于 `out/build/bounded-llvm/out/onnx_shape_source/profile/` 和 `out/build/bounded-llvm/out/bounded_projection_profile/`。失败输入不增加 allocation/submit，关联字段不替代形状和 ABI 校验。

复现：

```bash
out/venv/bin/python python/tools/make_minimind_projection_fixture.py --heads \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_heads
out/venv/bin/python python/tools/make_minimind_projection_fixture.py \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_projection_scalar
cmake --build out/build/bounded-llvm -j2
KXC_MINIMIND_HEADS_DIR="$PWD/out/fx_minimind_heads" \
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection_scalar" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure --no-tests=error \
  -R '^(onnx_shape_source_llvm_test|bounded_projection_llvm_test|shape_value_llvm_test)$'
```

完整回归和最终审计结果：

| 检查 | 结果 |
|---|---|
| 默认 CPU/LLVM 全量 CTest | **48/48** |
| bounded/dynamic ABI/restricted/exact/control gate-on 全量 CTest | **56/56**；显式提供 heads 与修复标量秩后的 projection fixture，两段实际模型子图均执行 |
| Python：`PYTHONPATH=python out/venv/bin/python -m pytest -q test/ python/` | **240/240** |
| Relay/Pass 生成物 freshness、契约、include layers、public headers | 全部通过 |
| NLP capability checker | 通过；normalization 只追加本子图证据，原 status/gate 和 CUDA 边界不变 |
| 文档索引与本地链接、`git diff --check` | 通过；覆盖 **47** 篇 Markdown |
| bounded 静态库重复 strong symbol 审计 | 输出为空 |

bounded 首轮的唯一失败来自上述旧元素数反例，修正后已重新执行完整 56 项。本轮没有重跑完整八层静态模型和容量 decode fixture，历史证据见 [M2 模型报告](M2_MINIMIND_STATE_REPORT.md)，不由本次子图结果替代。

## 三轮设计复核与边界

1. **复用**：现有 ONNX 解析器、二元 Concat、Relay shape 算子、ShapeProgram、TE/TIR 和 RuntimeSession；保留原依赖和实参顺序。
2. **权威**：source 格式区分尚未验证的图，输出声明在编译/cache 前检查；标量来源只属于证明链，不成为新 ABI；applicability 版本进入既有身份。
3. **消费者**：真实导出子图与强制运行的小型 LLVM fixture 同时存在，覆盖形状值实际输出、整数宽度、负索引、错误元数据、越界零启动和 cache 不变。

full Shape 只支持 rank 1–8、无 `start`/`end` 属性。固定向量控制长度、已知 Unsqueeze axes、可证明 reshape 元素数仍是边界。独立 shape 来源拼接、任意标量形状算术、`-1` 推断、数据相关 shape、CUDA 和状态合并未开放。下一段按实际 MiniMind 图接 RoPE/GQA 与 attention，再交接 M2 的 KV/valid extent。
