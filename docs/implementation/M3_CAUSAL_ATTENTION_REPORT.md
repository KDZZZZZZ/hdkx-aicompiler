# M3/M4 技术报告：真实 MiniMind 的变长因果 attention

日期：2026-09-09。范围：CPU/LLVM 上第一层从输入 RMSNorm 到 attention 输出投影的实际 ONNX 子图。

## 达到的效果

同一份编译产物现可执行真实 MiniMind 第一层的 RMSNorm → QKV → 分头与 Q/K 归一化 → RoPE → GQA → 因果 mask → Softmax → V 聚合 → 合头 → 输出投影。四组 B/S 共执行 **308 次 LLVM kernel 调用**，attention 输出和 K/V 两个中间输出与动态 ONNX ReferenceEvaluator 比较，最大绝对误差 **3.69549e-6**。同一产物再执行一次未来 token 扰动，共 **385 次实际模型 LLVM 调用**：修改三个序列各自后四个位置的激活，所有前四个位置的 attention 和 K/V 输出保持逐元素完全一致，后缀输出确实改变。所有 Run 前后 primitive cache 统计不变。

这把此前分别通过的 RoPE、GQA 和 attention 核心连成了真实导出图。输入仍是 embedding 后激活和已按位置选好的 cos/sin；本报告没有覆盖图内 embedding 查表、全局位置表动态切片、FFN/残差、完整八层变长模型，以及 bounded 图与 KV state/extent 的联合。B 变化表示一份产物接受不同 batch 尺寸；请求排队、合批和 KV 槽位管理仍待实施。没有新增 IR，也没有 CUDA 执行或速度提升声明。

## 使用的方法

### 让形状计算和数值计算使用同一个证明结果

实际导出按 `Shape → Gather(S) → Unsqueeze → Concat(S,S)` 构造目标，再由 `ConstantOfShape(-inf) → Trilu(upper=1,k=1)` 生成 mask。Python 的显式 shape-source 入口保留这些节点，C++ restricted producer 从输入轴证明目标为 `[S,S]`，TE 按实际 S 写出 mask。三角线之上为负无穷，主对角线及以下为零，因此当前 query 至少能看到自身。

扩展现有 `constant_of_shape` 的 schema 至 v2：静态形式仍是一个控制输入加明确 target；producer 为有界 float32 填充生成 `{数据来源, 形状控制}` 两个输入及已有的 Const/InputAxis 表达式。数据输入提供动态 extent 的唯一来源，控制向量保留在正式调用链中。源图不能自行携带 target 或表达式属性；普通 InferType 拒绝尚未准备的空 target。

float32 填充值允许正/负无穷、拒绝 NaN。JSON 用 `value_bits` 保存精确 IEEE-754 位模式，C++ 解码为同一浮点值，避免非标准 JSON `Infinity`；int64 控制填充继续使用精确整数 `value`，绝对值上限 2^53、向量长度上限 16。float32 目标 rank 为 1–8，利用现有 DimExpr 在符号上界计算最大元素数，共享既有 **256 MiB** 单输出字节上限；没有另一套 Python 形状求值器。语义参照 [ONNX ConstantOfShape](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html)。

### 用普通 Relay 算子计算三角区域

新增的 `trilu` 是普通 Relay operator，经过声明合同、attrs、InferType、TE compute、FFI、snapshot 和生产 lowering 接入现有链路。子集为 float32、rank≥2、固定 upper 和 int64 标量对角偏移 k。ONNX 可省略 k，此时为 0；opset 至少为 14，运行时 k、向量 k、错误 dtype 和非法属性均拒绝。

TE 以最后两轴的 `column-row` 和 k 比较，再用 Select 保留输入或写零。计算不使用 `row+k`，因此 int64 极值 k 不产生加法溢出；最后两轴以前的维度作为 batch 保留。十二组 CPU/LLVM 静态测试覆盖矩形、batch、上下三角、负/正/零偏移、两个 int64 极值及输入中的 `-inf`。参照 [ONNX Trilu](https://onnx.ai/onnx/operators/onnx__Trilu.html)。

### 支持重复 S 轴和仅用于 buffer 形状的 extent

attention 分数含 `[B,H,S,S]`，mask 含 `[S,S]`。同一输入的后一个 S 轴现在可以引用该输入更早的 S 轴。unit producer 和 ModuleInvocationContract 都只接受严格向前的轴引用，拒绝自引用、后向输入、越界和循环引用；RuntimeSession 和直接 `CompiledModule::Invoke` 都在分配前检查相等关系。

`shape_expr(S,S)` 的来源可能是 `[B,H,S,D]`。B 虽不进入结果元素，却仍是动态输入 buffer 形状和调用合同的一部分。lowering 保留这项 ABI 信息，并只对 Shape/ShapeExpr/ConstantOfShape 检查这种用途：它们必须不读取输入 payload，extent 必须实际出现在输入 buffer 形状中。仅被边界使用的 extent 进入 schedule canonical 内容。新增回归证明无用途的 extent 和伪装成形状算子的 payload 读取仍被拒绝，原有可执行性检查没有整体放宽。

### 在 Reshape 消费处证明固定的 -1

真实末尾导出的是 `[B,S,-1]`。先把 -1 保留为 Reshape 的推导指令；它不会成为负的 DimExpr。producer 对输入和目标中的直接符号按出现次数匹配，只在所有被约去的符号下界严格大于零时约去 B/S，再用现有 DimExpr 的 checked 乘除法证明剩余维度为固定正整数 `8×96=768`。仍含动态因子的结果、多个 -1、非整数商、零边界取消和小于 -1 的值均拒绝。

归一化后的 shape_expr 在每个 Reshape 消费处生成。无模型依赖的回归让两个不同宽度的张量共享原始 `[B,S,-1]` 控制，分别得到 `[B,S,6]` 和 `[B,S,10]`，验证没有把一个消费方的推导结果覆盖到另一个消费方。没有推导指令时继续复用原来的控制链。ONNX 中间 Reshape、Transpose、MatMul 和 Softmax 的未知 extent 保留为 metadata，最终由 C++ 普通类型推导及代表输出检查裁决。

### 身份和所有权

bounded applicability **v9**、unit shape contract **v5**、ModuleInvocationContract **ABI v4**（`KXC_MODULE_INVOKE_V4`）。preparation 保持 v3，`trilu` schema 为 v1；新增准入和参数用途通过已有 builder/canonical 链进入身份。唯一形状权威仍是 ShapeProgram/DimExpr，唯一值编号仍是 ValueGraph。snapshot cloner 复制 Trilu 属性和 ConstantOfShape 表达式数组；准备后修改调用方 Trilu 属性不会改变编译结果。

## 输入、产物与复核数据

模型源为 `jingyaogong/minimind` commit `6fc918beb68a0d8c40452338df6319fe168014ba` 的既有导出，沿用非原地 mask 与位置容量适配。8 层、hidden=768、Q heads=8、KV heads=4、head_dim=96、intermediate=2432、vocab=6400、位置表容量 128、opset 17、随机权重 seed=0。这里提取第一层，输入随机种子为 20260908；ONNX 1.22.0、NumPy 2.5.3。

| 产物 | 结构或 SHA-256 |
|---|---|
| 完整 prefill ONNX | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| 实际第一层子图 | 151 个 ONNX 节点；二元化 Concat 后 168 个 Relay source 调用、93 个源常量；准备后 77 个调用 |
| `attention_dynamic.onnx` | `eb7c543804c2fe8dbbfe720efa45d6e58f942b1563ae3c2c132bb9ab3ab493ce` |
| `attention_representative.onnx` | `497827f078a9bf426905b9947f5ea818873e53edfc4eec14fff4b616ac644de7` |
| `attention.json` | `3972768e02aaf71a5b5a55d02c57937d15f9b29c7d82a4ef76fedade25f92734` |
| `attention.params` | `153c459afab865d3c62267f60e3614d60e75bdf785bcdc6dc9381a94fd0add75` |

输入依次为 `/model/model/embed_tokens/Gather_output_0` 激活 `[B,S,768]`、全局两个 Slice 的 cos/sin `[S,96]`。三路共享 S；B∈[1,3]、S∈[1,8]。输出依次为 `o_proj/MatMul` 的 `[B,S,768]` 和 present K/V 各 `[B,S,4,96]`。四组 `(B,S,position_offset)` 为 `(1,1,0)`、`(1,4,1)`、`(2,3,2)`、`(3,8,3)`。每组执行 77 个调用。表片段在 fixture 端从真实 128×96 cos/sin 表截取，不算作图内动态位置选择。

`out/fx_minimind_attention/receipt.json` 保存全部实际节点名、顺序、参数与参考文件的 SHA。生成器用实际算子计数检查提取边界，防止模型导出变化后静默缩小测试范围。

## 验证结果和三轮质量检查

| 检查轮次 | 检查方法和结果 |
|---|---|
| 第一轮：合同和挂载 | 注册/attrs/InferType/TE/FFI/source/snapshot 使用同一算子合同；11 个 mask/Trilu 准备反例和 1 个原始 fill 类型反例；5 个固定 -1 推导反例；都在 cache 前拒绝。静态 Trilu 十二组数值及生产 primitive lowering 已通过 |
| 第二轮：独立数值与所有权 | 合成 attention 四组 B/S 加未来 K/V 扰动，45 次 LLVM 调用，double 参考最大误差 9.93489e-8；共享 -1 控制的两个 Reshape 四组 B/S、16 次调用，数值精确；重复 S/S 与 Trilu 属性快照三个尺寸、3 次调用；6 个 attention 非法输入和 session/直接 module 两个非方阵拒绝均无新增分配/启动，cache 不变。无用途 extent 与形状算子 payload 读取拒绝回归通过 |
| 第三轮：真实模型与完整回归 | 真实子图四组 ONNX 参考加未来激活扰动，共 385 次调用，误差及因果性结果见上。默认 48/48、bounded 59/59 CTest，Python 270/270；Relay/Pass 合同与 freshness、NLP 能力、include、公开头编译、文档索引和重复 strong symbol 检查全部通过 |

已读取 `out/build/bounded-llvm/out/bounded_causal_attention_profile/events.jsonl`：

| stage | 成功 Run | submit/exec 对数 | runtime alloc | 关联事件总数 |
|---|---:|---:|---:|---:|
| `causal_synthetic` | 4 | 36 | 104 | 180 |
| `causal_future` | 1 | 9 | 26 | 45 |
| `causal_fixed_reshape` | 4 | 16 | 48 | 84 |
| `causal_square` | 3 | 3 | 6 | 15 |
| `minimind_attention` | 4 | 308 | 912 | 1532 |
| `minimind_attention_future` | 1 | 77 | 228 | 383 |

这些事件带 stage、sequence 和 builder 生成的 plan ABI；有 B 的测试另带 batch，实际模型另带导出 SHA 和 position_offset。另有七个 RuntimeSession 失败 run，直接 module 的拒绝由专门断言验证。allocation 计数包含结果和 extent 参数存储，不包含 kernel 内部 scoped malloc；这里只用它审计关联及拒绝时没有新增运行时分配。

最终回归：默认 CPU/LLVM **48/48 CTest**、bounded/控制流 gate-on **59/59 CTest**、Python **270/270**，LLVM 版本为 20.1.2。bounded 全量回归显式启用投影、拆头、GQA、RoPE、因果 attention 五个实际 fixture。原有负例按新的已证明子集更新：固定整数商的 -1 可以接受，非整数商仍拒绝；float32 填充可接受，未声明的 float64 填充仍拒绝。

Relay **37/37**、Pass **20/20** 合同及生成物 freshness 通过；include layers 扫描 279 个文件、公开头检查编译 89 个安装头和 10 个实验头；文档检查覆盖 50 篇 Markdown；静态库 1,843 个 strong symbol 没有重复，`git diff --check` 通过。重新运行 fixture 生成器，30 个输出文件的 SHA-256 全部一致。能力矩阵的 mask_select 已按 CPU 实际证据更新，保留原有 Where 的范围和 CUDA 限制；全 mask Softmax 语义仍未开放。

日志：`/tmp/kxc-m3-causal-default-ctest.log`、`/tmp/kxc-m3-causal-bounded-ctest.log`、`/tmp/kxc-m3-causal-python-full.log`。完整 bounded LastTest 已另存 `/tmp/kxc-m3-causal-full-lasttest.log`，避免被后续测试覆盖。

## 复现

```bash
out/venv/bin/python python/tools/make_minimind_projection_fixture.py --attention \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_attention
cmake --build out/build/bounded-llvm -j4
KXC_MINIMIND_ATTENTION_DIR="$PWD/out/fx_minimind_attention" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure --no-tests=error \
  -R '^bounded_causal_attention_llvm_test$'
KXC_MINIMIND_ATTENTION_DIR="$PWD/out/fx_minimind_attention" \
KXC_MINIMIND_ROPE_DIR="$PWD/out/fx_minimind_rope" \
KXC_MINIMIND_GQA_DIR="$PWD/out/fx_minimind_gqa" \
KXC_MINIMIND_HEADS_DIR="$PWD/out/fx_minimind_heads" \
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection_scalar" \
  ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
cmake --build out/build/dev-ninja-cpu -j4
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
PYTHONPATH=python out/venv/bin/python -m pytest -q test/ python/
```

[主要回归](../../test/bounded_causal_attention_llvm_test.cpp)的合成、共享控制和拒绝测试无需模型下载；实际分支由 `KXC_MINIMIND_ATTENTION_DIR` 显式开启，本轮已运行。静态矩形 Trilu 在 [codegen_llvm_test.cpp](../../test/codegen_llvm_test.cpp)，extent 的正反例在 [te_schedule_test.cpp](../../test/te_schedule_test.cpp)，同输入轴引用的合同组装反例在 [compiled_module_test.cpp](../../test/compiled_module_test.cpp)。完整八层静态 prefill/decode 和 ResNet 的可选数值分支本轮为 SKIP，不记为新增整模型证据。
