# M3/M4 技术报告：真实 MiniMind 的变长 RoPE 组合链

日期：2026-09-09。范围：CPU/LLVM 上从第一层输入 RMSNorm、QKV 投影、分头、Q/K 归一化到 rotary position embedding（RoPE）的实际 ONNX 子图。

## 达到的效果

同一份编译产物现可执行四组 B/S 的上述完整组合链，返回旋转后的 Q/K 和分头后的 V。实际模型子图执行 **204 次 LLVM kernel 调用**，与动态 ONNX ReferenceEvaluator 比较的最大绝对误差为 **4.05312e-6**；每次 Run 前后 primitive cache 的所有统计均不变。独立的 Q/K 旋转公式测试执行 72 次 LLVM 调用，最大误差 **9.56492e-8**。

这使前一模块的分头与归一化结果可以继续进入位置旋转。GQA 的重复链已有独立证据，下一步需将这些结果接到实际 attention 的分数、mask 和输出路径。这里的输入仍包括外部 embedding 激活和按位置选好的 cos/sin 表片段；embedding 查表、全局位置表的变长切片、完整八层变长模型、KV state/extent 联合尚未完成。可变 batch 维不表示请求级动态合批，也没有 CUDA 或性能提升声明。

## 使用的方法

### 保留边界计算，由准备阶段证明

真实导出按下列方式旋转最后一维：

```text
Shape(Q/K) → Gather(head_dim) → Div(2) → Cast(int64) → Cast(int64)
  → Unsqueeze → Slice 的 starts/ends

Slice(后半维) → Neg ─┐
Slice(前半维) ───────┴→ Concat → Mul(sin) ─┐
Q/K ───────────────────────────→ Mul(cos) ┴→ Add → Cast(float32)
```

本模型的 head_dim 固定为 96，因此两段切片的边界 48 可在整个 B/S 合同内证明为常量。Python 的显式 `kxc.onnx_shape_source.v1` 保留原始控制输入，C++ restricted producer 折叠固定维度的 Gather、整数除法和恒等 int64 Cast。没有把代表输入 `[1,4,…]` 的 B/S 固化为常量。

现有 `slice` 的 schema v2 声明输入数范围 1..5，具体合法形式为：单输入加 `SliceAttrs`，或五输入且无 attrs 的待准备 source。2/3/4 输入拒绝；五输入形式必须由 producer 将四个非空、长度至多 16 的 int64 控制向量证明为常量，随后重写为普通单输入 Slice。普通 InferType 明确拒绝尚未准备的五输入形式。静态 loader 保持原有单输入/属性形式，source loader 也保留该形式的读取兼容性。

ONNX shape inference 有时无法推导 Slice 中间输出长度。source importer 将这些未知中间 extent 留给 C++，仍校验输入/输出的具体代表与可检查的 dtype、rank、attrs。下游 Cast 即使接收未知 extent，也不能绕过 dtype 转换边界。最终代表输出数量、形状和类型由 producer 在编译/cache 前校验。

### 扩展现有 Slice/Concat 的动态非操作轴

Slice 的操作轴必须静态、步长必须为 +1；没有被切片的 B/S 直接保留已有 TE extent。端点归一化继续由 Slice InferType 的既有规则负责：负轴归一化、负端点加维度后截断、正端点截断到维度上限；`INT64_MAX` 可表示切到末尾。参考 [ONNX Slice 规范](https://onnx.ai/onnx/operators/onnx__Slice.html)。

数据 Concat 的拼接轴必须静态，其他轴必须由 resolver 证明为相同 DimExpr；TE 复用现有 TOPI concatenate，保留非拼接轴的动态 extent，生成 fresh output。head 的两个半段因而能在不同 B/S 下重新拼接。非操作轴相等的要求来自 [ONNX Concat 规范](https://onnx.ai/onnx/operators/onnx__Concat.html)。负号仍通过已有 float32 Neg 计算。

这不是任意动态 Slice/Concat：依赖运行时数据的切片边界、在符号轴上切片、动态拼接轴和不同符号的非拼接轴均在准备阶段拒绝。

### 修复组合图暴露的两处所有权问题

1. **快照属性隔离**：在唯一 Relay snapshot cloner 中深复制 Slice 的 starts/ends/axes/steps。测试在准备后修改调用方四个数组，再编译并运行不同 B/S，结果仍使用已准备的合同。
2. **参数和常量的名称排序**：ValueGraph 先编号参数，但 ShapeProgram 会按名称排序，`value.10` 排在 `value.2` 前。此前按数组前缀区分参数和常量，在三个输入加权重的真实 RoPE 图上产生重复合同。现在按已有 ValueGraph 参数名称排除已覆盖项，常量保留静态合同；没有引入另一套值编号。无模型依赖的三参数回归显式覆盖这个交错顺序。

TE Slice 也构造独立的输出 shape 数组，避免编辑输出维度时改变输入 placeholder 的共享数组。

### 身份与挂载

继续使用现有 Relay、ShapeProgram/DimExpr、ValueGraph、TE/TIR、ExecutablePlan 和 RuntimeSession，没有新增 IR。`slice` 与 `concatenate` 的 operator schema 均升至 v2；bounded applicability 从 v7 升至 **v8**，使新准入行为进入既有身份构建链。preparation v3、unit shape contract v4 保持原有表达；Slice/Neg/Concat 接入 bounded unit 的正式白名单与真实 lowering consumer。

## 真实输入与产物

模型源为 `jingyaogong/minimind` commit `6fc918beb68a0d8c40452338df6319fe168014ba` 的既有导出。参数：8 层、hidden=768、Q heads=8、KV heads=4、head_dim=96、intermediate=2432、vocab=6400、位置表容量 128、opset 17、随机权重 seed=0；沿用非原地 mask 与位置容量的本地导出适配。

| 产物 | 结构或 SHA-256 |
|---|---|
| 完整动态 prefill ONNX | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| 实际第一层子图 | 75 个 ONNX 节点；有序二元化 Concat 后为 81 个 Relay source 调用、51 个源常量 |
| 编译后的计划 | 51 个调用，包含原有投影/拆头/QK 归一化的 33 个调用和 RoPE 的 18 个数据调用 |
| `rope_dynamic.onnx` | `f57f6d3896fa2853a691269f98c22f38ef93429e55c73a54995884662493c588` |
| `rope_representative.onnx` | `1df8f68569a6911b8aeb9523bf27de279ec041527c29fc39ff673b4540bc0631` |
| `rope.json` | `5c0d0bd87e37ba90ac6cdcb3e42c9f6d61b7c9109b2ad59c2e5b8729e0a4a047` |
| `rope.params` | `e617b24fc85249ddbf2c7f80e862c14619498ddd03165b7d47a3e0c8d3374e10` |

三个输入依次为实际 `embed_tokens/Gather` 的输出 `[B,S,768]`，以及 `/model/model/Slice_output_0`、`Slice_1_output_0` 的 cos/sin `[S,96]`。三路共享 S；B∈[1,3]、S∈[1,8]。输出为 Q `[B,S,8,96]`、K/V 各 `[B,S,4,96]`。

| B | S | cos/sin 起始位置 | LLVM 调用数 |
|---:|---:|---:|---:|
| 1 | 1 | 0 | 51 |
| 1 | 4 | 1 | 51 |
| 2 | 3 | 2 | 51 |
| 3 | 8 | 3 | 51 |

cos/sin 数据直接截取完整导出中的 128×96 常量表，激活输入使用 seed=20260908 的 float32 随机数据。生成工具使用 ONNX=1.22.0、NumPy=2.5.3；`out/fx_minimind_rope/receipt.json` 保存实际节点名、顺序和全部文件 SHA。位置表的截取在 fixture 生成端执行，未冒充图内动态位置选择。

## 验证和运行观测

[bounded_rope_llvm_test.cpp](../../test/bounded_rope_llvm_test.cpp) 在 bounded/LLVM 门禁下注册。合成和回归分支不需要下载；实际分支通过 `KXC_MINIMIND_ROPE_DIR` 显式开启，本次已开启并执行。

| 验证 | 结果 |
|---|---|
| 独立 Q/K 旋转参考 | 四组 B/S、72 次 LLVM 调用，和直接下标公式的 double 参考比较，最大误差 9.56492e-8；输出形状、fresh storage、有限值和 cache 不变均验证 |
| 实际 RMSNorm/QKV/heads/RoPE | 四组 B/S、204 次 LLVM 调用，Q/K/V 全输出和动态 ONNX 参考比较，最大误差 4.05312e-6，阈值 2e-5 |
| 参数/常量名称交错 | 三个参数和 `value.10` 常量，两组形状、20 次 LLVM 调用，数值精确一致 |
| Slice 属性深快照及截断 | 准备后修改调用方 starts/ends/axes/steps；两组形状、两次 LLVM 调用，负端点、超大终点、多轴切片结果精确一致 |
| 准备阶段反例 | 12 次拒绝：零/负步长、非法轴、符号切片轴、变量边界、标量控制、错误 arity、伪造 attrs、改变 shape-control dtype、动态拼接轴、非拼接轴独立符号、运行时边界输入；cache 不变 |
| 未准备 source | 普通 InferType 拒绝五输入 Slice |
| 运行时反例 | 10 次拒绝：B/S 范围、固定 heads/head_dim、Q/K B/S 不同、cos/sin 长度或宽度不符；无新增 allocation/submit，cache 全部统计不变 |
| Python 导入 | 新增 6 项 source 保留及负例检查；全量 **251/251** 通过 |

已读取 `out/build/bounded-llvm/out/bounded_rope_profile/events.jsonl`：

| stage | 成功 run | submit/exec 对数 | runtime alloc | 关联事件总数 |
|---|---:|---:|---:|---:|
| `rope_synthetic` | 4 | 72 | 208 | 356 |
| `rope_parameter_constants` | 2 | 20 | 60 | 102 |
| `rope_slice_snapshot` | 2 | 2 | 6 | 12 |
| `minimind_rope` | 4 | 204 | 604 | 1016 |

上述事件均带 stage、B/S 和 builder 生成的 plan ABI；实际模型事件另带动态导出 SHA 和 position_offset。另有 10 条失败 run 记录。runtime alloc 包含结果及 extent 参数存储，不包含 kernel 内部 scoped malloc。这里验证的是关联和执行正确性，没有进行速度对比。

最终回归：默认 CPU/LLVM 构建 **48/48 CTest**，bounded/控制流 gate-on 构建 **58/58 CTest**，Python **251/251**。两个构建均使用 LLVM 20.1.2。更新了旧注册断言与 Slice 类型反例：静态操作轴可以保留未知非操作轴，符号操作轴仍拒绝；五输入 source 在默认 gate-off 构建中也不能直接 InferType。

本轮 full bounded CTest 显式提供投影、分头、GQA、RoPE 四个实际 fixture 目录：它们分别执行 44、132、40、204 次 LLVM 调用。完整八层静态 prefill/decode 与 ResNet18 的可选数值分支本轮未重跑，不将这些 SKIP 记为新增整模型证据。

Relay/Pass 生成物 freshness、单注册检查、CTest 合同、include layers（279 个文件）、public headers（89 个安装头与 10 个实验头的编译检查）、NLP 能力检查、文档索引/链接（49 篇 Markdown）及 `git diff --check` 均通过。bounded 静态库的重复 strong symbol 审计结果为空。能力矩阵的 slice_concat 行已按前端、Relay、lowering、LLVM、CUDA、runtime、numeric、profile 八层更新 CPU 证据；CUDA 状态保留原有范围，没有新增硬件执行声明。

可复查本轮日志：`/tmp/kxc-m3-rope-default-ctest.log`、`/tmp/kxc-m3-rope-bounded-ctest.log`、`/tmp/kxc-m3-rope-full-lasttest.log`、`/tmp/kxc-m3-rope-python-full.log`。后续测试可能覆盖构建目录下的 LastTest.log，因此另保存了本轮完整副本。

## 复现

```bash
out/venv/bin/python python/tools/make_minimind_projection_fixture.py --rope \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_rope
cmake --build out/build/bounded-llvm -j2
KXC_MINIMIND_ROPE_DIR="$PWD/out/fx_minimind_rope" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure --no-tests=error \
  -R '^bounded_rope_llvm_test$'
KXC_MINIMIND_ROPE_DIR="$PWD/out/fx_minimind_rope" \
KXC_MINIMIND_GQA_DIR="$PWD/out/fx_minimind_gqa" \
KXC_MINIMIND_HEADS_DIR="$PWD/out/fx_minimind_heads" \
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection_scalar" \
  ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
PYTHONPATH=python out/venv/bin/python -m pytest -q test/ python/
```

## 三轮 QA

- **A：复用检查。** Slice 和 Concat 已有静态语义与 TE consumer；只补控制证明、动态非操作轴及快照复制。固定 head_dim 的整数求值复用 DimExpr，没有 Python shape evaluator 或新 IR。
- **B：权威与身份。** 源 Slice 的五输入形式进入正式算子 schema，准备后只保留单输入属性形式；未准备形式不能执行。参数/常量按 ValueGraph 名称区分，保留 ShapeProgram 的确定性排序。schema/applicability 版本覆盖新语义。
- **C：真实 consumer 与反例。** 测试经过 importer、adapter、Compiler::CompileBounded 和 RuntimeSession，包含实际权重与完整位置旋转链；独立参考、来源拒绝、运行时零副作用拒绝、深快照和名称交错回归共同覆盖。完整 attention 与状态联合继续推进。

下一段的实际导出已定位到 `ConstantOfShape([S,S], -inf) → Trilu → Add → Softmax`，还需让同一张量两个序列轴共享 S 的约束进入既有 unit/runtime guard 合同，再合入输出投影。
