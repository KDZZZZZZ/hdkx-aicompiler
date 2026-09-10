# M3/M4 技术报告：完整八层 MiniMind 的变长 prefill

日期：2026-09-09。范围：CPU/LLVM，真实导出图从 token ID 到 logits 和各层 K/V。

## 达到的效果

一份编译产物现在可以执行完整八层 MiniMind prefill：embedding → RMSNorm/QKV → RoPE/GQA/因果 attention → 输出投影与残差 → SwiGLU FFN 与残差 → 最终归一化和词表投影。输入只有 int64 `input_ids[B,S]`，B∈[1,3]、S∈[1,8]。位置表的 `0:S` 选择在图内完成。

同一 RuntimeSession 执行 `(B,S)=(1,1),(1,4),(2,3),(3,8)`，每次 **742 个 LLVM kernel 调用**，全部 **17 个输出**与动态 ONNX ReferenceEvaluator 比较，最大绝对误差 **8.46386e-6**。输出依次为 logits `[B,S,6400]`，以及 8 层各自的 K/V `[B,S,4,96]`。

同一产物另执行一次未来 token 扰动：把三个序列各自后四个 token 改为 `(id+13)%6400`。前四个位置的 logits 和全部层 K/V 逐元素完全一致，后缀 logits 确实改变。四次参考执行加扰动共 **3,710 次模型 kernel 调用**，Run 前后 primitive cache 的全部统计不变。六组非法完整模型输入在运行时分配和 kernel 提交前拒绝。

这是完整变长 prefill 的数值与执行验收。它仍使用 fresh-output plan；本报告不覆盖把有界输出接入持久 KV state、变长 decode、请求队列/合批、CUDA 或性能加速。权重来自固定随机种子的真实模型导出，不代表经过训练的语言质量。

## 使用的方法

### 复用 Gather、Sigmoid 和现有 FFN 路径

bounded producer 现在接受静态表上的数据 Gather。输出维度按数据轴前缀、索引张量维度、数据轴后缀组合；TE 使用输入的实际 B/S，避免从静态代表输出构造 shape。数据表所有维度仍须静态，axis、索引 dtype 和 rank 由既有 InferType 校验。Gather schema 升至 v2。

保留既有守卫语义：有效负索引加表长；超出 `[-N,N-1]` 的运行时索引返回相应 dtype 的零。LLVM 的惰性 Select 保证无效分支不会形成非法 load。ONNX 的有效索引域和输出形状规则见 [Gather 规范](https://onnx.ai/onnx/operators/onnx__Gather.html)；越界填零是 KXC 明确规定的扩展，不把它宣称为非法 ONNX 输入的数值对齐。静态导入的常量索引仍逐值校验并拒绝越界。

Sigmoid 已有 float32 InferType、TOPI 和 LLVM 实现，本次补齐 bounded 准入和形状透明证明；Sigmoid/Sqrt/Neg 的 source 不能携带额外属性。实际 FFN 按 `gate=xWg`、`up=xWu`、`down=(gate*sigmoid(gate)*up)Wd` 执行，再走现有 Add 残差；没有新增融合算子或另一套执行器。逐元素公式采用 [Sigmoid 规范](https://onnx.ai/onnx/operators/onnx__Sigmoid.html)。

### 证明位置表前缀，明确输入只提供形状元数据

实际 cos/sin 表固定为 `[128,96]`。源图保留 `Slice(table, [0], [S], [0], [1])`；producer 要求一个切片轴、start=0、step=1、end 为来自已知输入轴的直接 Symbol，并证明整个 S 上界不超过表容量。其他动态端点不进入该子集。一般端点归一化继续遵循既有静态 Slice；标准端点和轴规则见 [Slice 规范](https://onnx.ai/onnx/operators/onnx__Slice.html)。

准备后的 `slice` 使用 `{table, shape_anchor}` 两个输入，以及 `SliceAttrs.prefix_axis/extent_axis`。四个静态控制数组必须为空。InferType 和 TE 都检查轴、静态表、输出 rank/dtype/shape；snapshot 深拷贝并保存属性，canonical 序列化包含两个轴字段。Slice schema 升至 v3。调用方不能把两输入形式冒充为已证明的 source，仍必须由 producer 从五输入源图生成。

TE 只读取 table 的数据和 anchor 的 shape。沿用 `LowerTensorGraphToTIR`，增加内部的 metadata-only 输入清单；生产 lowering 从已验证的 Relay 调用推导清单。每项必须属于真实输入 ABI、是 placeholder、且未出现在 payload DAG 中。输入 B 虽未参与位置表输出 `[S,96]`，仍作为输入 buffer 形状的一部分保留在 extent ABI 和 `boundary_shape_extent` schedule identity 内。孤立 extent、缺失显式用途、外来 placeholder、元数据输入被读取 payload 都被拒绝。

Python 导入器对 Slice 后未知长度只读取 rank/dtype；Unsqueeze 复用同一个轴检查，Shape 只需要已知 rank。Sqrt/Pow/ReduceMean/Sigmoid 等保留 dtype/attrs 检查并把 extent 证明交回 C++。没有用代表长度 4 填充未知中间维。

### 消除完整模型触发的身份字节复制放大

首次整模型测试在 26.86 秒后被系统终止。调试栈定位到 `MintBoundedCompileRequest → MakeExactSpecializationRequests`：图身份包含 **275,598,940 字节**的规范内容，其中保留了原始权重字节；每个 unit request 复制 ShapeProfileKey，继而复制包含全图内容的字符串。742 个请求把模型级身份按 unit 数量重复存储，小型子图没有暴露这个规模问题。

在既有 GraphSemanticKey 和 ShapeProfileKey 中，将规范字节改为 `shared_ptr<const string>`。复制身份现在共享不可变存储，默认/移动状态、生命周期、digest、排序和完整字节比较保持原语义。没有改成只比较 hash，也没有新建 key 或缩减身份内容，所以规范字节和身份版本保持不变。C++ 对象存储布局变化，依赖这些头的对象文件需要重编译；默认和 bounded 构建均重新构建。

独立回归用 1 MiB 图身份生成 1,024 个 profile 副本，检查共享存储、原对象销毁后的有效性、独立构造的相同值、不同末字节的区分和排序。修复后的完整模型与身份专项测试通过；`/usr/bin/time -v` 记录峰值 RSS **4,512,816 KiB（约 4.30 GiB）**，两项测试总耗时 68.98 秒。该轮同时存在默认构建活动，这不是推理性能 benchmark。

### 身份与唯一所有者

bounded applicability **v10**、unit shape contract **v6**，preparation 保持 v3，ModuleInvocationContract 保持 ABI v4。准入、Gather/Slice schema 和元数据用途通过已有 compiler execution contract 与 schedule canonical 链进入身份。ShapeProgram/DimExpr 仍是唯一形状权威，ValueGraph 仍独占值编号，RuntimeSession 执行既有 plan 和 module。

## 验证与效果

| 验证 | 方法和结果 |
|---|---|
| 不依赖下载的入口/FFN | 静态 embedding `[7,4]`、位置表 `[8,4]`、4→6→4 SwiGLU 和残差；int32/int64 各四组 B/S，共 80 次 LLVM 调用；embedding/位置复制精确，double FFN 参考最大误差 1.9273e-8 |
| 索引边界 | 0、N-1、-1、-N、N、-N-1、INT32/INT64 最小/最大值；有效索引正确，无效值明确填零；准备后修改调用方 Gather axis 不影响产物 |
| Sigmoid | S=1/4/8，值域 `[-100,100]`，含 ±0；三次 LLVM 调用，double 参考最大误差 1.89347e-8 |
| 准备阶段反例 | 17 个反例覆盖容量上界、非零/负 start、非法 step/axis、伪造 prepared attrs/source、数据相关 end、动态表、错误 dtype 和无字段一元算子的额外属性；cache 不变 |
| 运行阶段反例 | 两种索引宽度各四个形状反例；完整模型另有六个越界 B/S、零长度、错误 rank/dtype 反例；均无新增 runtime alloc/submit，cache 不变 |
| ONNX 导入 | 新增包含 embedding、动态位置前缀和未知长度后续算子的 source 小图；五种后续算子的额外属性仍拒绝 |
| 完整模型 | 四次动态 ONNX 参考 + 一次未来 token 扰动；3,710 次调用，17 个输出，最大误差 8.46386e-6 |

三轮 Ponytail QA：第一轮复用普通 Gather/Slice/Sigmoid、已有 evaluator 和生产编译链；第二轮检查新字段的 InferType/TE/snapshot/canonical 消费及身份所有权，修复整模型暴露的不可变字节复制放大；第三轮执行独立数值、实际模型、失败前副作用、生成物与完整回归审计。没有测试专用执行器或新 IR。

最终回归：默认 CPU/LLVM **48/48 CTest**，bounded/控制流 gate-on **60/60 CTest**，Python **276/276**。bounded 全量明确启用完整 prefill、第一层 attention、RoPE、GQA、拆头和加权投影六个实际 fixture。旧负例将新准入的 Sigmoid 换为仍未开放的 Erf，保留“不支持算子必须拒绝”的检查；default gate-off 也单独复测通过。

Relay **37/37**、Pass **20/20** 合同及生成物 freshness 通过；include 检查 279 个文件，89 个安装头及 10 个实验头编译通过；文档索引覆盖 51 篇 Markdown；静态库 1,843 个 strong symbol 无重复，`git diff --check` 通过。fixture 重建后全部 **78 个文件 SHA 和 receipt 元数据完全一致**。当前模型 bundle 共有 14 个 RuntimeSession 预检失败 run，成功阶段统计见下表。

默认全量 43.56 秒，bounded 最终全量 97.30 秒。旧静态 MiniMind prefill/decode 及 ResNet18 可选数值分支本轮未启用，其先前证据保留；本轮完整变长 prefill 分支已执行。日志：`/tmp/kxc-m3-prefill-default-ctest.log`、`/tmp/kxc-m3-prefill-bounded-ctest.log`、`/tmp/kxc-m3-prefill-python-full.log`；最终完整 bounded LastTest 另存 `/tmp/kxc-m3-prefill-full-lasttest.log`。

实际 bundle 位于 `out/build/bounded-llvm/out/minimind_bounded_prefill_profile/events.jsonl`。已核对成功阶段：

| stage | 成功 Run | submit/exec 对数 | runtime alloc | 关联事件总数 |
|---|---:|---:|---:|---:|
| `prefill_embedding_ffn` | 8 | 80 | 240 | 408 |
| `prefill_sigmoid` | 3 | 3 | 6 | 15 |
| `minimind_bounded_prefill` | 4 | 2,968 | 8,864 | 14,804 |
| `minimind_bounded_prefill_future` | 1 | 742 | 2,216 | 3,701 |

模型运行关联 stage、batch、sequence、layers=8、导出 SHA、plan ABI 和 run_id；kernel 另有 call_index/symbol。alloc 包含结果与 extent 参数存储，不统计 kernel 内部 scoped malloc，不等同于峰值显存。能力矩阵 embedding_gather 刷新为实际 CPU 证据，CUDA 间接访问调度继续关闭；Slice/Concat 保留既有门禁并附加位置前缀证据。

## 输入和复现

模型沿用 MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba` 的既有导出：8 层、hidden=768、Q heads=8、KV heads=4、head_dim=96、intermediate=2432、vocab=6400、位置容量 128、随机权重 seed=0、opset 17。参考 token seed=20260909；ONNX 1.22.0、NumPy 2.5.3、LLVM 20.1.2。

原图 2,074 个节点，纯静态折叠后 1,338 个；二元化 Concat 后 1,474 个 Relay source 调用和 796 个源常量；C++ 证明/准备后 742 个 plan 调用。所有输入、参数、参考输出和 SHA 记录在 `out/fx_minimind_bounded_prefill/receipt.json`。

| 文件 | SHA-256 |
|---|---|
| 原始 `minimind_prefill.onnx` | `c468128df709d4c376e0ee4397e66f319b7cb76a033fad198737453eb486c286` |
| `prefill_dynamic.onnx` | `c52ef7d36dc5d37611341adf7d3e4ba0cc9255e665199acc1ec4712dd473a1d6` |
| `prefill_representative.onnx` | `ae2700aab4f8398e6bb3b409da1a823e0c38a5a6a15ce358c03d5f4d382efe7c` |
| `prefill.json` | `07e5933aa9d24956378a0140095c8e532234ae1eef134c244c879eac2ba78752` |
| `prefill.params` | `aa5b8c611c3ebc92ac42ab54cb941e773bdbef3307dd7f7b0572e287e742a134` |

```bash
OPENBLAS_NUM_THREADS=1 out/venv/bin/python python/tools/make_minimind_bounded_prefill_fixture.py \
  --onnx out/minimind_onnx_bounded/minimind_prefill.onnx --out out/fx_minimind_bounded_prefill
cmake --build out/build/bounded-llvm -j4
KXC_MINIMIND_BOUNDED_PREFILL_DIR="$PWD/out/fx_minimind_bounded_prefill" \
  ctest --test-dir out/build/bounded-llvm -V --output-on-failure --no-tests=error \
  -R '^minimind_bounded_prefill_llvm_test$'
```

未设置 fixture 环境变量时只执行内置数值和拒绝检查，并明确 SKIP 实际模型；本报告明确设置该变量并执行了完整分支。后续模块继续处理 bounded prefill 与 KV state/extent 的交接、变长 decode 和调度，不能由本轮的 prefill 结果直接推定已完成。
