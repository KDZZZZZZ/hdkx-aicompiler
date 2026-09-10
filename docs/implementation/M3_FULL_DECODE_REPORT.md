# M3/M4 技术报告：完整八层 MiniMind 的变长 decode

日期：2026-09-09。范围：CPU/LLVM，原始动态 decode 导出、外部 K/V 输入和 fresh-output plan。

后续已接通会话持有的 bounded KV 状态与实际 LLVM prefill 交接，见 [M2/M3 状态报告](M2_BOUNDED_STATE_REPORT.md)。本文保留 fresh-output 切片当时的验证记录与收据。

## 达到的效果

原始八层 MiniMind decode 已从 ONNX shape-source 导入、经过 C++ 有界形状证明、编译为 **774 个普通 LLVM kernel 单元**。输入为 int64 `input_ids[B,1]` 和 16 个 float32 `past_[kv]_[0..7][B,P,4,96]`；输出为 logits `[B,1,6400]` 和 16 个长度 P+1 的 present。

一份产物、同一个 RuntimeSession 执行 `(B,P)=(1,0),(1,1),(2,4),(3,8)`。全部 17 个输出与动态 ONNX ReferenceEvaluator 对齐，最大绝对误差 **1.56164e-5**；16 个 past 前缀在每个 batch 内逐元素保持原值。四组运行共 **3,096 次 LLVM kernel 调用**，没有按长度重新编译或访问 primitive cache。

连续 decode 的测试还将 ONNX prefill 的 K/V 与末位置 logits 作为起始外部输入，然后每一步使用实际 LLVM logits 计算 greedy token，并把实际 LLVM present 传入下一次 Run。四步 P 从 4 依次增长到 7，输出长度增长至 8；四步共 3,096 次 LLVM 调用，全部 17 个输出最大绝对误差 6.61612e-6，greedy token 为 5059、4840、4840、1763。

这次完成的是原始变长 decode 的图与执行链。RuntimeSession 当前仍使用 fresh-output plan，测试驱动持有外部 K/V；持久容量、有效长度、原地更新以及从实际 LLVM prefill 的状态交接继续由 M2 的既有 owner 接通。没有把外部数组循环计作持久 KV 状态完成，也没有新增模型专用状态引擎。

## 使用的方法

### 证明 P+1 和 total-current

Python shape-source 模式允许保留 int64 Sub，同时继续拒绝默认静态模式的 int64 Sub。Add/Sub 不接受额外属性；新增的属性反例发现 Add 原先会静默丢弃额外属性，现在在共有转换入口明确拒绝。

C++ resolver 对形状控制允许一个表达式与非负常量相加，或对已证明的 `Symbol+constant` 做不引入负常量项的消去。实际 mask 控制为 `(P+1)-1=P`。普通数据 Sub、两个动态表达式相加、P-1、运行时数据相关控制仍拒绝。

`ProveNonnegativeConstantOffset` 由现有 DimExpr 求常量项，再用完整规范结构相等验证 `expression == base + offset`。求值点本身不构成证明。该函数同时被 shape-value 编码和 unit 维度投影复用；没有再建 shape evaluator。

折叠控制优先保留原数据来源。减法消去后，若原来源只有 P+1 轴而不能直接表示 P，就按原函数参数顺序寻找能够表达整个控制向量的输入轴；只有规范等值证明成功才重新关联。实际 P 会关联到已有 past 输入。它仍是单个数据源上的 shape_expr，范围和所有输入间的 B/P 相等关系继续来自 GraphTemplate。

shape_expr 及共用的目标形状属性增加 kind=2：输入 0 的指定轴加非负常量，常量存于 expr_values。InferType、TE、快照和动态 unit value-expression 投影均消费该表示，已知长度相加检查 int64 溢出，符号值在整个上界范围检查溢出。生成的 LLVM 计算实际 P+1，未把代表长度写入控制张量。

### 在图内读取位置窗口

原图的 cos/sin 表固定为 `[128,96]`，decode 选择 `Slice(table,[P],[P+1],[0],[1])`。resolver 证明一个切片轴、step=1、start 是直接输入轴符号、end-start 是非负常量，并检查 `P_upper + count ≤ capacity`。

沿用 Slice 的两输入 prepared 形式，增加 `window_size`：-1 仍表示原有 `0:extent` 前缀，非负值表示 `extent:extent+window_size`。四个普通静态控制数组必须为空。InferType 和 TE 核对表容量、轴、输出 dtype/rank/shape；输出循环固定为 window_size，源地址加实际 extent。shape anchor 继续只提供元数据，既有 lowering 检查其 ABI 归属、extent 用途和 payload 不被读取。Slice 的一般轴与端点定义见 [ONNX Slice](https://onnx.ai/onnx/operators/onnx__Slice.html)；本子集通过整个范围证明避免动态端点钳制。

### 把派生长度带入 GQA、mask 和 Softmax

真实 decode 暴露了原先只查看直接 Symbol 下界的限制。范围比较现在调用 DimExpr 在原符号上下界求值，所以能证明 P+1 与 -1 不相等，也能在 P 可为零时证明 Softmax 的归约长度 P+1 至少为 1。

固定商的 Reshape -1 推导允许消去规范等值、下界严格为正的完整维度因子。例如 P+1 可消去，剩余 head 因子的商仍必须是正的静态整数。P 自身下界为零时仍不能据此消去；任意动态商没有进入支持范围。

mask 的 `(P+1)-1` 控制生成 `[1,P]` 的零前缀，再与当前 token 的 mask 拼接。现有 ConstantOfShape、Concatenate、Trilu、MatMul、Softmax 和 GQA 路径组成完整 attention；没有新的融合算子、IR 或执行器。

### 身份版本

bounded applicability **v12**，unit shape contract **v8**；Slice schema **v4**、ShapeExpr/ReshapeDynamic/ExpandDynamic schema **v2**、ConstantOfShape schema **v3**。Slice canonical 属性和快照包含 window_size；kind=2 的轴与 offset 进入已有 attrs 和 unit 合同身份。preparation 仍为 v3，bounded schedule 仍为 v3，ModuleInvocationContract 仍为 ABI v4，后者复用已有 Add 表达式。

## 验证与效果

| 验证 | 结果 |
|---|---|
| 无下载窗口/控制图 | C=0/1/2，各四组 B/P，合计 72 次 LLVM 调用；位置窗口、KV 拼接、P+1 控制和 total-current mask 全部精确 |
| 小图反例 | 三条图各四次运行拒绝和六次准备拒绝；覆盖容量、rank/dtype、未证明的减法/双动态加法、数据 Sub、伪造 prepared Slice；拒绝无 cache/alloc/submit 副作用 |
| 原始完整图 | 四组 B/P，774 calls/Run，3,096 次调用；全部 17 个输出最大误差 1.56164e-5，past 前缀精确 |
| 连续 greedy | 四步 P=4..7，3,096 次 LLVM 调用；实际 logits 选 token、实际 present 反馈下一步；全部 17 个输出最大误差 6.61612e-6，past 前缀精确 |
| 完整图反例 | 九组 B/P 范围、不同层 P/B 不一致、current 长度、rank/dtype 和输入数量的 preflight 检查通过；均零 alloc/submit，cache 不变 |
| Python | 280/280 通过，含新增四个导入/属性反例 |
| 默认 CPU/LLVM | 48/48 CTest 通过，44.94 秒 |
| bounded CPU/LLVM | 62/62 CTest 通过，168.34 秒；本次显式启用全部七组实际模型 fixture |
| 合同与架构 | Relay 37/37、pass 20/20；include 279 个文件、headers 89+10 编译、docs 53 篇通过；1,846 个强全局符号无重名，diff check 通过 |
| fixture 复现 | 独立重生成 230 个文件（含 receipt），逐文件 SHA-256 相同 |
| 能力矩阵 | decode_external_kv 的七个非 CUDA 格按本次 CPU 执行刷新；累计八行已审计，其余四行继续核对 |

生产测试为 [minimind_bounded_decode_llvm_test.cpp](../../test/minimind_bounded_decode_llvm_test.cpp)。完整图通过环境变量显式开启；不提供 fixture 时只运行无下载小图，不能据此声称模型验证通过。生成器为 [make_minimind_bounded_decode_fixture.py](../../python/tools/make_minimind_bounded_decode_fixture.py)。

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 out/venv/bin/python \
  python/tools/make_minimind_bounded_decode_fixture.py \
  --onnx out/minimind_onnx_bounded/minimind_decode.onnx \
  --out out/fx_minimind_bounded_decode
KXC_MINIMIND_BOUNDED_DECODE_DIR="$PWD/out/fx_minimind_bounded_decode" \
  ctest --test-dir out/build/bounded-llvm --output-on-failure \
  --no-tests=error -R '^minimind_bounded_decode_llvm_test$'
```

完整回归还显式启用已有 prefill、投影、拆头、GQA、RoPE、第一层 attention 六组 fixture。最终日志为 `/tmp/kxc-m3-decode-bounded-ctest-final.log`、`/tmp/kxc-m3-decode-default-ctest.log` 和 `/tmp/kxc-m3-decode-python-final.log`；完整 CTest 收据另存为 `/tmp/kxc-m3-decode-{bounded,default}-final-lasttest.log`。事件包位于 `out/build/bounded-llvm/out/minimind_bounded_decode_profile/events.jsonl`，按 stage、B/P、step、导出 receipt、plan ABI、run_id 和 kernel 关联。完整模型两种 stage 分别记录 3,096 对 submit/exec、4 次成功 Run 和 6,832 次分配事件；包含小图时共 6,264 对 submit/exec。它们提供数值和执行证据，不构成性能加速基准。

## 模型和产物来源

输入来自既有 MiniMind 导出：8 层、hidden=768、8 个 Q heads、4 个 KV heads、head_dim=96、FFN=2432、vocab=6400、位置表容量=128，opset=17。源图 2,184 个节点，纯静态折叠后 1,423 个节点，导入后保留 1,559 个 Relay 源调用和 821 个常量。权重来自固定种子的真实模型构造，不代表训练后的语言质量。

| 产物 | SHA-256 |
|---|---|
| 原始 decode | `c0733d61304970a37fff14fec8d56336baf945576398a0f5e0d14d8acd1d5fb2` |
| 动态 decode | `158ea389d1a332ee51df9901459c387034fd6b1eed4436fcbcd4740f9fba5fb9` |
| B=1/P=4 代表图 | `95bdca64b15d01ba881fc6f0fa5f0ec84984bfde6bddf4bff9908feda3586750` |
| decode.json | `e7f8d5c48211070f554a95c35823a5c90f2a3f439151cb2878d1e9974b2905f9` |
| decode.params | `df993ef65b9fcdb930b2cc158bbd60ca392d651a9884fd4d56985d66e22b6534` |

`receipt.json` 记录 229 项产物哈希、输入输出顺序、范围、参考程序、prefill seed 来源和 greedy 步骤。生成器不编译、调度或执行 KXC kernel。

## 三轮 Ponytail QA

1. **复用检查**：使用已有 DimExpr/ShapeProgram、shape_expr 属性、Slice、Concat 和 RuntimeSession；控制算术在 producer 内证明后折叠，无 int64 数据 Sub kernel 或新状态引擎。
2. **权威与身份检查**：图输入仍只绑定直接 Symbol/Const；来源变换须有完整等值证明，window/offset 有真实 consumer 和 canonical 版本，代表数据不能代替范围证明。
3. **执行与边界检查**：实际 ONNX 图、空 past、多 batch、全部层输出、连续 LLVM 反馈、非法输入、属性拒绝和无隐式编译均进入测试；持久状态交接、CUDA 与请求级 batching 保持未完成状态。
