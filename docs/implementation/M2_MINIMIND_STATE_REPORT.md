# M2 技术报告：MiniMind 的会话持有 KV 状态

> 状态：真实模型的 CPU/LLVM 状态绑定切片已完成（2026-09-08）。覆盖编译后的 prefill 输出初始化、同一 decode 会话连续追加、容量拒绝和模型 bundle。设备端更新、请求级批处理以及单个计划同时执行 prefill/decode 不在本次完成范围。

## 问题与结果

原来的容量型 decode 虽然能连续运行，但测试 driver 自己保存 K/V 数组、复制 present，并维护长度。现在由 `RuntimeSession` 分配、持有和更新 K/V；driver 每步只传 token、position、mask，取回 logits。长度通过 `StateExtent` 查询，缓存地址在整个 decode 循环中不变。

真实 8 层 MiniMind 的 prefill 和 decode 都经 ONNX importer → Relay → Compiler → LLVM → RuntimeSession。prefill 的全部 logits/K/V 与独立 ONNX 参考最大差 `8.82149e-06`；它实际计算出的 K/V 初始化 decode 会话后，连续 4 步的最大 logits 差为 `5.42402e-06`。有效长度从 16 增至 20，greedy token 序列逐步一致。

## 方法

`ExecutablePlan::BindStateOutputs` 把已编译静态计划中的 past 输入变为 `is_state`，把对应 present 输出变为私有、保留到状态提交结束的产物。`StateOutputBinding` 明确记录 state/source value ID、序列轴、source slot 和每次追加数量；没有按模型名维护第二张状态表。

独立版本 `kStaticStatefulExternalV1` 复用静态 kernel ABI。以本次布局 `[B,C,KV,D]=[1,32,4,96]` 为例，present 的序列长度为 33，新内容恒在下标 32。所有 kernel 成功完成后，runtime 只把这个新片段复制到旧 cursor 指定的位置，随后统一提交各层长度。kernel 对 state 的输入必须只读；alias/donation 及形状、容量、轴和角色冲突在执行前拒绝。

`InitializeState` 接收准备好的连续 prefill 输出和有效长度，复制有效前缀到会话自己的容量存储，保留无效区域的填充值。这是两个静态计划之间的显式初始化交接：prefill 和 decode 使用各自的计划，decode 后续各步始终复用同一份会话状态。没有借用 prefill 的裸指针，也不把 ONNX 参考 K/V 当作被测状态的生产者。

固定容量与有效长度分开、以及是否共享 past/present 存储的区别，参考了 [ONNX Runtime 的缓存说明](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)。本次仍分配 present 中间值；没有声明零复制或 past/present 共址。

## 契约与身份

- 旧的 `kDynamicStatefulV1` 仍通过 runtime extent ABI 和显式 alias 追加；fresh-output 模式继续拒绝持久状态。本切片没有改变这些合同。
- 新模式限定 CPU/LLVM。`RunAsync` 在返回前完成 CPU 状态复制，返回已完成句柄；这不是 CUDA 异步更新证据。
- 容量与参数错误发生在首个 launch 前，状态不变。开始执行后的失败不承诺回滚，会话进入不可继续状态。
- 私有 present 标记为 `is_async_live`，不能被内存复用覆盖；state storage、所有产物及模块由执行依赖保活。
- 绑定映射、extent 轴、source slot、append count 和 fill 进入 `PlanAbiFingerprint` v10，内存合同版本为 `static-external-stateful-session-state-v1`。运行 cursor 是数据，不进入编译身份。

三轮设计检查：A 复用 `ExecutablePlan`、`ValueTable`、`RuntimeSession` 和 Storage copy；B 没有第二个状态 owner，新增行为进入既有 identity builder；C 公共绑定/初始化 API 同时有真实 MiniMind、独立 LLVM 小图及执行前负例消费者。

能力矩阵只刷新 `kv_cache` 这一行：前端、Relay、lowering、LLVM、runtime 和 profile 标记为有明确 CPU 容量合同的实现；numeric 从参考/mock 口径改为真实编译运行证据。CUDA 保持关闭，检查器要求各格保留真实模型测试与本报告的证据链接。这一更新不能替代其余 11 行的独立验收。

## 验证与效果

复现真实模型：

```bash
PYTHONPATH=python out/venv/bin/python \
  python/tools/make_minimind_decode_loop_fixture.py \
  --onnx out/minimind_onnx_cap32 --out out/fx_decode_stateful --steps 4 --seed 0
KXC_MINIMIND_DECODE_LOOP_DIR=out/fx_decode_stateful \
  ./out/build/dev-ninja-cpu/minimind_decode_loop_llvm_test
```

| 检查 | 实测结果 |
|---|---|
| 真实 prefill | 650 个 LLVM kernel，全部 logits/K/V 最大差 `8.82149e-06` |
| 真实 decode | 683 个 kernel/步，4 步最大 logits 差 `5.42402e-06` |
| ownership/extent | 16 份 K/V 由 decode session 持有，地址稳定，长度 `16 → 20` |
| 无效区 | 填 `0.0` 与 `-1234.5` 的独立会话重放，logits 逐位相同；每步仍未写入的槽位保留 fill |
| 编译副作用 | decode 前后 primitive cache 统计完全相同 |
| 小图 LLVM | batch=2、容量=4、prefill extent=1 后三次追加，读回旧值、跨 batch 复制、容量边界与会话隔离通过 |
| 执行前负例 | 容量溢出与可写 state 输入的 fake launcher 计数为 0 |
| identity | 同合同确定；交换 state 的 source 或改变 fill 都改变 plan ABI |

回归：默认 CPU/LLVM 全量 48 项完成（初次两项测试/文档索引问题修正后复跑通过）；bounded gate-on 构建 **52/52**，包含原有 `kv_state_llvm_test`、动态 module ABI 与 shape-value 路径；Python **233/233**。NLP 检查器、生成物 freshness、include layers、公共头文件编译、文档链接和 `git diff --check` 通过；静态库重复 strong symbol 审计为空。

每步 16 次状态复制，每次复制 384 个 float32，共 **24 KiB/步**；runtime 不再经 host vector 读回并重写整份容量缓存。这是由布局与 copy 事件验证的复制量，未据此宣称端到端速度提升。

prefill bundle 有 650 对 submit/exec 和 1 个 runtime run；decode bundle 有 4098 对 submit/exec、96 次状态 copy 和 6 个 runtime run（4 步加两次哨兵重放）。分别有 1951、12396 条属于这些 runtime run 的事件，关联字段缺失数为 0。每条都携带 export receipt、stage、state extent 和从既有 builder 计算出的真实 plan ABI digest；关联键包含 trace/session/run，不能跨 bundle 只看 `run_id`。

## 剩余工作

本切片没有使任意形状的 prefill/decode 合并为单个计划，也没有移除容量型 attention 对 mask/position 的依赖。下一步继续 M3 的有界形状与注意力、逐格能力矩阵，以及 M6 的状态兼容热替换；请求排队、动态合批和 CUDA 状态更新仍需独立实现与报告。
