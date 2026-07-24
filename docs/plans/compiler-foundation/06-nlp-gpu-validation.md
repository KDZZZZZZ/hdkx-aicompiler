# 06：NLP / GPU 验收轨道

> **状态：** 规划中；不是 compiler foundation 的基础依赖，也不表示当前支持 Transformer  
> **所属路线：** [编译器基础路线图](README.md)  
> **历史输入：** [编译器基础架构基线审查](../../COMPILER_FOUNDATION_BASELINE_REVIEW.md)；本文为迁移计划归档。
> **范围：** 以 feature-gated Transformer 负载验证 shape、region、plan/runtime 与 CPU/LLVM/CUDA 路径；不实现代码。
## 1. 定位与基本原则
本文件不是“先支持 NLP 才能建设基础架构”的依赖倒置。04 的 structured control flow/shape
契约和 05 的 frozen task plan/runtime 可先以 synthetic fixture 与 fake executor 演进；06 可
同时准备 reference、fixture、profile schema 和 capability gate。功能只有在对应基础契约和
端到端证据同时满足时才能开放。
当前正式边界仍是静态 shape MVP：ONNX importer 支持有限 node，未知非 batch 维曾被暂定为
`1`；正式 Relay operator 数与 CUDA 证据也不足以代表 Transformer。未知/symbolic dim 必须
保留为 binding/constraint 或明确拒绝，绝不能静默 concretize 后宣称动态 NLP 支持。
验收优先验证正确性边界，而不是用“新增 op 数量”或单一 kernel benchmark 宣布成功：
稳定 softmax、mask、valid extent、KV cache alias/lifetime、prefill bucket、decode capacity/page、
dynamic batching，以及 CPU/LLVM 与 CUDA 的可复现证据都必须闭环。
## 2. 目标、非目标与验收模型
### 2.1 目标
1. 建立 Transformer support matrix：frontend、Relay type/shape、lowering、CPU/LLVM、CUDA、
   runtime、数值与 profile 的每项状态均可追溯。
2. 用 prefill 与 decode 两类 workload 验收 exact/bucket/plan variant 的 shape 和内存契约。
3. 将 KV capacity/page、valid context、alias 与 lifetime 作为一等测试对象。
4. 用 dynamic batching/长度分组验证请求级 route，而不让 RuntimeSession 承担服务控制面。
5. 建立性能、正确性、观测三类硬门禁，提供 CPU/LLVM、CUDA 和端到端证据。
### 2.2 非目标
- 不把 06 作为 04/05 的硬前置条件，也不等待全部 Transformer/ONNX opset 才验证基础接口。
- 不承诺训练、反向传播、通用 attention kernel、量化、beam search、分布式 serving 或所有 GPU。
- 不使用 unknown dim -> `1`、`-1` sentinel、fuzzy capacity 命中或未 masked padding 伪造成功。
- 不把 profiler 单次采样、无 reference 的吞吐数字、或仅 CPU pass 当 CUDA/端到端支持证据。
- 不让 RuntimeSession 按 token、模型名或 op 名决定 dynamic batch、compile 或 cache policy。
### 2.3 最小验收模型
| 层次 | fixture | 必测维度 |
|---|---|---|
| 算子微测 | embedding/gather、batched matmul、mask、stable softmax、norm、slice/concat | dtype、mask、极值、tail、layout |
| attention block | Q/K/V、mask、softmax、projection | `B,S,H,heads`、causal/padding mask |
| prefill | 可变 `S` 的一层/小型 Transformer | bucket、padding、有效 token、峰值内存 |
| decode | token=1 + past KV | context、KV capacity/page、alias、增长/回收 |
| 服务模拟 | 请求队列与长度分组 | dynamic batching、排队、routing、隔离 |
所有 fixture 都应有确定性 seeds、独立 CPU reference 和可导出的 shape/profile manifest；真实模型
仅是补充，不得替代可最小化的失败用例。
## 3. 数据结构草案与冻结接口
以下协议由 06 消费；它不自行定义 runtime 私有行为。
```text
TransformerProfile {
  batch, sequence, hidden, heads, head_dim, dtype, layout;
  past_kv_length, kv_capacity, kv_page_size, causal, padding_lengths;
}
WorkloadManifest {
  model_revision, fixture_id, target, driver, runtime_abi;
  compiler/pipeline/backend fingerprints; shape_profile; seeds; tolerance;
}
ValidationRecord {
  capability_gate, selected_variant, logical/physical/valid_extent;
  reference_error, output_hash, KV alias/lifetime trace;
  compile/cache/queue, latency, throughput, peak_memory, profile_artifacts;
}
BatchRequest { request_id, profile, arrival_time, deadline, input_lengths; }
BatchGroup { compatible_dispatch_key, members, padded_profile, policy_reason; }
```
`TransformerProfile` 绑定的是逻辑语义；bucket 后的 physical capacity 与 valid extent 必须另记。
例如 `S=97` 运行 `[B,128,H]` physical buffer 时，manifest 必须记录 `seq < 97` tail/mask
规则。KV 的 logical context、allocated capacity 和 page layout 也必须分开；capacity 绝不能
被误作当前有效 context。
`WorkloadManifest` 包含 target/driver/ABI/compiler/pipeline/backend fingerprints。artifact 或
profile 不匹配必须 fail closed。`ValidationRecord` 必须能定位到 05 的 plan variant/task 与
04 的 control-flow/shape gate，但不包含 compiler 内部对象地址或 graph-local id 作为 cache key。
## 4. Transformer op/backend 缺口矩阵
每个单元必须分开标为 `unsupported`、`contracted`、`implemented`、`validated`；只有最后一项
可对外宣称支持。当前状态以权威审查为准，不因本计划而改变。
| 能力 | 需要的语义契约 | CPU/LLVM 证据 | CUDA 证据 | 首要负例 |
|---|---|---|---|---|
| embedding / gather | index dtype/range、layout、OOB policy | reference + lowering | kernel + device test | negative/OOB index |
| batched matmul | rank/broadcast、transpose、accum dtype | varied B/S/H | varied B/S/H | rank-2 假设泄漏 |
| stable softmax | max-subtraction、mask、all-masked 行、precision | extreme logits | extreme logits + tail | NaN/Inf/overflow |
| normalization | epsilon、axis、accum dtype | fp32/fp16 cases | fp32/fp16 cases | tiny variance |
| mask/select | causal/padding valid extent | mask shapes | masked tail | padded token leaked |
| slice/concat | axis、bounds、alias/copy | shape cases | shape cases | overlap/empty slice |
| KV cache | capacity/page、valid context、alias/lifetime | decode trace | decode trace | stale/overwritten KV |
| copy/event | device/stream/event order | CPU no-op semantics | H2D/D2H/compute | missing wait |
不支持的 op/backend 必须由 capability verifier 在 partition 前拒绝，并附带缺失项；不允许
frontend 静默替代、host fallback 或把失败算子跳过以令 fixture “通过”。
## 5. 硬依赖、软依赖、可并行任务与集成点
### 5.1 硬依赖
- 正确性运行：operator contract、同源 type/shape rule 与 lowering 公式、exact shape binding。
- bucket/prefill：logical/physical/valid-extent、tail-safe kernel、ShapeProfile/dispatch gate。
- decode/KV：显式 capacity/page、alias/effect、dependency-aware memory lifetime。
- dynamic batching：冻结 PlanVariant、compatibility/dispatch key、请求边界控制面。
- CUDA 声明：目标 GPU、driver、ABI、backend fingerprint 与设备数值/性能 artifact。
### 5.2 软依赖
- 04 的 If/loop 仅在模型/adapter 实际使用数据控制流时需要；attention/pre-fill 可先纯 dataflow。
- 05 的 fusion/library region、多 stream、多 device 能提升性能，但 exact per-Call CPU/LLVM
  验收不等待它们。
- CompileCoordinator、持久 artifact、AOT metadata 对服务重启/compile cost 验收有用，不阻塞
  第一轮数值 fixture。
- production tokenizer、真实大模型权重和网络服务不是最小 fixture 的依赖。
### 5.3 可并行任务
| 工作包 | 可并行方式 | 产出 |
|---|---|---|
| reference/fixture | 不依赖 compiler 成功 | inputs、golden outputs、manifest、negative cases |
| support matrix | 审核现有 contract/backend | feature gates 与缺口报告 |
| profiler schema | 消费 05 冻结字段或 mock trace | ValidationRecord/报表断言 |
| CPU/LLVM path | exact static profile | 数值、IR/compile evidence |
| CUDA path | 独立 target fixture | device output、trace、kernel evidence |
| dynamic batching simulator | fake PlanVariant dispatcher | group policy、padding/queue evidence |
| 04/05 integration tests | feature-gated | control-flow/KV lifetime、task/event assertions |
### 5.4 集成点
06 从 04 消费 shape binding、control-flow capability 和 effect/alias trace；从 05 消费
frozen plan variant、task/event/memory profile 与 RuntimeSession 的静态执行结果。06 只通过
版本化 manifests、feature gates 和 profile events 接入；若接口尚未完成，fixture 以 mock/fake
executor 运行并明确标为 `pending`，不可制造支持结论。
## 6. 分阶段验收步骤
### Phase A：先建证据与门禁
1. 为每个 fixture 写 manifest、reference、tolerance、seed、目标/版本指纹与 expected gate。
2. 将 support matrix 连接 OperatorSpec、type/shape、lowering、backend、runtime 和测试。
3. 对未支持动态维、op、layout、target、KV/page、event 形成稳定负例，而非隐式降级。
4. 按 exact static CPU reference 建立 baseline，区分 test harness 成功与 compiler 成功。
**测试：** manifest 完整性、reference determinism、错误 gate、错误 fingerprint、未知 dim
未绑定、all-masked softmax、空/边界 sequence。此阶段可完全不依赖 GPU 或真实 executor。
### Phase B：CPU/LLVM 正确性闭环
1. 以 exact `B,S,H` profiles 打通 op 微测、attention block、prefill 和最小 decode。
2. 比较 fp32 及声明支持的低精度误差；stable softmax 必测大正/负 logits、all-masked 行和 tail。
3. 验证 output logical shape、padding crop、mask 以及 KV valid context；每步 decode 比较 cache
   内容与 reference。
4. 保存 compiler pipeline、LLVM/target、plan variant、kernel/signature 和 output hash 证据。
**测试：** 多 B/S、zero/one/token tail、causal/padding mask、KV grow/reuse/page boundary、
不同 layout、重复 Run 和非法 capacity/alias。任何 mismatch 必须最小化为 op/shape/plan 级 fixture。
### Phase C：prefill bucket
1. 选择有限 sequence bucket，例如由 profile policy 而非“最近更大 kernel”规则决定。
2. 对每个请求记录 logical `S`、physical bucket、valid token count、padding 比例和 selected variant。
3. 证明 kernel 对 tail 做 mask/predicate，输出 crop 后与 exact reference 相同。
4. 量化 unique artifact 数、compile time、cache hit、P50/P95 latency、tokens/s 与峰值内存。
**测试：** 每个 bucket 边界、bucket-1/bucket/bucket+1、短序列落入大 bucket、masked reduction、
attention tail 和 guard 域外拒绝。没有 tail-safe 证明的 bucket 一律 gate off。
### Phase D：decode 与 KV capacity/page
1. 固定 token=1 请求，逻辑 context 每步递增；物理 KV capacity/page 由显式 profile 分配。
2. 在 plan/trace 中验证 K/V 写入位置、read valid context、page crossing、alias、retire 和输出。
3. capacity 满时仅允许经过验证的 grow/repage/copy task；不能覆盖仍 live 的 KV 或以 capacity
   伪装 logical length。
4. 报告 decode latency、tokens/s、KV peak memory、page waste、copy/event wait 与每 token compile。
**测试：** 初始空 cache、capacity-1/capacity/capacity+1、page boundary、不同 batch、请求取消、
KV reuse、stale read、in-flight generation 保活。每 token 重编译或无效 context 均为失败。
### Phase E：dynamic batching 与服务级路由
1. 在 RuntimeSession 之上的控制面按 dispatch compatibility、dtype/layout、deadline、长度和
   KV 状态分组；不兼容请求不得强行拼 batch。
2. 记录队列等待、group reason、padding、selected PlanVariant、fallback（仅已验证 variant）和
   cancellation；无 variant 时等待/编译/报错，绝不 fuzzy 执行。
3. 先用 fake dispatcher/replay trace 验证公平性、隔离和 profile，再接真实 plan variants。
4. 比较 fixed batch、length-grouped 和动态 batch 的端到端 latency/throughput/内存/compile 代价。
**测试：** 混合长度、deadline、不同 dtype/layout、bucket 边界、队列饱和、cancellation、
compile singleflight 与负缓存。request 间 KV、storage、artifact generation 必须隔离。
### Phase F：CUDA 与端到端性能证据
1. 对每个可开启 CUDA gate 记录 GPU 型号、driver、CUDA/backend、ABI、clock/power 条件与命令。
2. 保存设备数值比较、kernel/stream trace、copy/event、compile/cache、peak memory 和失败日志。
3. 分别报告 CPU/LLVM 与 CUDA；不以 CPU reference 或单 kernel 时间替代端到端 prefill/decode。
4. 性能结论同时给出 latency、throughput、compile cost、peak memory、padding/page waste 与可用的
   硬件计数器/采样证据，标明样本数与噪声处理。
**测试：** clean target、错误 target/driver/ABI 拒绝、冷热 cache、bucket mix、decode 长序列、
copy/compute overlap 与数值回归。没有 target-匹配 artifact 和设备执行记录时 CUDA gate 保持关闭。
## 7. 硬门禁与 Done 条件
### 7.1 正确性门禁
- 所有开放 fixture 有 reference、固定输入、tolerance、logical/physical/valid-extent 与 output
  shape 断言；stable softmax 的极值/all-mask 不能 NaN/Inf 泄漏。
- prefill bucket 的每个 tail/crop/mask 与 exact 一致；decode 每步 KV content/context/page/alias
  与 reference 一致，且无越界、stale read 或非法 reuse。
- unsupported op/shape/backend/target/ABI 必须 fail closed，不能通过 silent fallback 获得绿灯。
### 7.2 性能门禁
- 用同一 manifest 比较 baseline 与候选，报告端到端 latency 分位数、throughput、compile/
  queue/cache、peak memory、padding/page waste；不得只报告最快 kernel。
- bucket/dynamic batch 的收益须大于明确的 padding、等待与编译成本；没有收益可保持 exact gate。
- CPU/LLVM 与 CUDA 的结论分别具备可复现命令、版本指纹、原始 profile 和目标设备证据。
### 7.3 观测门禁
- 每次 run 可关联 capability gate、fixture/manifest、dispatch/plan variant、logical/physical/
  valid extent、task/device/stream、artifact generation、compile/cache 和 memory lifecycle。
- profile 能区分 compile storm、bucket miss、padding、KV growth、copy/event wait、kernel time 与
  queue time；缺字段即不开放对应服务优化。
### 7.4 完成定义
1. support matrix 的每项开放能力同时具备 CPU/LLVM 与声明 CUDA target 的数值证据，或明确
   标为 target-specific，不允许泛化宣称。
2. 固定/可变序列 prefill、decode/KV 和 dynamic batching 通过上述三类门禁，且 symbolic dim
   从 frontend 到 plan 不曾被静默改为 `1`。
3. 06 的 fixture/profile 可在 04/05 未完成时以 feature-gated mock 维护；真实集成只接受
   冻结接口，不迫使 RuntimeSession 越界承担编译或路由。
4. 失败案例、原始证据和回退 gate 与成功结果同等可追溯；这条验收轨道始终不成为基础架构
   的阻断依赖。
