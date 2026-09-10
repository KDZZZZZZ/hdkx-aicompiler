# CUDA 有界 KV 状态接入技术报告

2026-09-10。状态：生产状态准入、完整模型 consumer、GPU 数值和设备复制顺序验收已完成。本报告复用同一状态 owner，不把编译成功或无设备跳过视为硬件验收。

本报告中较早的本地回归段落保留设备离线时的交接记录；“Windows GPU 设备证据”段落是本轮最终结论，覆盖同一源码与 fixture 的真实重建和执行。

## 达到的效果

既有 `kBoundedStatefulExternalV1` 现在允许同一 CUDA 设备上的 float32 状态、紧凑前缀和追加源。固定容量 KV 仍由 `RuntimeSession` 持有；同一个已编译 decoder 接收本次实际 P，产生 P+1 的私有输出，再将追加片段提交到会话缓存。没有增加模型状态表、形状求值器、运行时编译或新的 IR。

完整八层 GPU 模型的验证入口复用 [现有 consumer](../../test/minimind_bounded_decode_llvm_test.cpp)。Windows RTX 4070 Ti SUPER 实际通过四组 B/P、两类 greedy 循环、持久 KV 和 prefill→state 交接；共享 CPU 路径继续作为独立回归，不替代设备证据。

## 方法与合同

原状态实现已经使用 `CopyStateRange → StorageCopySync` 处理任意 extent 轴外的各行。真正需要扩展的是 [计划校验](../../src/runtime/executable_plan.cc)和 [module/plan 准入](../../src/runtime/session.cc)：

1. state、prefix input 和 produced source 必须全为 float32，且设备完全相同；只接受 CPU:0 或 CUDA。不同 CUDA ordinal 也不能混用。
2. module backend 必须与设备对应。原地、state-sourced extent 的另一种动态状态模式继续保持其原有边界。
3. GPU 请求批处理继续明确拒绝；它现在在自己的合同中检查全部 value 为 CPU:0，避免随着 unbatched state 的扩展被意外开放。

采用 ONNX Runtime [past/present 缓存说明](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)中的 past 长度与 present 增长关系。KXC 保留固定容量状态与私有 present 输出，通过显式复制维护有效区；这里没有引入共享 past/present 存储。

执行顺序沿用已有 owner：先校验 caller 输入、共享 B/P 和容量；将各 `[B,C,...]` 的有效行紧凑打包为 `[B,P,...]`；在调用者 stream 上运行全部 kernel；等待最后一个完成句柄；把每行的新片段复制回容量缓存；最后提交有效长度。`RunAsync` 返回时状态提交已完成，下一步可以换另一条 stream。运行失败继续传播并沿用会话 poison 规则。

这是同步提交的状态接口。它没有新增 GPU 状态排队器、跨会话共享或跨 worker 一致性，也没有给出吞吐加速结论。

## 身份与三轮实现审查

- A：直接复用 `StateOutputBinding`、`ExecutablePlan`、`RuntimeSession`、prefix buffers 和设备复制；没有新 registry 或 evaluator。
- B：既有 v11 bounded state ABI 已包含设备、物理容量、绑定和填充值；kernel 参数与内存分配算法未改变。因此本次不重新定义 ABI。CUDA 证明继续使用前一模块的 Pass v8/backend v8。
- C：真实 consumer 通过 `CompileBounded → RuntimeSession::RunAsync` 执行，CMake 独立注册 CUDA 数值与 CUPTI 门禁；合成设备正负例、CPU 真模型和无设备跳过分别记录，不混用证据。

## GPU consumer 要求通过什么

| 路径 | 覆盖范围与检查 |
|---|---|
| 普通 decode | `(B,P)=(1,0),(1,1),(2,4),(3,8)` 的全部 logits/16 KV；同一产物再执行四步外部 K/V greedy |
| 持久状态 decode | 四组 B/P 与普通结果逐位比较；实际 GPU prefill 输出初始化缓存，再执行 P=4..7 的四步 greedy |
| 状态所有权 | 六个小图会话覆盖 B=1/2/3 和两种哨兵，30 次追加；容量地址固定，保留的状态和首轮输出在会话析构后仍可读取 |
| stream 与错误 | 两条非默认 stream 交替；state RunAsync 必须返回 ready；错误输入、CPU 输入、CPU stream、层间 P 不一致和容量溢出在提交前拒绝 |
| 数值 | consumer 计数要求 692,224 个独立参考值；完整模型阈值 5e-5；请求批处理的 CPU 原入口继续运行 |
| 设备活动 | decode bundle 必须关联 16×774=12,384 个 kernel；seed prefill 单独关联 742 个 kernel |
| 复制顺序 | 每次 decode 的 934 个 uint64 extent DMA 完成后才能启动各自 kernel；持久状态共 336 次行复制，前缀在首个提交前完成，追加在全部 kernel 后进行 |

上表是已落地的验收要求，尚不是 GPU 实测结果。[CUPTI 检查器](../../test/cuda_profile_bundle_test.py)还要求 13 个无提交错误，并在真实事件上执行 13 种损坏检查，拒绝错误 shape/ABI/receipt、scalar、复制大小、父事件、完成事件和提交顺序。前缀复制与追加复制合计预期 1,622,016 字节；extent DMA 为 14,944 次、119,552 字节。

## 本地验证记录

| 检查 | 结果 |
|---|---|
| plan/session 定向测试 | 2/2，0.08 秒；含 CPU、CUDA:0、CUDA:1 准入及混合设备/请求批处理拒绝 |
| 共享完整 decode CPU 测试 | 通过，129.55 秒；四组 B/P 最大误差 1.56164e-5，greedy 6.61612e-6，实际 prefill → state → greedy 8.9407e-6 |
| Windows CUDA consumer | 通过：fresh/state/greedy、两 stream、16 cache、CUPTI 关联和 13 个负例 |
| 默认 CPU 全量 | 55/55，148.30 秒 |
| adaptive CPU 全量 | 55/55，216.36 秒 |
| bounded CPU 全量 | 70/70，226.62 秒；实际模型与 fixture 均未跳过 |
| 契约与架构 | Relay 40 项、Pass 20 项、include 284 个文件；三套构建的公共头检查通过；诊断测试 4/4 |
| 链接与旧证据 | CPU 1,603 / CUDA 1,894 个 strong symbol，无重复；旧四组 GPU bounded 记录仍通过更新后的关联检查 |

状态与普通输出仍逐位相同；请求合批仍为 7 个请求步合成 5 次运行，3,870 次提交，对照独立执行为 5,418 次。CPU 事件实际记录四组 owned decode 的 208 次状态复制和四步 greedy 的 128 次复制，合计 1,622,016 字节。GPU 检查器的每个 call/extent 数还与实际生成 CUDA 签名逐一核对：18 个静态 call、578 个单 extent call、178 个双 extent call，每次共 934 个参数。

定向日志在 `out/windows-gpu/logs/` 下：`bounded-state-cuda-admission-{build,ctest}.log`、`bounded-state-cuda-consumer-{ctest,lasttest}.log`；最终 CUDA consumer 重建为 `bounded-state-cuda-consumer-final-rebuild.log`。三套全量日志为 `cpu/kxc-bounded-state-cuda-{default,adaptive,bounded}-{build,ctest,lasttest}.log`，逐模型检查见 `bounded-state-cuda-cpu-regression-audit.json`。两份 seed/greedy 比较同时检查实际值和参考值有限，避免非有限参考被最大误差聚合忽略。

本地可交接源码为 `out/windows-gpu/bounded-state-cuda-source.tar.gz`，逐文件哈希为同目录 `bounded-state-cuda-source-manifest.json`；这些是设备恢复前的历史交接包。当前 Windows 结果见下方设备证据段落，源码和结果仍未同步到 GitHub。

decode fixture 已独立打包并核验 228 个 consumer 文件（279,064,099 字节，排除两个仅供导出的 ONNX 大文件）。`bounded-decode-fixture.tar.gz` 为 258,407,694 字节，SHA256 `cb7f6d9442e15a78633aca4d2505890cb923461c893e97d59865eccc0a595f27`；逐文件 manifest SHA256 为 `6f6e763e77b3fafccf0b90e31037944183ad62282bf3a629148c99ed394e7d75`。该包尚未上传 Windows。

## Windows GPU 设备证据（2026-09-10）

在隔离目录 `F:\kxc-gpu\20260910-goal` 使用 Release 配置重建后，完整 bounded decode consumer 输出：`fresh_runs=8 state_runs=8 calls=774 streams=2 reference_values=692224`；fresh 最大误差 `1.56164e-5`，greedy `6.19888e-6`，prefill→owned-state→greedy `8.58307e-6`。状态 cache 地址稳定，容量拒绝零分配/零提交，16 个 cache 的 extent 从 4 增至 8。

设备 bundle 由 CUPTI checker 逐项核验：12384 个 decode kernel、14944 次 8-byte extent DMA（119552 bytes）、336 次 state copy（1622016 bytes）、13 个 zero-submit rejection 和 13 个损坏证据反例；seed prefill 同 bundle 742 kernel。`cuda_state_runtime_test` 与 `request_batching_cuda_test` 另以 0.52s/0.60s 通过，验证 CUDA 状态准入和小图请求队列。最终标记为 `[PASS] bounded_minimind_decode_profile_verified` 与 `[PASS] cuda_profile_correlation_verified`。

## 复现与剩余工作

启用 bounded CUDA 及其前置 shape gates，设置 `KXC_MINIMIND_BOUNDED_PREFILL_DIR` 与 `KXC_MINIMIND_BOUNDED_DECODE_DIR` 后运行：

```sh
cmake --build out/build/bounded-cuda --parallel 2 --target minimind_bounded_decode_cuda_test
ctest --test-dir out/build/bounded-cuda --output-on-failure --no-tests=error \
  -R '^minimind_bounded_decode_cuda_test$'
```

Windows 对应增加 `--config Release` / `-C Release`。fixture 必须来自实际锁定导出；缺 fixture 或设备返回 77。默认注册使用 600 秒上限，同时要求数值结束标记与事件关联校验，原生进程仅返回 0 不足以通过。

GPU 请求批处理和状态热替换的完整设备验收分别见 [请求批处理报告](GPU_REQUEST_BATCHING_REPORT.md) 和 [热替换报告](M6_BOUNDED_REPORT.md)；跨设备状态迁移、分布式一致性和性能预算仍不在本证据范围。

2026-09-10 后续：GPU 请求批处理的设备验收已完成，见 [后续报告](GPU_REQUEST_BATCHING_REPORT.md)；本文保留本阶段的状态 owner、复制顺序和边界。
