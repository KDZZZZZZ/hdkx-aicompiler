# CUDA 请求批处理接入技术报告

2026-09-10。状态：CUDA 准入、显式 stream、请求生命周期、完整模型数值和 CUPTI 设备活动验收已完成。

## 达到的效果

原有请求进入、单步排队、等长同形合批、退出取消和 KV 槽位复用现在可用于同一 CUDA 设备。固定容量缓存仍归 `RuntimeSession` 所有，每个请求的提交长度独立保存；运行期不编译、不补 padding，也不混合不同 P。新增 `RunNextBatch(stream, metadata)` 允许调用方选择同设备 stream，返回前完成 kernel、KV 追加和长度提交。

完整八层 MiniMind 的共享 CPU 验证已经通过：三个 KV 槽位先后服务四个请求，七个请求步合为五次模型运行，提交次数为 **3,870**，独立执行为 **5,418**。每个请求的 logits 和全部 16 份 KV 与独立执行逐元素精确相等。这是执行次数的减少，尚不是 GPU 吞吐或延迟提升的结论。

## 方法与边界

沿用 [CPU 请求批处理](M2_REQUEST_BATCHING_REPORT.md)和 [bounded CUDA 状态](GPU_BOUNDED_STATE_REPORT.md)的现有 owner。`AdmitRequest` 初始化空闲槽位，`EnqueueRequest` 保存输入快照；`RunNextBatch` 选择最早请求及其相容同行，将有效前缀收集为紧凑输入。全部 kernel 完成后，追加段散写回各自物理槽位，最后提交各请求长度。退出的旧编号不能复活，返回输出和诊断副本不暴露可复用的槽位。

采用 [Triton sequence batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#sequence-batcher)中“有状态请求序列必须保留所属实例”的不变量。KXC 将它落实为同一 RuntimeSession 中的稳定请求槽位；没有引入 Triton、后台调度器、超时策略或跨会话状态共享。

生产变更集中于以下位置：

- [计划校验](../../src/runtime/executable_plan.cc)：请求计划所有 value 必须位于同一 CPU:0 或同一 CUDA ordinal。保留行独立性声明、共享 batch guard、统一容量/追加数和非 batch 状态轴检查。
- [运行时](../../src/runtime/session.cc)：普通运行与批处理共用 stream 校验；显式批处理 stream 在队列为空时也验证，并在合批分配/复制前拒绝非法 stream。原 metadata-only 接口继续使用默认 stream。
- [公共接口](../../include/kxc/runtime/session.h)：新重载要求显式 metadata 参数，保留旧 `RunNextBatch({})` 的源兼容性。初始状态及排队输入必须先完成其上游设备生产，接口才可同步复制。

同步提交不推断跨 stream 的未完成输入依赖。所有请求方法保留并发/重入拒绝；开始提交后的执行失败沿用会话 poison 规则，不允许读取部分写入的状态。

## 身份与三轮实现审查

1. A：复用 RequestBatchingContract、request slots、LeadingRow、CopyStateRange、StorageCopySync 和现有完成句柄；没有第二张 KV 表或 GPU 请求执行器。
2. B：既有请求计划 ABI v12 已记录设备、物理槽位、最大批量、绑定和 FIFO/等长规则。stream 选择不改变 kernel 参数、产物数学语义或内存规划，因此不升级 ABI。CPU 正例仍验证有/无批处理及不同最大批量的身份差异。
3. C：共享 CPU/CUDA consumer 从生产编译进入真实 RuntimeSession。小图覆盖快照、排队取消、容量、重入、并发、提交后失败、axis 2/C=2 和释放图/cache 后执行；完整模型对比全部输出。硬件缺失只使设备入口返回 77，不替换为 CPU 执行。

## 验证记录

| 检查 | 已取得的结果 |
|---|---|
| 计划/会话/请求队列/完整模型 CPU 定向回归 | 4/4；完整模型 137.23 秒 |
| 最终共享小图 CPU 请求测试 | 通过，0.24 秒；未定义 stream 在空队列和有队列时均在分配/复制前拒绝 |
| CUDA 小图源码/PTX | 两组各 3/3 个原语；axis 1/C=1 与 axis 2/C=2，均 0.54 秒 |
| CUDA consumer 构建 | 小图与完整模型均构建成功 |
| 实际 GPU 入口 | 小图请求、完整模型请求和 bounded decode/state 均通过 |
| CPU 请求事件审计 | 12 个模型 run 的分组 receipt 正确；五次合批内 192 次状态行复制，合计 540,672 字节 |
| 请求管理复制审计 | 图执行之外 159 次 Storage 复制、737,400 字节，覆盖初始前缀、入队快照、合并输入和诊断副本；CPU 记录通过 5 种损坏拒绝 |
| 完整模型 CUDA/PTX 回归 | prefill 742/742、decode 774/774，2/2 通过，共 99.51 秒；无 GPU 执行 |
| 默认 CPU 全量 | 55/55，147.92 秒 |
| adaptive CPU 全量 | 55/55，216.53 秒；实际 MiniMind 热替换 fixture 已执行 |
| bounded CPU 全量 | 70/70，232.14 秒；完整模型 fixture 均未跳过 |
| 契约、架构与诊断 | Relay 40/40、Pass 20/20、生成物 freshness、include 284 个文件；三套公共头检查、文档 78 篇、诊断 4/4 通过 |
| 强符号审计 | 默认/adaptive/bounded CPU/CUDA archive 分别 1,604/1,716/1,884/1,894 个，无重复定义 |

完整模型 CUDA 门禁复用 [consumer](../../test/minimind_bounded_decode_llvm_test.cpp)与 [CUPTI 检查器](../../test/cuda_profile_bundle_test.py)，独立使用 batching bundle。它要求五个合批 run 和七个独立参考 run，全部 9,288 个 kernel 一一关联，11,208 个 uint64 extent DMA 在各自 kernel 前完成，192 次状态行复制遵守前缀先于首个提交、追加晚于最后 kernel、提交完成先于接口返回的顺序。四个首次参考请求共核对 99,328 个 ONNX 参考值，CUDA 容差为 5e-5。

请求编号和顺序必须为 `1,3 → 2 → 4 → 2,4 → 3`，相应 P 为 `4 → 0 → 0 → 1 → 5`。图执行之外的 159 次 Storage 复制也必须逐条关联实际 D2D、同步完成和复制大小；每个合批 run 前须已完成对应数量的 token 行复制。此处计数不包含没有 Storage copy_id 的原始主机字节上传/下载，不据此声称整个请求延迟已被 graph span 覆盖。

GPU 验证器另对真实 bundle 做 17 种损坏检查，包括伪造请求编号/批量/P、错误 ABI、丢失复制完成、管理复制和错误提交顺序；本轮真实 bundle 的 17 项均被拒绝。数值 fixture 使用锁定随机权重的完整架构，验证编译执行一致性，不评价预训练模型质量。

能力矩阵的 CUDA 请求项可升级为 bounded hardware validated；任意 ragged、跨设备请求和吞吐/延迟提升仍不在本报告范围。

## 复现与后续

```sh
cmake --build out/build/bounded-cuda --parallel 2 --target \
  request_batching_cuda_test minimind_bounded_decode_cuda_test minimind_bounded_cuda_codegen_test
ctest --test-dir out/build/bounded-cuda --output-on-failure --no-tests=error \
  -R 'request_batching.*cuda|request_batching_axis[12]_cuda_codegen'
```

完整模型入口要求设置 `KXC_MINIMIND_BOUNDED_DECODE_DIR`，沿用原锁定 decode fixture。CPU 可运行 `minimind_bounded_decode_llvm_test --batching` 独立输出请求证据；普通 CPU 全量测试仍执行原完整模型场景。Windows 命令增加 Release 配置。

## Windows GPU 设备证据（2026-09-10）

隔离目录 `F:\kxc-gpu\20260910-goal` 的 Release 构建使用 RTX 4070 Ti SUPER/CUDA 12.9，并在 bounded decode fixture 通过 SHA-256 校验后执行。完整请求 consumer 通过 `[PASS] full_minimind_bounded_cuda_request_batching_no_runtime_compile`：3 个 KV 槽位服务 4 个请求，7 个 request step 合并为 5 个 batch，3870 次 CUDA submit（独立执行 5418 次），99,328 个参考值全部核对，logits/16 KV 逐位相等，最大 ONNX 误差 `1.20401e-5`。

CUPTI checker 实测 12 个 run、9288 个 kernel、11208 次 extent DMA（89664 bytes）、192 次 state copy（540672 bytes）、159 次管理复制（737400 bytes），管理 DMA 顺序验证为 true；17 个篡改反例全部拒绝。最终标记为 `[PASS] bounded_minimind_requests_profile_verified` 和 `[PASS] cuda_profile_correlation_verified`。同轮 `request_batching_cuda_test` 0.60s 通过，axis 1/2 codegen 也分别通过。

本轮日志位于 Windows 隔离构建的 `bounded-request-evidence` bundle；源码和结果尚未同步到 GitHub。状态热替换、跨设备状态迁移、分布式批处理和吞吐/延迟预算仍需独立实现或验收。

本地交接包为 `out/windows-gpu/bounded-request-cuda-source.tar.gz`，逐文件身份见同目录 `bounded-request-cuda-source-manifest.json`；CPU 日志、独立请求 bundle 与 CUDA/PTX 编译证据收录于 `bounded-request-cuda-local-evidence.tar.gz`。最终核对与包哈希见 `logs/bounded-request-cuda-progress-audit.json`，不覆盖上一模块的 704 文件快照。三套完整回归日志使用 `logs/cpu/kxc-bounded-request-cuda-{default,adaptive,bounded}-*` 前缀，逐模型核对见 `bounded-request-cuda-cpu-regression-audit.json`。
