# M1：CUDA 复制关联与异步保活技术报告

2026-09-09。Windows RTX 4070 Ti SUPER 上，Storage 复制的 CUPTI DMA 活动现在能通过唯一 `copy_id` 连接到主机提交和完成记录。真实 H2D → D2D → CUDA ReLU → D2H 数据链与三种受控未完成复制均已执行，**1,028 个 float32 结果完全一致**。本模块补齐 `copy_event.cuda` 的局部硬件证据，执行仍使用原有 DeviceAPI、AsyncOperation 和 RuntimeSession。

## 问题与实现方法

[CPU 复制观测](M1_COPY_EVENT_REPORT.md)已能把 submit/complete 配成一对，[CUDA 模型关联](M1_CUDA_CORRELATION_REPORT.md)也能把设备 kernel 对应到实际调用。但 Storage 复制只有调用者关联，没有在实际 DMA 提交时激活自己的 ID，因而同一个 run 中的多次复制不能独立对应 CUPTI 活动。

本次在 [ExecutionObserver](../../include/kxc/runtime/execution_observer.h)增加默认实现的 `OnCopyScopeEnter`，复用已有同步作用域/退出动作。它接收已有 CopyInfo，不生成运行时事件 ID。旧派生类默认转到通用 OnScopeEnter，可保持源码兼容；已编译的 C++ 自定义 observer 需要重新构建。

[StorageCopySync/Async](../../src/runtime/device_stream.cc)完成范围、方向、stream、重叠校验后才进入此作用域。原地同范围 no-op 在此前返回。profiling 适配器为实际提交建立 `copy_launch`，使用其 span_id 作为 copy_id，并由既有 ActivationScope/CUPTI external correlation 把设备 DMA 指向该 span。复制的字节数、端点及异步属性都来自原有 CopyInfo。

适配器只在当前同步作用域的 TLS 栈保存 copy_id 和进入作用域前的关联。提交/完成继续指向原调用者，避免把新建的 copy_launch 错当成调用者。完成回调按值捕获关联和 ProfileContext，在另一个线程/另一个活动 profile 下完成也不会读取当时的 TLS。同步完成另分配自己的 span_id，避免与 launch 重复。

| 记录 | 时间含义 | 连接方式 |
|---|---|---|
| copy_launch | host_submit，实际提交所在主机作用域 | span_id=copy_id，parent=原调用者 |
| copy，phase=submit | host_submit，异步 API 返回后的提交点 | fields.copy_id=launch ID，parent=原调用者 |
| copy，phase=complete，同步 | host_execute，runtime 测得的主机执行区间 | 独立 span_id，同一 copy_id/调用者 |
| copy，phase=complete，异步 | host_observed_complete，提交点至主机观察到完成 | 独立 span_id，同一 copy_id/调用者 |
| cuda_memcpy | device_execute，CUPTI 实际 DMA 时间 | parent_span_id=copy_id；沿 launch 找调用者 |

设备复制可能在异步 API 返回之前就已开始或完成，因此时间校验以 copy_launch 起点为下界，以主机观察完成为上界，允许既有独立时钟采样的 1 ms 容差。不能要求 DMA 晚于 submit 点，更不能把等待观察的时间当成 DMA 耗时。

schema_version 仍为 1，新增事件和 copy_id 使用已有扩展字段。kernel ABI、计划身份、内存计划版本及编译缓存不变。runtime 没有包含 profiling 头，也没有新增同步、event owner、状态表或隐式跨 stream 依赖。旧诊断器只消费明确的 host_execute copy，新的 launch/设备/等待区间不会重复计入主机复制量。

## 真实 consumer 与独立校验

[C++ consumer](../../test/cuda_copy_profiling_test.cpp)使用普通 Compiler（CUDA 既有 level 3 流水线）、NDArray 与 RuntimeSession：

- 数据链：257 个 float，CPU/GPU 各有带 4 字节偏移的连续视图，前后各保留保护值。H2D 在 producer stream，显式 Wait 后由 consumer stream 执行 D2D、ReLU、D2H。输入源/view 和运行输出的原始引用提前释放，数值与两端保护值保持正确。另一次同步 D2H 验证 GPU 保护值及同步事件。
- 前置拒绝：shape 不符、错误 stream device、异步和同步的 GPU 重叠区间、源范围越界，共五项；同 Storage 同范围是已完成 no-op。精确活动计数证明这些路径没有产生额外复制 launch 或 DMA。
- 三种完成路径：在真实非默认 stream 中，用测试专用 cudaLaunchHostFunc 挡住后面的 D2D。首次 IsReady 必须为 false；原始 Storage、observer、ProfileContext 引用均释放后，回调计数仍为 0，两个外部 Storage 的 deleter 尚未触发，ProfileContext 仍存活。
- 随后另一个线程分别执行 Wait、stream 完成后的 IsReady、最后句柄析构。该线程活动的是另一份关闭 CUPTI 的 profile。由主线程独立放行后，复制结果各比较 257 个 float；观测恰好结算一次，抛错回调不影响数值或其他回调，晚注册立即执行，重复 Wait/IsReady 不重复结算。完成后回调释放 profile；Storage 保留至完成句柄释放，两个 deleter 各执行一次。

测试门闩只使用 mutex/condition_variable，不在 CUDA host callback 中调用 CUDA API；等待有 10 秒失败上限，析构先放行并同步，避免测试悬挂。读回结果发生在普通完成观察线程。依据 NVIDIA [cudaLaunchHostFunc 合同](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EXECUTION.html)及[异步复制同步行为](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/api-sync-behavior.html)：pageable 主机内存的 Async 接口可能同步，因此受控 pending 用 D2D；本模块不把 H2D/D2H 返回快慢当成正确性条件。

[Python 验证器](../../test/cuda_profile_bundle_test.py)读取实际 Bundle，要求完整执行标记及 CUPTI available，验证每个 DMA 与 copy_launch、submit/complete、原调用者一一关联，核对端点、字节数、stream、正设备时长和时钟范围。每份 Bundle 注入九种破坏：空证据、错误 run、parent、时钟、重复活动、字节数、copy_id、缺完成、错误计时域，均必须拒绝。编译期常量快照沿用 copy_kind 语义，不冒充 Storage copy_id。

四份 GPU Bundle 共 **7 条关联 DMA、13 个主机 copy 点、7,204 个完成字节**；数据链有 4 次复制/4,120 字节，三个生命周期场景各 1 次/1,028 字节。三份 worker profile 没有混入复制或 DMA。CPU/LLVM 的实际复制/ReLU/复制用例也新增 copy_launch 与原调用方 parent 的配对验证。

## 回归与复现

Windows 专项 **11/11、12.78 秒**通过，包含新的复制验收、完整 MiniMind prefill 数值、1,300 条 kernel 关联及双 profile 并发测试。模型的 401,408 个输出元素最大绝对误差仍为 `1.001358032e-5`。复制 Bundle 的 150 条事件全部 status=ok；7 条 DMA 时长合计 6,945 ns，三个受控 pending 的 host_observed_complete 分别约 20.7–20.9 ms。这些延迟由测试门闩引入，不能用于复制性能比较。

| CPU/LLVM 完整回归 | 结果 |
|---|---|
| 默认配置 | 55/55，144.25 秒 |
| adaptive | 首轮 54/55，213.31 秒；唯一失败是新报告尚未加入文档索引，补齐后单独复查 docs 1/1，7.15 秒 |
| bounded | 70/70，228.60 秒 |

adaptive 的代码、数值和模型测试在完整回归中均通过；没有把首轮文档失败记成一次全绿运行。三种配置的实际视觉、单图联合模型和文本 decode 用例均核验未跳过，adaptive 还执行真实模型热替换，bounded 执行完整变长 prefill/decode 与请求批处理。可选历史模型 fixture 的 SKIP 不计入这些证据。

40 个 Relay operator、20 个 pass 及生成物新鲜度、284 个 include 边界、90 个 installed/10 个 experimental header 独立编译、71 篇文档索引与链接检查通过。9 种矩阵篡改被拒绝，包括移除新 consumer/验证器/报告、伪造 validated 和擅自开放 decode/state/batching CUDA。4 项 Python 诊断测试通过；7 份非空 GPU Bundle 经现有 schema 和公开离线诊断入口读取，3 份 worker Bundle 确认没有混入记录。等待场景没有被诊断成复制主机耗时异常，数据链的 copy_dominance 仅依据真实同步复制。CPU/CUDA 静态库分别 1,603/1,614 个强定义，均无重复。

源码 overlay `out/windows-gpu/copy-correlation-final.tar.gz` 包含 17 个本轮文件，SHA-256 为 `225ab3e042d117a7fe52d86dd5d9ecfd215a7567d268159723c4b2e35b72e9c7`。两端完整 694 文件通过逐项 SHA-256 核验；报告收尾另同步 metadata，完整清单为 `copy-correlation-source-manifest.json`。原始 Windows 日志和 Bundle 在 `out/windows-gpu/copy-correlation-evidence/`，归档 SHA-256 为 `d7fbd8bcfc295331f78d0a23341f25f4d08088733b02a4016f1b256061d5b835`。最终复制 Bundle 为 `build/cuda-profile-evidence/copy-ol2_hbul/`，其中 chain/events.jsonl SHA-256 为 `83f21fc1b85055331e6573893585dcc12a3b7eac8e21d4bf37dadedf2f991359`。

CPU 原始回执在 `out/windows-gpu/logs/cpu/kxc-cuda-copy-*`；schema/诊断、CPU 实际模型、矩阵反例、强定义和最终汇总分别见 `out/windows-gpu/logs/copy-correlation-*-audit.json`。

```sh
cmake --build <cuda-build> --config Release --target cuda_copy_profiling_test
ctest --test-dir <cuda-build> -C Release -V -R '^cuda_profile_copy_test$'
python3 python/tools/check_nlp_gpu_validation.py --root .
```

CTest 使用 Python 标准库验证器，每次在 `<cuda-build>/cuda-profile-evidence/copy-*/` 新建目录，不覆盖已有证据。设备不存在返回 77/Skipped；程序静默退出或缺最终成功标记不能通过。测试不需要外部模型 fixture。完整 prefill 与并发 kernel 关联仍通过已有测试单独验收。

## 三轮 QA 与限制

A：复用 CopyInfo、作用域钩子、ProfileContext ID 和 AsyncOperation，新增的仅是观测关联。B：没有第二套 runtime owner；copy 的异步资源和一次性完成仍由原句柄管理，TLS 只存同步提交事实，析构恢复深度。C：生产 Storage API、真实 CUDA kernel、受控未完成 DMA、跨线程完成、异常回调和回执篡改均有执行证据，CPU/LLVM 保留回归。

NVIDIA live catalog 本轮前两页可读（200/350），第三页返回 WAF 202 空正文，未据不完整目录声称不存在合适技能，也未安装技能；采用官方 CUDA 12.9.1 API 文档和已安装 SDK。设备仍为 Windows CUDA 12.9/驱动 576.57；Linux CUDA 仅提供编译证据。

这是局部正确性和观测完整性证据，没有复制加速或模型吞吐结论。受控 pending 证明单 stream D2D 生命周期；跨 stream 数据链使用显式 Wait，没有引入自动 event 依赖或异步队列调度。原始外部指针的 CopyFromBytes/CopyToBytes 不拥有 Storage 复制生命周期，不自动获得本次 copy_id 合同。GPU decode/state、bounded CUDA、长期 CUPTI 存储上限、时钟漂移与 GPU 热替换性能决策仍需后续模块。既有 Compute Sanitizer 环境限制未解决，不声称内存插桩通过。
