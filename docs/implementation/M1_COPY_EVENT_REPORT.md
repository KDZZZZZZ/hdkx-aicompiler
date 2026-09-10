# M1 复制完成观测与诊断技术报告

2026-09-09 后续：Windows GPU 的 DMA/copy_id 配对、偏移数据链、真实 pending 与跨线程完成已通过，见 [CUDA 复制报告](M1_CUDA_COPY_REPORT.md)。以下保留本轮 CPU 实施时的设备边界。

日期：2026-09-09。范围：StorageCopyAsync → AsyncOperation → Profile Bundle → 离线诊断的现有执行链。

## 解决的问题与效果

CPU 的异步复制接口实际上同步完成，runtime 已测得它的执行耗时，但 profiling 适配器原先忽略该值，记录的是复制之后的观测开销。没有 RuntimeSession run 关联的异步复制还只记录提交，不安装完成回调。请求入队、状态初始化及图前后复制因此可能缺少完成记录。

本次修复这两处，并让提交和完成携带同一个 `copy_id` 与提交时确定的 run/parent 关联。回调能在观测器被释放后完成记录；完成后释放其 ProfileContext 引用。即使提交时另一个 profile 正在活动，也不会借用它的 run id。

实际 CPU 路径已验证：从带偏移的输入视图异步复制四个 float，经过真实 LLVM ReLU，再异步复制结果；输入 `[-2,3,-4,5]` 得到精确的 `[0,3,0,5]`。两次 Storage 复制共 **32 个完成字节、两组提交/完成配对**，LLVM kernel 实际提交并完成一次。错误复制形状在复制与事件产生前被拒绝。

诊断器现在只把成功、完成、标明 `host_execute` 的事件用于主机复制耗时判断。延迟到 Wait/IsReady 才观察到的 `host_observed_complete` 区间不再被当作设备执行耗时；提交点和错误记录也不计入复制量或耗时。短 kernel 规则改为消费真实 `kernel_exec`，不再用旧 module-run 名称推测 kernel 数量。

## 方法与边界

实现只扩展现有 [profiling 适配器](../../include/kxc/profiling/runtime_observer.h)和 [诊断引擎](../../python/kxc_agent/services/diagnosis_engine.py)。`CopyInfo.duration_ns` 继续由 runtime 测量；AsyncOperation 仍是唯一完成句柄及异步依赖 owner，没有新 stream、event、状态表或同步机制。

| 记录 | 计时与关联方法 |
|---|---|
| 同步复制 | 已有主机执行耗时；新增 copy_id 使用既有 ProfileContext span id |
| CPU 异步接口 | 提交点仍是瞬时事件；立即完成的 span 使用 runtime 实测 duration，排除 profiling 回调开销 |
| 尚未观察完成的句柄 | 只记录提交；已有 Wait/IsReady/析构结算回调时记录完成，区间标为 host_observed_complete |
| RuntimeSession 外复制 | 不要求新建 run；若同一 ProfileContext 有显式活动区间，提交时捕获它；否则保持空关联 |
| 另一 profile 活动时 | 不继承外部 profile 的区间；短暂使用既有 ActivationScope 显式记录，随后恢复调用方上下文 |

回调按值捕获上下文、复制量、端点、copy_id、run/parent id 和 metadata，不读取回调触发时的线程局部关联。事件 schema 仍为 1，copy_id 位于原有可扩展 fields 中；kernel ABI、计划身份与编译缓存不变。没有在观测器中增加 Wait、Sync 或设备查询。

[NVIDIA Driver API 的 Event 定义](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)区分完成查询/等待与事件间设备计时。本次没有添加设备计时；主机后来观察到完成的时间包含观察延迟，不能称为 DMA 时间。NVIDIA 技能目录首页可读取，但分页 2 连续返回 HTTP 202 空正文，因此本次直接以官方 API 文档作为参考。

## 验证与三轮 QA

- QA A：复用既有 CopyInfo、AsyncOperation、ProfileContext 和完成回调，修复实测耗时丢失与空 run 漏记；没有新增 runtime 执行策略。
- QA B：[C++ 测试](../../test/runtime_profiling_test.cpp)以给定的一纳秒事实精确检查 duration 转发，验证未完成时不提前记录、无 run 复制、不同 profile 隔离、调用方活动上下文恢复、回调保活及完成后释放。这些人为句柄用例验证回调合同，不代表 CUDA 执行。
- QA C：同一测试另跑真实 CPU 复制/LLVM/复制数值链；[Python 测试](../../test/diagnosis_engine_test.py)通过公开 analyze_bundle/compare_bundles 入口检查提交不重复计数、错误/未知计时不参与性能判断、晚观察不构成复制性能退化、当前 kernel_exec 词表与旧 module/观察区间的区别。生成的真实 LLVM Bundle 也经过 schema 和配对/字节检查。

源码冻结后，三套完整 CPU/LLVM 回归与四项新增 Python 诊断测试均通过：

| 完整检查 | 实际结果 |
|---|---|
| 默认 CPU/LLVM | 51/51，41.51 秒 |
| adaptive，含实际静态 MiniMind fixture | 51/51，115.76 秒；两代各 650 次调用，17 个输出逐位一致 |
| bounded，含八组显式模型 fixture | 66/66，257.20 秒；完整 prefill/decode、状态交接、greedy 与请求批处理复跑 |
| Python 全量 | 284/284，1.06 秒 |
| Relay / Pass 合同与生成物 | 38/38、20/20，生成物 freshness 通过 |
| 架构与记录 | 90 个安装头、10 个实验头独立编译；284 个文件 include 扫描；60 篇文档链接、NLP 矩阵/指纹与 diff 检查通过 |
| 强定义 | 三套 CPU 构建与新增 CUDA 构建的静态库均无重复强 C++ 定义 |

真实 prefill 的 25,290 条 Bundle 事件还通过了公开诊断入口的 schema 检查，当前 kernel_exec 记录能被短 kernel 规则消费。这个 Bundle 同时包含编译和预期拒绝用例，不能直接当作纯推理延迟基准。

完整回归使用与 [全 mask 报告](M5_MASKED_SOFTMAX_REPORT.md)相同的三套配置和八组 fixture，完整命令也见该报告。本地最终回执为 `/tmp/kxc-copy-event-final-{default,adaptive,bounded}-lasttest.log`；检查器、Python 与 CUDA 专项分别记录在同前缀的 checks/python 日志以及 `/tmp/kxc-copy-event-cuda-lasttest.log`。临时日志是本轮回执，长期复现仍以代码、合同、导出 fixture 和命令为准。

## CUDA 的实际结果

独立构建 `out/build/copy-event-cuda` 已启用 CUDA 12.9 与 LLVM，四个编译器/运行时测试二进制均成功编译、链接。device_runtime_test、runtime_profiling_test、cuda_schedule_test 通过；其中 device_runtime_test 明确跳过设备路径。codegen_cuda_test 的源码发射、非法/未绑定 TIR 拒绝通过，进入硬件阶段时以 `CUDA device is unavailable` 失败；后续 NVRTC、加载和启动检查未执行。**这套 CUDA 专项为 3/4，通过的部分不能代替硬件执行门禁。**

实际设备探针：`nvidia-smi` 返回 Driver/library version mismatch，内核模块为 580.159.03，用户态库为 580.173.02；直接调用 `cuInit(0)` 返回 `804 CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`。本次没有更换驱动或修改系统库。CUDA DMA、跨 stream 完成及设备数值仍未验证，矩阵继续关闭该格。

## 同步刷新 prefill 与能力矩阵

`prefill_exact` 原记录仍写“本地 LLVM 不可用/只验证参考”，已经落后于实际状态。本次按 [完整变长 prefill 报告](M3_FULL_PREFILL_REPORT.md)、[静态热替换报告](M6_RUNTIME_REPORT.md)和 [状态交接报告](M2_BOUNDED_STATE_REPORT.md)刷新：真实导入、生产原语、完整 logits/16 KV、静态及有界执行和模型 profile 均有各自证据。这里没有新增另一条 prefill 实现。

`copy_event` 的前端/Relay/lowering 格明确标为 runtime API 范围，不虚构 copy Relay 算子；LLVM 格仅说明真实 LLVM 数据链消费了复制结果，复制本身由 DeviceAPI 执行。12 行矩阵完成当前证据审计，仍包含 contracted/unsupported 格，也不代表全部 PROJECT_GOAL 已实现。

## 复现

```sh
cmake --build out/build/dev-ninja-cpu --target runtime_profiling_test -j2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure -R '^runtime_profiling_test$'
PYTHONPATH=python out/venv/bin/python -m pytest -q test/diagnosis_engine_test.py
python3 python/tools/check_nlp_gpu_validation.py --root .
cmake -S . -B out/build/copy-event-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_LLVM=ON -DLLVM_DIR=/usr/lib/llvm-20/cmake \
  -DKXC_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build out/build/copy-event-cuda --target kxc_runtime device_runtime_test runtime_profiling_test codegen_cuda_test cuda_schedule_test -j2
ctest --test-dir out/build/copy-event-cuda --output-on-failure \
  -R '^(device_runtime_test|runtime_profiling_test|codegen_cuda_test|cuda_schedule_test)$'
```

真实数据链的 Bundle 在 `<build>/runtime_profiling_output/copy_event_llvm/`；另含编译期的空常量快照记录，统计上述两次数据传输时按 copy_id 选择 Storage 复制，再只计算 complete 的 bytes。修复提高观测完整性与诊断准确性，没有宣称复制加速或模型吞吐提升。
