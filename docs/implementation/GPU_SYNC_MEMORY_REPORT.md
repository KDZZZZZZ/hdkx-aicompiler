# CUDA 同步内存完成语义修复报告

2026-09-09。已修复 CUDA 同步复制与清零 API 提前返回、后续非阻塞 stream 可能读到旧数据的问题。Windows 最终专项中的独立内存检查通过 **96 次精确数据核对**；该问题是在多阶段归一化专项中由原有 float64 sum 测试暴露的。

## 问题与方法

`NDArray::CopyFromBytes`、`StorageCopySync` 以及 `ZeroData` 承诺调用返回时完成操作，但 [CUDA DeviceAPI](../../src/runtime/device/cuda_device_api.cc) 原来直接调用 cudaMemcpy/cudaMemset 后返回。[NVIDIA 的同步行为说明](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) 明确区分：pageable H2D 可能只完成 staging，D2D 不等待宿主，device memset 也是异步提交。[非阻塞 stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html) 不继承 legacy 默认流的同步顺序。

因此同步上传之后在新建 stream 启动 kernel，并不能仅凭 cudaMemcpy 返回就推断输入已就绪。失败回执中 float64 sum 第 0 项为 `0`，独立参考为 `-0.78378378378378388`。仅修复下面的底层完成语义后，原有 float32/64、标量及二维 sum/max 全部通过；之后才继续执行新的归一化用例。

修复在 `CopyDataSync` 的 cudaMemcpy 和 `ZeroData` 的 cudaMemset 后调用 `cudaStreamSynchronize(nullptr)` 并检查错误。只等待承载这些操作的默认流；没有改成设备全局同步，异步复制入口仍由其 stream/event 完成对象表达依赖。API 签名与 Storage/NDArray/RuntimeSession 的所有权不变。

## 验证与效果

[device_runtime_test](../../test/device_runtime_test.cpp) 在所有分配和 stream 创建完成后，交替产生两种长度的 16 种数据：85,272 字节和 4 MiB。每次同步 H2D、同步 D2D、同步清零后，立即在已有非阻塞 stream 上读回；只有消费者的完成对象负责等待。三个阶段、两种长度、16 轮，共 **96 次**，全部字节精确匹配。

测试有意避免在生产者与消费者之间做 cudaMalloc/cudaFree、默认流读回或额外 Wait；这些操作可能掩盖缺失的完成关系。不是靠固定延迟推测设备已经完成。新的 kernel 数值回归也继续使用原有同步上传 API，不在测试局部添加补救同步。

最终六项 CUDA 专项及完整 CPU 回归回执见 [归一化报告](GPU_MULTISTAGE_REDUCTION_REPORT.md)。初始失败和修复后的执行日志保留在 `out/windows-gpu/logs/cuda-multistage-*.log`。

这使同步 API 的返回语义与合同一致，代价是原本漏掉的默认流完成等待。需要流水线传输时仍应显式使用异步 API 及依赖；本修复不承诺替调用者等待别的非阻塞流上的未完成生产者，也不构成设备吞吐、内存插桩或完整 M1 copy/event profiling 的验收。
