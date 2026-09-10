# CUDA 有界 decode 编译接入技术报告

2026-09-10。状态：完整模型 CUDA 数值、bounded 会话状态、两条 stream 和 CUPTI 设备活动验收已完成。

前半部分保留 2026-09-09 的编译快照；下方 Windows GPU 数值与设备验收段落取代其中关于硬件尚未执行的历史描述。

## 达到的效果

实际八层 MiniMind bounded decode 的 CUDA 编译从 **750/774** 提升到 **774/774** 个原语。剩余的 24 个缺口全部是动态长度拼接：每层两份 K/V 追加与一份 mask 拼接。新增的分支范围证明闭合这些缺口，未修改 Relay/TE 的拼接数学定义，也未引入新 IR。

本次 decoder 使用 17 个输入：词元与 16 份 K/V；batch 上界为 3，历史长度 P 为 0..8，单步追加 1。输出是 logits 和 16 份长度 P+1 的 K/V。此处证明的是完整导出模型的编译可行性，尚无本版本的 GPU 执行结果。

## 使用的方法

扩展现有 [BindCudaThreads](../../src/tir/transforms/bind_cuda_threads.cc) 的有界地址证明。当循环坐标满足 `0 <= j < P+C`、C 是非负编译常量，且读取使用 `j<P` 分支时，分别证明两段：

- 前段的坐标范围为 `0..P-1`；P 为 0 时此段不可达。
- 后段在证明内部写成 `j=P+u`，其中 `0<=u<C`，使追加输入的 `j-P` 地址可被证明。C 为 0 时此段不可达。
- 坐标替换仅存在于证明过程；原始 TIR 和实际紧凑布局保持不变。分支范围与商/余数缓存随分支退出恢复，不能用于兄弟分支或后续读取。

这条规则只接收已证明的符号边界与常量尾段。错误阈值、错误偏移、窄化比较、其他轴、未初始化读取、越界写入，以及在 P 可为 0 的路径上除以 P，仍然拒绝。未开放一般符号不等式求解或任意动态间接访问。

复用 [完整模型 codegen consumer](../../test/minimind_bounded_cuda_codegen_test.cpp)，增加 decode 选择；同一 [Python wrapper](../../test/minimind_cuda_codegen_test.py) 检查源码与 PTX 的全部独立入口。生产 preparation、逐原语 lowering、已解析的 TIR pipeline 和 CUDA codegen 均实际执行；synthetic sm_89 目标只用于无设备编译证据。

Pass schema 升为 **8**，backend identity 升为 **cuda-nvrtc-driver-v8**，由既有 pipeline/artifact 身份消费。runtime extent 参数 ABI 不变，bounded state 与请求批处理沿用同一生产 RuntimeSession owner。

## 已运行的验证

| 检查 | 实际结果 |
|---|---|
| 新增拼接边界 | 6 个合法用例、15 个拒绝用例通过；包含空前段、空尾段、多元素追加及分支事实泄露 |
| CUDA 调度与 pipeline 身份 | 两项 CTest 通过；调度测试共 19 组 |
| 完整 bounded prefill 回归 | 742/742 个源码与 PTX 入口，52.19 秒 |
| 完整 bounded decode | 774/774 个源码与 PTX 入口，48.40 秒 |
| 上述本地 CTest 合计 | 4/4，100.75 秒；NVCC 12.9.86，无 GPU 执行 |

prefill 的 CUDA 源码和 PTX 与 [前一模块](GPU_BOUNDED_PREFILL_REPORT.md)逐字节相同。decode 源码为 1,836,274 字节，SHA256 为 `0b30a01b81cc353a896a8217ad96f3829162a63a596b466781ec26d25d685cba`；PTX 为 1,989,156 字节，SHA256 为 `5f2921b6444c4507172454e45380d888f72267aa371fa77b424edacac0cd95cc`。

日志保存在 `out/windows-gpu/logs/bounded-decode-v8-codegen-{ctest,lasttest}.log`。源码、PTX、编译命令和逐文件哈希保存在 `out/build/bounded-cuda/bounded-cuda-codegen-evidence/bounded-decode-bnstnvh0/` 的 `codegen-audit.json`；对应 prefill 为 `bounded-prefill-jewj4dgv/`。这些是本地生成的证据，不是 GitHub 上的交付文件。

复现时设置 `KXC_MINIMIND_BOUNDED_PREFILL_DIR` 和 `KXC_MINIMIND_BOUNDED_DECODE_DIR`，在已配置 bounded CUDA 的构建中运行：

```sh
ctest --test-dir out/build/bounded-cuda -j 1 --output-on-failure --no-tests=error \
  -R '^(cuda_schedule_test|pipeline_resolver_test|minimind_bounded_cuda_codegen_test|minimind_bounded_decode_cuda_codegen_test)$'
```

## Windows GPU 数值与设备验收（2026-09-10）

隔离构建目录为 `F:\kxc-gpu\20260910-goal`，设备为 RTX 4070 Ti SUPER、CUDA 12.9；bounded prefill/decode fixture 通过 SHA-256 校验后解压，源码和 fixture 与本轮构建分离于旧证据。代码生成门禁为 4/4：prefill 742/742、decode 774/774、请求复制 axis 1/2 各通过。

完整 decode consumer 通过 `[PASS] full_minimind_bounded_cuda_decode_and_owned_state_no_runtime_compile`：fresh decode 8 runs、owned-state 8 runs、每次 774 calls、两条 stream、692224 个参考值；四组 B/P 最大误差 `1.56164e-5`，greedy `6.19888e-6`，prefill→owned-state→greedy `8.58307e-6`。16 个 cache 地址稳定，4→8 extent 交接正确，容量拒绝在分配/提交前发生。position/control 三个 C=1/2/0 场景和 state bridge 也全部通过。

CUPTI 审计结果：12384 个关联 kernel、14944 次 extent DMA（119552 bytes）、336 次 state copy（1622016 bytes）、13 个零提交拒绝，13 个篡改反例全部拒绝；同一 bundle 的 seed prefill 为 742 kernels、6 个拒绝反例。最终输出 `[PASS] bounded_minimind_decode_profile_verified` 与 `[PASS] cuda_profile_correlation_verified`。

这项结果只覆盖已有 bounded 合同（`1≤B≤3, 0≤P≤8, C=1`）及显式状态 owner；任意 ragged、跨设备状态、分布式 decode、热替换和性能预算仍需各自证据。
