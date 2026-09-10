# M6：CUDA 热替换技术报告

2026-09-10，Windows RTX 4070 Ti SUPER 上的静态 CUDA 热替换已完成。测试走现有 `AdaptiveHotSwapController`、`GenerationLease` 和 `RuntimeSession`，没有新增一套 GPU 路由或状态 owner。该切片证明了无状态 CUDA kernel 的换代、lease、CUPTI 关联、one-shot health/rollback 和执行前拒绝；bounded KV 状态热替换、GPU 请求批处理热替换、跨设备迁移仍分别由各自模块负责。

## 方法和实现

候选使用同一静态 Relay 图和同一 target。generation 1 使用 CUDA TIR O3，generation 2 使用新增的 CUDA TIR O2 pipeline；两条 pipeline 都显式运行 `bind_cuda_threads`，因此候选能取得完整 grid/block launch metadata。只请求替换第 1 个 primitive，其他两个 artifact pin、DispatchKey 和 Plan ABI 必须保持不变。运行通过 `RunAsync` 获取当前 lease，提交到两个显式 CUDA stream；完成后从 CUPTI bundle 核对每个 host kernel launch、runtime completion 和 `cuda_kernel` activity 的一一关联。

原有 O2 CUDA pipeline 没有线程绑定，候选编译会在 `GetCudaLaunchConfig` 前 fail closed。修复把 `tir.compiler.cuda.o2` 接入生成的 pass contract，并让 CUDA TIR O2/O3 都消费 `bind_cuda_threads`。这不是绕过校验：pipeline identity、artifact key 和生成的 launch metadata 都随候选编译重新计算。

## 验证结果

- Windows 主机：RTX 4070 Ti SUPER 16 GB，CUDA 12.9，driver 576.57。
- 原生 consumer：`adaptive_cuda_test`，adaptive 开关、CUDA 和 bounded graph 同时开启。
- 代际序列：`1 → 2 → 1`；generation 2 只替换选定 kernel，generation 1 在 rollback 后继续服务。
- 每次成功运行提交并完成 3 个 CUDA kernel；first bundle 两次成功运行加一次错误 preflight，second bundle 一次成功运行，共 9 个 CUPTI kernel activity。
- route digest `3a26f09ba1536b41` 与 Plan ABI `f44eb68b455d4502` 在两代相同；selection plan identity 不同。
- 两个显式非默认 CUDA stream 实际出现（stream 13/14）。CUPTI 活动、launch span、kernel submit/exec 和 run 均按 `run_id`、`call_index`、generation、route、ABI、variant 和 validation receipt 关联。
- generation 2 的 one-shot health 读取真实 CUDA activity，写出 `health-evidence.tsv`，随后隔离并回滚；重复消费和过期 lease 均拒绝。
- 错误设备输入在 launch 前拒绝，错误 run 的 submit count 为 0；独立 Python 审计对六种证据篡改均 fail closed。

复现与审计：

```bash
cmake -S . -B <build> -G "Visual Studio 16 2019" -A x64 \
  -DKXC_ENABLE_CUDA=ON -DKXC_ENABLE_ADAPTIVE_HOT_SWAP=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON -DKXC_BUILD_PASS_TESTS=ON
cmake --build <build> --config Release --target adaptive_cuda_test -j 2
KXC_MINIMIND_BOUNDED_DECODE_DIR=<fixture> \
  <build>/Release/adaptive_cuda_test.exe <bundle-root>
python3 python/tools/check_cuda_adaptive_hot_swap.py \
  --root <bundle-root>
```

Windows 设备回执保存在 `out/windows-gpu/evidence-20260910/adaptive-cuda/`；其中 `adaptive-cuda-audit.json` 是本轮独立校验结果。源码和回执仍是本地工作区证据，没有提交或推送 GitHub。

## 边界

这里验证的是无状态静态 CUDA plan 的安全换代。KV state 的同设备复制/extent 合同已经由 GPU bounded state 报告验证，但把该 state 接到 CUDA generation replacement、请求队列和完整 MiniMind decode 仍需单独 consumer；不同容量、布局、Plan ABI 和跨设备迁移继续明确拒绝。该测试也不是吞吐或实时预算结论。
