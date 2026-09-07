# M7：分布式模块先补证据，再决定是否接入内核

第一波和第二波都以单机 CPU/LLVM MiniMind-L1 为验收边界。当前 distributed/ 有会话、worker、放置、JSON 计划和 CPU 集合通信代码，规模约 2,487 行；执行器对 compiled module kernel 仍会报 launch 未实现，也没有多 worker 数值证据。因此 MiniMind 单机通过不能升级成分布式能力，MiniMind-O 的多流会话也不属于本模块首版。

本模块先证明计划、放置、通信和失败诊断，再决定是否桥接一个已经编译的 CPU kernel。worker 只能执行外部提供的 artifact，不编译、不猜 shape、不维护 KV registry；若 ABI 无法安全表达就保留为 unsupported。

> 状态：待实施，低于 MiniMind-L1 核心波次优先级。当前安排见 [WAVE_2](WAVE_2.md)，边界见 [PROJECT_GOAL.md](../PROJECT_GOAL.md) §2.3。不处理 KV 一致性或跨机器服务化。

## S1：计划和通信证据

覆盖一个 2-worker CPU 计划：输入值放置、传给指定 worker、一个 CCL 操作、结果放回调用方。验证 JSON round-trip 后 canonical 内容和 worker 顺序稳定；同一 worker 集合不能依赖 map 遍历顺序。

加入：

- placement 的合法/非法 device 与 worker 映射；
- DRef/value 生命周期和缺失初始值；
- CCL CPU backend 的输入 shape/dtype/count 错误；
- 节点执行顺序、重复 value、循环/未知节点拒绝；
- worker 失败和通信失败的上下文诊断。

S1 的结果仍标记为“distributed plan/communication validated”，不更新 Transformer compiled-kernel numeric 能力。

## S2：compiled module 桥接的最小切片

桥接必须接收已经准备好的 `api::CompiledModule`、`KernelSignature` 和显式 value contract；不能从 op 名、TIR 文本或 JSON 临时猜参数。worker 只执行已提供的 artifact，不编译、不查询 primitive cache、不选择形状。

在执行前验证 worker device、模块 entry、KernelABI、输入/输出 value、常量和生命周期。任何 mismatch 都在通信或 launch 前失败。先接一个无状态的 CPU elementwise kernel；CUDA、KV state、多进程/多机器和跨 worker alias 延后。

执行结果与单机 RuntimeSession 的结果逐元素比较，记录 worker/span/plan 节点关联。若当前模块 API 无法安全表达这种桥接，完成 S1 后把生产 kernel 执行保留为 unsupported，而不是增加裸指针接口。

## S3：边界决策

评审时作出二选一：

1. **接入**：将 compiled module 节点、ABI、放置、失败和生命周期写入现有 distributed plan contract，并补真正的单机多 worker 数值测试。
2. **降级**：保留计划和 CCL 模块，明确 compiled module kernel 尚未支持，并在能力矩阵中记录原因和测试证据。

两者都不能改变单目标 compiler 的 identity，也不能把分布式执行器变成第二个编译器。

## 代码和验证

| 位置 | 工作 |
|---|---|
| [execution_plan](../../include/kxc/distributed/execution_plan.h)、[JSON](../../src/distributed/execution_plan_json.cc) | 稳定节点/序列化合同 |
| [placement](../../src/distributed/placement.cc)、[worker](../../src/distributed/worker.cc) | 放置和 worker 失败边界 |
| [executor](../../src/distributed/executor.cc)、[CCL CPU](../../src/distributed/ccl_cpu.cc) | 先补已有逻辑测试，再评估 compiled module bridge |
| 新 `test/distributed_*` | S1 正/负例和 S2 单机数值切片 |

测试使用确定性输入、worker 调度和通信顺序；不需要假造网络或 GPU。接入 profiling 时复用 M1 的事件字段和现有 distributed span，不建第二个 bundle 格式。

建议命令（测试目标加入 CMake 后执行）：

```bash
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'distributed|execution_plan|profile_bundle_test|runtime_session_test'
python3 tools/architecture/check_include_layers.py --root .
python3 tools/architecture/check_public_headers.py --root . --compile
```

## 完成条件

- [ ] S1 的计划、placement、JSON、CCL 和失败路径有独立测试。
- [ ] compiled module 尚未接入时，负例明确报错且不执行 kernel。
- [ ] 若接入 S2，单机多 worker 的 compiled CPU kernel 与 RuntimeSession 数值一致，ABI mismatch 为零 launch。
- [ ] 没有声称 CUDA、KV coherence、多机网络或 Transformer 全图已支持。
- [ ] 矩阵将计划/通信证据与 compiled-kernel 证据分开记录。
