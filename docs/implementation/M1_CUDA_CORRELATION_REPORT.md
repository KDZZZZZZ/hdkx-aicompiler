# M1：CUDA 设备活动关联与时钟对齐技术报告

2026-09-09。完整八层 MiniMind 的固定 B1/S16 prefill 已把 **1,300/1,300 条 CUPTI kernel 活动**关联到对应运行和计划调用，并保留导出 receipt、stage、输入长度及 repeat。两个独立 profile、四个主机线程和四条 CUDA stream 的 12 次运行也完成 36 条设备 kernel 的独立关联。运行仍通过原有 Compiler → CUDA module → RuntimeSession，未增加模型运行时或新 IR。

## 问题与方法

[上轮完整模型报告](GPU_MINIMIND_PREFILL_REPORT.md) 留下了具体负面证据：主机有 1,300 对 submit/exec，CUPTI 也有 1,300 条 kernel，但设备记录的 run 和 parent 都为空。原因是 runtime 观测 TLS 没有在真正调用 Driver 的线程上激活 CUPTI external correlation。另一个问题是 CUPTI 初始化晚于 ProfileContext，却只减去 CUPTI 自己的时间原点，使首个设备 kernel 看起来比模型提交提前约 14.7 ms。

本轮复用现有 ExecutionObserver、ActivationScope 和 ScopedSpan：

1. runtime 新增可选的 `OnScopeEnter` 钩子，返回同线程、同步执行的清理动作。run 作用域和三条实际 kernel 提交路径都调用它；runtime 仍只依赖自身的纯数据类型。清理不进入异步 completion，不迁移 CUPTI 线程栈。
2. profiling 适配器在实际提交作用域激活 run，并创建唯一 `kernel_launch` span。原有 `kernel_submit/kernel_exec` 仍以 run span 为父；设备 kernel 的父节点是对应 launch，launch 再指向 run。调用序号和模型 metadata 留在 launch/run 中，设备记录通过 ID 连接它们，不能依赖缓存可能复用的 kernel 名称。
3. 每个 ActivationScope 成对 push/pop CUSTOM0(run) 和 CUSTOM1(span)。空 context/span 也构成屏障，防止外层模型污染未观测的工作。嵌套、异常退出和 ScopedSpan 移动都恢复先前线程状态。
4. CUPTI 采集器按原有唯一 session_id 保存目的 context 和该 context 的时钟采样，不再用后创建的 profile 覆盖前一个。external ID 在进程内单调分配；关联表与目标对象的注销、事件写入使用同一把 mutex，避免异步回调访问已销毁对象。注销时回收该 profile 的映射。
5. 绑定时在 `cuptiGetTimestamp` 前后取 ProfileContext elapsed time，以中点对齐原点；转换检查早于原点及整数溢出。每个 activity buffer 先读取 external records，再解析 API/kernel/copy 记录。没有可确认 profile 的全局 CUDA 活动不猜测归属。

实际模型的 650 个计划调用仅对应 61 个不同的设备 kernel 名称；并发小图的 3 个调用也只有 2 个设备名称。这直接验证了必须按调用 ID 关联。原采集器中没有读取方的共享 `last_error_` 写入已移除，避免多个提交线程写同一字符串。

依据为 NVIDIA [CUPTI 12.9 Activity API](https://docs.nvidia.com/cupti/12.9.1/api/group__CUPTI__ACTIVITY__API.html) 与本机 12.9 SDK 头文件：external correlation 的 push/pop 属于调用线程，CUPTI 活动记录使用自己的时间戳。NVIDIA live skill catalog 第一页可访问，第二页被 WAF 返回 202，未据不完整目录宣称不存在相关 skill；最终以实际 SDK 和官方 API 文档为依据。

## 事件、身份与故障边界

| 事件 | 时间域 | 关联 |
|---|---|---|
| runtime_session_run | host_execute：主机提交路径 | 模型 run 根节点 |
| kernel_launch | host_submit：实际调用所在词法作用域 | parent=run，含 call_index 和模型 metadata |
| kernel_submit | host_submit：提交完成的瞬时事实 | parent=run，保持既有语义 |
| kernel_exec | host_execute 或 host_observed_complete | parent=run；后者不代表设备执行耗时 |
| cuda_kernel / cuda_memcpy / cuda_memset | device_execute | CUPTI external ID 指定的 span |
| cuda_driver_api / cuda_runtime_api | host_execute | 同一 external ID 指定的主机作用域 |

schema_version 仍为 1：原有字段、时间单位和父节点语义未改变，新事件与 timing 使用已有字符串字段；schema.py 与工作台 contract 的通用事件结构可以读取。kernel/plan ABI、pass/backend identity 均未改变，观测信息不进入编译缓存身份。现有诊断器继续只将明确的 host_execute kernel_exec 用于主机耗时判断，不把设备时间或等待时间混在其中。

ExecutionObserver 新增默认实现的 virtual 钩子，现有派生类源码可以继续编译；已编译的自定义 C++ observer 需要随运行库重新构建，不宣称该类的二进制 ABI 兼容。

观测钩子、清理和 CUPTI C 回调吞掉自身异常，原执行异常照常传播。ScopedSpan 写出失败不再使析构 terminate；ProfileContext 析构仍尝试 Flush 并注销，显式 Flush 可报告无效目录。新增 CPU 反例用普通文件阻挡 bundle 目录，确认数值仍正确、TLS 恢复、销毁安全；launcher 抛错及观测器进入/退出抛错也有独立用例。

## 真实 GPU 验证

设备仍为 Windows RTX 4070 Ti SUPER，驱动 576.57，CUDA 12.9 Update 1，MSVC 19.29，LLVM 关闭。本轮延续已核验的 23 文件完整模型 fixture；模型 ONNX SHA-256 为 `eb1c60c73980b0e56693876906e4e6d1a861b2609bd40469e2637d09b0eef455`，随机权重与独立 ONNX 参考均未更换。

- [并发 consumer](../../test/cuda_runtime_profiling_test.cpp)：两个同时存活的 profile，各有两个线程/session/非默认 stream，每个 session 重复三次三调用图，12 次共比较 3,084 个 float32 元素，逐元素精确一致。错误输入在 launch 前拒绝；调用者 TLS、移动后的 span 和空 context 屏障有真实 CUDA 检查。
- [完整模型 consumer](../../test/minimind_prefill_cuda_test.cpp)：两次 prefill 的全部 logits 和 16 个 KV，共 401,408 个元素，最大绝对误差仍为 `1.001358032e-5`。十项 runtime cache 统计不变，输出 storage 独立。
- [Bundle 验证器](../../test/cuda_profile_bundle_test.py)：要求实际执行完成标记、CUPTI available、每个 run 的完整调用集合、device→launch→run 一一连接、模型 metadata、独立 stream 和正的设备 duration。设备开始不可早于 launch 超过 1 ms，结束不可晚于主机观测完成超过 1 ms；不会错误要求异步设备在 RunAsync 提交返回前结束。
- 验证器对真实 bundle 分别注入空数据、错误 run、缺失 parent、错误时钟、重复 span、错误模型六种破坏，必须全部拒绝。模型 fixture 未提供时 CTest 明确返回 77/Skipped；静默 exit 0 或缺少完成标记不能通过。

最终 Windows 专项 **10/10、11.89 秒**。完整模型关联 bundle 有 28,125 条事件，全部 status=ok/severity=info，设备活动相对对应 launch 起点的最小间隔为 **56,045 ns**，最大为 6,468,915 ns；没有 CUPTI dropped-record 告警。两个并发 profile 各有 6 个成功 run、18 个 kernel、2 个输入被拒绝且零提交的 error run，以及 10 次恢复至 caller_parent 的同步复制；每个 profile 都观测到不同 worker 的运行区间重叠。

三套完整 CPU/LLVM 回归：默认 **55/55、145.01 秒**，adaptive **55/55、212.20 秒**，bounded **70/70、226.95 秒**。逐用例日志核验完整视觉、三组单图联合模型、文本状态/decode；adaptive 另执行真实模型热替换，bounded 另执行完整变长 prefill/decode 与请求批处理，以上分别 5/6/8 个模型用例均未跳过。可选独立 L1a/ResNet18 等未设置 fixture 的历史 SKIP 保持原样，不计入这些模型证据。

40 个 Relay operator、20 个 pass 和生成物新鲜度、NLP 门禁、284 个 include 边界、90 个 installed 与 10 个 experimental header 独立编译、70 篇文档链接及 git diff 检查通过。8 种能力矩阵篡改被拒绝，包含移除新关联 consumer/报告和擅自开放 decode/state/copy CUDA。CPU/CUDA 静态库分别 1,603/1,614 个强定义，重复数为 0。三份真实 GPU bundle 通过既有 schema 与离线诊断器，GPU 等待区间没有被误判为短主机 kernel。Linux CUDA 构建通过；实际设备执行证据来自 Windows。

复现入口：设置 `KXC_MINIMIND_CUDA_PREFILL_DIR` 指向既有完整静态 fixture，再运行 `ctest --test-dir <cuda-build> -C Release -V -R '^cuda_profile_(concurrent|prefill)_test$'`。每次自动创建独立 evidence 目录，保留旧 bundle。

源码验证使用 `out/windows-gpu/cupti-correlation-final.tar.gz` 的 19 文件 overlay，SHA-256 为 `db2ea03399f248f607121c075c22663436fb2d043b2cd3f91c92e06c3fefa514`；报告收尾另同步 metadata，完整 692 文件清单为 `cupti-correlation-source-manifest.json`。原始 Windows 日志与 bundle 位于 `out/windows-gpu/cupti-correlation-evidence/`，其归档 SHA-256 为 `1ccb5a43a5c1fde88378660d6eaab9e71aea7945985e18db2bb26da8ed448609`。模型 events.jsonl SHA-256 为 `db5589806764904ebd393316ac024be312e3fdb57ecb1da3d25256021343ec9d`。数值、schema、模型无跳过、矩阵反例与最终汇总见 `out/windows-gpu/logs/cupti-correlation-*-audit.json`。

## 三轮审查与限制

A：复用 runtime observer 和 profiling 的现有作用域，只增加同步进入/退出合同，没有第二套事件注册或运行引擎。B：CUPTI 全局采集器仍为唯一 owner，按现有 profile session ID 路由；锁顺序为 adapter→ProfileContext，Flush 在取得 context 锁之前调用 CUPTI，避免反向持锁。C：真实 MiniMind、并发多 profile、错误输入、观测故障、空作用域及篡改回执均有实际 consumer；runtime 的三种提交分支共同接入。

这是设备活动关联正确性的局部硬件证据，不是吞吐或加速比测量。关联表在 profile 生命周期内保留异步记录所需的 ID，长时间 profile 的内存上限/滚动清理及长期时钟漂移仍需单独处理。全局 mutex 串行化路由和事件写入，吞吐优化应依据实际争用再拆分锁与生命周期管理。没有把同步复制的 caller 关联检查当作完整 M1 异步复制配对验收，`copy_event.cuda` 保持关闭。GPU decode/state、bounded CUDA、GPU 热替换健康决策和 L3 实时预算未因本轮结果自动完成。Compute Sanitizer 的既有环境问题仍在，没有新增内存插桩通过声明。
