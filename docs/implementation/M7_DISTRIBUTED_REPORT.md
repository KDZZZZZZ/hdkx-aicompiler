# M7 技术报告：进程内多 worker 的已编译 CPU 执行

M7 的 S1/S2/S3 采用“接入”方案：`ExecutionPlanExecutor` 现在能消费调用方提供的 `CompiledModule`，在不同 worker 线程运行真实 LLVM kernel，再通过 CPU 集合通信传递结果。代表用例完成 **2 个 worker × 2 个 kernel、3 次集合通信**，输出与单机 `RuntimeSession` 精确一致。验收范围是进程内的静态无状态 CPU 小图，完整回归结果见下表。

源码入口为 [executor](../../src/distributed/executor.cc)，可运行证据为 [distributed_runtime_test.cpp](../../test/distributed_runtime_test.cpp)。

## 采用的方法

**先验证整份计划，再执行。** 既有 distributed `ExecutionPlan` 增加 `Validate()`，证明 value 元数据完整、使用顺序合法、每个输出只定义一次、worker 集合明确且无重复。循环、未知节点、隐式别名、缺失放置和未定义值均拒绝。执行器进一步检查初始 DRef 的会话归属、每个 worker 的值是否可用、通信尺寸、模块入口、完整 `KernelSignature` 字节、Target 能力、只读常量、输出预算和下游输入对齐。末尾 kernel 的错误也会在开头的通信之前被发现。

参数检查复用 runtime 的 `ValidateTensorArgument`；原 `ValidateKernelArgument` 委托给同一实现。常量继续由模块持有并注入，调用方拿到的是独立快照。执行器不读取 Relay/TIR 来猜 ABI，不编译、不查询 primitive cache，也不选择 shape profile。

**每个节点等待自己的 worker 作业完成。** 内核节点使用标准库 `std::async` 为显式 worker 集合创建作业，传播 worker id 和既有 profiling 上下文，调用 `CompiledModule::Invoke` 并等待 completion。所有已启动作业都结束后才发布输出 DRef，再进入下一个节点。每次 Execute 使用自己的 value Map，不再通过共享 Map 修改调用方初始绑定，也不保留上一轮输出表。

**引用负责张量生命周期。** DRef 节点持有 session；最后一个 DRef 节点引用消失时，session 清空对应寄存器上的张量。独立构造的同寄存器 DRef 也计入保活。跨会话引用、释放后的寄存器、关闭后的分配/读写/同步都会报错。测试通过外部 Storage deleter 验证实际释放，并验证 session 关闭后已取出的 NDArray 仍有效。寄存器替换、最后引用释放和 Shutdown 都在 session 锁外析构张量；外部 deleter 重入 SyncWorker 的用例验证不会因持锁回调而死锁。

**明确集合通信的参与者与分组。** worker 分组为连续等宽区间。allreduce 要求每个参与者存在，广播与 scatter 使用各组自己的第一个 worker，gather 验证相同分片尺寸后写入各组根节点。`in_group=false` 使用全部 worker；显式点对点路由在 `in_group=true` 时不得跨组。当前 collective worker_set 必须包含全部 session worker；不支持的 async、非零 group_id、root 和 reduce 属性均拒绝。CCL 先计算各组结果，再提交目的张量，避免后组的参数错误先改写前组目的值。

CPU allreduce 支持 float32/float64/int32/int64 的 sum。整数用对应无符号类型定义定宽回绕；通过字节复制读取元素，兼容合法的未对齐 NDArray view。直接 CCL 入口也检查 Storage 字节范围，不仅依赖执行器预检。

这些规则参考了 [TVM Disco 的 DRef、worker0 与 group 约定](https://tvm.apache.org/docs/reference/api/python/runtime/disco.html)，以及 [MPI Allreduce](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Allreduce.3.html) 和 [Scatter](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Scatter.3.html) 对参与数据及计数的一致性要求；实现仍使用仓库自己的 session 与 CPU CCL，没有引入这些框架或网络依赖。

## 合同与身份

distributed JSON 升为 **v2**，kernel 节点增加 `kernel_abi`，内容必须是绑定模块提供的完整 `KernelSignature::CanonicalBytes()`。JSON 只保存可检查的计划，模块仍由 C++ 调用方显式绑定；没有新增接受裸指针或自动编译的 FFI。

Target 的稳定规范字节从 compiler 内部实现移至 `Target::CanonicalBytes()`，原 compiler helper 委托到它。原 `target-snapshot-v1` 字段、顺序和字节格式保持一致；可用显存/内存等瞬时观察量仍不进入身份。placement 与模块使用同一个 owner 比较完整能力，worker id 不再从物理 device id 推断。

JSON 现在保存全部稳定 Target 属性，规范化 worker 顺序，保留全部通信属性，并拒绝未知字段、重复 value id、null 节点、非法 Unicode 转义和控制字符和过深嵌套。Unicode 转义正确还原为 UTF-8，不再静默替换为问号。无效计划在打开输出文件前被拒绝，写入和关闭错误显式报告。

v1 JSON 会明确拒绝，调用方须用 v2 重新导出并绑定模块。新增 session 引用生命周期虚函数要求 C++ 使用方重新构建。Kernel ABI、单目标图语义和 primitive artifact 身份版本没有因分布式调度改变；计划中声明的 ABI 是调用兼容性检查，不是跨进程来源认证。

## 达到的效果与证据矩阵

数值用例先将 `[-3,4,5,-6]` scatter 为两个长度 2 的输入。各 worker 运行 `relu(x + [1,-2])`，分别得到 `[0,2]` 和 `[6,0]`；allreduce 后每个 worker 得到 `[6,2]`，gather 的最终输出为 **`[6,2,6,2]`**。参考值来自同一原始编译结果的单机 RuntimeSession，两路结果再按同样通信数学组合，最大绝对差为 **0**。

| 层次 | 本次证据 | 可作出的结论 |
|---|---|---|
| 计划、放置、JSON | 反向输入的 worker 构造仍得到相同 JSON；完整 Target/ABI 往返；错误 use/def、字段、映射拒绝 | 计划/通信合同已验证 |
| session 与 CPU CCL | 4 worker / 2 group 的 sum、广播、scatter/gather，点对点复制、跨组拒绝、缺参与者/尺寸/dtype/范围错误，整数溢出和未对齐 view | 进程内 CPU 引用和通信具有执行证据 |
| compiled kernel | 两个独立 worker 线程的首对调用通过有界会合检查；4 次真实 LLVM 调用与单机结果精确一致 | 静态、无状态 CPU 多 worker kernel 已接入 |
| 生命周期、重复运行 | caller graph/module/session 释放并清空 primitive cache 后仍能执行；独立输出地址、初始 Map 不变、失败后重试通过 | 模块与张量由实际执行句柄保活，执行期间全部缓存统计不变 |
| 执行前拒绝 | 缺模块、晚节点错误 ABI/参数数/shape/Target、错误分组等；CCL 和 launcher 计数为 0 | 整份计划的合同错误不会触发前置通信或内核 |
| 运行失败与 profile | 注入 worker 1 失败，其他已启动作业收尾，后续节点不执行；成功、预检失败、worker 失败的 JSON span 自动校验 | worker/node/run、parent span、状态与 completion 有对应关系 |
| Transformer / CUDA / 多机器 | 本次未执行 | 不更新 Transformer 数值或 CUDA/网络能力格子 |

本表独立区分计划/通信与真实内核证据，不给 [Transformer 矩阵](../../test/nlp_validation/transformer_capability_matrix.json) 的十二项模型能力追加不相关的小图证明。

| 回归 | 结果 |
|---|---|
| 默认 CPU/LLVM | 49/49 CTest 通过，43.82 秒，含最终 DRef/Shutdown 释放回调重入验证 |
| adaptive CPU/LLVM | 49/49 全量通过，42.68 秒；最终生命周期专项再次通过 |
| bounded CPU/LLVM | 63/63 全量通过，250.42 秒；显式启用 projection、heads、RoPE、GQA、attention、完整 prefill/decode 及静态容量 decode-loop 八组模型 fixture；最终生命周期专项再次通过 |
| Python | 280/280 通过，1.75 秒 |
| 生成合同与 NLP 门禁 | Relay 37/37、pass 20/20、两项生成物 freshness、NLP reference/capability checker 与 diff check 通过 |
| 架构与单定义 | include 279 个文件、public headers 89+10 编译、docs 56 篇通过；默认/adaptive/bounded 静态库分别 1,574/1,686/1,854 个强全局符号，无重复 |

## 三轮 Ponytail QA

1. **复用**：扩展既有 ExecutionPlan、DiscoSession、CCLBackend 和 executor，调用既有 CompiledModule、参数检查及 Profile Bundle。只修正已有 JSON parser，不增加 parser/worker 框架、compiler callback 或 KV registry。
2. **权威与版本**：模块拥有 ABI/常量/可执行资源，Target 拥有能力字节，计划显式声明 worker/value 边界。distributed v2 隔离旧序列化；原 Target 字节不变。通信非默认行为均有 consumer 或明确拒绝。
3. **真实消费和失败**：真实 LLVM 数值、并行进入、缓存清空后的保活、整个计划的零执行拒绝、CCL/worker 错误与 profile 均形成可运行检查。清理了隐式 worker0、通信预克隆和多输出共享 DRef；构造异常由对象引用负责清理。

## 当前限制

这是一个进程内、静态、同一 CPU:0 Target 的执行切片。kernel 返回新输出；不接受动态 extent、部分有效区、state、跨 worker alias 或 CUDA kernel。尚无 Relay 全图自动分区、跨进程模块运输、网络容错或分布式 KV 一致性。

每个节点创建并等待 worker 线程，没有长期 worker pool；本次不声称降低延迟或提高吞吐。寄存器编号单调增长以阻止旧 DRef 复活：张量可及时释放，但寄存器元数据保留到 session 关闭。通信临时结果的复制也有内存开销。

执行开始后的 worker/CCL 异常会停止后续节点、等待已启动作业并释放本轮引用；没有分布式事务回滚。调用期间调用方须保持计划、模块和输入寄存器/存储不变。若上游分配不能保证下游内核所需对齐，预检会拒绝，不在运行时偷偷复制修复。

## 复现

```bash
cmake --build out/build/dev-ninja-cpu -j2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error \
  -R 'distributed_runtime_test|compiler_contract_test|compiler_identity_test|compiled_module_test|runtime_session_test'
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
```

默认构建的自动校验 bundle 位于 `out/build/dev-ninja-cpu/out/distributed_runtime/`。其中 `distributed-success` 是完整成功运行，`distributed-admission-error` 是末尾 ABI 不匹配的零执行拒绝，`distributed-worker-error` 是实际 worker 失败。测试中的 launcher 包装只负责会合、计数和故障注入，成功路径委托真实 LLVM executable。
