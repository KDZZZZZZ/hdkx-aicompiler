# M1：执行侧观测

现在编译期已经有完整的观测设施：[profiling 公共头](../../include/kxc/profiling/profiling.h) 定义了 `ProfileContext`、`ScopedSpan`、`ActivationScope` 和 bundle 写出（span 指一段有开始和结束时间的操作记录；bundle 是一次观测导出的一组文件：events.jsonl、trace.json、summary.json 等），Relay/TIR pass 与编译各阶段都会写入 span，[bundle 测试](../../test/profile_bundle_test.cpp) 已验证这些内容。但执行侧是盲区：[RuntimeSession](../../src/runtime/session.cc) 的 Run/RunAsync 从输入校验、输出分配到内核提交，全程不产生任何事件；CompiledModule 虽然已经存着编译期传来的 ProfileContext（见 [compiled_module_node.h](../../src/runtime/internal/compiled_module_node.h) 第 49 行），启动路径从未读过它。工作台和诊断引擎消费的 `runtime_session_run`、`kernel_exec`、`alloc`、`copy` 这类事件名，目前只来自手工 fixture 和分布式 executor 的模拟代码（见[工作台事件契约](../../tools/workbench/src/kxc/contract.ts)），没有一条来自真实 RuntimeSession。也不能直接在 session.cc 里加埋点：[分层检查器](../../tools/architecture/check_include_layers.py) 的 ALLOWED 表规定 runtime 层只能 include runtime/ffi/support，方向是 profiling 依赖 runtime，反过来就检查失败；而且 LLVM 内核虽在主机线程上同步执行（见 [llvm_jit.cc](../../src/codegen/llvm/llvm_jit.cc) 第 114-146 行），RunAsync 的接口却是异步的，只在调用外面包一层 RAII span（构造计时、析构落盘的守卫对象）会把"主机提交动作"误记成执行耗时，也表达不了"已提交、尚未观测到完成"的中间状态。

本模块要把执行观测接进真实运行路径：runtime 层定义一组与 profiling 无关的观察者钩子（callback，由上层注入的回调接口）和完成回调，profiling 层提供把钩子翻译成 ProfileContext 事件的适配器，RuntimeSession 从模块继承观测器。目标是一次 Run/RunAsync 的运行、内核提交、内存分配与复用、数据拷贝、完成观测和错误，都能用 run_id/span_id/parent_span_id（一次运行的编号 / 一个 span 的编号 / 它的父 span 编号）关联进同一个 bundle。明确不做：不扩展 GPU 计时和 CUPTI（CUDA 的设备活动采集接口）接入语义；不改 KV cache、shape 路由、调度策略和模块 ABI/identity；观测不触发编译、不改变路由、不强制设备同步；不改 events.jsonl 的既有字段集合，schema_version 保持 1。

> 状态：待实施。第一波 A 线，[M0](M0_BASELINE.md) 的 G0 基线完成后启动；分派与集成次序见[并行计划](WAVE_1.md)，模块定位见[实施总览](README.md)。当前架构事实以[架构总览](../ARCHITECTURE.md)为准。

## 当前代码与需要核实的事实

| 事项 | 当前核对结果 | 本模块动作 |
|---|---|---|
| 分层规则 | ALLOWED 表中 runtime 只能 include runtime/ffi/support，profiling 可以 include runtime；`src/runtime` 下文件名含 compiled_module 的文件被归为 runtime_executable，可以直接 include profiling | 观测接口放 runtime 层，记录适配器放 profiling 层，session 只依赖接口 |
| 事件模型与序列化 | [profiling.h](../../include/kxc/profiling/profiling.h) 的 `EventSpec`/`RecordCompletedSpan`/`RecordInstant` 固定写出 run_id/span_id/parent_span_id/ts_ns/duration_ns 等字段；phase 现有取值为 complete 与 instant；时间戳是相对 bundle 起点的单调时钟 | 复用既有序列化；phase 增加 "submit" 作为增量取值，字段集合不变 |
| ProfileContext 传递 | [compiler.cc](../../src/compiler/compiler.cc) 第 570-571 行在 AssembleCompiledGraph 时把 profile_context 传入 BuildCompiledModule 并存入模块节点；`GetProfileBundlePath` 已能读出 bundle 目录 | 不再新拉一条传递通道，观测器从模块继承 |
| 执行路径 | session.cc 的 Run（第 484 行）只是 RunAsync 加 Wait；RunAsync（第 491 行）依次完成校验、绑定、经 `ValueTable` 分配输出、逐内核调用 `InvokeOrderedModuleEntry`（第 316 行），全程无事件、无观测钩子 | 在这些落点接入钩子，不改校验顺序和拒绝行为 |
| 完成句柄 | [device_stream.h](../../include/kxc/runtime/device_stream.h) 的 AsyncOperation 有 Wait/IsReady/RetainDependencies，没有完成回调；`completed` 标志可读 | 增加完成观测回调，在 Wait/IsReady/析构真正观测到完成时恰好结算一次；注册时已完成则立即触发 |
| 分配与拷贝落点 | 运行期分配集中在 [value_table.h](../../src/runtime/internal/value_table.h) 的 Allocate（命中兼容存储则复用，否则 NDArray::Empty）与 Alias；拷贝公共入口是 [device_stream.cc](../../src/runtime/device_stream.cc) 的 StorageCopySync/StorageCopyAsync；会话构造用 NDArray::Zeros 初始化 state；模块构建期有常量快照复制（[compiled_module.cc](../../src/runtime/compiled_module.cc) 第 33-52 行） | 分配/复用/别名经会话观测器记账；拷贝经 runtime 线程本地观测器记账；常量快照复制在 runtime_executable 层直接记录 |
| 事件消费方 | [contract.ts](../../tools/workbench/src/kxc/contract.ts) 的 EVENT_TYPE 已定义 runtime_session_run/kernel_exec/alloc/copy 词表；[诊断引擎](../../python/kxc_agent/services/diagnosis_engine.py) 的 copy_dominance、sync_overhead、shape_fragmentation 规则已在消费 device_api/runtime_session 事件，但当前没有真实生产者 | 事件命名对齐既有词表；与诊断引擎期待值的差异写入交接 |
| bundle 结构校验 | [schema.py](../../python/kxc_agent/services/schema.py) 要求每个事件带 20 个固定字段；序列化器固定全量写出 | 不增删字段，schema_version 保持 1；新增的只是 event_type/phase 取值 |

## 边界设计：runtime 暴露钩子，profiling 装配记录器

观测接入必须维持"profiling 依赖 runtime"的方向，接入后 `python3 tools/architecture/check_include_layers.py --root .` 仍要通过。拟采用的结构：

1. 拟新增公共头 `include/kxc/runtime/execution_observer.h`（runtime 层，只依赖 runtime 自身类型）：声明 `ExecutionObserver` 接口，钩子覆盖运行开始/结束、内核提交、分配/复用/别名、拷贝、错误，参数全部是设备、字节数、符号、形状签名、状态、消息这类纯数据；同一头文件提供线程本地的"当前观测器"与 RAII 作用域，供分配和拷贝这类没有 session 参数的底层路径查询。
2. [session_node.h](../../src/runtime/internal/session_node.h) 的 RuntimeSessionNode 增加一个可为空的观测器成员；构造时经 compiled_module_node.h 的 `api::internal` 访问器从模块继承。模块没有启用 profiling 时访问器返回空，Run/RunAsync 行为与现在完全一致。
3. `include/kxc/profiling/runtime_observer.h` 与 `src/profiling/runtime_observer.cc`（拟新增，profiling 层，允许 include runtime）：把钩子翻译成 ProfileContext 记录，每次 RunAsync 用 `NextRunId("run")` 取新 run id，run span 作为本次内核与分配事件的父 span。适配器内部吞掉自身异常：观测永远不能改变执行结果，也不能覆盖原始错误。
4. [compiled_module.cc](../../src/runtime/compiled_module.cc) 属 runtime_executable，可以在 BuildCompiledModule 里用已存入节点的 ProfileContext 构造适配器，并直接记录常量快照复制事件。

这个结构下 session.cc、session_node.h、value_table.h、device_stream.cc 等 runtime 文件只 include runtime 头；测试也可以在手工 BuildCompiledModule 时传入 ProfileContext（该参数已存在），不需要给 RuntimeSession 增加公开构造参数。

## 事件语义约定

执行事件最容易出错的是"记的是什么时刻"。本模块只使用主机单调时钟，并按下表区分三种含义；每个事件用 fields 里的 `timing` 字段标明语义，评审时逐字段核对。

| component | event_type | 形态 | 记录的时刻含义 | 关键字段 |
|---|---|---|---|---|
| runtime_session | runtime_session_run | span | RunAsync 主机提交路径的起止：校验、装配、依次提交；同步 CPU 后端下近似整个 Run 的主机耗时 | timing="host_execute"；metrics 记输入数、内核调用数 |
| execution_plan | kernel_submit | 瞬时事件，phase="submit" | 主机完成一次内核提交动作的时刻，不代表执行完成 | kernel_symbol、call_index、timing="host_submit" |
| execution_plan | kernel_exec | span | 起点是主机提交前时刻，终点是主机观测到完成的时刻；同步后端该区间就是主机执行区间，异步后端是"提交到观测完成"的等待区间，两种情况都不是设备活动时间 | timing="host_execute" 或 "host_observed_complete"；kernel_symbol |
| device_api | alloc | span | 主机执行分配（含 ValueTable 复用判定）的耗时；view、alias、复用不重复计为新分配 | metrics.bytes、fields.reused、alignment |
| device_api | copy | span 或瞬时 | 同步复制记主机执行耗时；异步复制记提交与观测完成两个点 | metrics.bytes、两端 device、timing |
| 同所属 span | 原事件 | status="error" | 失败发生点；message 为原异常文本 | 校验失败发生在任何 kernel_submit 之前 |

错误不新增事件类型：run span 或 kernel span 以 status="error" 关闭，与 [profile_bundle_test.cpp](../../test/profile_bundle_test.cpp) 对编译失败的既有断言方式一致。运行期没有实际发生的拷贝不允许凭模型结构虚构成事件。第一波不记录设备活动时间（CUPTI 已有自己的 activity 事件，见 [cupti_smoke_test.cpp](../../test/cupti_smoke_test.cpp)），也不记录 free：存储释放由析构时机决定，本波不把它伪装成执行事件。

## 实施步骤

1. 先用未修改基线跑一个现有 LLVM fixture，记录输出、调用次数和当前 bundle 缺少哪些执行事件，作为改前证据。
2. 在 runtime 层落地观测接口：新增 `include/kxc/runtime/execution_observer.h`；在 [device_stream.h](../../include/kxc/runtime/device_stream.h) / [device_stream.cc](../../src/runtime/device_stream.cc) 给 AsyncOperation 增加完成观测回调，观测点为 Wait、IsReady 和析构，三者之间恰好结算一次，回调不得在内部再等待同一句柄。
3. 接线 session：RuntimeSessionNode 持有观测器；[session.cc](../../src/runtime/session.cc) 在构造（state 初始化分配）、RunAsync（run span、校验失败的 error、逐内核 submit 加完成回调、[value_table.h](../../src/runtime/internal/value_table.h) 的 Allocate/Alias 记账、最终 completion 的完成回调）接入钩子，不改任何校验顺序与异常文本。
4. 接入分配与拷贝：[storage.cc](../../src/runtime/storage.cc) 的 `Storage::Alloc`、[device_stream.cc](../../src/runtime/device_stream.cc) 的 `StorageCopySync`/`StorageCopyAsync` 读取线程本地当前观测器，未装配时只有一次空判断；记录路径本身分配内存不得递归触发钩子，需有窄的重入保护。compiled_module.cc 记录常量快照复制。
5. 写 profiling 适配器并装配：BuildCompiledModule 在 profile_context 存在且 enabled 时构造适配器存入模块节点；适配器把钩子译成 ProfileContext 记录。首提交就同时带上调用者和 bundle 正例，不允许只落接口。
6. 登记与检查：把两个新公共头加入 [CMakeLists.txt](../../CMakeLists.txt) 的 KXC_PUBLIC_HEADERS；跑 `check_include_layers.py` 和 `check_public_headers.py --compile` 验证依赖方向与头文件自包含。
7. 数值正例：新增 `test/runtime_profiling_test.cpp`（CMake 注册属集成负责人协调区，按既有 kxc_add_runtime_exe 与 kxc_add_ctest_if_target 方式登记，注册后用 `ctest -N` 确认用例存在）。先用 fake CompiledModule 加 RecordingLauncher（模式见 [runtime_session_test.cpp](../../test/runtime_session_test.cpp)）加手工 ProfileContext 验证事件关联；再用真实 Compiler::Compile 的 LLVM Where fixture（[op_numeric_llvm_test.cpp](../../test/op_numeric_llvm_test.cpp) 第 566 行，有矩阵归属）在启用 profiling 下运行，断言数值不变且 bundle 中出现同一 run_id 的 runtime_session_run 与 kernel_exec。通用 Add fixture 只能证明观测基础设施，不作为能力证据。
8. 负例与关闭态：错误输入仍抛出同样的异常，且该 run 在 bundle 中只有 status="error" 的 run span、没有任何 kernel_submit；launcher 注入失败产生 error 事件且异常不变；profiling 关闭时无 runtime 事件、输出与现状逐位一致；用一个会抛异常的观测器证明执行结果不受观测影响。

公共验证命令沿用[并行计划](WAVE_1.md)第 6 节；相关回归子集为 `profile_bundle_test|runtime_session_test|device_runtime_test|op_numeric_llvm_test` 加新测试。profiling 由 `KXC_PROFILE_ENABLE`/`KXC_PROFILE_BUNDLE_DIR` 控制（[options.cc](../../src/profiling/options.cc)），[profile_run 工具](../../python/kxc_agent/tools/profile_run.py) 已对任意命令设置这两个变量。

## 重点代码与测试

- 观测目标：[RuntimeSession](../../src/runtime/session.cc)、[执行计划](../../include/kxc/runtime/executable_plan.h)、[模块调用与失败分类](../../include/kxc/runtime/module_invocation.h)、[异步完成句柄](../../include/kxc/runtime/device_stream.h)。
- 观测设施：[profiling 公共头](../../include/kxc/profiling/profiling.h)、[profiling 实现](../../src/profiling/profiling.cc)、[环境变量覆盖](../../src/profiling/options.cc)。
- 既有测试：[bundle 断言](../../test/profile_bundle_test.cpp)、[会话行为](../../test/runtime_session_test.cpp)、[LLVM 数值](../../test/op_numeric_llvm_test.cpp)。
- 生产消费方证据：[编译器装配点](../../src/compiler/compiler.cc)、[热替换准备创建真实 session](../../src/compiler/adaptive/adaptive_hot_swap_preparation.cc)、[诊断引擎](../../python/kxc_agent/services/diagnosis_engine.py)、[工作台事件契约](../../tools/workbench/src/kxc/contract.ts)。
- 分层与公共头检查：[check_include_layers.py](../../tools/architecture/check_include_layers.py)、[check_public_headers.py](../../tools/architecture/check_public_headers.py)。

## M1 验收

- [ ] 真实 Compiler::Compile 加 RuntimeSession 的 LLVM 运行产出 bundle，events.jsonl 中 runtime_session_run、kernel_submit、kernel_exec、alloc 事件带有相同 run_id，内核与分配事件的 parent_span_id 指向 run span；重复运行和独立 session 不串数据。
- [ ] submit/complete/error 三种语义区分明确：kernel_submit 只表示主机完成提交动作；kernel_exec 终点是主机观测到完成的时刻，timing 字段区分 host_execute 与 host_observed_complete；完成回调在 Wait/IsReady/析构之间恰好结算一次有单测，延迟 completion 不会提前报完成。
- [ ] 输入校验失败仍抛出原异常，bundle 中该 run 只有 status="error" 的 run span 且实际 launch 次数为零；launcher 注入失败产生 error 事件且异常类型与文本不变。
- [ ] profiling 关闭时不产生任何 runtime 事件，输出与关闭前逐位一致；观测路径不调用 Wait/Sync/QueryEvent，不触发编译，不改变路由与调度策略。
- [ ] runtime 层文件不 include profiling 头，check_include_layers.py 通过；新公共头登记进 KXC_PUBLIC_HEADERS 并通过 check_public_headers.py --compile。
- [ ] events.jsonl 字段集合不变、schema_version 保持 1，schema.py 的 validate_bundle 通过；观测器与 ProfileContext 不进入模块与执行计划的 identity（identity 与缓存代码目前均不引用 profiling 选项，本模块不改变这一点）。
- [ ] 未扩展 GPU 计时，未改 KV cache、shape 路由与调度策略，未改模块 ABI；第一波范围外项在 PR 中逐项注明。

## 风险与交接

主要风险是把"主机提交耗时"当成"内核耗时"：CPU 上两者目前重合，一旦照搬到异步后端就会说谎。防法是上面的事件语义表——每个事件必须带 timing 字段，评审逐字段核对。其次是完成回调的安全性：回调不得抛出、不得再进入同一句柄的 Wait，析构路径触发时同样成立。第三个风险是观测器自身故障影响执行，适配器吞异常并由专门测试钉住；记录路径的重入保护也要有测试。

一个已知限制要如实交接：CPU 上构造不出能穿过模块启动边界的 pending 完成句柄——[compiled_module.cc](../../src/runtime/compiled_module.cc) 第 394 行会把"未完成且无后端 event"的句柄判为非法完成。因此异步的"提交后尚未观测完成"语义由 AsyncOperation 回调的单测钉住，真实异步证据留给后续 CUDA 波次，不能用 CPU 结果冒充。

交接给集成负责人三点：诊断引擎的 kernel_launch_overhead 规则期待 `compiled_module_run` 事件名，本波采用工作台词表 `runtime_session_run`/`kernel_exec`，分析器侧的对齐不属于本模块；新测试的 CMake 登记属共享文件协调区；bundle 产物只放 `out/` 或 CI artifact，本文不粘贴会过期的样例。M6 的替换决策将以本模块的事件语义表为输入；字段含义变更必须先升级 schema_version 并同步 schema.py 与 contract.ts。
