# M1 技术报告：运行 metadata 与模型证据关联

> 状态：**已完成本切片（2026-09-08）**。本报告覆盖 runtime run metadata 的透传；它不宣称 MiniMind 生成循环或 KV state owner 已经完成。

## 目标

第一波 profiling 已经能按 `run_id` 关联 `runtime_session_run`、kernel、分配和拷贝事件，但调用方无法把一次运行标注为某个导出回执、prefill/decode 阶段、generation 或 state extent。G2 第 4 项因此缺少可消费的模型级证据。

## 方法

在 runtime 层增加不解释的字符串键值 metadata 通道。`RuntimeSession::Run`/`RunAsync` 保留原有无 metadata 重载，并提供带 metadata 的重载；runtime 只把字段放入 `ExecutionRunStart`，不参与输入校验、kernel ABI、计划 identity 或内存布局。profiling 适配器在 run 建立时复制 metadata，并将它附加到该 run 的结束、kernel submit/exec、alloc 和 copy 事件；异步完成回调捕获同一份副本，避免依赖线程局部状态或已结束的 run。

字段使用方可以按需提供 `export_receipt`、`stage`、`generation`、`plan_abi`、`state_extent`、`state_version` 等键。runtime 保留字段（例如 `timing`、`call_index`）优先，防止调用方 metadata 改写观测事实。

## 改动

- `include/kxc/runtime/execution_observer.h`
  - 定义 `ExecutionMetadata`，并把它加入 `ExecutionRunStart`。
- `include/kxc/runtime/session.h`、`src/runtime/session.cc`
  - 增加带 metadata 的同步/异步运行入口，旧 API 委托到空 metadata 路径。
- `include/kxc/profiling/runtime_observer.h`
  - 将 metadata 复制到整个 run 的 runtime 事件和延迟完成回调。
- `test/runtime_profiling_test.cpp`
  - 新增 `run_metadata_association`，检查 run、kernel submit/exec 与 alloc 在同一 `run_id` 下继承导出回执、阶段、代际、ABI 和 state extent 字段。

## 验证与效果

执行：

```bash
cmake --build out/build/dev-ninja-cpu --target runtime_profiling_test -j2
./out/build/dev-ninja-cpu/runtime_profiling_test
```

结果：**全部 12 个 runtime profiling 用例通过**，其中新增关联用例通过。旧的无 metadata API 与关闭 profiling 路径保持原有行为；metadata 不会出现在 kernel 参数或编译身份中。

## 后续边界

本切片提供稳定的字段承载和 run 继承，但调用方仍需在真实 MiniMind prefill/decode 驱动中填入 receipt、stage 和有效长度；它不替代 M2 的 session-owned KV state，也不自动创建 generation 或热替换路由。
