# KXC Agent System V1

## Scope

V1 只落地性能分析和结构化日志控制面：

- 编译器侧统一事件模型
- bundle 产物写盘
- Python 控制面离线分析
- 面向后续 `DebugAgent / TuningAdvisor / PassAuthor / OperatorAuthor` 的稳定数据契约

## Bundle Layout

单次 profile/run 输出一个 bundle 目录，至少包含：

- `manifest.json`
- `events.jsonl`
- `trace.json`
- `summary.json`
- `diagnosis.json`
- `diagnosis.md`
- `artifacts/`

其中：

- `events.jsonl` 是 agent 主消费源
- `trace.json` 兼容 Perfetto / Chrome Trace
- `summary.json` 提供快速聚合
- `artifacts/` 保存 IR dump 和失败上下文

## Core Event Fields

基础字段固定包含：

- `trace_id`
- `session_id`
- `run_id`
- `span_id`
- `parent_span_id`
- `component`
- `event_type`
- `phase`
- `ts_ns`
- `duration_ns`
- `status`
- `severity`
- `device`
- `worker_id`
- `op_name`
- `pass_name`
- `kernel_symbol`
- `shape_signature`
- `message`
- `metrics`

## Current C++ Instrumentation Coverage

已接入：

- `Compiler::Compile`
- Relay pass pipeline
- `LowerToTIR`
- TIR pass pipeline
- `RuntimeSession`
- background compile submission/completion
- `ExecutionPlanExecutor`
- CPU/CUDA `DeviceAPI` alloc/free/copy/stream/sync

当前 bundle smoke test：

- `profile_bundle_test`

## CompileConfig / Env

`CompileConfig` 新增 `profile_options`，同时支持环境变量覆盖：

- `KXC_PROFILE_ENABLE=1`
- `KXC_PROFILE_BUNDLE_DIR=<dir>`
- `KXC_PROFILE_LOG_LEVEL=debug|info|warn|error`
- `KXC_PROFILE_IR_MODE=disabled|changed_or_failed|verbose`
- `KXC_PROFILE_NVTX=1`
- `KXC_PROFILE_CUPTI=1`
- `KXC_PROFILE_RECORD_PASS_IR=1`
- `KXC_PROFILE_EXEC_PLAN_DETAILS=1`

## Python Control Plane

入口在 `python/kxc_agent/`，使用前设置：

```powershell
$env:PYTHONPATH='python'
```

CLI：

```powershell
python -m kxc_agent.cli analyze_bundle --bundle <bundle_dir>
python -m kxc_agent.cli compare_bundles --base <base_bundle> --new <new_bundle>
python -m kxc_agent.cli explain_logs --bundle <bundle_dir>
python -m kxc_agent.cli inspect_pass_trace --bundle <bundle_dir> --stage relay
python -m kxc_agent.cli profile_run --bundle <bundle_dir> -- <command> ...
```

稳定 Python API：

- `profile_run(...) -> bundle_path`
- `analyze_bundle(bundle_path) -> diagnosis.json payload`
- `compare_bundles(base_bundle, new_bundle) -> regression report`

## Analysis / Memory

诊断器当前输出的结构化类别包括：

- `compile_hotspot`
- `pass_regression`
- `kernel_launch_overhead`
- `cache_miss_pattern`
- `background_compile_stall`
- `copy_dominance`
- `sync_overhead`
- `shape_fragmentation`
- `execution_plan_imbalance`
- `lowering_failure_context`

本地记忆目录默认写到：

- `.kxc_agent_memory/bundle_index.json`
- `.kxc_agent_memory/diagnosis_history.jsonl`

## Known Gaps

- NVTX 目前是 host span 映射的最小动态加载实现
- CUPTI 选项位已预留，但 v1 还未采集 activity
- 本机验证目前走 CPU-only；LLVM/CUDA/NVTX/CUPTI 实测放在远端 `pgx`
