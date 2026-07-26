# KXC 性能分析图表桌布

面向 KXC 编译器与推理性能分析的可组合 GUI 工作台。需求基线见
[docs/plans/2026-07-26-performance-workbench-requirements.md](../../docs/plans/2026-07-26-performance-workbench-requirements.md)。

## 快速开始

```bash
npm install --prefix tools/workbench
node tools/workbench/scripts/make-fixture.mjs   # 生成样例 bundle
npm run dev --prefix tools/workbench            # http://localhost:5273
```

顶栏可直接加载 `fixtures/bundles/` 下的样例，或用「打开目录」选择本机任意 KXC bundle
（走 File System Access API，文件不离开浏览器进程）。

```bash
npm run build --prefix tools/workbench
npm test --prefix tools/workbench
```

## 数据契约的几个坑

前端解析 bundle 时踩到的实际差异，全部对照 `src/base/profiling.cc` 核验过，
细节写在 [src/kxc/contract.ts](src/kxc/contract.ts) 的注释里。最容易出错的几条：

| 现象 | 实际情况 | 出处 |
| --- | --- | --- |
| 时间戳字段 | JSON key 是 `ts_ns`，**不是** `timestamp_ns` | profiling.cc:759 |
| 算子名 | 顶层 `op_name` 从未被赋值，恒为空串；真实算子名在 `fields.op_name` | executor.cc:128 |
| Pipeline 事件 | component 是 `relay_pipeline` / `tir_pipeline`，event_type 是 `run_pipeline`；按 `relay_pass` 过滤取不到 | 真实 bundle 实测 |
| `worker_id` | 未赋值时是 `-1`，不是 null | profiling.cc:764 |
| `summary.json` | `component_counts` 的值是**字符串**，直接相加会字符串拼接 | profiling.cc:1104-1125 |
| 缓存事件的 shape | `cache_*` span 只带 `fields.shape_hash`，没有 `shape_signature`，需靠 hash 表或父 span 补全 | runtime_session.cc:57/65/72 |
| `trace.json` args | 只有 status/severity/message，**没有** run_id/span_id | profiling.cc:789-793 |
| 诊断 evidence | 只有标量与文本，无 span/时间引用；C++ 自动写出的版本只有 4 个字段 | profiling.cc:1142-1146 |

## 当前数据能支撑到什么程度

Tile 的 `readiness` 标注在 [src/tiles/registry.ts](src/tiles/registry.ts)，UI 上会如实显示，
不会因为数据缺失就画一张看起来正常的空图。

**ready** — KPI、阶段耗时分解、Pass Waterfall、Pass 排行、Shape × Cache 热力图、日志、事件表

**partial（有已知缺口，UI 会说明原因）**
- Hotspot / Kernel 类：算子耗时来自 `execution_plan.kernel_exec` span，而该 span 当前包裹的是
  值拷贝而非真实 kernel 执行（executor.cc:153），只能用于结构分析，不能当性能结论
- Timeline：内置轻量时间线可用；嵌入 Perfetto 后无法从选中事件回跳（trace.json 缺 span_id）
- 诊断：evidence 无精确引用，只能按 pass/component 名字近似回跳
- 回归对比：Pass 维度可按名字自动对齐，算子/Kernel 维度因字段未填充无法对齐

想解除这些限制需要改 C++ 侧埋点（补 trace.json args、顶层 op_name、evidence 引用），
本轮刻意没有改动编译器代码。

## 目录

```
src/kxc/        bundle 解析与统一查询层（Worker 内聚合，主线程只拿结果）
src/state/      空间模型、zustand store、Undo/Redo、Gather 自动布局
src/tiles/      图表 Tile 与目录注册表
src/ui/         外壳、列、模式、命令面板、键盘
scripts/        fixture 生成器
test/           数据管线测试与应用冒烟测试
```

`fixtures/bundles/real-compile` 是 `profile_bundle_test` 跑出来的**真实产物**（22 个事件，
只有编译期事件、没有任何 runtime/cache 数据），用来验证 fixture 没有失真，
也用来检验各 Tile 在稀疏数据下的退化表现。
