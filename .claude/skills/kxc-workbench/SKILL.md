---
name: kxc-workbench
description: 操作 KXC 性能分析图表桌布（performance workbench）。读取 profiling bundle 并计算 KPI/Pass 耗时/缓存命中/回归对比（kxc-wb query/compare，无需浏览器）；或驱动用户正在浏览的工作台页面摆图表、下钻、聚合（kxc-wb ui）。当用户要求分析编译或推理性能、对比 baseline、定位性能回归、展示或调整性能图表、操作桌布/workbench 时使用。
---

# 操作 KXC 性能分析桌布

你有一个 CLI（`kxc-wb`）可以做两类事：**读数据**（不需要浏览器）和**驱动用户正在看的工作台页面**。两者共用同一份聚合实现，不会算出不同的数。

## 0. 先分辨任务形态

| 用户要什么 | 走哪条路 |
| --- | --- |
| "慢在哪 / 有没有回归 / 缓存命中率多少" | 只用 `query` / `compare`，直接汇报数字 |
| "把 XX 图摆出来 / 展示给我看 / 调一下布局" | 先 `query` 得出结论，再用 `ui` 摆证据 |
| "分析一下这次编译为什么慢" | 完整流程：query → 结论 → ui 摆证据 → 汇报 |

**先读数据再摆图。** 只会操作界面的 agent 是遥控器；先用 query 判断该看哪几张图，再把它们摆到用户面前。

## 1. 环境

```bash
# 读数据：什么都不用起，直接跑（cwd = 仓库根）
node tools/workbench/cli/kxc-wb.mjs query kpi --bundle <bundle目录> --json

# 驱动界面：需要 dev server + 控制服务都在跑
npm run dev:all --prefix tools/workbench     # 一条命令起两个
```

- `ui` 命令报"连不上控制服务" → 控制服务没起，跑上面的 dev:all（或 `npm run control --prefix tools/workbench`）。
- `ui` 命令报"没有页面连上" → 让用户打开 http://localhost:5273/ 并确认右下角有「● agent 已连接」。
- 样例数据在 `tools/workbench/fixtures/bundles/`；没有就先 `node tools/workbench/scripts/make-fixture.mjs`。
- `query`/`compare` 需要 Node 23.6+（22.6+ 配 `NODE_OPTIONS=--experimental-strip-types`）；`ui` 任何 Node 都行。版本不足时 CLI 会自己给出指引。

## 2. 硬规则（违反会输出错误结论）

1. **永远加 `--json`**，解析单个 JSON 对象，不要抓文本。
2. **动界面前先 `ui state`**，用返回的真实 id，不要猜 `col-2`/`tile-3` 这类 id。
3. **命令回执自带执行后的快照**（`state` 字段），执行完不要再单独查一次状态——单独查可能读到 debounce 前的旧值。
4. **`readiness` / `caveat` / `confidence` 字段必须转述给用户**，一个都不能吞：
   - Tile 的 `readiness: partial|blocked` → 说明数据缺口（如"算子耗时来自值拷贝 span，只能看结构，不能当性能结论"）。
   - 查询结果里的 `caveat`（hotspot、kernel-dist 会带）→ 原样转述。
   - `compare` 的 `confidence` → 见下一条。
5. **单次运行不许宣称"回归"**。`compare` 在每侧只有一次观测时只给 `suspected_regression`；实测同一份代码连跑两次，毫秒级 pass 耗时可差 1.4~1.9 倍。汇报时用"疑似回归，需重复运行确认"，绝不说"回归了 X 倍"。
6. **对齐边界**：`compare` 的 `alignment` 字段写明 Pass 维度可靠、算子与 Kernel 维度不可对齐（编译器未填充相应字段）。不要给出算子级或 kernel 级的跨 bundle 结论。

## 3. 读数据：query / compare

```bash
node tools/workbench/cli/kxc-wb.mjs query <kind> --bundle <目录> --json [过滤器]
```

| kind | 用途 |
| --- | --- |
| `kpi` | 编译/运行总耗时、Pass 数、缓存命中率、告警数——**先跑这个** |
| `phases` | 各阶段耗时 + 未归类耗时 |
| `pass-ranking` | Pass 耗时排行（`--limit N`） |
| `pass-waterfall` | 逐 Pass 起止/嵌套/IR 体积变化 |
| `hotspot` | 热点（`--by pass\|op\|kernel\|component`；op 维度会带 caveat） |
| `shape-cache` | 每个输入 shape 的缓存命中/未命中 |
| `kernel-dist` | kernel 耗时分布（无 CUPTI 时带 caveat） |
| `diagnostics` | 诊断；未分析时会提示先跑 `analyze_bundle` |
| `facets` | 各字段的实际取值分布——**过滤前先跑这个**，不要猜名字 |
| `meta` / `events` / `logs` / `timeline` | 元信息 / 原始事件分页 / 日志 / 时间桶 |

过滤器通用：`--pass --op --kernel --shape --component --event-type --device --run --severity --status --search --start-ns --end-ns`。

```bash
node tools/workbench/cli/kxc-wb.mjs compare --baseline <目录> --candidate <目录> --json
```

返回 KPI 对比 + 逐 Pass delta（`status`: `suspected_regression|suspected_improvement|stable|noise|only_in_*`）+ `alignment` + `confidence`。`--min-delta-ms`（默认 1）以下判 noise。

## 4. 驱动界面：ui

```bash
node tools/workbench/cli/kxc-wb.mjs ui state --json          # 永远先看这个
node tools/workbench/cli/kxc-wb.mjs ui load-bundle <样例id>   # 把数据装进页面
node tools/workbench/cli/kxc-wb.mjs ui set-baseline --bundle-id <已加载id>
node tools/workbench/cli/kxc-wb.mjs ui new-column --title "回归定位" --width 1/2
node tools/workbench/cli/kxc-wb.mjs ui add-tile <类型> [--column-id <id>]
node tools/workbench/cli/kxc-wb.mjs ui drill --kind pass --value <pass名> [--tile-id <id>]
node tools/workbench/cli/kxc-wb.mjs ui gather <tileId> <tileId> ...
node tools/workbench/cli/kxc-wb.mjs ui set-filter --patch '{"pass":"fold_constant"}'
node tools/workbench/cli/kxc-wb.mjs ui mode --mode strip|gather|overview|focus
node tools/workbench/cli/kxc-wb.mjs ui pin --column-id <id>   # 防止列被后续下钻替换
node tools/workbench/cli/kxc-wb.mjs ui undo | redo
```

Tile 类型（`ui state` 的 `availableTileTypes` 里有完整列表 + readiness）：
`kpi` `phase_breakdown` `hotspot_topn` `pass_waterfall` `pass_ranking` `shape_cache_heatmap` `kernel_duration_dist` `kernel_topn` `diagnostics` `regression_delta` `logs` `event_table` `perfetto_timeline` `ir_diff`

语义要点：
- `add-tile` 不给 `--column-id` 就加到聚焦列，一列都没有会自动建。
- `drill` 会在右侧建新列并**替换其右侧未 Pin 的列**——想保住某列先 `pin`。
- `gather` 里的图是引用；聚合前确认 tileId 都存在（`ui state`）。
- 命令失败的错误信息是可行动的（会列出可用值/已加载 bundle），照着改即可。

## 5. 标准工作流：定位一次回归

```bash
# 1. 数据先行
node tools/workbench/cli/kxc-wb.mjs compare --baseline <b> --candidate <c> --json
# → 记下：KPI 差异、suspected_regression 的 pass、缓存命中率变化、confidence

# 2. 摆证据（页面开着才做这步）
node tools/workbench/cli/kxc-wb.mjs ui state --json
node tools/workbench/cli/kxc-wb.mjs ui load-bundle candidate
node tools/workbench/cli/kxc-wb.mjs ui load-bundle baseline
node tools/workbench/cli/kxc-wb.mjs ui set-baseline --bundle-id url:bundles/baseline
node tools/workbench/cli/kxc-wb.mjs ui new-column --title "回归" --width 1/2 --json
node tools/workbench/cli/kxc-wb.mjs ui add-tile regression_delta --json
node tools/workbench/cli/kxc-wb.mjs ui add-tile shape_cache_heatmap --json
node tools/workbench/cli/kxc-wb.mjs ui gather <上面回执里的 tileId...>
```

汇报模板（三件事都要说）：
1. Pass 侧差异 + **这是单次运行，只能算线索**；
2. 缓存/运行时侧的量级对比（miss 次数 × 单次同步编译代价，往往比 pass 差异大）;
3. 哪些维度没有结论（算子/kernel 不可对齐；readiness 非 ready 的图数据有缺口）。

## 6. 数据边界（不要试图绕过）

| 限制 | 原因 |
| --- | --- |
| 算子耗时只能看结构 | `kernel_exec` span 包裹的是值拷贝，不是真实 kernel 执行 |
| Perfetto 选中无法回跳 | trace.json 的 args 缺 span_id |
| 诊断证据只能近似跳转 | evidence 无 span/时间引用 |
| 无 GPU 硬件耗时 | 需要 CUPTI（`manifest.cupti_available` 为 false 时） |
| runtime 类图表可能全空 | bundle 只含编译期事件时属正常（如 real-compile），不是故障 |

诊断显示"尚未分析"时，先跑：`PYTHONPATH=python python -m kxc_agent.cli analyze_bundle --bundle <目录>`，再重新加载。

## 7. 深入参考

- 完整命令与错误对照：[docs/workbench-cli.md](../../../docs/workbench-cli.md)
- 面向人的界面手册：[docs/workbench-user-guide.md](../../../docs/workbench-user-guide.md)
- bundle 解析陷阱（8 条）：[tools/workbench/README.md](../../../tools/workbench/README.md)
