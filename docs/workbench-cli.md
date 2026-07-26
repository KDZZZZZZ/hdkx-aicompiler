# kxc-wb —— 让 agent 使用性能分析桌布

> 最后核验：2026-07-26。每条命令都实跑过。

这份文档同时给人和 agent 看。如果你在指挥 Claude 分析性能，把这个文件的路径告诉它就够了。

---

## 1. 它能做两件事

```
读数据（不需要浏览器）          驱动界面（需要浏览器开着）
kxc-wb query ...                kxc-wb ui ...
kxc-wb compare ...
```

**为什么要分两族**：agent 需要先"看见"数据才能判断该看哪张图。只能操作界面的 agent
只是个遥控器；只能读数据的 agent 没法把结论摆到你面前。两者合起来才是
"你说一句，它查完、想明白、把对应的图摆好"。

两族共用同一份聚合实现（`src/kxc/query.worker.ts`），所以 CLI 和界面**不会算出不同的数**。

---

## 2. 启动

```bash
# 一条命令起 dev server + 控制服务
npm run dev:all --prefix tools/workbench
```

页面打开后右下角出现 **● agent 已连接** 就说明控制链路通了。

只想读数据的话什么都不用起：

```bash
node tools/workbench/cli/kxc-wb.mjs query kpi --bundle <目录>
```

---

## 3. 给 agent 的使用约定

**永远加 `--json`。** 输出是单个 JSON 对象，不掺日志，直接 `JSON.parse`。

**先 `ui state`，再动手。** 状态里带着当前所有列和图的真实 id，以及
`availableTileTypes`（含每种图的用途和数据可用性）。不要猜 id。

**命令回执自带执行后的状态。** `ui add-tile` 的响应里有 `state` 字段，
不必再单独查一次——中间隔着 debounce，单独查会读到旧值。

**注意 `readiness`。** 状态和 `availableTileTypes` 里每种图都带这个字段：

| 值 | 含义 | 对 agent 的要求 |
| --- | --- | --- |
| `ready` | 数据齐全 | 正常用 |
| `partial` | 能画但有已知缺口 | **必须在结论里说明缺口**，别把它当可靠证据 |
| `blocked` | 核心数据没有 | 别摆这张图 |

**注意 `caveat`。** `query hotspot` 等结果里可能带 `caveat` 字段。出现了就说明
这个数字有水分（比如算子耗时其实来自值拷贝 span），**必须转述给用户**，
不能当成性能结论直接用。

---

## 4. query —— 读数据

```bash
kxc-wb query <kind> --bundle <目录> [--json] [过滤器]
```

| kind | 返回什么 |
| --- | --- |
| `meta` | 事件总数、时间跨度、component 分布、损坏行数 |
| `kpi` | 编译/运行总耗时、Pass 数、缓存命中率、告警数 |
| `phases` | 各阶段耗时 + 未归类耗时 |
| `pass-ranking` | Pass 耗时排行（`--limit`） |
| `pass-waterfall` | 逐 Pass 的起止、嵌套深度、IR 体积变化 |
| `hotspot` | 热点排行（`--by pass\|op\|kernel\|component`） |
| `shape-cache` | 每个 shape 的命中/未命中 |
| `kernel-dist` | kernel 耗时分布与分位数 |
| `diagnostics` | 诊断条目；未分析时会告诉你怎么生成 |
| `logs` / `events` / `facets` / `timeline` | 日志、原始事件、可选值分布、时间桶 |

**过滤器**（所有 kind 通用）：
`--pass --op --kernel --shape --component --event-type --device --run
--severity --status --search --start-ns --end-ns`

先用 `facets` 看有哪些取值，再用它们过滤——比猜名字可靠。

---

## 5. compare —— 找回归

```bash
kxc-wb compare --baseline <目录> --candidate <目录> [--threshold 1.2] [--json]
```

返回 KPI 对比 + 逐 Pass delta，每条带 `status`：
`suspected_regression` / `suspected_improvement` / `stable` / `noise` /
`only_in_baseline` / `only_in_candidate`。

**为什么是 `suspected_`。** 用两份真实 bundle 实测过：同一份代码连跑两次，
毫秒级 pass 的耗时能差出 1.4~1.9 倍、绝对差 2ms 以上，全是运行间抖动。
每侧只有一次观测时无法区分噪声与真实回归，所以一律标 `suspected_`；
只有当同名 pass 在两侧都有多次观测（`repeats > 1`）时才去掉前缀。
`--min-delta-ms`（默认 1）以下的差异直接判 `noise`。

**结果里的 `confidence` 字段 agent 必须转述**：

```json
"confidence": {
  "level": "single_run",
  "caveat": "每个 pass 在两侧各只观测到一次，因此所有差异只标为 suspected_*，不能当作结论……"
}
```

要真正判定回归，需要同一配置重复运行多次后比较分位数——项目设计文档
也明确要求「不用单次运行宣称性能回归或收益」。

**结果里有个 `alignment` 字段，agent 必须读。** 它说明哪些维度可比：

```json
"alignment": {
  "pass": "reliable",
  "op": "unavailable：顶层 op_name 未被编译器填充，无法跨 bundle 对齐算子",
  "kernel": "unavailable：CPU 路径下 kernel_symbol 多为空"
}
```

也就是说：**基于 Pass 的回归结论可信，基于算子或 Kernel 的不要下**。

---

## 6. ui —— 驱动界面

```bash
kxc-wb ui state                                  # 先看这个
kxc-wb ui load-bundle <样例 id>                  # 把数据装进页面
kxc-wb ui new-column --title "回归定位" --width 1/2
kxc-wb ui add-tile pass_ranking [--column-id col-3]
kxc-wb ui drill --kind pass --value fold_tuple_get_item [--as-tab]
kxc-wb ui gather tile-3 tile-5 tile-7
kxc-wb ui set-bundle --bundle-id <id>
kxc-wb ui set-baseline --bundle-id <id>
kxc-wb ui mode --mode strip|gather|overview|focus
kxc-wb ui set-filter --patch '{"pass":"fold_constant"}'
kxc-wb ui focus --tile-id tile-3 [--fullscreen]
kxc-wb ui consume --tile-id tile-3 --direction right
kxc-wb ui expel --tile-id tile-3
kxc-wb ui pin --column-id col-2
kxc-wb ui remove-tile --tile-id tile-3
kxc-wb ui remove-column --column-id col-2
kxc-wb ui new-workspace --name "显存分析"
kxc-wb ui switch-workspace --workspace-id ws-1
kxc-wb ui undo | redo
```

`add-tile` 不指定 `--column-id` 就加到当前聚焦列；一列都没有会自动建一列。

命令是**白名单**的，不是把 store 的 action 全反射出去——那样重构内部实现就会
悄悄破坏 agent。要加新命令请改 `src/ui/agent/useAgentBridge.tsx` 里的 `COMMANDS`。

---

## 7. 一次完整的 agent 工作流

用户说："看看 candidate 比 baseline 慢在哪，把证据摆出来。"

```bash
# 1. 先读数据，得出结论
kxc-wb compare --baseline fixtures/bundles/baseline \
                --candidate fixtures/bundles/candidate --json
```

拿到：编译 16.56ms → 19.38ms；`fold_tuple_get_item` 回归 3.79×；
缓存命中率 71% → 42%，未命中 7 → 14。

```bash
# 2. 摆证据。先看现状，别猜 id
kxc-wb ui state --json

# 3. 把两份数据装进页面并进入对比
kxc-wb ui load-bundle candidate
kxc-wb ui load-bundle baseline
kxc-wb ui set-baseline --bundle-id url:bundles/baseline

# 4. 建一列放主证据
kxc-wb ui new-column --title "回归：fold_tuple_get_item" --width 1/2 --json
kxc-wb ui add-tile regression_delta --json     # 回执里就有新 tileId

# 5. 缓存那条线索单独一列
kxc-wb ui new-column --title "缓存退化" --json
kxc-wb ui add-tile shape_cache_heatmap --json

# 6. 并排比量级
kxc-wb ui gather <tileId1> <tileId2>
```

然后向用户报告时，**同时说清三件事**：
1. Pass 侧多花约 3.75 ms，但这是单次运行的观测，只能算线索；
2. 缓存多出的 7 次未命中每次触发一轮同步编译（约 12 ms），量级更大；
3. 算子维度无法对齐，所以没有给出算子级结论。

---

## 8. 出错时怎么办

CLI 的错误信息是**可行动的**，照着改就行：

| 错误 | 含义与做法 |
| --- | --- |
| `连不上控制服务` | 起 `npm run dev:all`，或只起 `npm run control` |
| `没有页面连上控制服务` | 浏览器没开或没连上；确认右下角有"agent 已连接" |
| `Bundle 未加载: X；已加载的有 ...` | 用列出的 id，或先在界面里加载 |
| `未知图表类型 X；可用：...` | 从列表里挑 |
| `列不存在 / Tile 不存在` | 先 `ui state` 拿真实 id |
| `页面未在 5 秒内回执` | 页面卡住或崩了，看浏览器控制台 |

---

## 9. 边界

- 控制服务只监听 `127.0.0.1`，不暴露到网络（§19 本地优先）
- CLI **不会**把任何数据发到外部；唯一外发路径是界面上手动点"在 Perfetto 中打开"
- `query` 直接读磁盘上的 bundle 目录，与浏览器无关
- 命令面板里的「搜索 Pass / Op / Kernel」仍是占位项，agent 请用 `query facets`
- P1/P2 图表（Execution DAG、内存曲线、Roofline 等）尚未实现，`add-tile` 会报未知类型

---

## 10. 相关文档

- [使用手册](workbench-user-guide.md) —— 给人用的界面说明
- [开发说明与数据契约](../tools/workbench/README.md) —— bundle 解析的 8 条陷阱
