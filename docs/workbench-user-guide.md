# 性能分析图表桌布 · 使用手册

> 适用版本：`tools/workbench` v0.1.0
> 最后核验：2026-07-26（每一条操作都对照实现核对过；尚未实现的会明确标注）

---

## 1. 它解决什么问题

分析一次编译或推理为什么慢，通常要来回看很多张图：先看总耗时，再看哪个阶段慢，再钻到具体 Pass，再对比基线判断是不是回归。传统仪表盘的做法是把图固定在格子里，你要么来回切页面，要么把所有图挤在一屏。

这个工作台换了个思路：**分析过程沿着一条横向条带向右生长**。每深入一层就在右边新开一列，来源列留在左边，你随时能回头看自己是怎么走到这一步的。等需要横向比较时，再把散落各处的图临时抓到同一屏（Gather）。

如果你用过 niri 或 PaperWM 这类滚动式平铺窗口管理器，这个交互模型会很熟悉。

---

## 2. 五分钟上手

### 2.1 启动

```bash
npm install --prefix tools/workbench
node tools/workbench/scripts/make-fixture.mjs    # 生成样例数据，只需跑一次
npm run dev --prefix tools/workbench
```

打开 http://localhost:5273/。

### 2.2 加载数据

顶栏左侧的 **Bundle** 下拉里选一个：

| 样例 | 内容 | 用途 |
| --- | --- | --- |
| `baseline` | 222 个事件，完整编译 + 运行时 | 日常试用 |
| `candidate` | 223 个事件，`fold_tuple_get_item` 被人为放慢 3.4× | 试对比功能 |
| `raw-unanalyzed` | 同 baseline，但没跑过诊断分析 | 看"未分析"状态长什么样 |
| `real-compile` | **22 个事件，编译器真实产物** | 看真实数据有多稀疏 |

分析自己的数据点 **打开目录…**，选中任意 KXC profiling bundle 目录（就是含 `events.jsonl` 的那个）。文件通过浏览器的 File System Access API 读取，**不上传、不出本机**，顶栏会显示 🔒 本地。

> 需要 Chrome / Edge 95+。Firefox 和 Safari 不支持目录选择，只能用样例。

### 2.3 摆出第一张图

按 `Ctrl/Cmd + K` 打开命令面板 → 输入图表名 → 回车。图会加到当前列；还没有列的话会先建一列。

建议的第一组：**KPI 指标组** → **阶段耗时分解** → **Pass 耗时排行**。

### 2.4 往下钻

在 Pass 排行里点中最慢的那个 Pass，按 `Enter`。右边会出现一个新列，标题是这个 Pass，里面装好了 Pass Waterfall 和事件表，过滤条件已经自动限定到它。

左边的来源列还在。这就是核心用法。

---

## 3. 空间模型

```
Workspace（一个分析问题）
  └── Strip（横向条带）
        └── Column（一个分析上下文）
              └── Tile（一张图）
```

- **Workspace** —— 一个独立的问题。"为什么这次编译慢了" 和 "显存峰值从哪来" 应该是两个 Workspace。
- **Column** —— 一组过滤条件。同一列里的所有图看的是同一个切片。
- **Tile** —— 一张图、一份日志或一段 IR。

关键区别：**Column 承载"看哪一部分数据"，Tile 承载"用什么角度看"**。想换角度就往列里加 Tile，想换数据范围就开新列。

---

## 4. 五种模式

| 模式 | 快捷键 | 什么时候用 |
| --- | --- | --- |
| **Strip** | 默认 | 沿分析路径一路向右深入 |
| **Gather** | `G` | 把散在各列的图抓到一屏横向比较 |
| **Overview** | `O` | 列太多找不着北时，鸟瞰所有 Workspace 与列 |
| **Focus** | `F` | 单张图占满屏幕细看 |
| **Compare** | `C` | 与基线对比，找回归 |

### 4.1 Strip：分析路径

- 单击图元 → 在**当前**上下文里选中/过滤
- 双击 或 `Enter` → **Drill Down**，右侧新建列
- `Shift + Enter` → 结果作为当前列的 Tab 打开，不新开列

从中间某列重新钻取时，**它右侧未固定的列会被替换掉**——这符合"换一条分析路径"的直觉。想保住某列不被冲掉，选中它按 `P` 固定（Pin）。

想让新分支独立出去，用命令面板的 **Fork Workspace**。

### 4.2 Gather：临时聚合

在 Strip 里选中若干 Tile，按 `G`。

Gather 里的图是**引用**，不是副本：

- 它和原图共享时间范围、Bundle、当前算子、hover、选择
- 在 Gather 里关掉一张图，**原图不受影响**
- 布局可以命名保存，一个 Workspace 能存多套

布局按选中数量自动排：

| 数量 | 排法 |
| --- | --- |
| 2 | 1/2 + 1/2 |
| 3 | 2/3 主视图 + 1/3 两个纵向 |
| 4 | 2 × 2 |
| 5–6 | 3 列，每列最多 2 个 |
| >6 | 不再继续缩小，改为自动建 Tab；Timeline / IR Diff 这类宽视图优先占大格 |

手工拖过之后会标记"已手工调整"，点 **恢复** 回到自动布局。

> **重型视图限流**：Timeline、IR Diff、事件表这类图同屏最多 3 个在活动渲染，其余显示"已暂停渲染，点击激活"。这是刻意的——20 张图同时跑 ECharts 会卡死。点一下就能换进来。

### 4.3 Overview：鸟瞰

看所有 Workspace 和列的分布，支持点击跳转、拖动重排、搜索、按图表类型和严重级别过滤。

缩略图只用于**认出这是哪张图**，坐标轴和标签不保证可读。视口外的缩略图不渲染真实内容，重型视图在 Overview 里一律不渲染——否则一百个缩略图能把浏览器拖垮。

### 4.4 Compare：找回归

顶栏 **Baseline** 下拉选一个已加载的 bundle，就进入对比状态。

试一下：Bundle 选 `candidate`，Baseline 选 `baseline`，加一个 **回归对比** Tile。`fold_tuple_get_item` 会明显标红。

**对齐能力有边界**：Pass 维度按名字自动对齐，可靠；算子和 Kernel 维度**对不齐**，因为当前编译器没有填充 `op_name` / `kernel_symbol` 顶层字段（详见 §8）。

---

## 5. 布局操作

| 操作 | 快捷键 | 说明 |
| --- | --- | --- |
| 上一列 / 下一列 | `H` / `L` | |
| 同列上/下一个 Tile | `J` / `K` | |
| Drill Down | `Enter` | 右侧新建列 |
| 作为 Tab 打开 | `Shift + Enter` | |
| Consume 到左/右列 | `[` / `]` | 把当前 Tile 收进相邻列 |
| Expel | `E` | 把 Tile 拆出成独立列 |
| Pin | `P` | 固定列，不被下钻替换 |
| Focus | `F` | 单图全屏 |
| Gather | `G` | |
| Overview | `O` | |
| Compare | `C` | |
| 搜索 | `/` | 聚焦顶栏搜索框 |
| 命令面板 | `Ctrl/Cmd + K` | |
| 返回上一级 | `Esc` | 关面板 → 退出 Focus → 回 Strip → 清除选择 |
| 撤销 / 重做 | `Ctrl/Cmd + Z` / `Ctrl/Cmd + Shift + Z` | |

单键快捷键在输入框里不会触发。

**改键**：默认表存在 `localStorage` 的 `kxc-keymap-v1`；把 `kxc-keymap-disabled` 设为 `"true"` 可整体关闭快捷键（目前只能在浏览器控制台改，还没有设置界面）。

### 鼠标与触控板

| 操作 | 效果 |
| --- | --- |
| 滚轮 | 滚动当前列内部 |
| `Shift` + 滚轮 | 横向移动条带 |
| 空白处拖动 / 按住 `Space` 拖动 | 横向移动条带 |
| 拖标题栏 | 移动 Tile 或列 |
| 双击标题栏 | Focus |
| 拖边界 | 调整尺寸 |
| 触控板横向手势 | 横移条带（带方向锁定，避免横纵抖动） |

---

## 6. 图表说明

### 编译分析

| 图 | 回答什么 |
| --- | --- |
| **KPI 指标组** | 总耗时、Pass 数、缓存命中率、告警数 |
| **阶段耗时分解** | 时间花在 Relay / Lowering / TIR / CodeGen 哪一段 |
| **Pass Waterfall** | 每个 Pass 按顺序的耗时与 IR 体积变化 |
| **Pass 耗时排行** | 最慢的 Pass，以及它是否真的改了 IR |

> Pass Waterfall 会**弱化显示没改动 IR 的 Pass**。一个 Pass 花了时间却没改 IR，这本身就是个结论。

阶段耗时用的是 pipeline 汇总事件，不是把逐个 Pass 加起来——后者会把嵌套 span 重复计入。图上单独标出的"未归类耗时"是总耗时减去各阶段之和，不为零说明有时间没被埋点覆盖。

### 运行时分析

| 图 | 回答什么 |
| --- | --- |
| **Shape × Cache 热力图** | 哪些输入 shape 反复未命中缓存、触发同步编译 |
| **Hotspot Top-N** | 按 Pass / 算子 / Kernel 排最耗时的对象 |
| **Kernel 耗时分布 / Top-N** | kernel 耗时形态与长尾 |

### 通用

| 图 | 回答什么 |
| --- | --- |
| **Timeline** | 事件在时间轴上的分布，可框选时间范围联动全局 |
| **事件表** | 按当前过滤条件浏览原始事件 |
| **日志** | 按级别看编译期日志与告警 |
| **诊断** | 自动诊断结论、证据、下一步建议 |
| **回归对比** | Baseline 与 Candidate 的逐 Pass 差异 |
| **IR Before/After** | 某个 Pass 前后的 IR 文本对比 |

---

## 7. 联动与过滤

### 全局过滤

顶栏的搜索框、时间范围，以及任意图里的选择，都会写进全局上下文，所有图跟着变。

### 局部过滤

每个 Tile 都能覆盖部分全局条件：

- **锁定** —— 之后不再跟随全局变化
- **重置** —— 回到完全跟随全局
- **提升为全局** —— 把这个 Tile 的条件推给所有图

典型用法：主区域跟着全局走，旁边固定一个锁定在基准 shape 上的图当参照。

### 联动组

默认所有图在同一个联动组里。需要两条互不干扰的分析线时，可以建新组——不同组之间时间范围、hover、选择互不影响。

---

## 8. 数据能支撑到什么程度（重要）

工作台会**如实标注数据缺口**，不会因为拿不到数据就画一张看起来正常的空图。带 ⚠ 的 Tile 会把原因直接写在标题栏或图上。

以下限制来自 KXC 编译器当前的埋点，不是前端问题：

| 限制 | 影响 | 原因 |
| --- | --- | --- |
| 算子耗时不可信 | Hotspot 的**算子维度**只能看结构，不能当性能结论 | `execution_plan.kernel_exec` span 包裹的是值拷贝，不是真实 kernel 执行（[executor.cc:153](../src/base/disco/executor.cc:153)） |
| Perfetto 无法回跳 | 在 Perfetto 里选中事件，工作台不知道对应哪条 | `trace.json` 的 args 只有 status/severity/message，没有 span_id（[profiling.cc:789](../src/base/profiling.cc:789)） |
| 诊断证据只能近似定位 | 点击证据按 pass/component 名字跳转，无法精确到事件 | evidence 只有标量和文本，不含 span_id 或时间范围 |
| 算子/Kernel 维度无法对比 | Compare 只有 Pass 维度可靠 | 顶层 `op_name` / `kernel_symbol` 未被填充 |
| 无 GPU 硬件耗时 | Kernel 分布是逻辑耗时 | 需要 CUPTI；当前 bundle `cupti_available: false` |

解除这些限制需要改编译器侧埋点，不是改前端能解决的。

### 诊断显示"尚未分析"？

C++ 侧只会写一条占位诊断。真正的分析要单独跑：

```bash
PYTHONPATH=python python -m kxc_agent.cli analyze_bundle --bundle <bundle 路径>
```

跑完重新加载 bundle 就能看到完整诊断（含 evidence 与 next_steps）。样例里的 `raw-unanalyzed` 就是未分析状态，`baseline` / `candidate` 是已分析的。

---

## 9. 保存与分享

- **自动保存** —— 布局变化 500ms 后存进浏览器 `localStorage`，下次打开自动恢复
- **命名快照** —— 命令面板 → 保存布局快照
- **导出 / 导入 JSON** —— 命令面板 → 导出当前视图
- **URL 分享** —— 只编码 Workspace、模式、聚焦列和主要过滤条件，**不含数据**

导出的布局**不嵌入 trace 和 artifact**，只记录 bundle 的标识。别人拿到后需要自己绑定同名 bundle；找不到时会提示重新绑定。

分享前会移除本地路径、用户名、主机名。

---

## 10. 隐私

- Bundle 全程在浏览器内处理，**默认不上传任何东西**
- 顶栏始终显示当前数据来源，本地数据标 🔒
- 唯一会把数据发出去的操作是 **在 Perfetto 中打开**：它会把 `trace.json` 发到 `ui.perfetto.dev`（Google 运营）。点击后会先弹确认框说明风险，取消就不会发送
- 整个工作台可离线使用（除了上面那个 Perfetto 功能）

---

## 11. 性能与规模

设计目标：单 Workspace 100 列 / 300 个 Tile 引用，10 万事件下流畅导航，100 万事件可通过虚拟化完成分析。

实现手段：

- 只完整渲染当前列及左右各两列，其余轻量占位
- `events.jsonl` 在 Web Worker 里解析，主线程只拿聚合结果
- 查询结果缓存；切换 bundle 时取消过期查询
- 同屏活动的重型视图不超过 3 个

**如果卡顿**：先看是不是同屏开了太多重型视图（Timeline / IR Diff / 事件表），把不看的收进 Tab；再看事件量（顶栏 Bundle 旁显示），十万级以上建议先用时间范围缩小再细看。

---

## 12. 已知未完成

诚实清单，避免你在这些地方浪费时间：

| 项 | 状态 |
| --- | --- |
| 命令面板里的"搜索 Pass / Op / Kernel" | 占位项，标着「待实现」。请改用顶栏搜索框 |
| 改键界面 | 无，只能改 `localStorage` |
| IR Diff 配合样例 bundle | 从样例（HTTP）加载时不会预读 artifact，IR 内容为空；用 **打开目录…** 加载则正常 |
| P1 / P2 图表 | 未实现（Execution DAG、内存曲线、Roofline、Occupancy 等），多数也缺数据支撑 |
| 图表渲染的实机验证 | 尚未在真实浏览器中逐张确认过；测试里 ECharts 是被 mock 的。遇到空白图请反馈控制台输出 |

---

## 13. 排错

**下拉里没有样例** —— 没生成 fixture，跑 `node tools/workbench/scripts/make-fixture.mjs`。

**"打开目录"没反应** —— 浏览器不支持 File System Access API，换 Chrome / Edge 95+。

**提示"有 N 行解析失败"** —— bundle 可能是写到一半的（进程还没退出或崩了）。工作台会跳过坏行继续加载，但数据不完整。

**图是空的** —— 先确认这类事件在 bundle 里存在。`real-compile` 只有编译期事件，所有运行时相关的图（缓存热力图、Kernel 系列）本来就是空的，这是数据如此，不是故障。

**布局乱了想重来** —— 命令面板恢复某个快照；或在控制台执行 `localStorage.removeItem('kxc-workbench-doc-v1')` 后刷新。

---

## 14. 相关文档

- [需求基线](plans/2026-07-26-performance-workbench-requirements.md)
- [开发说明与数据契约](../tools/workbench/README.md) —— 包含 8 条 bundle 解析陷阱的完整列表
- [性能可视化与 AI 反馈模型](PERFORMANCE_VISUALIZATION_AND_AI_FEEDBACK.md)
