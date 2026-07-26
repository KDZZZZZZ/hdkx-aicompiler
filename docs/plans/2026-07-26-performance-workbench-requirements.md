# 性能分析图表桌布需求文档

> 状态：需求基线（用户提供）
> 记录时间：2026-07-26
> 说明：本文为产品需求原文，后续设计与实现计划另文记录。文中"KXC 当前已经…"类陈述为需求提出时的假设，需以代码核验结论为准。

## 1. 产品定义
### 1.1 产品名称
性能分析图表桌布（Performance Analysis Workbench Canvas）。
### 1.2 产品目标
系统必须提供一个面向编译器与模型推理性能分析的可组合 GUI 工作台，支持：
1. 在横向无限空间中持续展开分析过程。
2. 将分散在不同位置的图表临时聚合到同一屏幕。
3. 保留从总览、时间线、算子、Kernel 到 IR 的分析路径。
4. 支持单次运行分析、基线对比和性能回归定位。
5. 支持图表、Trace、IR、诊断结果之间的联动。
6. 支持保存、恢复、复制和分享完整分析桌布。
### 1.3 非目标
第一版本不要求：
1. 自研完整 Trace Viewer。
2. 自研 GPU 硬件计数器采集器。
3. 提供任意方向的二维白板式无限画布。
4. 支持多人实时协作编辑。
5. 替代 Nsight Compute、Nsight Systems 或 Perfetto 的全部功能。

---

## 2. 数据输入需求
### 2.1 KXC Bundle 输入
系统必须支持导入一个 KXC profiling bundle。
第一版本必须识别：
* `manifest.json`
* `events.jsonl`
* `trace.json`
* `summary.json`
* `diagnosis.json`
* `diagnosis.md`
* `artifacts/`

KXC 当前 profiling 流程已经生成上述主要文件，并验证 Relay Pass、Lowering 和 TIR Pass 事件能够进入 bundle。
### 2.2 事件字段
第一版本必须支持以下事件字段：
* `component`
* `event_type`
* `phase`
* `status`
* `severity`
* `device`
* `worker_id`
* `op_name`
* `pass_name`
* `kernel_symbol`
* `shape_signature`
* `message`
* `fields`
* `metrics`
* `run_id`
* `span_id`
* `parent_span_id`
* `timestamp_ns`
* `duration_ns`

其中 KXC 公共事件模型已经定义主要分析字段和扩展字段。
### 2.3 全局分析上下文
系统必须维护统一的 `AnalysisContext`，至少包含：
* 当前 bundle
* baseline bundle
* candidate bundle
* run ID
* 时间范围
* component
* event type
* device
* worker
* op
* pass
* kernel
* shape signature
* status
* severity
* 搜索关键词

所有图表必须声明自己读取和修改的上下文字段。

---

## 3. 空间模型需求
系统必须采用以下四层空间结构：
```text
Workspace
  └── Strip
        └── Column
              └── Tile
```
### 3.1 Workspace
Workspace 表示一个独立的分析问题。
每个 Workspace 必须具有：
* 唯一 ID
* 名称
* 描述
* Bundle 绑定
* Baseline 绑定
* 当前聚焦列
* 列顺序
* Gather 配置
* 全局过滤器
* 创建时间
* 最近更新时间

系统必须支持：
* 新建 Workspace
* 重命名 Workspace
* 复制 Workspace
* 删除 Workspace
* 上下切换 Workspace
* 将当前分析分支 Fork 为新 Workspace
* 自动创建一个空 Workspace
### 3.2 Strip
每个 Workspace 必须拥有一个横向滚动的 Strip。
Strip 必须满足：
1. Column 从左到右排列。
2. 新建 Column 时不得自动压缩已有 Column。
3. Strip 可持续向右扩展。
4. 当前聚焦 Column 应自动移动到视口偏左位置。
5. 聚焦 Column 左侧必须尽量保留来源 Column。
6. 聚焦 Column 右侧必须预留新 Column 的出现空间。
7. 支持鼠标、触控板和键盘横向导航。
8. 支持快速跳转到第一列、最后一列和指定列。

横向滚动平铺、动态 Workspace、列式无限条带和 Overview 的交互可参考 Niri。
### 3.3 Column
Column 表示一个完整分析上下文。
每个 Column 必须具有：
* 唯一 ID
* 标题
* Context
* 来源 Column ID
* 来源事件或图表 ID
* 宽度档位
* Tile 列表
* Tile 排列方式
* Pin 状态
* Compare 状态
* 创建原因
* 创建时间

Column 必须支持以下宽度档位：
* `1/3`
* `1/2`
* `2/3`
* `Full`
* `Content Fit`

Column 内部必须支持：
* 纵向平铺
* Tab 展示
* 单 Tile 聚焦
* Baseline/Candidate 对照
* 调整 Tile 高度
* 将 Tile 收入相邻 Column
* 将 Tile 从当前 Column 拆出
* 将整个 Column 移动到其他位置
### 3.4 Tile
Tile 是图表、Trace、IR、日志、指标或诊断视图的最小布局单位。
每个 Tile 必须声明：
* Tile 类型
* 最小宽度
* 推荐宽度
* 最小高度
* 推荐高度
* 是否允许纵向堆叠
* 是否允许 Tab
* 是否允许 Gather
* 是否允许 Compare
* 是否为重型视图
* 数据查询配置
* 当前局部过滤器
* 当前选择状态

---

## 4. 工作模式需求
系统必须提供以下五种模式。
### 4.1 Strip Mode
Strip Mode 用于沿分析路径逐步深入。
必须满足：
1. 单击图表元素时，在当前上下文中执行选择或过滤。
2. 双击或执行 Drill Down 时，在右侧新建 Column。
3. 新 Column 必须继承来源 Column 的上下文。
4. 新 Column 必须保存来源关系。
5. 从中间 Column 重新钻取时，默认替换其右侧未固定 Column。
6. 用户可选择 Fork，将新分支放入独立 Workspace。
7. 固定 Column 不得被自动替换。
### 4.2 Gather Mode
Gather Mode 用于将分散图表聚合到同一屏幕。
必须满足：
1. 用户可从任意 Column 选择多个 Tile。
2. 进入 Gather Mode 后，系统必须创建所选 Tile 的同步引用视图。
3. Gather 不得移动或删除原始 Tile。
4. Gather Tile 与原 Tile 必须共享：
   * 全局时间范围
   * Bundle
   * Baseline
   * 当前算子
   * 当前 Kernel
   * Hover
   * Selection
   * Cross-filter
5. Gather 中关闭 Tile，不得关闭原始 Tile。
6. Gather 中的布局可单独保存。
7. 一个 Workspace 可保存多个命名 Gather Layout。
8. Gather Layout 必须支持恢复默认布局。
9. Gather Layout 必须支持复制为新的 Workspace。
#### 4.2.1 自动聚合布局
选中 2 个 Tile 时：
```text
1/2 + 1/2
```
选中 3 个 Tile 时：
```text
2/3 主视图 + 1/3 两个纵向 Tile
```
选中 4 个 Tile 时：
```text
2 × 2
```
选中 5 至 6 个 Tile 时：
```text
3 列，每列最多 2 个 Tile
```
超过 6 个 Tile 时：
1. 不得继续无限缩小。
2. 系统必须根据图表类型和优先级自动建立 Tab。
3. Timeline、IR Diff、Execution DAG 等宽视图优先获得 `2/3` 或 `Full` 宽度。
4. KPI、诊断、排行榜等紧凑视图优先填充剩余区域。
#### 4.2.2 Gather 布局调整
Gather 中必须支持：
* 拖动换位
* 调整列宽
* 调整 Tile 高度
* Stack/Tab 切换
* Tile 最大化
* 恢复自动布局
* 固定 Tile
* 添加新的引用 Tile
* 移除引用 Tile
### 4.3 Overview Mode
Overview Mode 用于查看所有 Workspace 和 Column 的空间分布。
必须展示：
* Workspace 名称
* Column 缩略图
* Column 标题
* 当前过滤器摘要
* Pin 状态
* Warning 数量
* 当前聚焦 Column
* Gather Layout
* 来源关系

Overview 必须支持：
* 点击跳转
* 拖动重排 Column
* 跨 Workspace 移动或复制 Column
* 删除 Column
* 创建 Workspace
* 搜索 Column
* 按图表类型过滤
* 按诊断严重级别过滤
* 返回原缩放位置

Overview 中的图表缩略图只用于识别，不要求坐标轴和数据标签完整可读。
### 4.4 Focus Mode
Focus Mode 必须让单个 Tile 或 Column 占据主要视口。
必须支持：
* 全屏查看
* 保留上下文面包屑
* 快速返回原位置
* 左右切换相邻 Tile
* 切换同列 Tab
* 保留联动和筛选
* 聚焦 Trace、IR Diff 或 DAG 时隐藏非必要 UI
### 4.5 Compare Mode
Compare Mode 必须支持两个 Bundle 的并行分析。
必须提供：
* Baseline
* Candidate
* Absolute delta
* Relative delta
* Regression 状态
* Improvement 状态

Compare Mode 必须支持：
1. 同一 Tile 内叠加对比。
2. 同一 Column 内上下或左右对比。
3. Baseline 和 Candidate 时间线同步缩放。
4. Baseline 和 Candidate 共享游标。
5. 同一 Pass、Op、Kernel 和 Shape 自动对齐。
6. 不可对齐对象必须明确标记。
7. Compare 结果必须能够直接生成诊断 Tile。

KXC 当前比较逻辑已经覆盖 Pass 耗时、缓存命中、设备拷贝和同步开销，可作为第一版 Compare 数据来源。

---

## 5. 页面框架需求
### 5.1 顶部上下文栏
顶部上下文栏必须固定显示：
* Workspace
* Bundle
* Baseline
* Candidate
* Model
* Input Shape
* Target
* Device
* Run
* Time Range
* Search
* Strip/Gather/Overview 切换
* 保存状态
* 命令面板入口
### 5.2 Workspace 导航区
必须支持：
* Workspace 列表
* Workspace 顺序调整
* 警告数量
* 未保存状态
* 当前活动标记
* 快速新建
* 快速 Fork
* 快速进入 Gather

导航区必须允许折叠。
### 5.3 Strip 主区域
必须支持：
* 横向滚动
* Column 吸附
* Column 间距
* 聚焦边框
* 来源关系提示
* Drop Preview
* 空白区域拖动
* Minimap
* 当前视口范围提示
### 5.4 Minimap
Minimap 必须展示：
* 所有 Column 的相对位置
* 当前视口范围
* 当前焦点
* Pin Column
* Warning Column
* Gather 中使用的 Column

Minimap 必须支持点击和拖动导航。

---

## 6. Tile 标题栏需求
每个 Tile 标题栏必须提供：
* 标题
* 图表类型
* 数据范围摘要
* Bundle 标识
* Baseline 状态
* Loading 状态
* Warning 状态
* Refresh
* Drill Down
* Pin
* Add to Gather
* Duplicate
* Move
* Tab/Stack
* Focus
* More
* Close

Tile 标题栏必须在紧凑模式下折叠低频操作。

---

## 7. 图表联动需求
### 7.1 全局联动
图表必须能够发布以下事件：
* Time range selected
* Event selected
* Op selected
* Pass selected
* Kernel selected
* Shape selected
* Device selected
* Worker selected
* Severity selected
* Hover changed
* Filter changed
### 7.2 联动组
系统必须支持创建 Link Group。
同一 Link Group 内的 Tile 必须能够共享：
* 时间范围
* Hover
* Selection
* Zoom
* Cursor
* Filter
* Baseline

不同 Link Group 可使用不同分析上下文。
### 7.3 局部过滤器
每个 Tile 必须支持：
* 继承全局过滤器
* 覆盖部分过滤器
* 锁定局部过滤器
* 重置到全局过滤器
* 将局部过滤器提升为全局过滤器
### 7.4 Drill Down
执行 Drill Down 时必须：
1. 基于当前选中对象构造新 Context。
2. 在右侧创建适合该对象的默认 Column。
3. 记录来源 Tile。
4. 记录来源事件。
5. 支持返回来源位置。
6. 支持将 Drill Down 结果改为当前 Column 的 Tab。

---

## 8. 图表目录需求
### 8.1 P0 图表
| Tile                         | 最小宽度 | 推荐宽度 | 必须支持 Gather | 必须支持 Compare |
| ---------------------------- | ---: | ---: | ----------- | ------------ |
| KPI 指标组                      |  1/3 |  1/3 | 是           | 是            |
| 阶段耗时分解                       |  1/3 |  1/2 | 是           | 是            |
| Hotspot Top-N                |  1/3 |  1/3 | 是           | 是            |
| Perfetto Timeline            |  2/3 | Full | 是           | 是            |
| Pass Waterfall               |  1/2 |  2/3 | 是           | 是            |
| Pass Duration Ranking        |  1/3 |  1/3 | 是           | 是            |
| Shape × Cache Heatmap        |  1/2 |  1/2 | 是           | 是            |
| Kernel Duration Distribution |  1/3 |  1/2 | 是           | 是            |
| Kernel Top-N                 |  1/3 |  1/3 | 是           | 是            |
| Diagnostics                  |  1/3 |  1/3 | 是           | 是            |
| Regression Delta             |  1/3 |  1/2 | 是           | 是            |
| Logs                         |  1/3 |  1/2 | 是           | 否            |
| Event Table                  |  1/2 |  2/3 | 是           | 是            |
### 8.2 P1 图表
| Tile                 | 数据前置要求                  |
| -------------------- | ----------------------- |
| IR Before/After Diff | IR artifact             |
| IR 变化矩阵              | IR node、byte、op count   |
| Execution Plan DAG   | 节点和依赖边                  |
| Memory 时间曲线          | allocation/free counter |
| Worker 利用率           | worker state event      |
| Copy/Compute Overlap | device 与 stream 时间      |
| Critical Path        | 完整依赖和时间                 |
| Compile Queue        | queue length counter    |
| Cache 生命周期           | cache insert/hit/evict  |
| Flame Graph          | perf 或 pprof 数据         |
### 8.3 P2 图表
| Tile                    | 数据前置要求                    |
| ----------------------- | ------------------------- |
| Roofline                | FLOPs、实际内存流量、设备峰值         |
| Memory Hierarchy        | DRAM/L2/L1/Shared counter |
| Occupancy               | GPU counter               |
| Warp Stall              | GPU counter               |
| Source/SASS Correlation | 源码和硬件 profile             |
| 多设备拓扑                   | 完整设备与通信数据                 |

---

## 9. Trace 需求
1. 系统必须能够加载 KXC 生成的 `trace.json`。
2. 第一版本优先嵌入 Perfetto，不重写完整 Trace Viewer。
3. Trace 必须能够接收工作台传入的时间范围和查询条件。
4. Trace 中选择时间范围后，必须更新工作台全局 Context。
5. Trace 中选择事件后，必须能打开对应：
   * Pass
   * Op
   * Kernel
   * Worker
   * Device
   * Log
6. Trace Tile 在非活动状态下必须暂停重型渲染。
7. 同一视口默认最多保留一个完整活动 Trace 实例。
8. Gather 或 Compare 中需要多个 Trace 时，可使用静态预览或同步轻量视图。

Perfetto 提供大型 Trace 的浏览器时间线、Trace Processor 和本地分析能力，可作为 Trace 页面基础。

---

## 10. IR 需求
IR Tile 必须支持：
* Relay IR
* TIR
* LLVM IR
* C/CUDA Source
* Before/After
* Side-by-side Diff
* Inline Diff
* 搜索
* 跳转定义
* 行号
* 高亮 Pass 修改区域
* 复制
* 下载 artifact
* 与 Pass Waterfall 联动

IR Tile 必须支持销毁离屏编辑器实例并保留模型状态。

---

## 11. 诊断需求
诊断 Tile 必须展示：
* Severity
* Category
* Component
* Summary
* Evidence
* Next Steps
* 关联事件
* 关联时间范围
* 关联 Pass
* 关联 Op
* 关联 Kernel
* 关联 artifact

KXC 当前诊断结构已经包含 `category`、`severity`、`component`、`summary`、`evidence` 和 `next_steps`。

诊断 Tile 必须支持：
* 点击证据跳转到对应图表
* 将诊断加入 Gather
* 将诊断标记为已确认
* 将诊断标记为误报
* 添加用户备注
* 生成新的分析 Workspace

---

## 12. 布局操作需求
### 12.1 Consume
用户必须能够将一个 Tile 收入左侧或右侧 Column。
Consume 后必须：
* 保留 Tile 状态
* 保留查询 Context
* 保留来源关系
* 根据空间自动选择 Stack 或 Tab
* 支持撤销
### 12.2 Expel
用户必须能够将 Tile 从 Column 中拆出。
Expel 后必须：
* 创建新 Column
* 保持原始宽度偏好
* 放置在当前 Column 相邻位置
* 支持撤销
### 12.3 Pin
Pin Column 必须：
* 不被自动分支替换
* 在 Overview 中明确标记
* 可选择在 Gather 中始终出现
* 可选择固定在视口左侧
### 12.4 Focus Position
聚焦 Column 时必须提供：
* 左对齐
* 居中
* 偏左居中
* 保持当前位置

默认使用偏左居中。

---

## 13. 鼠标和触控板需求
### 13.1 鼠标
* 普通滚轮：滚动当前 Column
* `Shift + 滚轮`：横向移动 Strip
* 空白区域拖动：横向移动 Strip
* 标题栏拖动：移动 Tile 或 Column
* 双击标题栏：Focus
* 拖动边界：调整尺寸
* 图表内拖动：优先交给图表
* 按住 Space 拖动：强制移动 Strip
### 13.2 触控板
* 明显纵向手势：Column 内滚动
* 明显横向手势：Strip 横移
* 捏合：仅在 Overview 或支持缩放的图表内生效
* 四指手势：进入或退出 Overview
* 手势方向锁定必须避免横纵抖动

---

## 14. 键盘需求
默认快捷键必须支持：
| 快捷键                    | 操作               |
| ---------------------- | ---------------- |
| `H / L`                | 上一列 / 下一列        |
| `J / K`                | 同列上一个 / 下一个 Tile |
| `Enter`                | Drill Down       |
| `Shift + Enter`        | 作为 Tab 打开        |
| `[` / `]`              | Consume 到左列 / 右列 |
| `E`                    | Expel            |
| `P`                    | Pin              |
| `F`                    | Focus            |
| `G`                    | 进入或退出 Gather     |
| `O`                    | 进入或退出 Overview   |
| `C`                    | Compare          |
| `/`                    | 搜索               |
| `Ctrl/Cmd + K`         | 命令面板             |
| `Esc`                  | 返回上一级状态          |
| `Ctrl/Cmd + Z`         | 撤销               |
| `Ctrl/Cmd + Shift + Z` | 重做               |

所有快捷键必须可配置。

---

## 15. 命令面板需求
命令面板必须支持搜索和执行：
* 打开图表
* 添加到 Gather
* 切换 Workspace
* 切换 Column
* 切换 Bundle
* 设置 Baseline
* 搜索 Pass
* 搜索 Op
* 搜索 Kernel
* 设置时间范围
* 保存 Layout
* 恢复 Layout
* Consume
* Expel
* Pin
* Focus
* Compare
* Fork Workspace
* 导出当前视图

---

## 16. 状态保存需求
系统必须持久化：
* Workspace
* Column 顺序
* Tile 顺序
* Column 宽度
* Tile 高度
* Tab 状态
* 聚焦位置
* Pin 状态
* Gather Layout
* Compare 配置
* Link Group
* 全局过滤器
* 局部过滤器
* 用户备注

必须支持：
* 自动保存
* 手动保存快照
* 命名 Layout
* 恢复历史 Layout
* 导出 JSON
* 导入 JSON
* 通过 URL 恢复可分享的轻量状态
* Bundle 路径失效时提示重新绑定

---

## 17. 撤销与重做需求
以下操作必须进入 Undo/Redo 历史：
* 新建或删除 Column
* 移动 Column
* Consume
* Expel
* 调整宽度
* 调整高度
* 切换 Stack/Tab
* Pin
* 修改 Gather
* 修改 Link Group
* 修改过滤器
* Fork Workspace

数据加载、Hover 和临时游标不得进入历史。

---

## 18. 性能需求
### 18.1 布局虚拟化
系统必须：
1. 只完整渲染当前视口附近的 Column。
2. 默认完整渲染当前列及左右各两列。
3. 对离屏 Column 使用轻量占位或静态快照。
4. 对固定 Column 保持必要状态，不要求持续绘制。
5. 横向虚拟化必须支持动态列宽。
6. Column 进入视口时必须恢复 Tile 状态。
### 18.2 重型 Tile
以下 Tile 必须被标记为重型：
* Perfetto
* IR Editor
* IR Diff
* Flame Graph
* Execution DAG
* 大型 Heatmap
* 大型事件表

重型 Tile 必须支持：
* Lazy Load
* Pause
* Dispose
* Snapshot
* Restore
* Worker 解析
* 渐进式渲染
### 18.3 数据处理
系统必须：
* 在 Worker 中解析大型 JSONL
* 支持增量读取
* 支持按需聚合
* 缓存常用查询
* 避免每个 Tile 重复解析 Bundle
* 使用统一查询层
* 在 Bundle 切换时取消过期查询
* 为每个查询提供 Loading、Error 和 Cancelled 状态
### 18.4 规模目标
第一版本必须满足：
* 单 Workspace 至少支持 100 个 Column
* 单 Workspace 至少支持 300 个 Tile 引用
* Gather 至少支持 20 个 Tile
* 同时完整活动的重型 Tile 不超过 3 个
* 10 万事件下保持基本流畅导航
* 100 万事件下能够通过虚拟化和渐进加载完成分析
* 布局操作不得触发所有图表重新渲染

---

## 19. 本地优先与隐私需求
1. Bundle 默认在本地处理。
2. 未经用户操作，不得上传 Trace、IR、日志或模型信息。
3. 必须明确显示当前数据是否离开本机。
4. 导出 Layout 时，默认不嵌入大型 Trace 和 artifact。
5. 分享状态必须能够移除：
   * 本地路径
   * 用户名
   * 主机名
   * 环境变量
   * 敏感日志
6. 支持离线使用。

---

## 20. 视觉需求
### 20.1 总体风格
* 深色优先
* 低饱和度
* 高信息密度
* 不使用装饰性渐变
* 不使用大面积高亮色
* Warning 和 Error 必须有稳定语义色
* Baseline 与 Candidate 必须始终使用一致视觉编码
### 20.2 Column
* 聚焦 Column 必须有明确边界
* 来源 Column 必须可识别
* Pin Column 必须有固定标识
* Gather 引用 Tile 必须标记来源
* Compare Column 必须标明 Baseline/Candidate
### 20.3 图表响应式
每个图表必须提供：
* Full 状态
* Normal 状态
* Compact 状态
* Thumbnail 状态

Compact 状态必须减少：
* 次要坐标轴
* 次要标签
* 次要图例
* 非必要交互控件

不得仅通过整体缩放生成 Compact 状态。

---

## 21. 可访问性需求
* 所有布局操作必须可通过键盘完成
* 聚焦顺序必须稳定
* 图表必须提供文本摘要
* Warning 不得仅通过颜色表达
* Tooltip 必须支持键盘触发
* Overview 必须支持屏幕阅读器识别 Workspace 和 Column
* 快捷键必须可关闭或修改
* 动画必须尊重减少动态效果设置

---

## 22. MVP 验收需求
MVP 必须完成：
1. 导入 KXC Bundle。
2. 创建和切换 Workspace。
3. 横向滚动 Strip。
4. 新建、删除、移动 Column。
5. Column 宽度档位。
6. Column 内 Stack 和 Tab。
7. Consume 和 Expel。
8. Overview。
9. Gather Mode。
10. Gather 自动布局。
11. Gather 引用视图。
12. KPI Tile。
13. 阶段耗时 Tile。
14. Hotspot Top-N。
15. Pass Waterfall。
16. Shape × Cache Heatmap。
17. Diagnostics Tile。
18. Event Table。
19. Perfetto Trace 嵌入。
20. Baseline/Candidate Compare。
21. 全局过滤器联动。
22. Tile Drill Down。
23. Layout 保存与恢复。
24. Undo/Redo。
25. Column 虚拟化。
26. 命令面板。
27. 键盘导航。

---

## 23. 可参考 Repo
### 23.1 项目数据源
| 对应需求                                             | Repo                                                                    |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| KXC profiling bundle、事件模型、诊断、Trace 和 IR artifact | [KDZZZZZZ/hdkx-aicompiler](https://github.com/KDZZZZZZ/hdkx-aicompiler) |
### 23.2 横向滚动平铺与 Overview
| 对应需求                                                     | Repo                                                            |
| -------------------------------------------------------- | --------------------------------------------------------------- |
| 横向无限 Strip、Column 模型、动态 Workspace、Overview、Tabbed Column | [niri-wm/niri](https://github.com/niri-wm/niri)                 |
| 滚动平铺、焦点位置、鼠标与触控板交互                                       | [paperwm/PaperWM](https://github.com/paperwm/PaperWM)           |
| Niri 生态中的 Minimap、Session、Sidebar 和 Workspace 工具         | [niri-wm/awesome-niri](https://github.com/niri-wm/awesome-niri) |
### 23.3 Dock、Tab 与桌面布局
| 对应需求                                | Repo                                                                                          |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| Tabset、Splitter、Dock、拖动、最大化、布局序列化   | [caplin/FlexLayout](https://github.com/caplin/FlexLayout)                                     |
| 多窗口 Web 布局、Docking、可保存布局            | [golden-layout/golden-layout](https://github.com/golden-layout/golden-layout)                 |
| Gather Mode、可拖动和缩放 Grid、响应式布局、布局持久化 | [react-grid-layout/react-grid-layout](https://github.com/react-grid-layout/react-grid-layout) |
### 23.4 Trace 与性能分析
| 对应需求                                                | Repo                                                  |
| --------------------------------------------------- | ----------------------------------------------------- |
| Trace Viewer、时间线、Trace Processor、SQL 分析、外部 Trace 格式 | [google/perfetto](https://github.com/google/perfetto) |
### 23.5 图表渲染
| 对应需求                                             | Repo                                                |
| ------------------------------------------------ | --------------------------------------------------- |
| 柱状图、折线图、Heatmap、Scatter、自定义 Series、Canvas/SVG 渲染 | [apache/echarts](https://github.com/apache/echarts) |
| 超大数据集的 GPU 可视化和自定义 Layer                         | [visgl/deck.gl](https://github.com/visgl/deck.gl)   |
### 23.6 DAG 与执行计划
| 对应需求                                   | Repo                                              |
| -------------------------------------- | ------------------------------------------------- |
| Execution Plan DAG、节点编辑、Minimap、选择与边交互 | [xyflow/xyflow](https://github.com/xyflow/xyflow) |
### 23.7 IR 与源码查看
| 对应需求                                     | Repo                                                                  |
| ---------------------------------------- | --------------------------------------------------------------------- |
| IR 编辑器、Side-by-side Diff、搜索、语法高亮、大文件模型管理 | [microsoft/monaco-editor](https://github.com/microsoft/monaco-editor) |
### 23.8 虚拟化
| 对应需求                 | Repo                                                            |
| -------------------- | --------------------------------------------------------------- |
| 横向、纵向和 Grid 虚拟化、动态尺寸 | [TanStack/virtual](https://github.com/TanStack/virtual)         |
| 大型列表和表格的轻量虚拟化        | [bvaughn/react-window](https://github.com/bvaughn/react-window) |
### 23.9 本地数据查询
| 对应需求                                   | Repo                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| 浏览器本地 SQL、JSON/CSV/Parquet 分析、Wasm 查询层 | [duckdb/duckdb-wasm](https://github.com/duckdb/duckdb-wasm) |
### 23.10 状态管理
| 对应需求                                      | Repo                                                |
| ----------------------------------------- | --------------------------------------------------- |
| Workspace、Column、Tile、Link Group 和持久化状态管理 | [pmndrs/zustand](https://github.com/pmndrs/zustand) |
### 23.11 Tooltip、菜单与浮层
| 对应需求                                | Repo                                                                  |
| ----------------------------------- | --------------------------------------------------------------------- |
| Tooltip、Context Menu、Popover、边界碰撞处理 | [floating-ui/floating-ui](https://github.com/floating-ui/floating-ui) |
### 23.12 无限画布交互参考
| 对应需求                            | Repo                                              |
| ------------------------------- | ------------------------------------------------- |
| Minimap、缩放、选择、拖动和画布状态模型，仅作为交互参考 | [tldraw/tldraw](https://github.com/tldraw/tldraw) |
