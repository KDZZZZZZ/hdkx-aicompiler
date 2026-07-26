/**
 * 桌布空间模型：Workspace → Strip → Column → Tile（需求 §3）。
 *
 * 设计取舍：
 * - Strip 不是独立实体，它就是 Workspace 的 columnIds 顺序 + 滚动状态。多引入一层
 *   只会让每个 action 多穿一次壳，没有收益。
 * - Tile 全局扁平存放在 tiles 表里，Column 只持有 tileIds。Gather 的“引用视图”
 *   （§4.2.2）因此天然成立：引用就是同一个 tile id，不需要复制状态。
 * - 局部过滤器用 Partial<AnalysisContext> 覆盖全局，而不是各存一份完整 context，
 *   这样 §7.3 的“重置到全局”只是删掉覆盖键。
 */

// ---------------------------------------------------------------------------
// 分析上下文（§2.3）
// ---------------------------------------------------------------------------

/** 时间范围，单位纳秒，对齐 events.jsonl 的 ts_ns（steady_clock 相对时间）。 */
export interface TimeRangeNs {
  startNs: number
  endNs: number
}

/**
 * 全局分析上下文。所有图表声明自己读写哪些键（见 TileSpec.reads / writes）。
 * 每一项都可为空，空表示“不过滤”。
 */
export interface AnalysisContext {
  bundleId: string | null
  baselineBundleId: string | null
  candidateBundleId: string | null
  runId: string | null
  timeRange: TimeRangeNs | null
  component: string | null
  eventType: string | null
  device: string | null
  workerId: number | null
  op: string | null
  pass: string | null
  kernel: string | null
  shapeSignature: string | null
  status: string | null
  severity: string | null
  search: string | null
}

export type ContextKey = keyof AnalysisContext

export const EMPTY_CONTEXT: AnalysisContext = {
  bundleId: null,
  baselineBundleId: null,
  candidateBundleId: null,
  runId: null,
  timeRange: null,
  component: null,
  eventType: null,
  device: null,
  workerId: null,
  op: null,
  pass: null,
  kernel: null,
  shapeSignature: null,
  status: null,
  severity: null,
  search: null,
}

/** 局部过滤器：只记录被覆盖的键（§7.3）。 */
export type ContextOverride = Partial<AnalysisContext>

export function resolveContext(
  global: AnalysisContext,
  override: ContextOverride | undefined,
  locked: boolean,
): AnalysisContext {
  if (!override) return global
  // 锁定后局部过滤器不再跟随全局变化，但 bundle 绑定始终跟随，
  // 否则切换 bundle 时被锁的 Tile 会继续查一个已经卸载的 bundle。
  const base = locked ? { ...global, ...override } : { ...global, ...override }
  base.bundleId = override.bundleId ?? global.bundleId
  return base
}

// ---------------------------------------------------------------------------
// Column 与 Tile（§3.3 / §3.4）
// ---------------------------------------------------------------------------

/** 列宽档位（§3.3）。'fit' 即 Content Fit，由内容测量决定。 */
export type WidthTier = '1/3' | '1/2' | '2/3' | 'full' | 'fit'

export const WIDTH_TIERS: readonly WidthTier[] = ['1/3', '1/2', '2/3', 'full', 'fit']

/** 各档位占视口的比例；'fit' 不在此表，需实际测量后回填。 */
export const WIDTH_FRACTION: Record<Exclude<WidthTier, 'fit'>, number> = {
  '1/3': 1 / 3,
  '1/2': 1 / 2,
  '2/3': 2 / 3,
  full: 1,
}

/** Column 内 Tile 的排列方式（§3.3）。 */
export type TileArrange = 'stack' | 'tabs'

/** Column 的创建原因，Overview 里用来解释来源（§3.3、§4.3）。 */
export type ColumnOrigin =
  | 'initial'
  | 'drill_down'
  | 'expel'
  | 'manual'
  | 'fork'
  | 'diagnostic'
  | 'compare'

export interface Column {
  id: string
  title: string
  /** 该列自己的上下文覆盖，叠加在 Workspace 全局过滤器之上。 */
  context: ContextOverride
  /** 来源列，Drill Down 时记录（§4.1.4、§7.4.5）。 */
  sourceColumnId: string | null
  /** 来源 Tile / 事件，支持“返回来源位置”。 */
  sourceTileId: string | null
  sourceEventId: string | null
  width: WidthTier
  tileIds: string[]
  arrange: TileArrange
  /** tabs 模式下当前激活的 tile。 */
  activeTileId: string | null
  pinned: boolean
  /** Compare 时该列代表哪一侧；null 表示非对比列。 */
  compareRole: 'baseline' | 'candidate' | 'delta' | null
  origin: ColumnOrigin
  createdAt: number
}

/** Tile 的选择状态，用于联动（§7.1）与 Drill Down 取值。 */
export interface TileSelection {
  kind: 'pass' | 'op' | 'kernel' | 'shape' | 'event' | 'device' | 'worker' | 'severity' | 'range'
  value: string
  /** kind==='range' 时携带。 */
  range?: TimeRangeNs
  /** 触发选择的事件 span_id，便于回跳。 */
  eventId?: string
}

export interface Tile {
  id: string
  type: TileType
  /** 覆盖默认标题；为空时用 TileSpec.title。 */
  title: string | null
  /** 局部过滤器（§7.3）。 */
  context: ContextOverride
  /** 锁定后不再继承全局过滤器变化。 */
  contextLocked: boolean
  /** stack 模式下的高度，px；null 表示按推荐高度均分。 */
  heightPx: number | null
  selection: TileSelection | null
  /** 联动组（§7.2）。默认全部 tile 在 'global' 组。 */
  linkGroupId: string
  /** 来源关系。 */
  sourceTileId: string | null
  /** Compare 时该 tile 显示哪一侧数据。 */
  compareMode: 'single' | 'overlay' | 'baseline' | 'candidate' | 'delta'
  createdAt: number
}

// ---------------------------------------------------------------------------
// Tile 类型与静态声明（§3.4、§8）
// ---------------------------------------------------------------------------

export type TileType =
  // P0
  | 'kpi'
  | 'phase_breakdown'
  | 'hotspot_topn'
  | 'perfetto_timeline'
  | 'pass_waterfall'
  | 'pass_ranking'
  | 'shape_cache_heatmap'
  | 'kernel_duration_dist'
  | 'kernel_topn'
  | 'diagnostics'
  | 'regression_delta'
  | 'logs'
  | 'event_table'
  // P1（数据可用时才注册）
  | 'ir_diff'

/** 图表的响应式状态（§20.3）。 */
export type TileDensity = 'full' | 'normal' | 'compact' | 'thumbnail'

export interface TileSpec {
  type: TileType
  title: string
  /** 一句话说明这个图回答什么问题，用于命令面板与 Overview 的可访问描述。 */
  purpose: string
  minWidth: WidthTier
  preferredWidth: WidthTier
  minHeightPx: number
  preferredHeightPx: number
  stackable: boolean
  tabbable: boolean
  gatherable: boolean
  comparable: boolean
  /** 重型视图（§18.2）：离屏时必须 dispose，同屏活动数受限。 */
  heavy: boolean
  /** 该图读取哪些上下文键。 */
  reads: readonly ContextKey[]
  /** 该图会修改哪些上下文键（点击选择时向外发布）。 */
  writes: readonly ContextKey[]
  /**
   * 当前 KXC 数据下的可用性。用来在 UI 上如实标注，而不是画一个空图骗人。
   * 'ready'   —— 数据齐全
   * 'partial' —— 能画但有已知缺口，需在 Tile 上显示说明
   * 'blocked' —— 当前 bundle 拿不到核心数据
   */
  readiness: 'ready' | 'partial' | 'blocked'
  /** readiness 非 ready 时必须给出原因，直接显示给用户。 */
  readinessNote?: string
}

// ---------------------------------------------------------------------------
// Gather（§4.2）
// ---------------------------------------------------------------------------

/** Gather 中一个格子的位置，列优先。 */
export interface GatherSlot {
  tileId: string
  /** 第几列（从 0 起）。 */
  col: number
  /** 列内第几行。 */
  row: number
  /** 该列宽度档位。 */
  width: WidthTier
  /** 同一 (col,row) 允许多个 tile 形成 tab 组时，标记是否为激活项。 */
  tabbed: boolean
  pinned: boolean
  heightFr: number
}

export interface GatherLayout {
  id: string
  name: string
  /** 引用的 tile（不复制，§4.2.3）。 */
  slots: GatherSlot[]
  /** 是否处于自动布局状态；手工调整后置 false，可一键恢复（§4.2.2）。 */
  auto: boolean
  createdAt: number
}

// ---------------------------------------------------------------------------
// Link Group（§7.2）
// ---------------------------------------------------------------------------

export interface LinkGroup {
  id: string
  name: string
  color: string
  /** 该组共享哪些维度。 */
  shares: {
    timeRange: boolean
    hover: boolean
    selection: boolean
    zoom: boolean
    cursor: boolean
    filter: boolean
    baseline: boolean
  }
}

export const DEFAULT_LINK_GROUP_ID = 'global'

// ---------------------------------------------------------------------------
// Workspace（§3.1）
// ---------------------------------------------------------------------------

export interface Workspace {
  id: string
  name: string
  description: string
  bundleId: string | null
  baselineBundleId: string | null
  /** 全局过滤器（§3.1）。bundle 绑定单独存于上面两个字段。 */
  filters: ContextOverride
  columnIds: string[]
  focusedColumnId: string | null
  focusedTileId: string | null
  gatherLayouts: GatherLayout[]
  activeGatherLayoutId: string | null
  /** 聚焦位置策略（§12.4），默认偏左居中。 */
  focusPosition: 'left' | 'center' | 'center_left' | 'keep'
  compareEnabled: boolean
  createdAt: number
  updatedAt: number
}

// ---------------------------------------------------------------------------
// 模式（§4）
// ---------------------------------------------------------------------------

export type WorkbenchMode = 'strip' | 'gather' | 'overview' | 'focus'

// ---------------------------------------------------------------------------
// 可持久化文档（§16）
// ---------------------------------------------------------------------------

/**
 * 进入 Undo/Redo 与持久化的部分。刻意不含 hover、加载状态、查询缓存——
 * 需求 §17 明确要求这些不进历史。
 */
export interface WorkbenchDoc {
  workspaces: Record<string, Workspace>
  workspaceOrder: string[]
  columns: Record<string, Column>
  tiles: Record<string, Tile>
  linkGroups: Record<string, LinkGroup>
  activeWorkspaceId: string | null
  /** 用户对诊断的标注（§11）。key 为 `${bundleId}:${category}:${index}`。 */
  diagnosticNotes: Record<string, DiagnosticNote>
}

export interface DiagnosticNote {
  state: 'none' | 'confirmed' | 'false_positive'
  note: string
  updatedAt: number
}

/** 导出/导入的信封（§16）。 */
export interface WorkbenchSnapshot {
  kind: 'kxc-workbench-layout'
  version: 1
  savedAt: number
  doc: WorkbenchDoc
  /** 只记录 bundle 的标识与来源提示，不内嵌 trace/artifact（§19.4）。 */
  bundleRefs: Array<{ id: string; label: string; hint: string }>
}
