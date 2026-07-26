/**
 * 统一查询层协议（需求 §18.3）。
 *
 * 所有 Tile 都只通过这里取数，不允许自己再解析一遍 bundle。Worker 侧持有解析结果，
 * 主线程只拿聚合后的小对象，这样 100 万事件也不会把主线程堵死。
 */

import type { KxcDiagnostic, KxcEvent, KxcManifest } from './contract'

/** 查询用的过滤条件，是 AnalysisContext 的可序列化投影。 */
export interface QueryFilter {
  runId?: string | null
  startNs?: number | null
  endNs?: number | null
  component?: string | null
  eventType?: string | null
  device?: string | null
  workerId?: number | null
  op?: string | null
  pass?: string | null
  kernel?: string | null
  shapeSignature?: string | null
  status?: string | null
  severity?: string | null
  search?: string | null
}

export type QuerySpec =
  | { kind: 'meta' }
  | { kind: 'kpi' }
  | { kind: 'phase_breakdown' }
  | { kind: 'pass_ranking'; limit: number }
  | { kind: 'pass_waterfall' }
  | { kind: 'hotspot'; by: 'op' | 'pass' | 'kernel' | 'component'; limit: number }
  | { kind: 'kernel_duration_dist'; buckets: number }
  | { kind: 'shape_cache' }
  | { kind: 'events'; offset: number; limit: number }
  | { kind: 'logs'; limit: number }
  | { kind: 'diagnostics' }
  | { kind: 'timeline'; buckets: number }
  | { kind: 'facets' }
  | { kind: 'artifact'; path: string }
  | { kind: 'event_by_span'; spanId: string }

export type QueryKind = QuerySpec['kind']

// ---------------------------------------------------------------------------
// 各查询的结果类型
// ---------------------------------------------------------------------------

export interface MetaResult {
  manifest: KxcManifest
  /** 全部事件的时间跨度（ts_ns 相对时间）。 */
  spanNs: { startNs: number; endNs: number }
  eventCount: number
  /** 解析失败被跳过的行数，非 0 时必须在 UI 上提示（§18.3 Error 状态）。 */
  malformedLines: number
  componentCounts: Record<string, number>
  runIds: string[]
  hasTrace: boolean
  hasArtifacts: boolean
}

export interface KpiResult {
  /** 端到端编译耗时；无 compile_module 事件时为 null。 */
  compileNs: number | null
  /** 所有 pass 事件耗时之和（含 relay 与 tir）。 */
  passTotalNs: number
  passCount: number
  /** 运行时会话总耗时；无 runtime_session_run 事件时为 null。 */
  runtimeNs: number | null
  runCount: number
  cacheExactHit: number
  cacheFuzzyHit: number
  cacheMiss: number
  /** 命中率；分母为 0 时为 null，不要显示成 0%。 */
  cacheHitRate: number | null
  errorCount: number
  warnCount: number
  /** 观测到的事件总数（已应用过滤器）。 */
  eventCount: number
}

export interface PhaseBreakdownItem {
  phase: string
  component: string
  durationNs: number
  /** 该阶段包含的事件数。 */
  count: number
}
export interface PhaseBreakdownResult {
  items: PhaseBreakdownItem[]
  totalNs: number
  /** 阶段耗时之和与顶层 compile_module 的差值，用于提示"存在未归类耗时"。 */
  unattributedNs: number | null
}

export interface PassRankItem {
  passName: string
  component: string
  totalNs: number
  count: number
  maxNs: number
  changed: boolean
  irBeforeBytes: number | null
  irAfterBytes: number | null
}
export interface PassRankingResult {
  items: PassRankItem[]
  totalNs: number
}

export interface WaterfallItem {
  spanId: string
  passName: string
  component: string
  startNs: number
  durationNs: number
  status: string
  changed: boolean
  irBeforeBytes: number | null
  irAfterBytes: number | null
  depth: number
}
export interface PassWaterfallResult {
  items: WaterfallItem[]
  startNs: number
  endNs: number
}

export interface HotspotItem {
  key: string
  totalNs: number
  count: number
  selfNs: number
}
export interface HotspotResult {
  items: HotspotItem[]
  totalNs: number
  /** by='op' 时，若数据来自 fields.op_name 而非真实 kernel 执行，需在 UI 标注。 */
  caveat: string | null
}

export interface KernelDurationDistResult {
  buckets: Array<{ lowNs: number; highNs: number; count: number }>
  totalCount: number
  p50Ns: number | null
  p95Ns: number | null
  maxNs: number | null
  caveat: string | null
}

export interface ShapeCacheCell {
  shapeSignature: string
  shapeHash: string
  exactHit: number
  fuzzyHit: number
  miss: number
  totalNs: number
}
export interface ShapeCacheResult {
  cells: ShapeCacheCell[]
  /** 无法解析出 shape_signature 的缓存事件数（父 span 缺失时会发生）。 */
  unresolved: number
}

export interface EventsResult {
  rows: KxcEvent[]
  total: number
  offset: number
}

export interface LogsResult {
  rows: KxcEvent[]
  total: number
}

export interface DiagnosticsResult {
  diagnostics: KxcDiagnostic[]
  /** 只有 C++ 占位诊断时为 true，UI 应提示"尚未运行分析"。 */
  notAnalyzed: boolean
}

export interface TimelineResult {
  startNs: number
  endNs: number
  bucketNs: number
  /** 每个 component 一条序列，值为该桶内的事件耗时之和。 */
  series: Array<{ component: string; values: number[] }>
  counts: number[]
}

export interface FacetsResult {
  components: Array<{ value: string; count: number }>
  eventTypes: Array<{ value: string; count: number }>
  devices: Array<{ value: string; count: number }>
  passes: Array<{ value: string; count: number }>
  ops: Array<{ value: string; count: number }>
  kernels: Array<{ value: string; count: number }>
  shapes: Array<{ value: string; count: number }>
  severities: Array<{ value: string; count: number }>
  statuses: Array<{ value: string; count: number }>
}

export interface ArtifactResult {
  path: string
  content: string | null
}

export interface EventBySpanResult {
  event: KxcEvent | null
  /** 祖先链，从最近的父到根，用于面包屑与上下文补全。 */
  ancestors: KxcEvent[]
}

export type QueryResultMap = {
  meta: MetaResult
  kpi: KpiResult
  phase_breakdown: PhaseBreakdownResult
  pass_ranking: PassRankingResult
  pass_waterfall: PassWaterfallResult
  hotspot: HotspotResult
  kernel_duration_dist: KernelDurationDistResult
  shape_cache: ShapeCacheResult
  events: EventsResult
  logs: LogsResult
  diagnostics: DiagnosticsResult
  timeline: TimelineResult
  facets: FacetsResult
  artifact: ArtifactResult
  event_by_span: EventBySpanResult
}

export type QueryResult<K extends QueryKind = QueryKind> = QueryResultMap[K]

// ---------------------------------------------------------------------------
// Worker 消息
// ---------------------------------------------------------------------------

export type WorkerRequest =
  | { type: 'load'; id: number; bundleId: string; files: LoadPayload }
  | { type: 'unload'; id: number; bundleId: string }
  | { type: 'query'; id: number; bundleId: string; spec: QuerySpec; filter: QueryFilter }
  | { type: 'cancel'; id: number }

/** 文件内容由主线程读好后传入，Worker 不碰 IO，便于同时支持目录句柄与 fetch。 */
export interface LoadPayload {
  manifest: string | null
  events: string
  summary: string | null
  diagnosis: string | null
  trace: string | null
  /** 相对路径 → 内容。artifact 通常按需加载，这里允许为空。 */
  artifacts: Record<string, string>
}

export type WorkerResponse =
  | { type: 'loaded'; id: number; bundleId: string; meta: MetaResult }
  | { type: 'result'; id: number; data: unknown }
  | { type: 'error'; id: number; message: string }
  | { type: 'cancelled'; id: number }
  | { type: 'progress'; id: number; parsed: number; total: number }
