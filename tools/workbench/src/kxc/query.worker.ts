/// <reference lib="webworker" />
/**
 * Bundle 解析与聚合（需求 §18.3）。
 *
 * 职责边界：只做纯计算，不做任何 IO。文件内容由调用方读好后传进来，
 * 这样同一份代码既能服务本地目录句柄，也能服务 fetch 来的 fixture。
 *
 * 这个文件有两个消费者：
 * 1. 浏览器 —— Vite 把它打成 Worker，走 onmessage 那套壳；
 * 2. CLI（kxc-wb query）—— 直接 import parseBundle / runQuery，在 Node 里跑。
 *    Node 24 原生剥离类型，所以不需要为 CLI 单独构建一份。
 *
 * 因此 self 必须做存在性判断：Node 里没有 self，模块顶层直接取会抛错。
 * 聚合口径只有这一份实现，CLI 和界面不会算出不同的数。
 */

import {
  CACHE_EVENT_TYPES,
  COMPONENT,
  EVENT_TYPE,
  PASS_COMPONENTS,
  PIPELINE_COMPONENTS,
  parseEventLine,
  readKernelSymbol,
  readOpName,
  severityRank,
  type KxcDiagnosisFile,
  type KxcEvent,
  type KxcManifest,
  type KxcSummary,
  // 带 .ts 扩展名是为了让 CLI 能用 Node 原生 ESM 直接 import 这个模块：
  // Node 不做扩展名推断，而 Vite 两种写法都吃。
} from './contract.ts'
import type {
  ArtifactResult,
  DiagnosticsResult,
  EventBySpanResult,
  EventsResult,
  FacetsResult,
  HotspotResult,
  KpiResult,
  KernelDurationDistResult,
  LoadPayload,
  LogsResult,
  MetaResult,
  PassRankingResult,
  PassWaterfallResult,
  PhaseBreakdownResult,
  QueryFilter,
  QuerySpec,
  ShapeCacheResult,
  TimelineResult,
  WorkerRequest,
  WorkerResponse,
} from './query-protocol'

export interface ParsedBundle {
  meta: MetaResult
  events: KxcEvent[]
  bySpanId: Map<string, KxcEvent>
  diagnostics: KxcDiagnosisFile | null
  artifacts: Record<string, string>
  /** shape_hash → shape_signature，从带 shape_signature 的事件反建，用于补全缓存事件。 */
  shapeByHash: Map<string, string>
}

const bundles = new Map<string, ParsedBundle>()
const cancelled = new Set<number>()

// Node 里没有 self；CLI 直接 import 本模块时不能在顶层解引用它。
const ctx =
  typeof self !== 'undefined' ? (self as unknown as DedicatedWorkerGlobalScope) : null

function post(msg: WorkerResponse) {
  ctx?.postMessage(msg)
}

// ---------------------------------------------------------------------------
// 解析
// ---------------------------------------------------------------------------

function safeJson<T>(text: string | null): T | null {
  if (!text) return null
  try {
    return JSON.parse(text) as T
  } catch {
    return null
  }
}

export function parseBundle(bundleId: string, files: LoadPayload, reqId = 0): ParsedBundle {
  const events: KxcEvent[] = []
  let malformed = 0

  // 逐行解析，避免一次性构造巨大的中间数组。
  const text = files.events
  let lineStart = 0
  let parsedCount = 0
  const totalBytes = text.length
  while (lineStart <= text.length) {
    let lineEnd = text.indexOf('\n', lineStart)
    if (lineEnd === -1) lineEnd = text.length
    const line = text.slice(lineStart, lineEnd)
    lineStart = lineEnd + 1
    if (!line.trim()) {
      if (lineEnd >= text.length) break
      continue
    }
    const e = parseEventLine(line)
    if (e) events.push(e)
    else malformed += 1
    parsedCount += 1
    // 每 20000 行汇报一次，供 UI 显示渐进加载进度。
    if (parsedCount % 20000 === 0) {
      post({ type: 'progress', id: reqId, parsed: lineStart, total: totalBytes })
    }
    if (lineEnd >= text.length) break
  }

  events.sort((a, b) => a.ts_ns - b.ts_ns)

  const bySpanId = new Map<string, KxcEvent>()
  const shapeByHash = new Map<string, string>()
  const componentCounts: Record<string, number> = {}
  const runIdSet = new Set<string>()
  let startNs = Number.POSITIVE_INFINITY
  let endNs = 0

  for (const e of events) {
    if (e.span_id) bySpanId.set(e.span_id, e)
    componentCounts[e.component] = (componentCounts[e.component] ?? 0) + 1
    if (e.run_id) runIdSet.add(e.run_id)
    if (e.ts_ns < startNs) startNs = e.ts_ns
    const end = e.ts_ns + Math.max(0, e.duration_ns)
    if (end > endNs) endNs = end
    // 建立 shape_hash → shape_signature 映射：缓存事件只有 hash，靠这张表补签名。
    const hash = e.fields.shape_hash
    if (hash && e.shape_signature && !shapeByHash.has(hash)) {
      shapeByHash.set(hash, e.shape_signature)
    }
  }
  if (!Number.isFinite(startNs)) startNs = 0

  const manifest = safeJson<KxcManifest>(files.manifest)
  const summary = safeJson<KxcSummary>(files.summary)

  const meta: MetaResult = {
    manifest:
      manifest ??
      ({
        schema_version: 0,
        trace_id: bundleId,
        session_id: '',
        bundle_dir: '',
        log_level: 'info',
        ir_capture_mode: 'unknown',
        enable_nvtx: false,
        enable_cupti: false,
        cupti_available: false,
        record_execution_plan_details: false,
        event_count: events.length,
      } as KxcManifest),
    spanNs: { startNs, endNs },
    eventCount: events.length,
    malformedLines: malformed,
    // summary.json 的 component_counts 值是字符串，这里以自算结果为准，
    // 只在 events.jsonl 缺失时才回退到 summary。
    componentCounts:
      events.length > 0
        ? componentCounts
        : Object.fromEntries(
            Object.entries(summary?.component_counts ?? {}).map(([k, v]) => [k, Number(v) || 0]),
          ),
    runIds: [...runIdSet].sort(),
    hasTrace: Boolean(files.trace),
    hasArtifacts: Object.keys(files.artifacts).length > 0,
  }

  return {
    meta,
    events,
    bySpanId,
    diagnostics: safeJson<KxcDiagnosisFile>(files.diagnosis),
    artifacts: files.artifacts,
    shapeByHash,
  }
}

// ---------------------------------------------------------------------------
// 过滤
// ---------------------------------------------------------------------------

function makePredicate(f: QueryFilter, b: ParsedBundle): (e: KxcEvent) => boolean {
  const search = f.search?.trim().toLowerCase() || null
  const minSeverity = f.severity ? severityRank(f.severity) : null

  return (e) => {
    if (f.runId && e.run_id !== f.runId) return false
    if (f.startNs != null && e.ts_ns + Math.max(0, e.duration_ns) < f.startNs) return false
    if (f.endNs != null && e.ts_ns > f.endNs) return false
    if (f.component && e.component !== f.component) return false
    if (f.eventType && e.event_type !== f.eventType) return false
    if (f.device && e.device !== f.device) return false
    if (f.workerId != null && e.worker_id !== f.workerId) return false
    if (f.pass && e.pass_name !== f.pass) return false
    if (f.op && readOpName(e) !== f.op) return false
    if (f.kernel && readKernelSymbol(e) !== f.kernel) return false
    if (f.status && e.status !== f.status) return false
    if (minSeverity != null && severityRank(e.severity) < minSeverity) return false
    if (f.shapeSignature) {
      const sig = e.shape_signature || resolveShape(e, b)
      if (sig !== f.shapeSignature) return false
    }
    if (search) {
      const hay =
        `${e.component} ${e.event_type} ${e.pass_name} ${readOpName(e)} ${readKernelSymbol(e)} ${e.message} ${e.shape_signature}`.toLowerCase()
      if (!hay.includes(search)) return false
    }
    return true
  }
}

/**
 * 补全事件的 shape_signature。
 *
 * cache_exact_hit / cache_fuzzy_hit / cache_miss_sync_compile 这三个 span 在 C++ 侧
 * 只写了 fields.shape_hash，没写 shape_signature（src/runtime/runtime_session.cc:57/65/72）。
 * 因此必须靠两条路补：先查 shape_hash → signature 表，再沿 parent_span 上溯。
 * 这也是 Shape × Cache 热力图不需要改 C++ 就能做出来的原因。
 */
function resolveShape(e: KxcEvent, b: ParsedBundle): string {
  if (e.shape_signature) return e.shape_signature
  const hash = e.fields.shape_hash
  if (hash) {
    const sig = b.shapeByHash.get(hash)
    if (sig) return sig
  }
  let cur = e.parent_span_id
  for (let depth = 0; depth < 8 && cur; depth += 1) {
    const parent = b.bySpanId.get(cur)
    if (!parent) break
    if (parent.shape_signature) return parent.shape_signature
    cur = parent.parent_span_id
  }
  return ''
}

// ---------------------------------------------------------------------------
// 各查询实现
// ---------------------------------------------------------------------------

function isPassEvent(e: KxcEvent): boolean {
  return e.event_type === EVENT_TYPE.runPass && PASS_COMPONENTS.includes(e.component)
}

function queryKpi(evts: KxcEvent[]): KpiResult {
  let compileNs: number | null = null
  let passTotalNs = 0
  let passCount = 0
  let runtimeNs: number | null = null
  let runCount = 0
  let exact = 0
  let fuzzy = 0
  let miss = 0
  let errors = 0
  let warns = 0

  for (const e of evts) {
    if (e.event_type === EVENT_TYPE.compileModule) {
      compileNs = (compileNs ?? 0) + e.duration_ns
    }
    if (isPassEvent(e)) {
      passTotalNs += e.duration_ns
      passCount += 1
    }
    if (e.event_type === EVENT_TYPE.runtimeSessionRun) {
      runtimeNs = (runtimeNs ?? 0) + e.duration_ns
      runCount += 1
    }
    if (e.event_type === EVENT_TYPE.cacheExactHit) exact += 1
    else if (e.event_type === EVENT_TYPE.cacheFuzzyHit) fuzzy += 1
    else if (e.event_type === EVENT_TYPE.cacheMissSyncCompile) miss += 1
    if (e.severity === 'error' || e.status === 'error') errors += 1
    else if (e.severity === 'warn') warns += 1
  }

  const totalCache = exact + fuzzy + miss
  return {
    compileNs,
    passTotalNs,
    passCount,
    runtimeNs,
    runCount,
    cacheExactHit: exact,
    cacheFuzzyHit: fuzzy,
    cacheMiss: miss,
    cacheHitRate: totalCache > 0 ? (exact + fuzzy) / totalCache : null,
    errorCount: errors,
    warnCount: warns,
    eventCount: evts.length,
  }
}

function queryPhaseBreakdown(evts: KxcEvent[]): PhaseBreakdownResult {
  // 阶段取自 pipeline 汇总事件与 lowering / codegen，而不是把逐 pass 耗时相加，
  // 后者会把嵌套 span 重复计入（需求文档非目标里也点了这个坑）。
  const acc = new Map<string, { component: string; ns: number; count: number }>()
  let compileNs = 0

  const label: Record<string, string> = {
    [COMPONENT.relayPipeline]: 'Relay Pass',
    [COMPONENT.tirPipeline]: 'TIR Pass',
    [COMPONENT.lowering]: 'Lowering',
  }

  for (const e of evts) {
    if (e.event_type === EVENT_TYPE.compileModule) {
      compileNs += e.duration_ns
      continue
    }
    let name: string | null = null
    if (PIPELINE_COMPONENTS.includes(e.component) && e.event_type === EVENT_TYPE.runPipeline) {
      name = label[e.component] ?? e.component
    } else if (e.component === COMPONENT.lowering && e.event_type === EVENT_TYPE.lowerToTir) {
      name = 'Lowering'
    } else if (e.event_type === EVENT_TYPE.llvmJitCompile) {
      name = 'CodeGen'
    } else if (e.event_type === EVENT_TYPE.runtimeSessionRun) {
      name = 'Runtime'
    }
    if (!name) continue
    const cur = acc.get(name) ?? { component: e.component, ns: 0, count: 0 }
    cur.ns += e.duration_ns
    cur.count += 1
    acc.set(name, cur)
  }

  const items = [...acc.entries()]
    .map(([phase, v]) => ({ phase, component: v.component, durationNs: v.ns, count: v.count }))
    .sort((a, b) => b.durationNs - a.durationNs)
  const totalNs = items.reduce((s, i) => s + i.durationNs, 0)

  // Runtime 不属于编译，算未归类耗时时要排除。
  const compileAttributed = items
    .filter((i) => i.phase !== 'Runtime')
    .reduce((s, i) => s + i.durationNs, 0)

  return {
    items,
    totalNs,
    unattributedNs: compileNs > 0 ? Math.max(0, compileNs - compileAttributed) : null,
  }
}

function queryPassRanking(evts: KxcEvent[], limit: number): PassRankingResult {
  const acc = new Map<string, { component: string; ns: number; n: number; max: number; changed: boolean; before: number | null; after: number | null }>()
  for (const e of evts) {
    if (!isPassEvent(e) || !e.pass_name) continue
    const key = `${e.component} ${e.pass_name}`
    const cur =
      acc.get(key) ??
      { component: e.component, ns: 0, n: 0, max: 0, changed: false, before: null, after: null }
    cur.ns += e.duration_ns
    cur.n += 1
    if (e.duration_ns > cur.max) cur.max = e.duration_ns
    if (e.fields.ir_changed === 'true') cur.changed = true
    if (e.metrics.ir_before_bytes != null) cur.before = e.metrics.ir_before_bytes
    if (e.metrics.ir_after_bytes != null) cur.after = e.metrics.ir_after_bytes
    acc.set(key, cur)
  }
  const all = [...acc.entries()].map(([key, v]) => ({
    passName: key.split(' ')[1] ?? '',
    component: v.component,
    totalNs: v.ns,
    count: v.n,
    maxNs: v.max,
    changed: v.changed,
    irBeforeBytes: v.before,
    irAfterBytes: v.after,
  }))
  all.sort((a, b) => b.totalNs - a.totalNs)
  return { items: all.slice(0, limit), totalNs: all.reduce((s, i) => s + i.totalNs, 0) }
}

function queryPassWaterfall(evts: KxcEvent[], b: ParsedBundle): PassWaterfallResult {
  const items = evts
    .filter((e) => isPassEvent(e) || e.event_type === EVENT_TYPE.lowerToTir)
    .map((e) => ({
      spanId: e.span_id,
      passName: e.pass_name || e.event_type,
      component: e.component,
      startNs: e.ts_ns,
      durationNs: e.duration_ns,
      status: e.status,
      changed: e.fields.ir_changed === 'true',
      irBeforeBytes: e.metrics.ir_before_bytes ?? null,
      irAfterBytes: e.metrics.ir_after_bytes ?? null,
      depth: spanDepth(e, b),
    }))
    .sort((x, y) => x.startNs - y.startNs)

  const startNs = items.length ? Math.min(...items.map((i) => i.startNs)) : 0
  const endNs = items.length ? Math.max(...items.map((i) => i.startNs + i.durationNs)) : 0
  return { items, startNs, endNs }
}

function spanDepth(e: KxcEvent, b: ParsedBundle): number {
  let depth = 0
  let cur = e.parent_span_id
  while (cur && depth < 16) {
    const p = b.bySpanId.get(cur)
    if (!p) break
    depth += 1
    cur = p.parent_span_id
  }
  return depth
}

function queryHotspot(
  evts: KxcEvent[],
  by: 'op' | 'pass' | 'kernel' | 'component',
  limit: number,
): HotspotResult {
  const acc = new Map<string, { ns: number; n: number }>()
  let caveat: string | null = null

  for (const e of evts) {
    let key = ''
    if (by === 'op') {
      key = readOpName(e)
      if (key && e.event_type === EVENT_TYPE.kernelExec) {
        // executor.cc 的 kernel_exec span 包裹的是 ccl_backend_->Copy，不是真实算子执行，
        // 直接当算子热点会误导。如实标注而不是悄悄画出来。
        caveat =
          '算子耗时来自 execution_plan.kernel_exec span，该 span 当前包裹的是值拷贝而非真实 kernel 执行，仅可用于结构分析，不能作为性能结论'
      }
    } else if (by === 'pass') key = e.pass_name
    else if (by === 'kernel') key = readKernelSymbol(e)
    else key = e.component
    if (!key) continue
    const cur = acc.get(key) ?? { ns: 0, n: 0 }
    cur.ns += e.duration_ns
    cur.n += 1
    acc.set(key, cur)
  }

  const all = [...acc.entries()]
    .map(([key, v]) => ({ key, totalNs: v.ns, count: v.n, selfNs: v.ns }))
    .sort((a, b2) => b2.totalNs - a.totalNs)
  return {
    items: all.slice(0, limit),
    totalNs: all.reduce((s, i) => s + i.totalNs, 0),
    caveat,
  }
}

function queryKernelDist(evts: KxcEvent[], buckets: number): KernelDurationDistResult {
  const durations: number[] = []
  let sawCupti = false
  for (const e of evts) {
    if (e.event_type === EVENT_TYPE.cudaKernel) {
      sawCupti = true
      durations.push(e.duration_ns)
    } else if (e.event_type === EVENT_TYPE.kernelExec) {
      durations.push(e.duration_ns)
    }
  }
  if (durations.length === 0) {
    return { buckets: [], totalCount: 0, p50Ns: null, p95Ns: null, maxNs: null, caveat: null }
  }
  durations.sort((a, b) => a - b)
  const max = durations[durations.length - 1] ?? 0
  const min = durations[0] ?? 0
  const width = Math.max(1, (max - min) / Math.max(1, buckets))
  const hist = Array.from({ length: buckets }, (_, i) => ({
    lowNs: min + i * width,
    highNs: min + (i + 1) * width,
    count: 0,
  }))
  for (const d of durations) {
    const idx = Math.min(buckets - 1, Math.floor((d - min) / width))
    const slot = hist[idx]
    if (slot) slot.count += 1
  }
  const pick = (p: number) => durations[Math.min(durations.length - 1, Math.floor(durations.length * p))] ?? null

  return {
    buckets: hist,
    totalCount: durations.length,
    p50Ns: pick(0.5),
    p95Ns: pick(0.95),
    maxNs: max,
    caveat: sawCupti
      ? null
      : '当前 bundle 无 CUPTI kernel activity，分布基于 execution_plan.kernel_exec 的逻辑耗时，不是 GPU 硬件耗时',
  }
}

function queryShapeCache(evts: KxcEvent[], b: ParsedBundle): ShapeCacheResult {
  const acc = new Map<string, ShapeCacheResult['cells'][number]>()
  let unresolved = 0
  for (const e of evts) {
    if (!CACHE_EVENT_TYPES.includes(e.event_type)) continue
    const hash = e.fields.shape_hash ?? ''
    const sig = resolveShape(e, b)
    if (!sig && !hash) {
      unresolved += 1
      continue
    }
    if (!sig) unresolved += 1
    const key = sig || `hash:${hash}`
    const cell =
      acc.get(key) ??
      { shapeSignature: sig || `(未解析 ${hash})`, shapeHash: hash, exactHit: 0, fuzzyHit: 0, miss: 0, totalNs: 0 }
    if (e.event_type === EVENT_TYPE.cacheExactHit) cell.exactHit += 1
    else if (e.event_type === EVENT_TYPE.cacheFuzzyHit) cell.fuzzyHit += 1
    else cell.miss += 1
    cell.totalNs += e.duration_ns
    acc.set(key, cell)
  }
  const cells = [...acc.values()].sort(
    (a, b2) => b2.exactHit + b2.fuzzyHit + b2.miss - (a.exactHit + a.fuzzyHit + a.miss),
  )
  return { cells, unresolved }
}

function queryTimeline(evts: KxcEvent[], buckets: number, meta: MetaResult): TimelineResult {
  const startNs = meta.spanNs.startNs
  const endNs = Math.max(meta.spanNs.endNs, startNs + 1)
  const bucketNs = Math.max(1, (endNs - startNs) / buckets)
  const perComponent = new Map<string, number[]>()
  const counts = new Array<number>(buckets).fill(0)

  for (const e of evts) {
    const idx = Math.min(buckets - 1, Math.max(0, Math.floor((e.ts_ns - startNs) / bucketNs)))
    counts[idx] = (counts[idx] ?? 0) + 1
    let arr = perComponent.get(e.component)
    if (!arr) {
      arr = new Array<number>(buckets).fill(0)
      perComponent.set(e.component, arr)
    }
    arr[idx] = (arr[idx] ?? 0) + Math.max(0, e.duration_ns)
  }
  return {
    startNs,
    endNs,
    bucketNs,
    series: [...perComponent.entries()].map(([component, values]) => ({ component, values })),
    counts,
  }
}

function queryFacets(evts: KxcEvent[], b: ParsedBundle): FacetsResult {
  const mk = () => new Map<string, number>()
  const components = mk()
  const eventTypes = mk()
  const devices = mk()
  const passes = mk()
  const ops = mk()
  const kernels = mk()
  const shapes = mk()
  const severities = mk()
  const statuses = mk()
  const bump = (m: Map<string, number>, k: string) => {
    if (!k) return
    m.set(k, (m.get(k) ?? 0) + 1)
  }
  for (const e of evts) {
    bump(components, e.component)
    bump(eventTypes, e.event_type)
    bump(devices, e.device)
    bump(passes, e.pass_name)
    bump(ops, readOpName(e))
    bump(kernels, readKernelSymbol(e))
    bump(shapes, resolveShape(e, b))
    bump(severities, e.severity)
    bump(statuses, e.status)
  }
  const out = (m: Map<string, number>) =>
    [...m.entries()]
      .map(([value, count]) => ({ value, count }))
      .sort((a, c) => c.count - a.count)
  return {
    components: out(components),
    eventTypes: out(eventTypes),
    devices: out(devices),
    passes: out(passes),
    ops: out(ops),
    kernels: out(kernels),
    shapes: out(shapes),
    severities: out(severities),
    statuses: out(statuses),
  }
}

function queryDiagnostics(b: ParsedBundle): DiagnosticsResult {
  const list = b.diagnostics?.diagnostics ?? []
  const notAnalyzed = list.length > 0 && list.every((d) => d.category === 'not_analyzed')
  return { diagnostics: list, notAnalyzed }
}

function queryEventBySpan(spanId: string, b: ParsedBundle): EventBySpanResult {
  const event = b.bySpanId.get(spanId) ?? null
  const ancestors: KxcEvent[] = []
  let cur = event?.parent_span_id
  while (cur && ancestors.length < 16) {
    const p = b.bySpanId.get(cur)
    if (!p) break
    ancestors.push(p)
    cur = p.parent_span_id
  }
  return { event, ancestors }
}

// ---------------------------------------------------------------------------
// 分发
// ---------------------------------------------------------------------------

export function runQuery(b: ParsedBundle, spec: QuerySpec, filter: QueryFilter): unknown {
  if (spec.kind === 'meta') return b.meta
  if (spec.kind === 'diagnostics') return queryDiagnostics(b)
  if (spec.kind === 'artifact') {
    const res: ArtifactResult = { path: spec.path, content: b.artifacts[spec.path] ?? null }
    return res
  }
  if (spec.kind === 'event_by_span') return queryEventBySpan(spec.spanId, b)

  const pred = makePredicate(filter, b)
  const evts = b.events.filter(pred)

  switch (spec.kind) {
    case 'kpi':
      return queryKpi(evts)
    case 'phase_breakdown':
      return queryPhaseBreakdown(evts)
    case 'pass_ranking':
      return queryPassRanking(evts, spec.limit)
    case 'pass_waterfall':
      return queryPassWaterfall(evts, b)
    case 'hotspot':
      return queryHotspot(evts, spec.by, spec.limit)
    case 'kernel_duration_dist':
      return queryKernelDist(evts, spec.buckets)
    case 'shape_cache':
      return queryShapeCache(evts, b)
    case 'timeline':
      return queryTimeline(evts, spec.buckets, b.meta)
    case 'facets':
      return queryFacets(evts, b)
    case 'events': {
      const res: EventsResult = {
        rows: evts.slice(spec.offset, spec.offset + spec.limit),
        total: evts.length,
        offset: spec.offset,
      }
      return res
    }
    case 'logs': {
      const logs = evts.filter((e) => e.event_type === EVENT_TYPE.log || e.severity === 'warn' || e.severity === 'error')
      const res: LogsResult = { rows: logs.slice(0, spec.limit), total: logs.length }
      return res
    }
    default:
      throw new Error(`未实现的查询: ${(spec as { kind: string }).kind}`)
  }
}

// 只有真在 Worker 里才挂消息处理；CLI 直接调 parseBundle / runQuery。
if (ctx) ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data
  if (msg.type === 'cancel') {
    cancelled.add(msg.id)
    return
  }
  try {
    if (msg.type === 'load') {
      const parsed = parseBundle(msg.bundleId, msg.files, msg.id)
      bundles.set(msg.bundleId, parsed)
      post({ type: 'loaded', id: msg.id, bundleId: msg.bundleId, meta: parsed.meta })
      return
    }
    if (msg.type === 'unload') {
      bundles.delete(msg.bundleId)
      post({ type: 'result', id: msg.id, data: true })
      return
    }
    if (msg.type === 'query') {
      if (cancelled.has(msg.id)) {
        cancelled.delete(msg.id)
        post({ type: 'cancelled', id: msg.id })
        return
      }
      const b = bundles.get(msg.bundleId)
      if (!b) throw new Error(`bundle 未加载: ${msg.bundleId}`)
      const data = runQuery(b, msg.spec, msg.filter)
      if (cancelled.has(msg.id)) {
        cancelled.delete(msg.id)
        post({ type: 'cancelled', id: msg.id })
        return
      }
      post({ type: 'result', id: msg.id, data })
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, message: err instanceof Error ? err.message : String(err) })
  }
}
