/**
 * KXC profiling bundle 的数据契约。
 *
 * 本文件的每一条都对照 src/base/profiling.cc 的序列化实现核验过，不是照抄需求文档。
 * 需求文档与实现不一致的地方以实现为准，并在此处注明，避免前端按文档写出解析不到的字段。
 *
 * 核验基线：worktree-perf-workbench @ origin/main
 *   - SerializeEventLine      src/base/profiling.cc:741-773
 *   - SerializePerfettoEvent  src/base/profiling.cc:775-795
 *   - WriteAllOutputsLocked   src/base/profiling.cc:1058-1157
 */

/** bundle 里由 C++ 直接写出的文件名。 */
export const BUNDLE_FILES = {
  manifest: 'manifest.json',
  events: 'events.jsonl',
  trace: 'trace.json',
  summary: 'summary.json',
  diagnosisJson: 'diagnosis.json',
  diagnosisMd: 'diagnosis.md',
  artifactsDir: 'artifacts',
} as const

/**
 * events.jsonl 的一行。字段顺序与 SerializeEventLine 一致。
 *
 * 契约陷阱（前端必须知道）：
 * 1. 时间戳的 JSON key 是 `ts_ns`，不是需求文档写的 `timestamp_ns`。(profiling.cc:759)
 * 2. `worker_id` 是数字，未赋值时序列化为 -1 而不是 null。(profiling.cc:764)
 * 3. `op_name` / `kernel_symbol` / `shape_signature` / `device` / `message` 恒被写出，
 *    未赋值时是空字符串，不会缺键。
 * 4. 顶层 `op_name` 目前在整个编译器里从未被赋值；算子名实际藏在 `fields.op_name`
 *    （src/base/disco/executor.cc:128、165）。取算子名必须两处都看，见 readOpName()。
 * 5. `fields` 恒含 `thread_id` 与 `schema_version`。(profiling.cc:746-747)
 * 6. `ts_ns` 是 steady_clock 相对时间，不是墙钟；墙钟起点在 trace.json 的
 *    metadata.start_time_ns。跨 bundle 比较绝不能直接比 ts_ns 绝对值。
 */
export interface KxcEvent {
  trace_id: string
  session_id: string
  run_id: string
  span_id: string
  parent_span_id: string
  component: string
  event_type: string
  phase: string
  ts_ns: number
  duration_ns: number
  status: string
  severity: KxcSeverity
  device: string
  worker_id: number
  op_name: string
  pass_name: string
  kernel_symbol: string
  shape_signature: string
  message: string
  fields: Record<string, string>
  metrics: Record<string, number>
}

/** LogSeverityToString 的取值域。(profiling.cc:762) */
export type KxcSeverity = 'debug' | 'info' | 'warn' | 'error'

export const SEVERITY_ORDER: readonly KxcSeverity[] = ['debug', 'info', 'warn', 'error']

export function severityRank(s: string): number {
  const i = SEVERITY_ORDER.indexOf(s as KxcSeverity)
  return i < 0 ? 0 : i
}

/** manifest.json。(profiling.cc:1060-1080) */
export interface KxcManifest {
  schema_version: number
  trace_id: string
  session_id: string
  /** 生成机器上的绝对路径。分享前必须脱敏，见 §19.5。 */
  bundle_dir: string
  log_level: KxcSeverity
  ir_capture_mode: string
  enable_nvtx: boolean
  enable_cupti: boolean
  cupti_available: boolean
  record_execution_plan_details: boolean
  event_count: number
}

/**
 * summary.json。(profiling.cc:1119-1125)
 *
 * 陷阱：`component_counts` 的值是**字符串**而非数字——C++ 侧用 StringMap 承载计数，
 * ToJsonObject 会加引号。前端必须 Number() 转换，直接相加会得到字符串拼接。
 */
export interface KxcSummary {
  trace_id: string
  session_id: string
  event_count: number
  component_counts: Record<string, string>
}

/**
 * diagnosis.json 的一条诊断。
 *
 * 存在两种形态，前端必须同时容忍：
 * - C++ 自动写出的占位版本只有 category/severity/component/summary 四个字段，
 *   且在没跑分析时是一条 category="not_analyzed" 的提示。(profiling.cc:1127-1148)
 * - Python 侧 `python -m kxc_agent.cli analyze_bundle` 覆盖写入后才有
 *   evidence 与 next_steps。(python/kxc_agent/services/diagnosis_engine.py)
 *
 * evidence 目前只是标量/文本映射，不含 span_id、时间范围或 artifact 路径，
 * 因此需求 §11「点击证据跳转到对应图表」在当前数据下只能做到按 pass_name /
 * component 这类名字回跳，做不到精确到事件。见 evidenceToContext()。
 */
export interface KxcDiagnostic {
  category: string
  severity: string
  component: string
  summary: string
  evidence?: Record<string, string | number>
  next_steps?: string[]
}

export interface KxcDiagnosisFile {
  trace_id: string
  diagnostics: KxcDiagnostic[]
}

/** trace.json。(profiling.cc:1088-1102) */
export interface KxcTraceFile {
  traceEvents: KxcTraceEvent[]
  metadata: {
    trace_id: string
    session_id: string
    /** 墙钟起点，纳秒。 */
    start_time_ns: number
  }
}

/**
 * trace.json 里的一条 Chrome Trace 事件。(profiling.cc:775-795)
 *
 * 陷阱：args 只有 status / severity / message 三项，**没有 run_id / span_id**，
 * 所以在 Perfetto 里选中一个事件无法反查回 events.jsonl 的那一条。
 * 需求 §9.5「Trace 中选择事件后打开对应 Pass/Op/Kernel」在当前产物下不可实现，
 * 只能按 (name, cat, ts) 做近似匹配，见 matchTraceEventToKxcEvent()。
 *
 * 另外 ts/dur 是整数微秒（纳秒整除 1000），亚微秒事件的 dur 会变成 0。
 */
export interface KxcTraceEvent {
  name: string
  cat: string
  ph: 'X' | 'i'
  /** 微秒 */
  ts: number
  /** 微秒 */
  dur: number
  pid: number
  tid: number
  s?: string
  args: { status?: string; severity?: string; message?: string }
}

// ---------------------------------------------------------------------------
// 字段读取助手：把上面那些契约陷阱收敛在一处，UI 层不要再各自处理。
// ---------------------------------------------------------------------------

/** 顶层 op_name 恒为空，真实算子名在 fields.op_name。 */
export function readOpName(e: KxcEvent): string {
  return e.op_name || e.fields.op_name || ''
}

/** kernel_symbol 顶层只在 CUPTI 路径有值，逻辑执行路径写在 fields 里。 */
export function readKernelSymbol(e: KxcEvent): string {
  return e.kernel_symbol || e.fields.kernel_symbol || ''
}

/** worker_id 未赋值时是 -1，视为“无 worker 信息”。 */
export function readWorkerId(e: KxcEvent): number | null {
  return e.worker_id < 0 ? null : e.worker_id
}

/** shape_hash 只在 fields 里，且是十进制字符串。 */
export function readShapeHash(e: KxcEvent): string | null {
  return e.fields.shape_hash ?? null
}

export function eventEndNs(e: KxcEvent): number {
  return e.ts_ns + (e.duration_ns > 0 ? e.duration_ns : 0)
}

export function isInstant(e: KxcEvent): boolean {
  return e.duration_ns <= 0
}

/**
 * 一行 JSONL 解析成 KxcEvent，并补齐缺省值。
 * 解析失败返回 null，由调用方计入「损坏行」计数而不是中断整个 bundle 加载。
 */
export function parseEventLine(line: string): KxcEvent | null {
  const trimmed = line.trim()
  if (!trimmed) return null
  let raw: unknown
  try {
    raw = JSON.parse(trimmed)
  } catch {
    return null
  }
  if (typeof raw !== 'object' || raw === null) return null
  const o = raw as Record<string, unknown>
  const str = (k: string): string => (typeof o[k] === 'string' ? (o[k] as string) : '')
  const num = (k: string, fallback = 0): number =>
    typeof o[k] === 'number' && Number.isFinite(o[k] as number) ? (o[k] as number) : fallback

  // component 与 ts_ns 是最低限度的可用性要求，缺了这两个无法参与任何聚合。
  if (typeof o.component !== 'string' || typeof o.ts_ns !== 'number') return null

  const fields: Record<string, string> = {}
  if (o.fields && typeof o.fields === 'object') {
    for (const [k, v] of Object.entries(o.fields as Record<string, unknown>)) {
      fields[k] = typeof v === 'string' ? v : String(v)
    }
  }
  const metrics: Record<string, number> = {}
  if (o.metrics && typeof o.metrics === 'object') {
    for (const [k, v] of Object.entries(o.metrics as Record<string, unknown>)) {
      const n = typeof v === 'number' ? v : Number(v)
      if (Number.isFinite(n)) metrics[k] = n
    }
  }

  const severity = str('severity')
  return {
    trace_id: str('trace_id'),
    session_id: str('session_id'),
    run_id: str('run_id'),
    span_id: str('span_id'),
    parent_span_id: str('parent_span_id'),
    component: str('component'),
    event_type: str('event_type'),
    phase: str('phase'),
    ts_ns: num('ts_ns'),
    duration_ns: num('duration_ns'),
    status: str('status') || 'ok',
    severity: (SEVERITY_ORDER as readonly string[]).includes(severity)
      ? (severity as KxcSeverity)
      : 'info',
    device: str('device'),
    worker_id: num('worker_id', -1),
    op_name: str('op_name'),
    pass_name: str('pass_name'),
    kernel_symbol: str('kernel_symbol'),
    shape_signature: str('shape_signature'),
    message: str('message'),
    fields,
    metrics,
  }
}

// ---------------------------------------------------------------------------
// 组件与事件类型常量：来自实际埋点调用点，不是猜的。
// ---------------------------------------------------------------------------

// 以下常量取自 out/build 跑出的真实 bundle（profile_bundle_test_output），
// 不是从需求文档推断的。两处容易搞错：
//   - pipeline 事件的 component 是 relay_pipeline / tir_pipeline，**不是** relay_pass，
//     所以按 component=relay_pass 过滤拿不到 pipeline 总耗时。
//   - pipeline 的 event_type 是 run_pipeline，不是 pipeline。
export const COMPONENT = {
  compiler: 'compiler',
  relayPass: 'relay_pass',
  relayPipeline: 'relay_pipeline',
  tirPass: 'tir_pass',
  tirPipeline: 'tir_pipeline',
  lowering: 'lowering',
  executionPlan: 'execution_plan',
  runtimeSession: 'runtime_session',
  runtime: 'runtime',
  deviceApi: 'device_api',
  backgroundCompiler: 'background_compiler',
  cuda: 'backend.cuda',
  analysis: 'analysis',
} as const

export const EVENT_TYPE = {
  // 编译
  compileModule: 'compile_module',
  llvmJitCompile: 'llvm_jit_compile',
  runPass: 'run_pass',
  runPipeline: 'run_pipeline',
  lowerToTir: 'lower_to_tir',
  // 运行时缓存（这三个 span 只带 fields.shape_hash，shape_signature 要向父 span 取）
  cacheExactHit: 'cache_exact_hit',
  cacheFuzzyHit: 'cache_fuzzy_hit',
  cacheMissSyncCompile: 'cache_miss_sync_compile',
  runtimeSessionRun: 'runtime_session_run',
  runtimeSessionWarmup: 'runtime_session_warmup',
  backgroundCompileSubmitted: 'background_compile_submitted',
  backgroundCompileCompleted: 'background_compile_completed',
  // 执行计划
  executePlan: 'execute_plan',
  executeKernelNode: 'execute_kernel_node',
  executeCommNode: 'execute_comm_node',
  executeBarrierNode: 'execute_barrier_node',
  kernelExec: 'kernel_exec',
  commExec: 'comm_exec',
  // 设备
  alloc: 'alloc',
  free: 'free',
  copy: 'copy',
  gpuSync: 'gpu_sync',
  // CUPTI
  cudaKernel: 'cuda_kernel',
  cudaMemcpy: 'cuda_memcpy',
  // 日志
  log: 'log',
} as const

/** 缓存事件集合，Shape × Cache 热力图与缓存健康度都依赖它。 */
export const CACHE_EVENT_TYPES: readonly string[] = [
  EVENT_TYPE.cacheExactHit,
  EVENT_TYPE.cacheFuzzyHit,
  EVENT_TYPE.cacheMissSyncCompile,
]

/** 逐 Pass 事件所在的 component，用于 Pass 瀑布与排行。 */
export const PASS_COMPONENTS: readonly string[] = [COMPONENT.relayPass, COMPONENT.tirPass]

/** pipeline 汇总事件所在的 component，用于阶段耗时分解（注意与上面不同）。 */
export const PIPELINE_COMPONENTS: readonly string[] = [
  COMPONENT.relayPipeline,
  COMPONENT.tirPipeline,
]

/**
 * IR artifact 的实际落盘路径规则，取自真实 bundle：
 *   artifacts/<run_id>/relay/<pass_name>.{before,after}.relay.txt
 *   artifacts/<run_id>/tir/<pass_name>.{before,after}.tir.txt
 *   artifacts/<run_id>/lower/lower_to_tir.before.relay.txt
 *   artifacts/<run_id>/lower/lower_to_tir.after.tir.txt
 * 注意 lower 阶段 before 是 relay、after 是 tir，两侧后缀不同。
 */
export function irArtifactPath(
  runId: string,
  stage: 'relay' | 'tir' | 'lower',
  passName: string,
  side: 'before' | 'after',
): string {
  const dialect = stage === 'lower' ? (side === 'before' ? 'relay' : 'tir') : stage
  return `artifacts/${runId}/${stage}/${passName}.${side}.${dialect}.txt`
}

/** 由事件反推它属于哪个 IR 阶段目录。 */
export function irStageOf(e: KxcEvent): 'relay' | 'tir' | 'lower' | null {
  if (e.component === COMPONENT.relayPass) return 'relay'
  if (e.component === COMPONENT.tirPass) return 'tir'
  if (e.component === COMPONENT.lowering) return 'lower'
  return null
}
