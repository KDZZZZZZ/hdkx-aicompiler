#!/usr/bin/env node
/**
 * 生成契约级 KXC profiling bundle fixture。
 *
 * 输出格式逐字段对齐 src/base/profiling.cc 的序列化实现（不是对齐需求文档）：
 *   - SerializeEventLine      profiling.cc:741-773  → events.jsonl
 *   - SerializePerfettoEvent  profiling.cc:775-795  → trace.json 的 traceEvents
 *   - WriteAllOutputsLocked   profiling.cc:1058-1157 → manifest/summary/diagnosis
 *
 * 刻意保留的真实产物特征（前端必须能吃下这些，不要在 fixture 里“修好”它们）：
 *   1. 时间戳 key 是 ts_ns，不是 timestamp_ns。
 *   2. summary.json 的 component_counts 值是字符串。
 *   3. 顶层 op_name 恒为空串；算子名只在 fields.op_name。
 *   4. worker_id 未赋值时是 -1。
 *   5. trace.json 的 args 只有 status/severity/message，没有 run_id/span_id。
 *   6. trace.json 的 ts/dur 是整数微秒，亚微秒事件 dur 会变 0。
 *   7. cache_* span 只带 fields.shape_hash，shape_signature 要靠父 span 取。
 *   8. diagnosis.json 提供两种形态：C++ 占位版（4 字段）与 Python 分析版（含 evidence）。
 *
 * 用法：
 *   node scripts/make-fixture.mjs                 # 默认 baseline + candidate
 *   node scripts/make-fixture.mjs --scale 40      # 放大事件量做虚拟化压测
 *   node scripts/make-fixture.mjs --out ../other  # 指定输出根目录
 */

import { mkdirSync, writeFileSync, rmSync, readdirSync, readFileSync, existsSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const DEFAULT_OUT = resolve(HERE, '..', 'fixtures')

// ---------------------------------------------------------------------------
// 确定性随机：同样的输入永远生成同样的 bundle，方便做快照测试。
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function jitter(rand, base, spread) {
  return Math.max(1, Math.round(base * (1 + (rand() - 0.5) * 2 * spread)))
}

/** 真实 bundle 的 ir hash 形如 "f420c27ddbc366c7"：16 位小写十六进制，无 0x 前缀。 */
function hash16(seed) {
  let h = BigInt(Math.abs(Math.round(seed)) + 0x9e3779b9)
  h = (h * 0x100000001b3n) & 0xffffffffffffffffn
  return h.toString(16).padStart(16, '0').slice(-16)
}

function relayIrText(passName, side, bytes) {
  return (
    `#[version = "0.0.5"]\n` +
    `// ${passName} ${side} (${bytes} bytes)\n` +
    `fn @main(%x: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {\n` +
    (side === 'before'
      ? `  %0 = nn.conv2d(%x, meta[relay.Constant][0], padding=[3, 3, 3, 3], kernel_size=[7, 7]);\n` +
        `  %1 = nn.batch_norm(%0, meta[relay.Constant][1], meta[relay.Constant][2]);\n` +
        `  %2 = nn.relu(%1.0);\n`
      : `  %0 = fused_nn_conv2d_batch_norm_relu(%x, meta[relay.Constant][0]);\n` +
        `  %2 = %0;\n`) +
    `  %3 = nn.max_pool2d(%2, pool_size=[3, 3], strides=[2, 2]);\n` +
    `  nn.dense(nn.batch_flatten(%3), meta[relay.Constant][3])\n}\n`
  )
}

function tirIrText(passName, side, bytes) {
  return (
    `// ${passName} ${side} (${bytes} bytes)\n` +
    `primfn(x_handle: handle, out_handle: handle) -> ()\n` +
    `  attr = {"global_symbol": "main", "tir.noalias": True}\n` +
    `  buffers = {x: Buffer(x_data, float32, [1, 3, 224, 224], [])}\n` +
    (side === 'before'
      ? `  for (i: int32, 0, 150528) {\n    out[i] = (x[i] * 2f32)\n  }\n`
      : `  for (i.outer: int32, 0, 9408) {\n    for (i.inner: int32, 0, 16) "vectorized" {\n` +
        `      out[((i.outer*16) + i.inner)] = (x[((i.outer*16) + i.inner)] * 2f32)\n    }\n  }\n`)
  )
}

// ---------------------------------------------------------------------------
// 事件构造。字段顺序刻意与 SerializeEventLine 一致，便于肉眼 diff。
// ---------------------------------------------------------------------------
class BundleBuilder {
  constructor({ traceId, sessionId, runId, seed }) {
    this.traceId = traceId
    this.sessionId = sessionId
    this.runId = runId
    this.rand = mulberry32(seed)
    this.events = []
    this.artifacts = []
    this.spanCounter = 0
    this.clockNs = 0
  }

  nextSpanId() {
    this.spanCounter += 1
    return `span-${String(this.spanCounter).padStart(6, '0')}`
  }

  /** 推进相对时钟（ts_ns 是 steady_clock 相对时间，从 0 起算）。 */
  advance(ns) {
    this.clockNs += ns
    return this.clockNs
  }

  push(spec) {
    const fields = { ...(spec.fields ?? {}) }
    // profiling.cc:746-747 恒注入这两项。真实产物里 thread_id 是十进制字符串。
    fields.thread_id = spec.threadId ?? '6772'
    fields.schema_version = '1'

    this.events.push({
      trace_id: this.traceId,
      session_id: this.sessionId,
      run_id: spec.runId ?? this.runId,
      span_id: spec.spanId ?? '',
      parent_span_id: spec.parentSpanId ?? '',
      component: spec.component,
      event_type: spec.eventType,
      phase: spec.phase ?? (spec.durationNs > 0 ? 'complete' : 'instant'),
      ts_ns: spec.tsNs,
      duration_ns: spec.durationNs ?? 0,
      status: spec.status ?? 'ok',
      severity: spec.severity ?? 'info',
      device: spec.device ?? '',
      // 未赋值即 -1，不是 null。
      worker_id: spec.workerId ?? -1,
      // 顶层 op_name 在真实产物里恒为空串。
      op_name: spec.opName ?? '',
      pass_name: spec.passName ?? '',
      kernel_symbol: spec.kernelSymbol ?? '',
      shape_signature: spec.shapeSignature ?? '',
      message: spec.message ?? '',
      fields,
      metrics: spec.metrics ?? {},
    })
  }

  artifact(relativePath, content) {
    this.artifacts.push({ relativePath, content })
    return `artifacts/${relativePath}`
  }
}

// ---------------------------------------------------------------------------
// 场景：一次 ResNet18 编译 + 多 shape 运行时会话。
// ---------------------------------------------------------------------------

// Pass 名与真实 bundle 一致：全部 snake_case，取自 profile_bundle_test 的实际流水线。
const RELAY_PASSES = [
  ['infer_type', 120_000],
  ['fold_constant', 480_000],
  ['simplify_expr', 210_000],
  ['fold_tuple_get_item', 1_450_000],
  ['eliminate_common_subexpr', 620_000],
  ['canonicalize_cast', 180_000],
  ['eliminate_dead_let', 240_000],
  ['annotate_memory_scope', 890_000],
  ['remove_standalone_reshapes', 320_000],
  ['capture_post_dfs_index_in_spans', 150_000],
]

const TIR_PASSES = [
  ['fold_constant', 140_000],
  ['simplify_expr', 760_000],
  ['vectorize_loop', 1_120_000],
  ['unroll_loop', 540_000],
  ['force_narrow_index_to_i32', 220_000],
  ['loop_partition', 980_000],
  ['convert_for_loops_serial', 310_000],
  ['remove_no_op', 190_000],
]

/** 制造回归的那个 pass，candidate 侧被放大。 */
const REGRESSED_PASS = 'fold_tuple_get_item'

const OPS = [
  'nn_conv2d',
  'nn_relu',
  'nn_max_pool2d',
  'add',
  'nn_global_avg_pool2d',
  'nn_flatten',
  'nn_gemm',
]

const SHAPES = [
  { sig: '[1,3,224,224]', hash: '1844674407370955161' },
  { sig: '[4,3,224,224]', hash: '9223372036854775807' },
  { sig: '[8,3,224,224]', hash: '3141592653589793238' },
  { sig: '[16,3,224,224]', hash: '2718281828459045235' },
  { sig: '[1,3,299,299]', hash: '1414213562373095048' },
]

/**
 * @param variant 'baseline' | 'candidate'
 *   candidate 刻意制造两处可被 Compare 抓到的差异：
 *   - FuseOps 明显回退（pass 耗时回归）
 *   - 缓存命中率下降（更多 cache_miss_sync_compile）
 */
function buildScenario(variant, scale) {
  const isCandidate = variant === 'candidate'
  const b = new BundleBuilder({
    traceId: `kxc-${variant}-0001`,
    sessionId: `sess-${variant}`,
    runId: `run-${variant}-0001`,
    seed: isCandidate ? 20260726 : 20260721,
  })

  // ---- 编译总入口 ----------------------------------------------------------
  const compileSpan = b.nextSpanId()
  const compileStart = b.advance(50_000)

  // ---- Relay pass pipeline -------------------------------------------------
  const relayPipelineSpan = b.nextSpanId()
  const relayStart = b.advance(30_000)
  let irBytes = 184_320
  for (const [passName, baseNs] of RELAY_PASSES) {
    const regression = isCandidate && passName === REGRESSED_PASS ? 3.4 : 1
    const dur = jitter(b.rand, baseNs * regression, 0.08)
    const start = b.advance(12_000)
    const changed = passName !== 'infer_type' && passName !== 'canonicalize_cast'
    const afterBytes = changed ? Math.round(irBytes * (0.92 + b.rand() * 0.06)) : irBytes

    b.push({
      component: 'relay_pass',
      eventType: 'run_pass',
      spanId: b.nextSpanId(),
      parentSpanId: relayPipelineSpan,
      tsNs: start,
      durationNs: dur,
      passName,
      fields: {
        ir_before_hash: hash16(irBytes),
        ir_after_hash: hash16(afterBytes),
        ir_changed: changed ? 'true' : 'false',
      },
      metrics: { ir_before_bytes: irBytes, ir_after_bytes: afterBytes },
    })

    // artifact 路径与真实 bundle 一致：artifacts/<run_id>/relay/<pass>.<side>.relay.txt
    b.artifact(
      `${b.runId}/relay/${passName}.before.relay.txt`,
      relayIrText(passName, 'before', irBytes),
    )
    b.artifact(
      `${b.runId}/relay/${passName}.after.relay.txt`,
      relayIrText(passName, 'after', afterBytes),
    )
    irBytes = afterBytes
    b.advance(dur)
  }
  // pipeline 汇总事件的 component 是 relay_pipeline，event_type 是 run_pipeline。
  b.push({
    component: 'relay_pipeline',
    eventType: 'run_pipeline',
    spanId: relayPipelineSpan,
    parentSpanId: compileSpan,
    tsNs: relayStart,
    durationNs: b.clockNs - relayStart,
    metrics: { pass_count: RELAY_PASSES.length },
  })

  // ---- Lowering ------------------------------------------------------------
  // 真实 bundle 里 lowering 事件不带 pass_name，artifact 的 before 是 relay、after 是 tir。
  const loweringStart = b.advance(20_000)
  const loweringDur = jitter(b.rand, 2_300_000, 0.1)
  const tirAfterBytes = Math.round(irBytes * 2.4)
  b.push({
    component: 'lowering',
    eventType: 'lower_to_tir',
    spanId: b.nextSpanId(),
    parentSpanId: compileSpan,
    tsNs: loweringStart,
    durationNs: loweringDur,
    fields: { ir_before_hash: hash16(irBytes), ir_after_hash: hash16(tirAfterBytes) },
    metrics: { ir_before_bytes: irBytes, ir_after_bytes: tirAfterBytes },
  })
  b.artifact(`${b.runId}/lower/lower_to_tir.before.relay.txt`, relayIrText('lower_to_tir', 'before', irBytes))
  b.artifact(`${b.runId}/lower/lower_to_tir.after.tir.txt`, tirIrText('lower_to_tir', 'after', tirAfterBytes))
  b.advance(loweringDur)

  // ---- TIR pass pipeline ---------------------------------------------------
  const tirPipelineSpan = b.nextSpanId()
  const tirStart = b.advance(15_000)
  let tirBytes = Math.round(irBytes * 2.4)
  for (const [passName, baseNs] of TIR_PASSES) {
    const dur = jitter(b.rand, baseNs, 0.08)
    const start = b.advance(9_000)
    const changed = passName !== 'remove_no_op'
    const afterBytes = changed ? Math.round(tirBytes * (0.95 + b.rand() * 0.08)) : tirBytes
    b.push({
      component: 'tir_pass',
      eventType: 'run_pass',
      spanId: b.nextSpanId(),
      parentSpanId: tirPipelineSpan,
      tsNs: start,
      durationNs: dur,
      passName,
      fields: {
        ir_before_hash: hash16(tirBytes),
        ir_after_hash: hash16(afterBytes),
        ir_changed: changed ? 'true' : 'false',
      },
      metrics: { ir_before_bytes: tirBytes, ir_after_bytes: afterBytes },
    })
    b.artifact(`${b.runId}/tir/${passName}.before.tir.txt`, tirIrText(passName, 'before', tirBytes))
    b.artifact(`${b.runId}/tir/${passName}.after.tir.txt`, tirIrText(passName, 'after', afterBytes))
    tirBytes = afterBytes
    b.advance(dur)
  }
  b.push({
    component: 'tir_pipeline',
    eventType: 'run_pipeline',
    spanId: tirPipelineSpan,
    parentSpanId: compileSpan,
    tsNs: tirStart,
    durationNs: b.clockNs - tirStart,
    metrics: { pass_count: TIR_PASSES.length },
  })

  // ---- CodeGen -------------------------------------------------------------
  const jitStart = b.advance(18_000)
  const jitDur = jitter(b.rand, 4_600_000, 0.12)
  b.push({
    component: 'compiler',
    eventType: 'llvm_jit_compile',
    spanId: b.nextSpanId(),
    parentSpanId: compileSpan,
    tsNs: jitStart,
    durationNs: jitDur,
    kernelSymbol: 'kxc_main_kernel',
    fields: { opt_level: '3' },
  })
  b.advance(jitDur)

  b.push({
    component: 'compiler',
    eventType: 'compile_module',
    spanId: compileSpan,
    parentSpanId: '',
    tsNs: compileStart,
    durationNs: b.clockNs - compileStart,
    fields: {
      compile_mode: 'aot',
      target_kind: 'llvm',
      backend: 'llvm',
      opt_level: '3',
    },
  })

  // ---- 运行时会话：多 shape，命中/未命中混合 --------------------------------
  // 关键结构：cache_* span 只带 fields.shape_hash，shape_signature 在父 span
  // runtime_session_run 上。前端必须靠 parent_span_id 或 shape_hash 做 join。
  const runCount = 24 * scale
  const seen = new Map()
  for (let i = 0; i < runCount; i += 1) {
    const shape = SHAPES[Math.floor(b.rand() * SHAPES.length)]
    const count = (seen.get(shape.hash) ?? 0) + 1
    seen.set(shape.hash, count)

    const sessionSpan = b.nextSpanId()
    const sessionStart = b.advance(40_000)

    // candidate 的模糊命中退化成未命中，制造缓存回归。
    const firstTime = count === 1
    const fuzzyChance = isCandidate ? 0.12 : 0.34
    let cacheEvent
    if (firstTime) cacheEvent = 'cache_miss_sync_compile'
    else if (b.rand() < fuzzyChance) cacheEvent = 'cache_fuzzy_hit'
    else if (b.rand() < (isCandidate ? 0.55 : 0.82)) cacheEvent = 'cache_exact_hit'
    else cacheEvent = 'cache_miss_sync_compile'

    const cacheDur =
      cacheEvent === 'cache_miss_sync_compile'
        ? jitter(b.rand, 12_400_000, 0.15)
        : jitter(b.rand, 86_000, 0.3)

    const cacheStart = b.advance(5_000)
    b.push({
      component: 'runtime_session',
      eventType: cacheEvent,
      spanId: b.nextSpanId(),
      parentSpanId: sessionSpan,
      tsNs: cacheStart,
      durationNs: cacheDur,
      // 刻意不写 shapeSignature —— 真实代码里这三个 span 没有。
      fields: { shape_hash: shape.hash },
    })
    b.advance(cacheDur)

    // 执行计划节点：算子名藏在 fields.op_name，顶层 op_name 为空。
    for (let n = 0; n < 3; n += 1) {
      const op = OPS[Math.floor(b.rand() * OPS.length)]
      const kdur = jitter(b.rand, 240_000, 0.5)
      const kstart = b.advance(3_000)
      b.push({
        component: 'execution_plan',
        eventType: 'kernel_exec',
        spanId: b.nextSpanId(),
        parentSpanId: sessionSpan,
        tsNs: kstart,
        durationNs: kdur,
        fields: { op_name: op, kernel_symbol: `kxc_${op}_kernel` },
        metrics: { input_count: 1 + Math.floor(b.rand() * 2), output_count: 1 },
      })
      b.advance(kdur)
    }

    // 设备侧 alloc/copy/free
    for (const [evt, base, bytes] of [
      ['alloc', 42_000, 3_211_264],
      ['copy', 310_000, 3_211_264],
      ['free', 28_000, 3_211_264],
    ]) {
      const d = jitter(b.rand, base, 0.35)
      const s = b.advance(2_000)
      b.push({
        component: 'device_api',
        eventType: evt,
        spanId: b.nextSpanId(),
        parentSpanId: sessionSpan,
        tsNs: s,
        durationNs: d,
        device: 'cpu:0',
        metrics: { bytes },
      })
      b.advance(d)
    }

    b.push({
      component: 'runtime_session',
      eventType: 'runtime_session_run',
      spanId: sessionSpan,
      parentSpanId: '',
      tsNs: sessionStart,
      durationNs: b.clockNs - sessionStart,
      // 只有这里有 shape_signature。
      shapeSignature: shape.sig,
      fields: { shape_hash: shape.hash },
      metrics: { shape_seen_count: count },
    })

    // 触发后台编译（instant 事件，duration_ns = 0）
    if (count === 2) {
      b.push({
        component: 'runtime_session',
        eventType: 'background_compile_submitted',
        spanId: b.nextSpanId(),
        parentSpanId: sessionSpan,
        tsNs: b.advance(1_000),
        durationNs: 0,
        phase: 'instant',
        shapeSignature: shape.sig,
        fields: { shape_hash: shape.hash },
        metrics: { priority: count, seen_count: count },
      })
    }
  }

  // ---- 日志事件（Logs Tile 与 Diagnostics 用） ------------------------------
  b.push({
    component: 'lowering',
    eventType: 'log',
    spanId: b.nextSpanId(),
    tsNs: b.advance(9_000),
    durationNs: 0,
    phase: 'instant',
    severity: 'warn',
    message: 'nn_gemm 未命中特化模板，回退到通用实现',
  })
  if (isCandidate) {
    b.push({
      component: 'relay_pass',
      eventType: 'log',
      spanId: b.nextSpanId(),
      tsNs: b.advance(4_000),
      durationNs: 0,
      phase: 'instant',
      severity: 'error',
      status: 'error',
      passName: REGRESSED_PASS,
      message: `${REGRESSED_PASS} 触发了 2 次重试，融合组规模超过阈值`,
    })
  }
  b.push({
    component: 'backend.cuda',
    eventType: 'log',
    spanId: b.nextSpanId(),
    tsNs: b.advance(3_000),
    durationNs: 0,
    phase: 'instant',
    severity: 'info',
    message: 'CUPTI 不可用，GPU activity 未采集',
  })

  return b
}

// ---------------------------------------------------------------------------
// 序列化：严格复刻 C++ 的写出方式
// ---------------------------------------------------------------------------

function writeBundle(outDir, b, { variant, analyzed }) {
  rmSync(outDir, { recursive: true, force: true })
  mkdirSync(join(outDir, 'artifacts'), { recursive: true })

  // events.jsonl —— 按 ts_ns 排序，与真实产物的写入顺序一致（记录即写）。
  const ordered = [...b.events].sort((x, y) => x.ts_ns - y.ts_ns)
  const eventsText = ordered.map((e) => JSON.stringify(e)).join('\n') + '\n'
  writeFileSync(join(outDir, 'events.jsonl'), eventsText, 'utf8')

  // trace.json —— args 只有三项；ts/dur 整除 1000 变微秒。
  const traceEvents = ordered.map((e) => {
    const isDuration = e.duration_ns > 0
    return {
      name: e.event_type || e.component,
      cat: e.component,
      ph: isDuration ? 'X' : 'i',
      ts: Math.floor(e.ts_ns / 1000),
      dur: Math.floor(e.duration_ns / 1000),
      pid: 4242,
      tid: 1911,
      s: 't',
      args: { status: e.status, severity: e.severity, message: e.message },
    }
  })
  writeFileSync(
    join(outDir, 'trace.json'),
    JSON.stringify({
      traceEvents,
      metadata: {
        trace_id: b.traceId,
        session_id: b.sessionId,
        // 固定墙钟起点，保证 fixture 可复现。
        start_time_ns: variant === 'candidate' ? 1774483200000000000 : 1774396800000000000,
      },
    }),
    'utf8',
  )

  // summary.json —— component_counts 的值是字符串。
  const counts = {}
  for (const e of ordered) counts[e.component] = String((Number(counts[e.component] ?? 0) || 0) + 1)
  writeFileSync(
    join(outDir, 'summary.json'),
    JSON.stringify({
      trace_id: b.traceId,
      session_id: b.sessionId,
      event_count: ordered.length,
      component_counts: counts,
    }),
    'utf8',
  )

  // manifest.json
  writeFileSync(
    join(outDir, 'manifest.json'),
    JSON.stringify({
      schema_version: 1,
      trace_id: b.traceId,
      session_id: b.sessionId,
      bundle_dir: `/home/kxc/profile_bundles/${b.traceId}`,
      log_level: 'info',
      ir_capture_mode: 'on_change',
      enable_nvtx: false,
      enable_cupti: false,
      cupti_available: false,
      record_execution_plan_details: true,
      event_count: ordered.length,
    }),
    'utf8',
  )

  // diagnosis.json —— analyzed=false 复刻 C++ 占位版；true 复刻 Python 分析版。
  const diagnostics = analyzed
    ? buildAnalyzedDiagnostics(ordered, variant)
    : [
        {
          category: 'not_analyzed',
          severity: 'info',
          component: 'analysis',
          summary:
            'Bundle created. Run PYTHONPATH=python python -m kxc_agent.cli analyze_bundle --bundle <path> to populate diagnostics.',
        },
      ]
  writeFileSync(
    join(outDir, 'diagnosis.json'),
    JSON.stringify({ trace_id: b.traceId, diagnostics }),
    'utf8',
  )

  const md =
    '# Diagnostics\n\n' +
    diagnostics
      .map((d) => `- [${d.severity}] ${d.category} (${d.component}): ${d.summary}\n`)
      .join('')
  writeFileSync(join(outDir, 'diagnosis.md'), md, 'utf8')

  for (const a of b.artifacts) {
    const p = join(outDir, 'artifacts', a.relativePath)
    mkdirSync(dirname(p), { recursive: true })
    writeFileSync(p, a.content, 'utf8')
  }

  return ordered.length
}

/** 复刻 kxc_agent diagnosis_engine 的输出形态：多了 evidence 与 next_steps。 */
function buildAnalyzedDiagnostics(events, variant) {
  const passTotals = new Map()
  for (const e of events) {
    if (e.event_type !== 'run_pass' || !e.pass_name) continue
    passTotals.set(e.pass_name, (passTotals.get(e.pass_name) ?? 0) + e.duration_ns)
  }
  const [topPass, topNs] = [...passTotals.entries()].sort((a, c) => c[1] - a[1])[0] ?? ['', 0]

  let exact = 0
  let miss = 0
  for (const e of events) {
    if (e.event_type === 'cache_exact_hit') exact += 1
    if (e.event_type === 'cache_miss_sync_compile') miss += 1
  }

  const out = [
    {
      category: 'compile_hotspot',
      severity: 'warn',
      component: 'pass_pipeline',
      summary: `${topPass} 占编译 pass 总耗时最高`,
      evidence: { pass_name: topPass, duration_ns: topNs, component: 'relay_pass' },
      next_steps: [
        `检查 ${topPass} 的输入 IR 规模`,
        '对比 baseline bundle 判断是否为回归',
      ],
    },
    {
      category: 'cache_miss_pattern',
      severity: miss > exact ? 'error' : 'info',
      component: 'runtime_session',
      summary: `缓存未命中 ${miss} 次，精确命中 ${exact} 次`,
      evidence: { cache_exact_hit: exact, cache_miss_sync_compile: miss },
      next_steps: ['考虑扩大 shape 特化范围', '检查 warmup 是否覆盖热点 shape'],
    },
  ]
  if (variant === 'candidate') {
    out.push({
      category: 'pass_regression',
      severity: 'error',
      component: 'pass_pipeline',
      summary: `${REGRESSED_PASS} 相对基线回归 3.4x`,
      evidence: {
        pass_name: REGRESSED_PASS,
        base_duration_ns: 1_450_000,
        new_duration_ns: 4_930_000,
      },
      next_steps: [`定位 ${REGRESSED_PASS} 的输入规模变化`, '回滚最近的相关改动验证'],
    })
  }
  return out
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const out = { scale: 1, outDir: DEFAULT_OUT }
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--scale') out.scale = Math.max(1, Number(argv[++i]) || 1)
    else if (argv[i] === '--out') out.outDir = resolve(process.cwd(), argv[++i] ?? DEFAULT_OUT)
  }
  return out
}

const { scale, outDir } = parseArgs(process.argv.slice(2))

const targets = [
  { variant: 'baseline', dir: 'bundles/baseline', analyzed: true },
  { variant: 'candidate', dir: 'bundles/candidate', analyzed: true },
  // 第三份刻意不跑分析，用来验证前端对 not_analyzed 占位诊断的处理。
  { variant: 'baseline', dir: 'bundles/raw-unanalyzed', analyzed: false },
]

const index = []
for (const t of targets) {
  const b = buildScenario(t.variant, scale)
  const dir = join(outDir, t.dir)
  const n = writeBundle(dir, b, { variant: t.variant, analyzed: t.analyzed })
  index.push({ id: t.dir.split('/').pop(), path: t.dir, variant: t.variant, event_count: n })
  console.log(`  ${t.dir.padEnd(26)} ${String(n).padStart(6)} events`)
}

// 把手工放进来的真实 bundle（目录名以 real- 开头）也纳入索引。
// 这些是 profile_bundle_test 跑出来的真产物，不由本脚本生成，但前端必须能直接加载，
// 用来验证 fixture 没有失真。
for (const dirent of readdirSync(join(outDir, 'bundles'), { withFileTypes: true })) {
  if (!dirent.isDirectory() || !dirent.name.startsWith('real-')) continue
  const manifestPath = join(outDir, 'bundles', dirent.name, 'manifest.json')
  if (!existsSync(manifestPath)) continue
  const m = JSON.parse(readFileSync(manifestPath, 'utf8'))
  index.push({
    id: dirent.name,
    path: `bundles/${dirent.name}`,
    variant: 'real',
    event_count: m.event_count ?? 0,
  })
  console.log(`  ${`bundles/${dirent.name}`.padEnd(26)} ${String(m.event_count ?? 0).padStart(6)} events (真实产物)`)
}

// 供前端在没有文件选择器时直接按 URL 加载。
writeFileSync(
  join(outDir, 'bundles', 'index.json'),
  JSON.stringify({ bundles: index }, null, 2),
  'utf8',
)
console.log(`\nfixture 写入 ${outDir}/bundles（scale=${scale}）`)
