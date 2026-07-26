#!/usr/bin/env node
/**
 * kxc-wb —— 性能分析桌布命令行。
 *
 * 面向两类调用者：人，和 agent。所以有两条硬规则：
 * 1. `--json` 输出永远是**单个 JSON 对象**，不掺任何日志。agent 直接解析，不用切文本。
 * 2. 出错时给的是**能指导下一步**的信息（可用值列表、已加载的 bundle 等），
 *    而不是一句"失败"。agent 拿到错误应该知道该怎么改。
 *
 * 两族命令：
 *   query / compare  —— 无头读数据，不需要浏览器（复用界面同一份聚合实现）
 *   ui               —— 驱动正在浏览的那个页面（需要控制服务）
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, resolve, basename } from 'node:path'
import { parseBundle, runQuery } from '../src/kxc/query.worker.ts'

const PORT = process.env.KXC_WB_PORT ?? 5274
const CONTROL = `http://127.0.0.1:${PORT}`

// ---------------------------------------------------------------------------
// 参数解析
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const positional = []
  const flags = {}
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const key = a.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) flags[key] = true
      else {
        flags[key] = next
        i += 1
      }
    } else positional.push(a)
  }
  return { positional, flags }
}

const { positional, flags } = parseArgs(process.argv.slice(2))
const asJson = flags.json === true || flags.json === 'true'

function out(obj, humanFn) {
  if (asJson) {
    process.stdout.write(JSON.stringify(obj, null, 2) + '\n')
  } else if (humanFn) {
    humanFn(obj)
  } else {
    process.stdout.write(JSON.stringify(obj, null, 2) + '\n')
  }
}

function fail(message, extra = {}) {
  const payload = { ok: false, error: message, ...extra }
  if (asJson) process.stdout.write(JSON.stringify(payload, null, 2) + '\n')
  else process.stderr.write(`错误：${message}\n`)
  process.exit(1)
}

// ---------------------------------------------------------------------------
// bundle 读取
// ---------------------------------------------------------------------------

function loadBundleFromDisk(dir) {
  const root = resolve(process.cwd(), dir)
  if (!existsSync(root)) fail(`目录不存在：${root}`)
  const read = (f) => (existsSync(join(root, f)) ? readFileSync(join(root, f), 'utf8') : null)
  const events = read('events.jsonl')
  if (events == null) {
    fail(`${root} 下没有 events.jsonl，这不是一个 KXC profiling bundle`, {
      hint: 'bundle 目录应当包含 events.jsonl / manifest.json / trace.json 等文件',
    })
  }

  // artifact 按需读：IR 文本可能很大，query 命令通常用不到。
  const artifacts = {}
  const artRoot = join(root, 'artifacts')
  if (existsSync(artRoot)) {
    const walk = (d, prefix) => {
      for (const name of readdirSync(d)) {
        const p = join(d, name)
        if (statSync(p).isDirectory()) walk(p, `${prefix}/${name}`)
        else artifacts[`${prefix}/${name}`] = ''
      }
    }
    walk(artRoot, 'artifacts')
  }

  return parseBundle(basename(root), {
    manifest: read('manifest.json'),
    events,
    summary: read('summary.json'),
    diagnosis: read('diagnosis.json'),
    trace: read('trace.json'),
    artifacts,
  })
}

function buildFilter(f) {
  const num = (v) => (v === undefined || v === true ? null : Number(v))
  return {
    runId: typeof f.run === 'string' ? f.run : null,
    startNs: num(f['start-ns']),
    endNs: num(f['end-ns']),
    component: typeof f.component === 'string' ? f.component : null,
    eventType: typeof f['event-type'] === 'string' ? f['event-type'] : null,
    device: typeof f.device === 'string' ? f.device : null,
    workerId: f.worker === undefined ? null : Number(f.worker),
    op: typeof f.op === 'string' ? f.op : null,
    pass: typeof f.pass === 'string' ? f.pass : null,
    kernel: typeof f.kernel === 'string' ? f.kernel : null,
    shapeSignature: typeof f.shape === 'string' ? f.shape : null,
    status: typeof f.status === 'string' ? f.status : null,
    severity: typeof f.severity === 'string' ? f.severity : null,
    search: typeof f.search === 'string' ? f.search : null,
  }
}

// ---------------------------------------------------------------------------
// 显示助手
// ---------------------------------------------------------------------------

const ms = (ns) => (ns == null ? '—' : `${(ns / 1e6).toFixed(2)} ms`)
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`)

function table(rows, cols) {
  if (rows.length === 0) return '（无数据）\n'
  const widths = cols.map((c) =>
    Math.max(c.header.length, ...rows.map((r) => String(c.get(r)).length)),
  )
  const line = (cells) => cells.map((c, i) => String(c).padEnd(widths[i])).join('  ') + '\n'
  let s = line(cols.map((c) => c.header))
  s += widths.map((w) => '-'.repeat(w)).join('  ') + '\n'
  for (const r of rows) s += line(cols.map((c) => c.get(r)))
  return s
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

const QUERY_KINDS = {
  meta: () => ({ kind: 'meta' }),
  kpi: () => ({ kind: 'kpi' }),
  phases: () => ({ kind: 'phase_breakdown' }),
  'pass-ranking': (f) => ({ kind: 'pass_ranking', limit: Number(f.limit ?? 20) }),
  'pass-waterfall': () => ({ kind: 'pass_waterfall' }),
  hotspot: (f) => ({ kind: 'hotspot', by: String(f.by ?? 'pass'), limit: Number(f.limit ?? 20) }),
  'shape-cache': () => ({ kind: 'shape_cache' }),
  'kernel-dist': (f) => ({ kind: 'kernel_duration_dist', buckets: Number(f.buckets ?? 20) }),
  diagnostics: () => ({ kind: 'diagnostics' }),
  logs: (f) => ({ kind: 'logs', limit: Number(f.limit ?? 50) }),
  events: (f) => ({
    kind: 'events',
    offset: Number(f.offset ?? 0),
    limit: Number(f.limit ?? 50),
  }),
  facets: () => ({ kind: 'facets' }),
  timeline: (f) => ({ kind: 'timeline', buckets: Number(f.buckets ?? 60) }),
}

function cmdQuery() {
  const kind = positional[1]
  if (!kind || !QUERY_KINDS[kind]) {
    fail(`未知查询 ${kind ?? '(空)'}`, { available: Object.keys(QUERY_KINDS) })
  }
  if (!flags.bundle) fail('需要 --bundle <目录>')

  const bundle = loadBundleFromDisk(String(flags.bundle))
  const data = runQuery(bundle, QUERY_KINDS[kind](flags), buildFilter(flags))

  out({ ok: true, kind, bundle: String(flags.bundle), data }, () => {
    if (kind === 'kpi') {
      process.stdout.write(
        `编译总耗时  ${ms(data.compileNs)}\n` +
          `Pass 总耗时 ${ms(data.passTotalNs)}（${data.passCount} 个）\n` +
          `运行时耗时  ${ms(data.runtimeNs)}（${data.runCount} 次）\n` +
          `缓存命中率  ${pct(data.cacheHitRate)}（精确 ${data.cacheExactHit} / 模糊 ${data.cacheFuzzyHit} / 未命中 ${data.cacheMiss}）\n` +
          `告警        ${data.errorCount} error，${data.warnCount} warn\n`,
      )
      return
    }
    if (kind === 'pass-ranking') {
      process.stdout.write(
        table(data.items, [
          { header: 'Pass', get: (r) => r.passName },
          { header: '组件', get: (r) => r.component },
          { header: '总耗时', get: (r) => ms(r.totalNs) },
          { header: '次数', get: (r) => r.count },
          { header: '改动 IR', get: (r) => (r.changed ? '是' : '否') },
        ]),
      )
      return
    }
    if (kind === 'phases') {
      process.stdout.write(
        table(data.items, [
          { header: '阶段', get: (r) => r.phase },
          { header: '耗时', get: (r) => ms(r.durationNs) },
          { header: '事件数', get: (r) => r.count },
        ]),
      )
      if (data.unattributedNs) process.stdout.write(`\n未归类耗时 ${ms(data.unattributedNs)}\n`)
      return
    }
    if (kind === 'shape-cache') {
      process.stdout.write(
        table(data.cells, [
          { header: 'Shape', get: (r) => r.shapeSignature },
          { header: '精确命中', get: (r) => r.exactHit },
          { header: '模糊命中', get: (r) => r.fuzzyHit },
          { header: '未命中', get: (r) => r.miss },
        ]),
      )
      if (data.unresolved) process.stdout.write(`\n${data.unresolved} 条缓存事件无法解析出 shape\n`)
      return
    }
    if (kind === 'hotspot') {
      process.stdout.write(
        table(data.items, [
          { header: '对象', get: (r) => r.key },
          { header: '总耗时', get: (r) => ms(r.totalNs) },
          { header: '次数', get: (r) => r.count },
        ]),
      )
      if (data.caveat) process.stdout.write(`\n⚠ ${data.caveat}\n`)
      return
    }
    if (kind === 'diagnostics') {
      if (data.notAnalyzed) {
        process.stdout.write(
          '该 bundle 尚未分析。请先运行：\n  PYTHONPATH=python python -m kxc_agent.cli analyze_bundle --bundle <路径>\n',
        )
        return
      }
      for (const d of data.diagnostics) {
        process.stdout.write(`[${d.severity}] ${d.category}（${d.component}）：${d.summary}\n`)
        for (const s of d.next_steps ?? []) process.stdout.write(`    → ${s}\n`)
      }
      return
    }
    process.stdout.write(JSON.stringify(data, null, 2) + '\n')
  })
}

// ---------------------------------------------------------------------------
// compare —— agent 最常用的一条，直接给出可行动的结论
// ---------------------------------------------------------------------------

function cmdCompare() {
  if (!flags.baseline || !flags.candidate) fail('需要 --baseline <目录> --candidate <目录>')
  const base = loadBundleFromDisk(String(flags.baseline))
  const cand = loadBundleFromDisk(String(flags.candidate))
  const filter = buildFilter(flags)
  const q = (b, spec) => runQuery(b, spec, filter)

  const bk = q(base, { kind: 'kpi' })
  const ck = q(cand, { kind: 'kpi' })
  const bp = q(base, { kind: 'pass_ranking', limit: 500 })
  const cp = q(cand, { kind: 'pass_ranking', limit: 500 })

  const baseByPass = new Map(bp.items.map((i) => [i.passName, i]))
  const candByPass = new Map(cp.items.map((i) => [i.passName, i]))
  const threshold = Number(flags.threshold ?? 1.2)

  const passDeltas = []
  for (const [name, c] of candByPass) {
    const b = baseByPass.get(name)
    if (!b) {
      // 只在 candidate 出现的 pass：不可对齐，必须明确标出（§4.5.6）
      passDeltas.push({ passName: name, baseNs: null, candNs: c.totalNs, ratio: null, status: 'only_in_candidate' })
      continue
    }
    const ratio = b.totalNs === 0 ? null : c.totalNs / b.totalNs
    passDeltas.push({
      passName: name,
      baseNs: b.totalNs,
      candNs: c.totalNs,
      deltaNs: c.totalNs - b.totalNs,
      ratio,
      status: ratio == null ? 'unknown' : ratio >= threshold ? 'regression' : ratio <= 1 / threshold ? 'improvement' : 'stable',
    })
  }
  for (const [name, b] of baseByPass) {
    if (!candByPass.has(name)) {
      passDeltas.push({ passName: name, baseNs: b.totalNs, candNs: null, ratio: null, status: 'only_in_baseline' })
    }
  }
  passDeltas.sort((a, b) => Math.abs(b.deltaNs ?? 0) - Math.abs(a.deltaNs ?? 0))

  const regressions = passDeltas.filter((d) => d.status === 'regression')
  const result = {
    ok: true,
    baseline: String(flags.baseline),
    candidate: String(flags.candidate),
    threshold,
    kpi: {
      compileNs: { base: bk.compileNs, candidate: ck.compileNs },
      cacheHitRate: { base: bk.cacheHitRate, candidate: ck.cacheHitRate },
      cacheMiss: { base: bk.cacheMiss, candidate: ck.cacheMiss },
    },
    passDeltas: passDeltas.slice(0, Number(flags.limit ?? 20)),
    regressionCount: regressions.length,
    // 这段是给 agent 直接用的：哪些维度可比、哪些不可比。
    alignment: {
      pass: 'reliable',
      op: 'unavailable：顶层 op_name 未被编译器填充，无法跨 bundle 对齐算子',
      kernel: 'unavailable：CPU 路径下 kernel_symbol 多为空',
    },
  }

  out(result, () => {
    process.stdout.write(
      `编译总耗时  ${ms(bk.compileNs)} → ${ms(ck.compileNs)}\n` +
        `缓存命中率  ${pct(bk.cacheHitRate)} → ${pct(ck.cacheHitRate)}（未命中 ${bk.cacheMiss} → ${ck.cacheMiss}）\n\n`,
    )
    process.stdout.write(
      table(result.passDeltas, [
        { header: 'Pass', get: (r) => r.passName },
        { header: 'baseline', get: (r) => ms(r.baseNs) },
        { header: 'candidate', get: (r) => ms(r.candNs) },
        { header: '倍数', get: (r) => (r.ratio == null ? '—' : `${r.ratio.toFixed(2)}x`) },
        { header: '判定', get: (r) => r.status },
      ]),
    )
    process.stdout.write(`\n算子与 Kernel 维度无法对齐（编译器未填充相应字段）\n`)
  })
}

// ---------------------------------------------------------------------------
// ui —— 驱动正在浏览的页面
// ---------------------------------------------------------------------------

async function control(path, init) {
  try {
    const res = await fetch(`${CONTROL}${path}`, init)
    return { status: res.status, body: await res.json() }
  } catch {
    fail(`连不上控制服务 ${CONTROL}`, {
      hint: '先启动：npm run control --prefix tools/workbench（或 npm run dev:all）',
    })
  }
}

async function cmdUi() {
  const sub = positional[1]

  if (sub === 'state') {
    const { status, body } = await control('/state', { method: 'GET' })
    if (status !== 200) fail(body.error ?? '读取状态失败')
    out(body, () => printState(body.state))
    return
  }

  if (!sub) {
    fail('需要子命令', { available: ['state', ...UI_COMMANDS] })
  }
  if (!UI_COMMANDS.includes(sub)) {
    fail(`未知 ui 命令 ${sub}`, { available: ['state', ...UI_COMMANDS] })
  }

  // 除 --json 外的 flag 原样作为命令参数传给页面。
  const args = {}
  for (const [k, v] of Object.entries(flags)) {
    if (k === 'json') continue
    if (k === 'tileIds' || k === 'tile-ids') args.tileIds = String(v).split(',')
    else args[k.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v === 'true' ? true : v
  }
  if (positional.length > 2 && sub === 'gather') args.tileIds = positional.slice(2)
  if (positional.length > 2 && sub === 'add-tile') args.type = positional[2]

  const { status, body } = await control('/command', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ cmd: sub, args }),
  })
  if (status !== 200) fail(body.error ?? '命令失败')
  // body.state 是执行后的快照，agent 不必再单独查一次。
  out(body, () => {
    process.stdout.write(`已执行 ${sub}${body.result ? ` → ${JSON.stringify(body.result)}` : ''}\n`)
    if (body.state) printState(body.state)
  })
}

function printState(s) {
  process.stdout.write(
    `\n模式 ${s.mode}｜Workspace「${s.workspace?.name ?? '—'}」｜Bundle ${s.workspace?.bundleId ?? '未绑定'}\n` +
      `已加载 bundle：${s.bundles.map((b) => `${b.id}(${b.eventCount})`).join(', ') || '（无）'}\n\n`,
  )
  for (const c of s.columns) {
    process.stdout.write(`${c.pinned ? '📌' : '  '} ${c.id}  ${c.title}  [${c.width}]\n`)
    for (const t of c.tiles) {
      process.stdout.write(
        `      ${t.id}  ${t.type}${t.readiness !== 'ready' ? `  ⚠${t.readiness}` : ''}\n`,
      )
    }
  }
}

const UI_COMMANDS = [
  'new-column',
  'remove-column',
  'add-tile',
  'remove-tile',
  'set-width',
  'drill',
  'consume',
  'expel',
  'pin',
  'focus',
  'gather',
  'mode',
  'set-filter',
  'clear-filters',
  'set-bundle',
  'set-baseline',
  'new-workspace',
  'switch-workspace',
  'undo',
  'redo',
]

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

const HELP = `kxc-wb —— 性能分析桌布命令行

读数据（不需要浏览器）
  kxc-wb query <kind> --bundle <目录> [--json] [过滤器]
      kind: ${Object.keys(QUERY_KINDS).join(', ')}
  kxc-wb compare --baseline <目录> --candidate <目录> [--threshold 1.2] [--json]

过滤器（query 与 compare 通用）
  --pass <名>  --op <名>  --kernel <名>  --shape <签名>
  --component <名>  --event-type <名>  --device <名>  --run <id>
  --severity <debug|info|warn|error>  --status <ok|error>  --search <关键词>
  --start-ns <n>  --end-ns <n>

驱动界面（需要控制服务与已打开的页面）
  kxc-wb ui state                                  查看页面当前有哪些列和图
  kxc-wb ui add-tile <类型> [--column-id <id>]
  kxc-wb ui new-column [--title <标题>] [--width 1/3|1/2|2/3|full|fit]
  kxc-wb ui drill --kind pass --value <pass 名> [--as-tab]
  kxc-wb ui gather <tileId> <tileId> ...
  kxc-wb ui set-bundle --bundle-id <id>
  kxc-wb ui set-baseline --bundle-id <id>
  kxc-wb ui mode --mode strip|gather|overview|focus
  kxc-wb ui set-filter --patch '{"pass":"fold_constant"}'
  kxc-wb ui pin --column-id <id>   |   undo   |   redo

约定
  --json 输出单个 JSON 对象，不掺日志，供 agent 直接解析。
  控制服务端口用 KXC_WB_PORT 覆盖（默认 5274）。
`

const cmd = positional[0]
try {
  if (!cmd || cmd === 'help' || flags.help) process.stdout.write(HELP)
  else if (cmd === 'query') cmdQuery()
  else if (cmd === 'compare') cmdCompare()
  else if (cmd === 'ui') await cmdUi()
  else fail(`未知命令 ${cmd}`, { available: ['query', 'compare', 'ui', 'help'] })
} catch (err) {
  fail(err instanceof Error ? err.message : String(err))
}
