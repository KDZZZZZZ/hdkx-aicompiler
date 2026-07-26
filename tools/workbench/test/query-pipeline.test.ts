/**
 * 数据管线测试：拿真实 fixture（含 profile_bundle_test 跑出来的真产物）喂给
 * Worker 的聚合逻辑，验证口径正确。
 *
 * 这是整个工作台里最该被测的一层——图表画错了肉眼能看出来，
 * 但聚合口径错了（比如把嵌套 span 重复计入）只会得到一个看起来很合理的错数字。
 */

import { describe, it, expect, beforeAll } from 'vitest'
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import type {
  KpiResult,
  LoadPayload,
  MetaResult,
  PassRankingResult,
  PhaseBreakdownResult,
  ShapeCacheResult,
  WorkerRequest,
  WorkerResponse,
} from '../src/kxc/query-protocol'

const FIXTURES = join(__dirname, '..', 'fixtures', 'bundles')

/** Worker 模块顶层引用了 self，Node 里要先把它垫上才能 import。 */
const responses: WorkerResponse[] = []
beforeAll(async () => {
  ;(globalThis as unknown as { self: unknown }).self = {
    postMessage: (m: WorkerResponse) => responses.push(m),
    onmessage: null,
  }
  await import('../src/kxc/query.worker')
})

function send(req: WorkerRequest): WorkerResponse {
  responses.length = 0
  const handler = (globalThis as unknown as { self: { onmessage: (e: { data: WorkerRequest }) => void } })
    .self.onmessage
  handler({ data: req })
  const last = responses[responses.length - 1]
  if (!last) throw new Error('worker 没有回应')
  return last
}

function readBundle(name: string): LoadPayload {
  const dir = join(FIXTURES, name)
  const read = (f: string) => (existsSync(join(dir, f)) ? readFileSync(join(dir, f), 'utf8') : null)
  const events = read('events.jsonl')
  if (!events) throw new Error(`fixture ${name} 缺 events.jsonl，先跑 npm run fixture`)
  return {
    manifest: read('manifest.json'),
    events,
    summary: read('summary.json'),
    diagnosis: read('diagnosis.json'),
    trace: read('trace.json'),
    artifacts: {},
  }
}

function load(id: string, name: string): MetaResult {
  const res = send({ type: 'load', id: 1, bundleId: id, files: readBundle(name) })
  if (res.type !== 'loaded') throw new Error(`加载失败: ${JSON.stringify(res)}`)
  return res.meta
}

function query<T>(bundleId: string, spec: WorkerRequest extends { spec: infer S } ? S : never): T {
  const res = send({ type: 'query', id: 2, bundleId, spec, filter: {} } as WorkerRequest)
  if (res.type !== 'result') throw new Error(`查询失败: ${JSON.stringify(res)}`)
  return res.data as T
}

describe('真实产物 real-compile', () => {
  beforeAll(() => load('real', 'real-compile'))

  it('解析出全部 22 个事件且没有损坏行', () => {
    const meta = load('real', 'real-compile')
    expect(meta.eventCount).toBe(22)
    expect(meta.malformedLines).toBe(0)
  })

  it('识别出 relay_pipeline / tir_pipeline 是独立 component', () => {
    const meta = load('real', 'real-compile')
    // 这是真实产物的关键特征：pipeline 汇总事件不在 relay_pass 下。
    expect(meta.componentCounts['relay_pipeline']).toBe(1)
    expect(meta.componentCounts['tir_pipeline']).toBe(1)
    expect(meta.componentCounts['relay_pass']).toBe(10)
  })

  it('阶段耗时用 pipeline 汇总而不是逐 pass 相加，避免嵌套重复计入', () => {
    const r = query<PhaseBreakdownResult>('real', { kind: 'phase_breakdown' })
    const relay = r.items.find((i) => i.phase === 'Relay Pass')
    expect(relay).toBeDefined()
    // pipeline 只有 1 条事件，如果实现错误地把 10 条 run_pass 加起来，count 就会是 10。
    expect(relay!.count).toBe(1)

    const ranking = query<PassRankingResult>('real', { kind: 'pass_ranking', limit: 100 })
    const passSum = ranking.items.reduce((s, i) => s + i.totalNs, 0)
    // 逐 pass 之和必须 <= pipeline 耗时，否则说明把 pipeline 也算进 pass 了。
    expect(passSum).toBeLessThanOrEqual(relay!.durationNs + (r.items.find((i) => i.phase === 'TIR Pass')?.durationNs ?? 0))
  })

  it('这份 bundle 没有 runtime 事件，缓存与 shape 查询必须返回空而不是报错', () => {
    const kpi = query<KpiResult>('real', { kind: 'kpi' })
    expect(kpi.runCount).toBe(0)
    // 分母为 0 时必须是 null，不能是 0——否则 UI 会显示"命中率 0%"这种假结论。
    expect(kpi.cacheHitRate).toBeNull()

    const shape = query<ShapeCacheResult>('real', { kind: 'shape_cache' })
    expect(shape.cells).toEqual([])
  })
})

describe('fixture baseline', () => {
  it('Shape × Cache 能靠 shape_hash 与父 span 补出签名（无需改 C++）', () => {
    load('base', 'baseline')
    const r = query<ShapeCacheResult>('base', { kind: 'shape_cache' })
    expect(r.cells.length).toBeGreaterThan(0)
    // 关键断言：cache_* 事件自身不带 shape_signature，必须全部被解析出来。
    expect(r.unresolved).toBe(0)
    for (const c of r.cells) {
      expect(c.shapeSignature).toMatch(/^\[\d/)
      expect(c.exactHit + c.fuzzyHit + c.miss).toBeGreaterThan(0)
    }
  })

  it('KPI 缓存计数与 shape 热力图总数一致', () => {
    load('base', 'baseline')
    const kpi = query<KpiResult>('base', { kind: 'kpi' })
    const shape = query<ShapeCacheResult>('base', { kind: 'shape_cache' })
    const shapeTotal = shape.cells.reduce((s, c) => s + c.exactHit + c.fuzzyHit + c.miss, 0)
    expect(shapeTotal).toBe(kpi.cacheExactHit + kpi.cacheFuzzyHit + kpi.cacheMiss)
  })

  it('过滤器生效：按 component 过滤后只剩该 component 的事件', () => {
    load('base', 'baseline')
    const res = send({
      type: 'query',
      id: 3,
      bundleId: 'base',
      spec: { kind: 'events', offset: 0, limit: 500 },
      filter: { component: 'relay_pass' },
    })
    if (res.type !== 'result') throw new Error('查询失败')
    const data = res.data as { rows: Array<{ component: string }>; total: number }
    expect(data.total).toBeGreaterThan(0)
    expect(data.rows.every((r) => r.component === 'relay_pass')).toBe(true)
  })
})

describe('candidate 相对 baseline 的回归可被检出', () => {
  it('fold_tuple_get_item 在 candidate 侧明显更慢', () => {
    load('base', 'baseline')
    load('cand', 'candidate')
    const b = query<PassRankingResult>('base', { kind: 'pass_ranking', limit: 100 })
    const c = query<PassRankingResult>('cand', { kind: 'pass_ranking', limit: 100 })
    const bp = b.items.find((i) => i.passName === 'fold_tuple_get_item')
    const cp = c.items.find((i) => i.passName === 'fold_tuple_get_item')
    expect(bp).toBeDefined()
    expect(cp).toBeDefined()
    expect(cp!.totalNs / bp!.totalNs).toBeGreaterThan(2)
  })
})

describe('损坏输入', () => {
  it('半行 JSON 只计入 malformedLines，不会让整个 bundle 加载失败', () => {
    const good = readBundle('baseline')
    const broken: LoadPayload = {
      ...good,
      events: good.events.split('\n').slice(0, 5).join('\n') + '\n{"component":"x", TRUNCATED',
    }
    const res = send({ type: 'load', id: 9, bundleId: 'broken', files: broken })
    expect(res.type).toBe('loaded')
    if (res.type === 'loaded') {
      expect(res.meta.malformedLines).toBe(1)
      expect(res.meta.eventCount).toBe(5)
    }
  })
})
